from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import base64
import traceback
import pypdfium2.raw as pdfium_c

import pandas as pd

try:
    import pypdfium2 as pdfium
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]
    _PDFIUM_IMPORT_ERROR = e
else:  # pragma: no cover
    _PDFIUM_IMPORT_ERROR = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]



def _render_page_to_base64(page: Any, *, dpi: int = 300, image_format: str = "png") -> Dict[str, Any]:
    """
    Render a page to an image and return base64 plus minimal metadata.

    Returns dict with:
    - image_b64: str
    - encoding: str ("png" or "raw")
    - orig_shape_hw: tuple[int,int] (H,W) of the rendered raster
    - shape/dtype: optional (for raw)
    """
    scale = max(float(dpi) / 72.0, 0.01)
    bitmap = page.render(scale=scale)

    # Prefer numpy output when available.
    arr = None
    if hasattr(bitmap, "to_numpy"):
        try:
            arr = bitmap.to_numpy()
        except Exception:
            arr = None

    if arr is None and hasattr(bitmap, "to_pil"):
        try:
            pil_img = bitmap.to_pil()
        except Exception:
            pil_img = None
        if pil_img is not None:
            w, h = pil_img.size
            buf = BytesIO()
            pil_img.save(buf, format=image_format.upper())
            return {
                "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "encoding": image_format.lower(),
                "orig_shape_hw": (int(h), int(w)),
            }

    # Fallback: if we got a numpy-like array, try encoding with Pillow; else raw bytes.
    if arr is not None and Image is not None:
        try:
            # Expect typical image layouts like (H,W,C) or (H,W).
            h = int(arr.shape[0]) if hasattr(arr, "shape") and len(arr.shape) >= 2 else 0
            w = int(arr.shape[1]) if hasattr(arr, "shape") and len(arr.shape) >= 2 else 0
            pil = Image.fromarray(arr)
            buf = BytesIO()
            pil.save(buf, format=image_format.upper())
            return {
                "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "encoding": image_format.lower(),
                "orig_shape_hw": (int(h), int(w)),
            }
        except Exception:
            pass

    if arr is not None:
        # Last resort: raw bytes base64 (still a base64 string, but not a standard image container).
        raw = arr.tobytes()
        meta: Dict[str, Any] = {"image_b64": base64.b64encode(raw).decode("ascii"), "encoding": "raw"}
        if hasattr(arr, "shape"):
            meta["shape"] = tuple(arr.shape)
            try:
                if len(arr.shape) >= 2:
                    meta["orig_shape_hw"] = (int(arr.shape[0]), int(arr.shape[1]))
            except Exception:
                pass
        if hasattr(arr, "dtype"):
            meta["dtype"] = str(arr.dtype)
        return meta

    raise RuntimeError("Failed to render page to an image representation.")


def _error_record(
    *,
    source_path: Optional[str],
    stage: str,
    exc: BaseException,
    page_number: int = 0,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Return a single output record with the same shape as a normal page record,
    but with error details in metadata.

    This is used to prevent one PDF/page failure from aborting an entire Ray job.
    """
    return {
        "page_number": int(page_number),
        "text": "",
        "page_image": None,
        "images": [],
        "tables": [],
        "charts": [],
        "infographics": [],
        "metadata": {
            "has_text": False,
            "needs_ocr": False,
            "dpi": int(dpi),
            "source_path": source_path,
            "error": {
                "stage": str(stage),
                "type": exc.__class__.__name__,
                "message": str(exc),
                # Keep traceback as a string for debugging; consumers can drop it.
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            },
        },
    }


def _is_scanned_page(page) -> bool:
    tp = page.get_textpage()
    text = tp.get_text_bounded() or ""
    num_chars = len(text.strip())
    num_images = sum(1 for obj in page.get_objects() if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE)

    return num_chars == 0 and num_images > 0

def _extract_page_text(page) -> str:
    """
    Always extract text from the given page and return it as a raw string.
    The caller decides whether to use per-page or doc-level logic.
    """
    textpage = page.get_textpage()
    return textpage.get_text_bounded()

def pdf_extraction(
    pdf_binary: Any,
    extract_text: bool = False,
    extract_images: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    dpi: int = 300,
    image_format: str = "png",
    text_extraction_method: str = "pdfium_hybrid",
    text_depth: str = "page",
    **kwargs: Any) -> Any:
    """
    Here are the steps for pdf extraction that should be implemented:
    1. Load the pdf from the binary data using pypdfium2
    2. Iterate through each page of the pdf using pypdfium2
    3. Extract the text from each page and save each page's text to a list of strings if extract_text is True
    4. If extract_images, extract_tables, extract_charts, or extract_infographics are True, convert the page to a numpy array using pypdfium2 and convert it to a base64 string and save it to a list of strings.
    5. IF extract_text is True but pypdfium2 does not detect text on the page also convert the page to a numpy array using pypdfium2 so that it can later be passed to a OCR model and indicate this in the metadata.
    6. Return a list of dictionaries containing the text, images, tables, charts, infographics, and page numbers.
    """

    # Assumption: PDF splitting ran earlier and produced a dataset where each row
    # contains a *single-page* PDF in the `"bytes"` column. We therefore open the
    # document and only process page 0 for each row.
    if isinstance(pdf_binary, pd.DataFrame):
        if pdfium is None:  # pragma: no cover
            # Best-effort: return error records for the whole batch rather than raising.
            outputs: List[Dict[str, Any]] = []
            for _, row in pdf_binary.iterrows():
                pdf_path = row["path"] if "path" in pdf_binary.columns else None
                outputs.append(
                    _error_record(
                        source_path=str(pdf_path) if pdf_path is not None else None,
                        stage="import_pypdfium2",
                        exc=_PDFIUM_IMPORT_ERROR if _PDFIUM_IMPORT_ERROR is not None else RuntimeError("pypdfium2 unavailable"),
                        page_number=0,
                        dpi=dpi,
                    )
                )
            return outputs

        outputs: List[Dict[str, Any]] = []

        for _, row in pdf_binary.iterrows():
            pdf_bytes = row["bytes"] if "bytes" in pdf_binary.columns else None
            pdf_path = row["path"] if "path" in pdf_binary.columns else None
            page_number = int(row["page_number"]) if "page_number" in pdf_binary.columns else 1

            try:
                if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
                    raise RuntimeError(f"Unsupported bytes payload type: {type(pdf_bytes)!r}")

                # Step 1: load the *single-page* PDF bytes.
                try:
                    doc = pdfium.PdfDocument(pdf_bytes)
                except Exception:
                    doc = pdfium.PdfDocument(BytesIO(bytes(pdf_bytes)))

                # TODO: Extend to support more image formats
                if image_format not in {"png"}:
                    raise ValueError(f"Unsupported image_format: {image_format!r}")

                # Step 2: process only the first page (single-page doc).
                page = None
                try:
                    # we can safely assume page[0] only because pre-splitting has already occurred.
                    page = doc.get_page(0)
                    is_scanned_page = _is_scanned_page(page)

                    ocr_extraction_needed_for_text = extract_text and (
                        (text_extraction_method == "pdfium_hybrid" and is_scanned_page) or text_extraction_method == "ocr"
                    )

                    extraction_needed_for_structured = extract_tables or extract_charts or extract_infographics

                    # Text extraction
                    if extract_text and not ocr_extraction_needed_for_text:
                        page_text = _extract_page_text(page)
                        # TODO: Tiddy up logic here for document depth option
                        if text_depth == "page":
                            text = page_text
                        else:
                            text = page_text

                    has_text = bool(text.strip()) if extract_text else False

                    want_any_raster = bool(
                        extract_images or extract_tables or extract_charts or extract_infographics or ocr_extraction_needed_for_text
                    )
                    render_info: Optional[Dict[str, Any]] = None
                    if want_any_raster:
                        render_info = _render_page_to_base64(page, dpi=dpi, image_format=image_format)

                    page_record: Dict[str, Any] = {
                        "page_number": page_number,
                        "text": text if extract_text else "",
                        "page_image": None,
                        "images": [],
                        "tables": [],
                        "charts": [],
                        "infographics": [],
                        "metadata": {
                            "has_text": has_text,
                            "needs_ocr_for_text": ocr_extraction_needed_for_text,
                            "dpi": dpi,
                            "source_path": pdf_path,
                            "error": None,
                        },
                    }

                    if want_any_raster and render_info is not None:
                        # Store the rendered page raster only here; leave the other
                        # fields empty so downstream stages have a single canonical
                        # place to find the page image.
                        page_record["page_image"] = render_info

                    outputs.append(page_record)
                finally:
                    try:
                        if page is not None and hasattr(page, "close"):
                            page.close()
                    except Exception:
                        pass
                    try:
                        doc.close()
                    except Exception:
                        pass
            except BaseException as e:
                outputs.append(
                    _error_record(
                        source_path=str(pdf_path) if pdf_path is not None else None,
                        stage="page_processing",
                        exc=e,
                        page_number=page_number,
                        dpi=dpi,
                    )
                )

        # Return a batch-shaped dataframe so Ray Data produces one output row per input page.
        return pd.DataFrame(outputs)
    else:
        raise NotImplementedError("pdf_extraction currently only supports pandas.DataFrame input.")


@dataclass(slots=True)
class PDFExtractionActor:
    """
    Skeleton PDF extraction callable.

    `__call__` uses `pdf_extract_config_from_kwargs()` to normalize configuration
    before running the (not yet implemented) extraction logic.
    """

    extract_kwargs: Dict[str, Any]

    def __init__(self, **extract_kwargs: Any) -> None:
        self.extract_kwargs = dict(extract_kwargs)

    def __call__(self, pdf: Any, **override_kwargs: Any) -> Optional[Any]:
        try:
            return pdf_extraction(pdf, **self.extract_kwargs, **override_kwargs)
        except BaseException as e:
            # As a last line of defense, never let the Ray UDF raise.
            source_path = None
            try:
                if isinstance(pdf, pd.DataFrame) and "path" in pdf.columns and len(pdf.index) > 0:
                    source_path = str(pdf.iloc[0]["path"])
            except Exception:
                source_path = None
            return [
                _error_record(
                    source_path=source_path,
                    stage="actor_call",
                    exc=e,
                    page_number=0,
                )
            ]
