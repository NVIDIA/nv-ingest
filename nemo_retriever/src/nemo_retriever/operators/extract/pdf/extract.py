# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import base64
import traceback

try:
    import cv2
except ImportError:
    cv2 = None

from nemo_retriever.common.api.util.metadata.aggregators import Base64Image
from nemo_retriever.common.api.util.pdf.pdfium import (
    convert_bitmap_to_corrected_numpy,
    extract_image_like_objects_from_pdfium_page,
    extract_nested_simple_images_from_pdfium_page,
    is_scanned_page as _is_scanned_page,
)

import nvtx
import pandas as pd

from nemo_retriever.models.nim.error_reporter import report_error
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.operator_archetype import ArchetypeOperator

if TYPE_CHECKING:
    from nemo_retriever.common.params import ExtractParams

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

# Default model input size used by `nemo_retriever.api` for page-element detection.
_MODEL_INPUT_SIZE: Tuple[int, int] = (1024, 1024)

# Allowed render-mode values.
RenderMode = Literal["full_dpi", "fit_to_model"]


def build_pdf_extraction_kwargs(params: ExtractParams) -> dict[str, Any]:
    """Build keyword arguments for ``PDFExtractionActor``.

    Args:
        params: PDF extraction configuration to convert.

    Returns:
        Keyword arguments accepted by ``PDFExtractionActor``.
    """
    return {
        "method": params.method,
        "dpi": int(params.dpi),
        "image_format": params.image_format,
        "jpeg_quality": int(params.jpeg_quality),
        "render_mode": params.render_mode,
        "extract_text": params.extract_text,
        "extract_images": params.extract_images,
        "extract_nested_images": params.extract_nested_images,
        "extract_tables": params.extract_tables,
        "extract_charts": params.extract_charts,
        "extract_infographics": params.extract_infographics,
        "extract_page_as_image": params.extract_page_as_image,
        "api_key": params.api_key,
    }


def _base64_image_to_record(image: Base64Image) -> Dict[str, Any]:
    """Convert an extracted image to the normalized page-record schema.

    Args:
        image: Extracted image and its page-space bounding box.

    Returns:
        Image payload with a normalized bounding box.
    """
    max_width = float(image.max_width) if image.max_width else 1.0
    max_height = float(image.max_height) if image.max_height else 1.0
    x0, y0, x1, y1 = image.bbox
    return {
        "bbox_xyxy_norm": [x0 / max_width, y0 / max_height, x1 / max_width, y1 / max_height],
        "text": "",
        "image_b64": image.image,
    }


def _compute_fit_to_model_scale(
    page: Any,
    target_wh: Tuple[int, int] = _MODEL_INPUT_SIZE,
    max_dpi: int = 300,
) -> float:
    """Compute a pdfium render scale that fits the page within *target_wh* pixels.

    This mirrors the logic in ``nemo_retriever.api.util.pdf.pdfium._compute_render_scale_to_fit``
    combined with the ``min(base_scale, fit_scale)`` cap applied in
    ``pdfium_pages_to_numpy`` when ``scale_tuple`` is provided.

    For a US-Letter page (612×792 pt) fitting into 1024×1024 the result is
    ``min(300/72, min(1024/612, 1024/792)) ≈ 1.293`` → ~93 effective DPI.
    """
    target_w, target_h = target_wh
    page_w = float(page.get_width())
    page_h = float(page.get_height())
    if page_w <= 0 or page_h <= 0 or target_w <= 0 or target_h <= 0:
        return max(float(max_dpi) / 72.0, 0.01)

    fit_scale = max(min(target_w / page_w, target_h / page_h), 1e-3)
    base_scale = max(float(max_dpi) / 72.0, 0.01)
    return min(base_scale, fit_scale)


def _render_page_to_base64(
    page: Any,
    *,
    dpi: int = 200,
    image_format: str = "jpeg",
    jpeg_quality: int = 100,
    render_mode: RenderMode = "fit_to_model",
) -> Dict[str, Any]:
    """Render a page and encode as JPEG or PNG.

    Parameters
    ----------
    render_mode:
        ``"full_dpi"`` – render at *dpi* (default 300 → 2550×3300 for US Letter).
        ``"fit_to_model"`` – render at the `nemo_retriever.api` fit-to-1024 scale (~93 DPI
        for US Letter) so the raster is already close to the model's input size,
        avoiding a large bilinear down-scale in ``resize_pad``.

    Returns dict with:
    - image_b64: str
    - encoding: str ("jpeg" or "png")
    - orig_shape_hw: tuple[int,int] (H,W) of the rendered raster
    """
    if render_mode == "fit_to_model":
        render_scale = _compute_fit_to_model_scale(page, _MODEL_INPUT_SIZE, max_dpi=dpi)
    else:
        render_scale = max(float(dpi) / 72.0, 0.01)
    bitmap = page.render(scale=render_scale)

    arr = convert_bitmap_to_corrected_numpy(bitmap)

    orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])

    # Strip alpha channel (RGBA→RGB) for JPEG compatibility.
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Encode.
    from io import BytesIO
    from PIL import Image as _PILImage

    fmt = image_format.lower()
    if cv2 is not None:
        if fmt == "jpeg":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise RuntimeError("cv2.imencode failed for JPEG")
            encoded_bytes = buf.tobytes()
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError("cv2.imencode failed for PNG")
            encoded_bytes = buf.tobytes()
    else:
        # PIL fallback when opencv-python is not installed (e.g. Intel Mac slim install).
        pil_img = _PILImage.fromarray(arr)
        buf = BytesIO()
        if fmt == "jpeg":
            pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=int(jpeg_quality))
        else:
            pil_img.save(buf, format="PNG", compress_level=3)
        encoded_bytes = buf.getvalue()

    return {
        "image_b64": base64.b64encode(encoded_bytes).decode("ascii"),
        "encoding": fmt,
        "orig_shape_hw": (orig_h, orig_w),
    }


def _error_record(
    *,
    source_path: Optional[str],
    stage: str,
    exc: BaseException,
    page_number: int = 0,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Return a single output record with the same shape as a normal page record,
    but with error details in metadata.

    This is used to prevent one PDF/page failure from aborting an entire Ray job.
    """
    return {
        "path": source_path,
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
    extract_nested_images: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    extract_page_as_image: bool = False,
    dpi: int = 200,
    image_format: str = "jpeg",
    jpeg_quality: int = 100,
    text_extraction_method: str = "pdfium_hybrid",
    text_depth: str = "page",
    render_mode: RenderMode = "fit_to_model",
    **kwargs: Any,
) -> Any:
    """
    Here are the steps for pdf extraction that should be implemented:
    1. Load the pdf from the binary data using pypdfium2
    2. Iterate through each page of the pdf using pypdfium2
    3. Extract the text from each page and save each page's text to a list of strings if extract_text is True
    4. If extract_images, extract_tables, extract_charts, or extract_infographics are True, convert the page to a numpy array using pypdfium2 and convert it to a base64 string and save it to a list of strings.  # noqa: E501, W505
    5. IF extract_text is True but pypdfium2 does not detect text on the page also convert the page to a numpy array using pypdfium2 so that it can later be passed to a OCR model and indicate this in the metadata.  # noqa: E501, W505
    6. Return a list of dictionaries containing the text, images, tables, charts, infographics, and page numbers.
    """

    # Allow callers to pass `method=` (the ExtractParams field name) as an
    # alias for `text_extraction_method`.
    text_extraction_method = kwargs.pop("method", None) or text_extraction_method

    # Assumption: PDF splitting ran earlier and produced a dataset where each row
    # contains a *single-page* PDF in the `"bytes"` column. We therefore open the
    # document and only process page 0 for each row.
    _EXTRACT_EXPLICIT_COLS = frozenset(
        (
            "bytes",
            "path",
            "page_number",
            "source_id",
            "text",
            "page_image",
            "images",
            "tables",
            "charts",
            "infographics",
            "metadata",
        )
    )

    if isinstance(pdf_binary, pd.DataFrame):
        if pdfium is None:  # pragma: no cover
            # Best-effort: return error records for the whole batch rather than raising.
            outputs: List[Dict[str, Any]] = []
            for _, row in pdf_binary.iterrows():
                pdf_path = row["path"] if "path" in pdf_binary.columns else None
                extra = {k: v for k, v in row.to_dict().items() if k not in _EXTRACT_EXPLICIT_COLS}
                err = _error_record(
                    source_path=str(pdf_path) if pdf_path is not None else None,
                    stage="import_pypdfium2",
                    exc=(
                        _PDFIUM_IMPORT_ERROR
                        if _PDFIUM_IMPORT_ERROR is not None
                        else RuntimeError("pypdfium2 unavailable")
                    ),
                    page_number=0,
                    dpi=dpi,
                )
                err.update(extra)
                outputs.append(err)
            return outputs

        outputs: List[Dict[str, Any]] = []

        for _, row in pdf_binary.iterrows():
            pdf_bytes = row["bytes"] if "bytes" in pdf_binary.columns else None
            pdf_path = row["path"] if "path" in pdf_binary.columns else None
            page_number = int(row["page_number"]) if "page_number" in pdf_binary.columns else 1
            source_id = row["source_id"] if "source_id" in pdf_binary.columns else None

            extra = {k: v for k, v in row.to_dict().items() if k not in _EXTRACT_EXPLICIT_COLS}

            try:
                if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
                    raise RuntimeError(f"Unsupported bytes payload type: {type(pdf_bytes)!r}")

                # Step 1: load the *single-page* PDF bytes.
                try:
                    doc = pdfium.PdfDocument(pdf_bytes)
                except Exception:
                    doc = pdfium.PdfDocument(BytesIO(bytes(pdf_bytes)))

                # TODO: Extend to support more image formats
                if image_format not in {"png", "jpeg"}:
                    raise ValueError(f"Unsupported image_format: {image_format!r}")

                # Step 2: process only the first page (single-page doc).
                page = None
                try:
                    # we can safely assume page[0] only because pre-splitting has already occurred.
                    page = doc.get_page(0)
                    is_scanned_page = _is_scanned_page(page)

                    ocr_extraction_needed_for_text = extract_text and (
                        (text_extraction_method == "pdfium_hybrid" and is_scanned_page)
                        or text_extraction_method == "ocr"
                    )

                    # extraction_needed_for_structured = (
                    #     extract_tables or extract_charts or extract_infographics
                    # )  # noqa: F841

                    # Default to empty so scanned/OCR pages don't hit a NameError below.
                    text = ""

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
                        extract_images
                        or extract_tables
                        or extract_charts
                        or extract_infographics
                        or extract_page_as_image
                        or ocr_extraction_needed_for_text
                    )
                    render_info: Optional[Dict[str, Any]] = None
                    if want_any_raster:
                        render_info = _render_page_to_base64(
                            page,
                            dpi=dpi,
                            image_format=image_format,
                            jpeg_quality=jpeg_quality,
                            render_mode=render_mode,
                        )

                    # Extract cropped images from pdfium page objects.
                    detected_images: List[Dict[str, Any]] = []
                    nested_image_error: Optional[Dict[str, Any]] = None
                    if extract_images:
                        try:
                            base64_images = extract_image_like_objects_from_pdfium_page(page)
                            detected_images.extend(_base64_image_to_record(img) for img in base64_images)
                        except Exception:
                            pass  # Image extraction failure should not crash the pipeline.

                        if extract_nested_images:
                            try:
                                nested_images = extract_nested_simple_images_from_pdfium_page(page)
                            except Exception as e:
                                report_error("pdf_extraction:nested_images", e)
                                nested_image_error = {
                                    "stage": "nested_images",
                                    "type": e.__class__.__name__,
                                    "message": str(e),
                                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                                }
                            else:
                                detected_images.extend(_base64_image_to_record(img) for img in nested_images)

                    page_record: Dict[str, Any] = {
                        "path": pdf_path,
                        "page_number": page_number,
                        "source_id": source_id,
                        "text": text if extract_text else "",
                        "page_image": None,
                        "images": detected_images,
                        "tables": [],
                        "charts": [],
                        "infographics": [],
                        "metadata": {
                            "has_text": has_text,
                            "needs_ocr_for_text": ocr_extraction_needed_for_text,
                            "dpi": dpi,
                            "source_path": pdf_path,
                            "error": nested_image_error,
                        },
                    }

                    if want_any_raster and render_info is not None:
                        # Store the rendered page raster only here; leave the other
                        # fields empty so downstream stages have a single canonical
                        # place to find the page image.
                        page_record["page_image"] = render_info

                    page_record.update(extra)
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
                report_error("pdf_extraction:page_processing", e)
                err = _error_record(
                    source_path=str(pdf_path) if pdf_path is not None else None,
                    stage="page_processing",
                    exc=e,
                    page_number=page_number,
                    dpi=dpi,
                )
                err.update(extra)
                outputs.append(err)

        # Return a batch-shaped dataframe so Ray Data produces one output row per input page.
        return pd.DataFrame(outputs)
    else:
        raise NotImplementedError("pdf_extraction currently only supports pandas.DataFrame input.")


@designer_component(
    name="PDF Extractor",
    category="Document Processing",
    compute="gpu",
    description="Extracts text, tables, and images from PDF pages",
    category_color="#64b4ff",
)
class PDFExtractionCPUActor(AbstractOperator, CPUOperator):
    """
    Skeleton PDF extraction callable.

    `__call__` uses `pdf_extract_config_from_kwargs()` to normalize configuration
    before running the (not yet implemented) extraction logic.
    """

    def __init__(self, **extract_kwargs: Any) -> None:
        super().__init__(**extract_kwargs)
        self.extract_kwargs = dict(extract_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::pdf_extract.batch", color="blue")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return pdf_extraction(data, **self.extract_kwargs, **kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pdf: Any, **override_kwargs: Any) -> Optional[Any]:
        try:
            return self.run(pdf, **override_kwargs)
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


class PDFExtractionActor(ArchetypeOperator):
    _cpu_variant_class = PDFExtractionCPUActor

    def __init__(self, **extract_kwargs: Any) -> None:
        super().__init__(**extract_kwargs)
