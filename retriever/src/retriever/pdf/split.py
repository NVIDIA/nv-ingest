# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import traceback

import pandas as pd
from retriever.params import PdfSplitParams

try:
    import pypdfium2 as pdfium
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]
    _PDFIUM_IMPORT_ERROR = e
else:  # pragma: no cover
    _PDFIUM_IMPORT_ERROR = None


def _error_record(
    *,
    source_path: Optional[str],
    stage: str,
    exc: BaseException,
    page_number: int = 0,
) -> Dict[str, Any]:
    return {
        "page_number": int(page_number),
        "bytes": b"",
        "path": source_path,
        "metadata": {
            "source_path": source_path,
            "error": {
                "stage": str(stage),
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            },
        },
    }


def _split_pdf_to_single_page_bytes(pdf_binary: Any) -> List[bytes]:
    """
    Split a PDF into single-page PDFs (raw bytes) using pypdfium2.
    """
    if pdfium is None:  # pragma: no cover
        raise ImportError(
            "pypdfium2 is required for PDF splitting but could not be imported."
        ) from _PDFIUM_IMPORT_ERROR

    try:
        doc = pdfium.PdfDocument(pdf_binary)
    except Exception:
        if isinstance(pdf_binary, (bytes, bytearray, memoryview)):
            doc = pdfium.PdfDocument(BytesIO(bytes(pdf_binary)))
        else:
            raise

    out: List[bytes] = []
    try:
        for page_idx in range(len(doc)):
            single = pdfium.PdfDocument.new()
            try:
                single.import_pages(doc, pages=[page_idx])
                buf = BytesIO()
                single.save(buf)
                out.append(buf.getvalue())
            finally:
                try:
                    single.close()
                except Exception:
                    pass
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return out


def pdf_path_to_pages_df(path: str) -> pd.DataFrame:
    """
    Convert a multi-page PDF at `path` into a DataFrame where each row
    contains a single-page PDF's raw bytes.

    Columns: bytes, path, page_number (1-indexed).
    Compatible with pdf_extraction and downstream inprocess/batch stages.
    """
    if pdfium is None:  # pragma: no cover
        raise ImportError(
            "pypdfium2 is required for PDF splitting but could not be imported."
        ) from _PDFIUM_IMPORT_ERROR

    abs_path = str(Path(path).resolve())
    out_rows: List[Dict[str, Any]] = []
    try:
        raw_bytes = Path(abs_path).read_bytes()
        pages = _split_pdf_to_single_page_bytes(raw_bytes)
        for page_idx, page_bytes in enumerate(pages):
            out_rows.append(
                {
                    "bytes": page_bytes,
                    "path": abs_path,
                    "page_number": page_idx + 1,
                }
            )
    except BaseException as e:
        out_rows.append({"bytes": b"", "path": abs_path, "page_number": 0, "error": str(e)})
    return pd.DataFrame(out_rows)


def split_pdf_batch(pdf_batch: Any, params: PdfSplitParams | None = None) -> pd.DataFrame:
    """
    Split a batch of PDFs into per-page single-page PDFs (bytes), without rendering.

    Expected input is a pandas DataFrame batch containing at least a bytes-like
    column `"bytes"` (and optionally `"path"`).
    """
    if not isinstance(pdf_batch, pd.DataFrame):
        raise NotImplementedError("split_pdf_batch currently only supports pandas.DataFrame input.")

    # Optional bounds (1-indexed inclusive).
    split_params = params or PdfSplitParams()
    start_page = split_params.start_page
    end_page = split_params.end_page

    out_rows: List[Dict[str, Any]] = []
    for _, row in pdf_batch.iterrows():
        pdf_path = row["path"] if "path" in pdf_batch.columns else None
        pdf_bytes = row["bytes"] if "bytes" in pdf_batch.columns else None

        try:
            if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
                raise ValueError(f"Unsupported bytes payload type: {type(pdf_bytes)!r}")

            pages = _split_pdf_to_single_page_bytes(pdf_bytes)
            start_idx = 0 if start_page is None else max(int(start_page) - 1, 0)
            end_idx = (len(pages) - 1) if end_page is None else min(int(end_page) - 1, len(pages) - 1)
            if len(pages) == 0 or start_idx > end_idx:
                continue

            for page_idx in range(start_idx, end_idx + 1):
                out_rows.append(
                    {
                        "bytes": pages[page_idx],
                        "path": pdf_path,
                        "page_number": page_idx + 1,
                        "metadata": {"source_path": pdf_path},
                    }
                )
        except BaseException as e:
            out_rows.append(
                _error_record(
                    source_path=str(pdf_path) if pdf_path is not None else None,
                    stage="split_pdf",
                    exc=e,
                    page_number=0,
                )
            )

    return pd.DataFrame(out_rows)


@dataclass(slots=True)
class PDFSplitActor:
    split_params: PdfSplitParams

    def __init__(self, split_params: PdfSplitParams | None = None) -> None:
        self.split_params = split_params or PdfSplitParams()

    def __call__(self, pdf_batch: Any) -> Any:
        return split_pdf_batch(pdf_batch, params=self.split_params)


def split_pdf(pdf_ds: Any, params: PdfSplitParams | None = None) -> Any:
    """
    Dataset-level splitter.

    Takes a Ray Dataset of PDFs (rows with `"bytes"` and optionally `"path"`) and
    returns a new Ray Dataset with one row per page (single-page PDF bytes).
    """
    try:
        import ray.data as rd  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("split_pdf() requires Ray Data (`ray`).") from e

    # Note: returning a Dataset here creates the new dataset representing pages.
    return pdf_ds.map_batches(PDFSplitActor(split_params=params), batch_format="pandas")
