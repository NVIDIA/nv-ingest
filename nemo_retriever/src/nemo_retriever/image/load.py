# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core image loading: convert standalone image files into the PDF page DataFrame schema.

A standalone image is conceptually a single PDF page.  The functions here
produce a one-row DataFrame with exactly the same columns as
``nemo_retriever.pdf.extract.pdf_extraction()`` so that all downstream GPU
stages (page-element detection, OCR, table/chart/infographic extraction)
work without modification.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Dict

import pandas as pd
from PIL import Image

# Raster formats handled natively by Pillow.
_RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# SVG requires cairosvg (optional dependency).
_SVG_EXTENSIONS = {".svg"}

SUPPORTED_IMAGE_EXTENSIONS = _RASTER_EXTENSIONS | _SVG_EXTENSIONS

# Default DPI value matching the PDF extraction pipeline default.
_DEFAULT_DPI = 200


def _error_record(path: str, error: str) -> pd.DataFrame:
    """Return a single-row DataFrame signalling a load error."""
    return pd.DataFrame(
        [
            {
                "path": path,
                "page_number": 1,
                "source_id": None,
                "text": "",
                "page_image": None,
                "images": [],
                "tables": [],
                "charts": [],
                "infographics": [],
                "metadata": {
                    "has_text": False,
                    "needs_ocr_for_text": False,
                    "dpi": _DEFAULT_DPI,
                    "source_path": path,
                    "error": error,
                },
            }
        ]
    )


def _svg_to_pil(content_bytes: bytes) -> Image.Image:
    """Convert SVG bytes to a PIL RGB Image via cairosvg."""
    try:
        import cairosvg  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("cairosvg is required for SVG image support. " "Install it with: pip install cairosvg")
    png_bytes = cairosvg.svg2png(bytestring=content_bytes)
    return Image.open(BytesIO(png_bytes)).convert("RGB")


def image_bytes_to_pages_df(content_bytes: bytes, path: str) -> pd.DataFrame:
    """
    Convert raw image bytes into a single-row DataFrame matching the PDF extraction schema.

    Parameters
    ----------
    content_bytes : bytes
        Raw file content.
    path : str
        Original file path (used for format detection via extension and metadata).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: path, page_number, source_id, text,
        page_image, images, tables, charts, infographics, metadata.
    """
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext in _SVG_EXTENSIONS:
            img = _svg_to_pil(content_bytes)
        elif ext in _RASTER_EXTENSIONS:
            img = Image.open(BytesIO(content_bytes)).convert("RGB")
        else:
            return _error_record(path, f"Unsupported image format: {ext}")

        # Encode as PNG for downstream stages (matches PDF pipeline image encoding).
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        h, w = img.height, img.width

        page_image: Dict[str, Any] = {
            "image_b64": image_b64,
            "encoding": "png",
            "orig_shape_hw": (h, w),
        }

        page_record: Dict[str, Any] = {
            "path": path,
            "page_number": 1,
            "source_id": None,
            "text": "",
            "page_image": page_image,
            "images": [],
            "tables": [],
            "charts": [],
            "infographics": [],
            "metadata": {
                "has_text": False,
                "needs_ocr_for_text": True,
                "dpi": _DEFAULT_DPI,
                "source_path": path,
                "error": None,
            },
        }

        return pd.DataFrame([page_record])

    except ImportError:
        raise
    except Exception as exc:
        return _error_record(path, f"{type(exc).__name__}: {exc}")


def image_file_to_pages_df(path: str) -> pd.DataFrame:
    """
    Read an image file from disk and convert to a single-row page DataFrame.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame matching the PDF extraction output schema.
    """
    abs_path = os.path.abspath(path)
    try:
        with open(abs_path, "rb") as f:
            content_bytes = f.read()
    except Exception as exc:
        return _error_record(abs_path, f"{type(exc).__name__}: {exc}")
    return image_bytes_to_pages_df(content_bytes, abs_path)
