"""gpu_pdfium — pypdfium2-compatible drop-in backed by the native _gpu_pdfium (PDFium in C++).

Exposes the subset of the pypdfium2 surface the nemo_retriever codebase uses:
  PdfDocument, PdfPage, PdfTextPage, PdfBitmap, PdfImage (alias), PdfiumError, and `raw`.

Drop-in activation (no source edits): call `activate()` BEFORE nemo_retriever imports pypdfium2,
or rely on `NEMO_PDF_BACKEND=gpu` via `maybe_activate_from_env()`. It injects this module as
`pypdfium2` and `pypdfium2.raw` into sys.modules.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# --- locate and import the native extension ----------------------------------
try:
    import _gpu_pdfium as _core
except ImportError:
    # Fall back to the in-tree CMake build dir.
    _build = Path(__file__).resolve().parents[2] / "native" / "build"
    if _build.is_dir():
        sys.path.insert(0, str(_build))
    import _gpu_pdfium as _core  # noqa: E402

from . import raw  # noqa: E402

PdfDocument = _core.PdfDocument
PdfPage = _core.PdfPage
PdfTextPage = _core.PdfTextPage
PdfBitmap = _core.PdfBitmap
PdfiumError = _core.PdfiumError
# pypdfium2 distinguishes PdfImage; our page objects carry the same methods, so alias it.
PdfImage = _core.PdfPageObject
PdfPageObject = _core.PdfPageObject

# Cosmetic parity with `pypdfium2.PDFIUM_INFO`.
PDFIUM_INFO = "gpu_pdfium (PDFium C++ via nanobind)"

__all__ = [
    "PdfDocument", "PdfPage", "PdfTextPage", "PdfBitmap", "PdfImage", "PdfPageObject",
    "PdfiumError", "raw", "PDFIUM_INFO", "activate", "maybe_activate_from_env",
]


def activate() -> None:
    """Inject this module as `pypdfium2` (and `.raw`) so existing imports resolve here."""
    sys.modules["pypdfium2"] = sys.modules[__name__]
    sys.modules["pypdfium2.raw"] = raw


def maybe_activate_from_env() -> bool:
    """Activate iff NEMO_PDF_BACKEND=gpu. Returns whether activation happened."""
    if os.environ.get("NEMO_PDF_BACKEND", "pypdfium").strip().lower() == "gpu":
        activate()
        return True
    return False
