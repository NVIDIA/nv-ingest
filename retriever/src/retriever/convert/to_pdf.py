"""Convert DOCX/PPTX files to PDF bytes via LibreOffice headless."""

from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx"})


def _error_record(
    *,
    source_path: Optional[str],
    stage: str,
    exc: BaseException,
) -> Dict[str, Any]:
    return {
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


def convert_to_pdf_bytes(file_bytes: bytes, extension: str) -> bytes:
    """Convert file bytes to PDF bytes.

    If *extension* is ``".pdf"``, return *file_bytes* unchanged.
    For ``".docx"`` / ``".pptx"``, write to a temp dir, invoke
    ``libreoffice --headless --convert-to pdf``, and return the
    resulting PDF bytes.

    Raises
    ------
    FileNotFoundError
        If the ``libreoffice`` binary is not on ``$PATH``.
    subprocess.CalledProcessError
        If LibreOffice conversion fails.
    RuntimeError
        If the expected PDF output file is missing after conversion.
    """
    ext = extension.lower()
    if ext == ".pdf":
        return file_bytes

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {extension!r}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Strip leading dot for the filename suffix (e.g. ".docx" -> "docx").
        input_path = os.path.join(tmp_dir, f"input{ext}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        command = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            input_path,
            "--outdir",
            tmp_dir,
        ]

        subprocess.run(command, check=True, capture_output=True, text=True)

        pdf_path = os.path.join(tmp_dir, "input.pdf")
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"LibreOffice conversion produced no output for {extension} file")

        with open(pdf_path, "rb") as f:
            return f.read()


def convert_batch_to_pdf(batch_df: Any) -> pd.DataFrame:
    """Convert a batch of files to PDF, passing PDFs through unchanged.

    Expects a :class:`pandas.DataFrame` with at least ``bytes`` and ``path``
    columns (the same schema produced by ``ray.data.read_binary_files``).
    Rows whose path ends with a supported non-PDF extension are converted;
    rows that are already PDFs are returned as-is.  On error, an error record
    is emitted (matching the pattern in ``pdf/split.py``).
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("convert_batch_to_pdf currently only supports pandas.DataFrame input.")

    out_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        file_path = row.get("path") or ""
        file_bytes = row.get("bytes", b"")

        ext = os.path.splitext(file_path)[1].lower() if file_path else ".pdf"

        if ext not in SUPPORTED_EXTENSIONS:
            # Unsupported extension — pass through unchanged (let downstream
            # stages handle or error).
            out_rows.append({"bytes": file_bytes, "path": file_path})
            continue

        if ext == ".pdf":
            out_rows.append({"bytes": file_bytes, "path": file_path})
            continue

        try:
            if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
                raise ValueError(f"Unsupported bytes payload type: {type(file_bytes)!r}")
            pdf_bytes = convert_to_pdf_bytes(bytes(file_bytes), ext)
            # Preserve original path so downstream metadata tracks the source file.
            out_rows.append({"bytes": pdf_bytes, "path": file_path})
        except BaseException as e:
            out_rows.append(
                _error_record(
                    source_path=str(file_path) if file_path else None,
                    stage="convert_to_pdf",
                    exc=e,
                )
            )

    return pd.DataFrame(out_rows)


@dataclass(slots=True)
class DocToPdfConversionActor:
    """Ray Data actor that converts DOCX/PPTX batches to PDF.

    Used with ``ray.data.Dataset.map_batches`` in the same style as
    ``PDFSplitActor``.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, batch_df: Any) -> Any:
        return convert_batch_to_pdf(batch_df)
