# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
File type routing for the online ingest server.

Maps file extensions / MIME types to the appropriate byte-based DataFrame loader
so that the pipeline can handle PDF, TXT, HTML, images, audio, and office docs.
"""

from __future__ import annotations

import os
import tempfile
from typing import Literal, Optional

import pandas as pd

FileCategory = Literal["pdf", "txt", "html", "image", "audio"]

_EXT_TO_CATEGORY: dict[str, FileCategory] = {
    ".pdf": "pdf",
    ".docx": "pdf",
    ".pptx": "pdf",
    ".txt": "txt",
    ".md": "txt",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".svg": "image",
    ".gif": "image",
    ".webp": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".mp4": "audio",
    ".mov": "audio",
    ".avi": "audio",
    ".mkv": "audio",
}

_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def detect_file_category(filename: str) -> FileCategory:
    ext = os.path.splitext(filename)[1].lower()
    category = _EXT_TO_CATEGORY.get(ext)
    if category is None:
        raise ValueError(f"Unsupported file extension: {ext!r} (filename={filename!r})")
    return category


def mime_for_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_TO_MIME.get(ext, "application/octet-stream")


def bytes_to_df(
    content_bytes: bytes,
    filename: str,
    category: Optional[FileCategory] = None,
) -> pd.DataFrame:
    """Convert raw file bytes into the initial DataFrame expected by the pipeline.

    Each file category maps to a specific byte-based loader that already exists
    in the codebase. For audio, which requires a file path for ffmpeg, a temp
    file is written and cleaned up after loading.
    """
    if category is None:
        category = detect_file_category(filename)

    if category == "pdf":
        return _load_pdf_bytes(content_bytes, filename)
    if category == "txt":
        return _load_txt_bytes(content_bytes, filename)
    if category == "html":
        return _load_html_bytes(content_bytes, filename)
    if category == "image":
        return _load_image_bytes(content_bytes, filename)
    if category == "audio":
        return _load_audio_bytes(content_bytes, filename)
    raise ValueError(f"Unknown file category: {category!r}")


def _load_pdf_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".docx", ".pptx"):
        from nemo_retriever.utils.convert import convert_to_pdf_bytes

        content_bytes = convert_to_pdf_bytes(content_bytes, ext)

    from nemo_retriever.ingest_modes.inprocess import pages_df_from_pdf_bytes

    return pages_df_from_pdf_bytes(content_bytes, filename)


def _load_txt_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    from nemo_retriever.txt.split import txt_bytes_to_chunks_df

    return txt_bytes_to_chunks_df(content_bytes, filename)


def _load_html_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    from nemo_retriever.html.convert import html_bytes_to_chunks_df

    return html_bytes_to_chunks_df(content_bytes, filename)


def _load_image_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    from nemo_retriever.image.load import image_bytes_to_pages_df

    return image_bytes_to_pages_df(content_bytes, filename)


def _load_audio_bytes(content_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        os.write(tmp_fd, content_bytes)
        os.close(tmp_fd)
        from nemo_retriever.audio.chunk_actor import audio_path_to_chunks_df

        return audio_path_to_chunks_df(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
