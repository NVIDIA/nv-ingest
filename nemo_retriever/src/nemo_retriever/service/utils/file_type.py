# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify uploaded files by filename suffix and reject unsupported types."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from fastapi import HTTPException, UploadFile

from nemo_retriever.common.input_files import INPUT_FORMAT_BY_SUFFIX
from nemo_retriever.common.inline_text import is_inline_text_source
from nemo_retriever.common.schemas.base import RichModel


class FileCategory(str, Enum):
    """High-level media category for an ingested file."""

    DOCUMENT = "document"
    TEXT = "text"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


_CATEGORY_BY_INPUT_TYPE: dict[str, FileCategory] = {
    "pdf": FileCategory.DOCUMENT,
    "doc": FileCategory.DOCUMENT,
    "txt": FileCategory.TEXT,
    "html": FileCategory.HTML,
    "image": FileCategory.IMAGE,
    "audio": FileCategory.AUDIO,
    "video": FileCategory.VIDEO,
}


class FileClassification(RichModel):
    """Result of classifying an uploaded file."""

    filename: str
    suffix: str
    category: FileCategory
    content_type: str


class FileClassifier:
    """Identify uploaded file types from filename suffix.

    Supported extensions and media types are derived from the canonical
    definitions in :mod:`nemo_retriever.common.input_files`.
    """

    SUFFIX_MAP: ClassVar[dict[str, tuple[FileCategory, str]]] = {
        suffix: (_CATEGORY_BY_INPUT_TYPE[item.input_type], item.content_type)
        for suffix, item in INPUT_FORMAT_BY_SUFFIX.items()
    }

    @classmethod
    def supported_extensions(cls) -> frozenset[str]:
        return frozenset(cls.SUFFIX_MAP)

    @classmethod
    def classify(cls, upload: UploadFile, *, filename_override: str = "") -> FileClassification:
        """Determine file category from suffix, raising 415 for unsupported types.

        Parameters
        ----------
        upload:
            The FastAPI ``UploadFile`` from the request.
        filename_override:
            If provided, use this instead of ``upload.filename`` (useful for
            the ``/ingest/page`` endpoint where the original document filename
            is submitted separately).
        """
        name = filename_override or upload.filename or "unknown"
        dot = name.rfind(".")
        suffix = name[dot:].lower() if dot != -1 else ""

        entry = cls.SUFFIX_MAP.get(suffix)
        if entry is None:
            supported = ", ".join(sorted(cls.SUFFIX_MAP))
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{suffix or '(none)'}' for file '{name}'. "
                    f"Supported extensions: {supported}"
                ),
            )

        category, content_type = entry
        return FileClassification(
            filename=name,
            suffix=suffix,
            category=category,
            content_type=content_type,
        )


_MEDIA_CATEGORIES: frozenset[FileCategory] = frozenset({FileCategory.AUDIO, FileCategory.VIDEO})


def infer_extraction_mode_from_filename(filename: str) -> str | None:
    """Map a filename suffix to a GraphIngestor ``extraction_mode`` string.

    Returns ``"text"`` / ``"html"`` / ``"image"`` / ``"audio"`` / ``"video"``
    / ``"pdf"`` for known extensions, or ``None`` when the suffix is not in
    :attr:`FileClassifier.SUFFIX_MAP`. Used by the service worker to avoid
    routing text-like uploads through the PDF or audio-only graphs when the
    client leaves ``extraction_mode`` at the default ``"auto"``.
    """
    if is_inline_text_source(filename):
        return "text"

    dot = filename.rfind(".")
    suffix = filename[dot:].lower() if dot != -1 else ""
    entry = FileClassifier.SUFFIX_MAP.get(suffix)
    if entry is None:
        return None
    category, _ = entry
    if category == FileCategory.TEXT:
        return "text"
    if category == FileCategory.HTML:
        return "html"
    if category == FileCategory.IMAGE:
        return "image"
    if category == FileCategory.AUDIO:
        return "audio"
    if category == FileCategory.VIDEO:
        return "video"
    if category == FileCategory.DOCUMENT:
        return "pdf"
    return None


def is_text_like_filename(filename: str) -> bool:
    """True when *filename* is a supported plain-text or HTML ingest type."""
    mode = infer_extraction_mode_from_filename(filename)
    return mode in {"text", "html"}


def category_requires_media_deps(category: FileCategory) -> bool:
    """True when *category* needs ``ffmpeg``/``ffprobe`` to ingest.

    Only audio and video uploads exercise the ``MediaChunkActor`` /
    ``MediaInterface`` code paths that shell out to ``ffmpeg`` and
    ``ffprobe``. PDF / image / text / HTML uploads are unaffected by
    media-dependency availability.
    """
    return category in _MEDIA_CATEGORIES


def enforce_media_dependencies(classification: FileClassification) -> None:
    """Reject media uploads up-front when the container is missing FFmpeg.

    Translates what would otherwise surface as a Ray worker crash
    (``RuntimeError: MediaChunkActor requires media dependencies; missing:
    ffmpeg, ffprobe``) into an HTTP 501 with an actionable Helm value
    and ``apt-get`` command. The check is local to this process — the
    gateway, realtime and batch pods all share the same container image,
    so an inconsistency between them is not possible under the standard
    chart layout.

    Audio / video ingestion only — other file categories are passed
    through without invoking the FFmpeg probe.
    """
    if not category_requires_media_deps(classification.category):
        return

    from nemo_retriever.common.modality.audio.media_interface import (
        HELM_FFMPEG_INSTALL_VALUE,
        MANUAL_FFMPEG_INSTALL_COMMAND,
        is_media_available,
        missing_media_dependencies,
    )

    if is_media_available():
        return

    missing = ", ".join(missing_media_dependencies()) or "ffmpeg, ffprobe"
    raise HTTPException(
        status_code=501,
        detail=(
            f"Audio and video ingestion require FFmpeg in the retriever "
            f"service container, but the following dependencies are "
            f"missing: {missing}. Re-deploy the Helm chart with "
            f"`--set {HELM_FFMPEG_INSTALL_VALUE}` to install FFmpeg at "
            f"container startup, install it manually inside the container "
            f"with `{MANUAL_FFMPEG_INSTALL_COMMAND}`, or build a custom "
            f"image that already includes ffmpeg/ffprobe (recommended for "
            f"air-gapped clusters). See the Helm chart README "
            f'("Audio / video extraction") for details. File: '
            f"{classification.filename!r}."
        ),
    )
