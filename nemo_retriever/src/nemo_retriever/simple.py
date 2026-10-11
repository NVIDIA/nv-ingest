# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simple, verb-first API for NeMo Retriever.

This module is the friendliest way to use the library. It reads like plain
English and hides every moving part behind a handful of verbs:

    from nemo_retriever import simple as nr

    # See what is inside your files.
    content = nr.extract("reports/")

    # Make your files searchable.
    nr.ingest("reports/", into="my-library")

    # Find the passages that match a question.
    hits = nr.search("my-library", "What were the Q3 results?")

    # Ask a question and get a written answer.
    answer = nr.ask("my-library", "What were the Q3 results?", model="...")

Every verb accepts any supported file - documents, text, web pages, images,
audio, or video - and figures out how to read each one for you. Point a verb at
a single file, a folder, a wildcard pattern, or a list of any of those.

The verbs describe *what happens*, never *how*. Speed, hardware, models, and
storage are chosen for you with sensible defaults so a first call just works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

__all__ = [
    "MEDIA_TYPES",
    "supported_media",
    "extract",
    "extract_documents",
    "extract_text",
    "extract_web_pages",
    "extract_images",
    "extract_audio",
    "extract_video",
    "ingest",
    "search",
    "ask",
]

# The kinds of files you can hand to any verb, described in plain language.
# Each entry maps a friendly media family to the file extensions it covers.
MEDIA_TYPES: Dict[str, Tuple[str, ...]] = {
    "documents": (".pdf", ".docx", ".pptx"),
    "text": (".txt",),
    "web_pages": (".html",),
    "images": (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".svg"),
    "audio": (".mp3", ".wav", ".m4a"),
    "video": (".mp4", ".mov", ".mkv"),
}

# What a source may be: a file, a folder, a wildcard pattern, or a list of them.
Source = Union[str, Path, List[Union[str, Path]]]


def supported_media() -> Dict[str, Tuple[str, ...]]:
    """List every kind of file the library can read.

    Returns a mapping of friendly media family (``"documents"``, ``"images"``,
    ``"audio"``, ...) to the file extensions it accepts, so you can check at a
    glance whether your files are supported.
    """
    return {family: tuple(extensions) for family, extensions in MEDIA_TYPES.items()}


def _as_patterns(source: Source) -> List[str]:
    """Turn a file, folder, pattern, or list of them into concrete file patterns.

    A folder is read all the way through; a single file or wildcard is used as
    given. This lets every verb accept whatever is most convenient for you.
    """
    items = source if isinstance(source, (list, tuple)) else [source]
    patterns: List[str] = []
    for item in items:
        text = str(item)
        if Path(text).is_dir():
            patterns.append(str(Path(text) / "**" / "*"))
        else:
            patterns.append(text)
    return patterns


def _new_ingestor(source: Source) -> Any:
    """Start a fresh ingestion pointed at *source* (kept private on purpose)."""
    from nemo_retriever.ingestor import create_ingestor

    return create_ingestor().files(_as_patterns(source))


# ---------------------------------------------------------------------------
# extract - read the content out of your files
# ---------------------------------------------------------------------------


def extract(source: Source) -> Any:
    """Read your files and return everything found inside them.

    Hand this any supported files - documents, text, web pages, images, audio,
    or video, in any mix - and get back a table of their contents: written
    text, tables, charts, pictures, and spoken words turned into text.

    Use this when you want to see or work with what is in your files. To make
    the content searchable instead, use :func:`ingest`.
    """
    return _new_ingestor(source).extract().ingest()


def extract_documents(source: Source) -> Any:
    """Read documents - PDFs, Word files, and PowerPoint slides.

    Returns the written text, tables, and charts found across every page.
    """
    return _new_ingestor(source).extract(extraction_mode="pdf").ingest()


def extract_text(source: Source) -> Any:
    """Read plain-text files and return their contents."""
    return _new_ingestor(source).extract_txt().ingest()


def extract_web_pages(source: Source) -> Any:
    """Read saved web pages (HTML) and return their readable content."""
    return _new_ingestor(source).extract_html().ingest()


def extract_images(source: Source) -> Any:
    """Read pictures and scans and return the text and content found in them."""
    return _new_ingestor(source).extract_image_files().ingest()


def extract_audio(source: Source) -> Any:
    """Read audio recordings and return the spoken words as text."""
    return _new_ingestor(source).extract_audio().ingest()


def extract_video(source: Source) -> Any:
    """Read videos and return their on-screen text and spoken words as text."""
    return _new_ingestor(source).extract_video().ingest()


# ---------------------------------------------------------------------------
# ingest - make your files searchable
# ---------------------------------------------------------------------------


def ingest(source: Source, *, into: str | None = None) -> Any:
    """Read your files and make their content searchable.

    Works on any supported files in any mix. The content is read and prepared
    so it can be searched by meaning rather than exact words.

    Pass ``into="a-name"`` to save the result under a named library you can
    come back to later with :func:`search` and :func:`ask`. Leave ``into`` out
    to get the prepared content back without saving it anywhere.
    """
    pipeline = _new_ingestor(source).extract().embed()
    if into is not None:
        pipeline = pipeline.vdb_upload(vdb_kwargs={"table_name": into})
    return pipeline.ingest()


# ---------------------------------------------------------------------------
# search / ask - use what you ingested
# ---------------------------------------------------------------------------


def _open_library(index: str) -> Any:
    from nemo_retriever.graph.retriever import Retriever

    return Retriever(vdb_kwargs={"table_name": index})


def search(index: str, question: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Find the passages in a library that best match a question or phrase.

    Point this at a library you created with ``ingest(..., into=...)`` and get
    back the most relevant passages, most relevant first. Use ``limit`` to say
    how many to return.
    """
    return _open_library(index).query(question, top_k=limit)


def ask(index: str, question: str, *, model: str) -> str:
    """Ask a question in plain words and get an answer written from a library.

    The most relevant passages in the library are found for you and used to
    write the answer, so the reply stays grounded in your own content.
    ``model`` names the assistant that writes the answer.
    """
    from nemo_retriever.models.llm.clients.litellm import LiteLLMClient

    library = _open_library(index)
    writer = LiteLLMClient.from_kwargs(model=model)
    return library.answer(question, llm=writer).answer
