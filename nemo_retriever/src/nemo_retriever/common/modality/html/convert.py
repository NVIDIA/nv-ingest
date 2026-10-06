# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTML to markdown conversion via markitdown, then tokenizer-based chunking.

Produces chunk DataFrames compatible with embed_text_from_primitives_df
and the LanceDB row builder (text, path, page_number, metadata).
"""

from __future__ import annotations

import io
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from nemo_retriever.common.params import HtmlChunkParams

from nemo_retriever.common.modality.txt.split import (
    DEFAULT_TOKENIZER_MODEL_ID,
    split_text_by_tokens,
)
from nemo_retriever.common.modality.txt.split import _get_tokenizer as _get_txt_tokenizer


class _HTMLTextFallbackParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def _html_plain_text_fallback(html_text: str) -> str:
    parser = _HTMLTextFallbackParser()
    parser.feed(html_text)
    parser.close()
    return "\n\n".join(parser.parts)


def _is_existing_file_path(value: str) -> bool:
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _read_html_file(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def html_to_markdown(html_content: Union[str, bytes, Path]) -> str:
    """
    Convert HTML to markdown using markitdown.

    Parameters
    ----------
    html_content : str | bytes | Path
        HTML as a string, bytes, or path to an .html file.

    Returns
    -------
    str
        Markdown text.
    """
    from markitdown import MarkItDown, StreamInfo

    md = MarkItDown()
    if isinstance(html_content, Path):
        result = md.convert(str(html_content))
        fallback_text = _read_html_file(html_content)
    elif isinstance(html_content, str):
        if _is_existing_file_path(html_content):
            result = md.convert(html_content)
            fallback_text = _read_html_file(html_content)
        else:
            result = md.convert_stream(
                io.BytesIO(html_content.encode("utf-8", errors="replace")),
                stream_info=StreamInfo(
                    mimetype="text/html",
                    extension=".html",
                    charset="utf-8",
                ),
            )
            fallback_text = html_content
    elif isinstance(html_content, bytes):
        result = md.convert_stream(io.BytesIO(html_content))
        fallback_text = html_content.decode("utf-8", errors="replace")
    else:
        result = md.convert(html_content)
        fallback_text = str(html_content)
    markdown_text = result.text_content or ""
    return markdown_text if markdown_text.strip() else _html_plain_text_fallback(fallback_text)


def html_file_to_chunks_df(
    path: str,
    params: HtmlChunkParams | None = None,
) -> pd.DataFrame:
    chunk_params = params or HtmlChunkParams()
    max_tokens = chunk_params.max_tokens
    overlap_tokens = chunk_params.overlap_tokens
    tokenizer_model_id = chunk_params.tokenizer_model_id
    tokenizer_cache_dir = chunk_params.tokenizer_cache_dir
    encoding = chunk_params.encoding

    """
    Read an .html file, convert to markdown via markitdown, chunk by tokens.

    Columns: text, path, page_number (chunk index, 1-based), metadata.
    Shape is compatible with embed_text_from_primitives_df and LanceDB row build.

    Parameters
    ----------
    path : str
        Path to the .html file.
    max_tokens : int
        Max tokens per chunk (default 512).
    overlap_tokens : int
        Overlap between consecutive chunks (default 0).
    tokenizer_model_id : str, optional
        HuggingFace model id for tokenizer (default: same as embedder).
    encoding : str
        File encoding (default utf-8).
    tokenizer_cache_dir : str, optional
        HuggingFace cache directory for tokenizer.

    Returns
    -------
    pd.DataFrame
        Columns: text, path, page_number, metadata.
    """
    path = str(Path(path).resolve())
    html_text = Path(path).read_text(encoding=encoding, errors="replace")
    markdown_text = html_to_markdown(html_text)
    return _markdown_to_chunks_df(
        markdown_text,
        path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )


def html_bytes_to_chunks_df(
    content_bytes: bytes,
    path: str,
    params: HtmlChunkParams | None = None,
) -> pd.DataFrame:
    chunk_params = params or HtmlChunkParams()
    max_tokens = chunk_params.max_tokens
    overlap_tokens = chunk_params.overlap_tokens
    tokenizer_model_id = chunk_params.tokenizer_model_id
    tokenizer_cache_dir = chunk_params.tokenizer_cache_dir
    encoding = chunk_params.encoding

    """
    Convert HTML bytes to markdown and return a DataFrame of chunks (same shape as html_file_to_chunks_df).

    Used by batch HtmlSplitActor when input is bytes + path from read_binary_files.
    """
    path = path if "://" in path else str(Path(path).resolve())
    html_text = content_bytes.decode(encoding, errors="replace")
    markdown_text = html_to_markdown(html_text)
    return _markdown_to_chunks_df(
        markdown_text,
        path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )


def _markdown_to_chunks_df(
    markdown_text: str,
    path: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer_model_id: Optional[str],
    tokenizer_cache_dir: Optional[str],
) -> pd.DataFrame:
    """Shared logic: markdown string -> tokenizer split -> chunk DataFrame."""
    if not markdown_text or not markdown_text.strip():
        return pd.DataFrame(
            columns=["text", "path", "page_number", "metadata"],
        ).astype({"page_number": "int64"})

    model_id = tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID
    tokenizer = _get_txt_tokenizer(model_id, cache_dir=tokenizer_cache_dir)
    chunk_texts = split_text_by_tokens(
        markdown_text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunk_texts:
        return pd.DataFrame(
            columns=["text", "path", "page_number", "metadata"],
        ).astype({"page_number": "int64"})

    rows: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunk_texts):
        rows.append(
            {
                "text": chunk,
                "content": chunk,
                "path": path,
                "page_number": i + 1,
                "metadata": {
                    "source_path": path,
                    "chunk_index": i,
                    "content_metadata": {"type": "text"},
                    "content": chunk,
                },
            }
        )
    return pd.DataFrame(rows)
