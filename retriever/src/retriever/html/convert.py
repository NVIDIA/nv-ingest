"""
HTML to markdown conversion via markitdown, then tokenizer-based chunking.

Produces chunk DataFrames compatible with embed_text_from_primitives_df
and the LanceDB row builder (text, path, page_number, metadata).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..txt.split import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TOKENIZER_MODEL_ID,
    split_text_by_tokens,
)
from ..txt.split import _get_tokenizer as _get_txt_tokenizer


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
    from markitdown import MarkItDown

    md = MarkItDown()
    if isinstance(html_content, Path):
        html_content = str(html_content)
    if isinstance(html_content, str):
        if Path(html_content).is_file():
            result = md.convert(html_content)
        else:
            result = md.convert_stream(io.BytesIO(html_content.encode("utf-8", errors="replace")))
    elif isinstance(html_content, bytes):
        result = md.convert_stream(io.BytesIO(html_content))
    else:
        result = md.convert(html_content)
    return result.text_content or ""


def html_file_to_chunks_df(
    path: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    tokenizer_model_id: Optional[str] = None,
    encoding: str = "utf-8",
    tokenizer_cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
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
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(path)
    markdown_text = result.text_content or ""
    return _markdown_to_chunks_df(
        markdown_text,
        path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
        **kwargs,
    )


def html_bytes_to_chunks_df(
    content_bytes: bytes,
    path: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    tokenizer_model_id: Optional[str] = None,
    encoding: str = "utf-8",
    tokenizer_cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convert HTML bytes to markdown and return a DataFrame of chunks (same shape as html_file_to_chunks_df).

    Used by batch HtmlSplitActor when input is bytes + path from read_binary_files.
    """
    path = str(Path(path).resolve())
    # Use temp file so markitdown can detect HTML by extension
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(content_bytes)
        tmp_path = f.name
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(tmp_path)
        markdown_text = result.text_content or ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return _markdown_to_chunks_df(
        markdown_text,
        path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
        **kwargs,
    )


def _markdown_to_chunks_df(
    markdown_text: str,
    path: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer_model_id: Optional[str],
    tokenizer_cache_dir: Optional[str],
    **kwargs: Any,
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
