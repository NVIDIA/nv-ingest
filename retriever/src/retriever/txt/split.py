# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer-based text splitting for .txt ingestion.

Produces chunk DataFrames compatible with embed_text_from_primitives_df
and the LanceDB row builder (text, path, page_number, metadata).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_TOKENIZER_MODEL_ID = "nvidia/llama-3.2-nv-embedqa-1b-v2"
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 0


def _get_tokenizer(model_id: str, cache_dir: Optional[str] = None):  # noqa: ANN201
    """Lazy-load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )


def split_text_by_tokens(
    text: str,
    *,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    """
    Split text into chunks by token count with optional overlap.

    Chunk boundaries align with tokenizer token boundaries. Uses a sliding
    window: each chunk has at most max_tokens tokens; consecutive chunks
    overlap by overlap_tokens tokens.

    Parameters
    ----------
    text : str
        Input text to split.
    tokenizer
        HuggingFace tokenizer (e.g. AutoTokenizer) with encode/decode.
    max_tokens : int
        Maximum tokens per chunk.
    overlap_tokens : int
        Number of tokens to overlap between consecutive chunks (default 0).

    Returns
    -------
    list[str]
        Chunk strings in order.
    """
    if not text or not text.strip():
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    enc = tokenizer.encode(text, add_special_tokens=False)
    if not enc:
        return []

    step = max(1, max_tokens - overlap_tokens)
    chunks: List[str] = []
    start = 0
    while start < len(enc):
        end = min(start + max_tokens, len(enc))
        chunk_ids = enc[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += step
        if end >= len(enc):
            break

    return chunks if chunks else [text]


def txt_file_to_chunks_df(
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
    Read a .txt file and return a DataFrame of chunks (one row per chunk).

    Columns: text, path, page_number (chunk index, 1-based), metadata.
    Shape is compatible with embed_text_from_primitives_df and LanceDB row build.

    Parameters
    ----------
    path : str
        Path to the .txt file.
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
    raw = Path(path).read_text(encoding=encoding, errors="replace")
    model_id = tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID
    tokenizer = _get_tokenizer(model_id, cache_dir=tokenizer_cache_dir)
    chunk_texts = split_text_by_tokens(
        raw,
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


def txt_bytes_to_chunks_df(
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
    Decode bytes to text and return a DataFrame of chunks (same shape as txt_file_to_chunks_df).

    Used by batch TxtSplitActor when input is bytes + path from read_binary_files.
    """
    path = str(Path(path).resolve())
    raw = content_bytes.decode(encoding, errors="replace")
    model_id = tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID
    tokenizer = _get_tokenizer(model_id, cache_dir=tokenizer_cache_dir)
    chunk_texts = split_text_by_tokens(
        raw,
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
