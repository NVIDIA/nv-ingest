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
from nemo_retriever.params import TextChunkParams

DEFAULT_TOKENIZER_MODEL_ID = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_OVERLAP_TOKENS = 0


def _get_tokenizer(model_id: str, cache_dir: Optional[str] = None):  # noqa: ANN201
    """Lazy-load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    from nemo_retriever.utils.hf_model_registry import get_hf_revision

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=get_hf_revision(model_id),
        cache_dir=cache_dir,
        trust_remote_code=True,
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


def split_df(
    df: pd.DataFrame,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    tokenizer_model_id: Optional[str] = None,
    tokenizer_cache_dir: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Re-chunk a DataFrame's ``text`` column by token count.

    This is a **post-extraction** transform: it takes rows that already have a
    ``text`` column (produced by ``extract`` / ``extract_txt`` / etc.) and
    splits long texts into multiple rows using :func:`split_text_by_tokens`.
    All other columns (``path``, ``page_number``, ``metadata``, …) are
    preserved on every output row.  Each chunk row's ``metadata`` dict is
    updated with ``chunk_index`` and ``chunk_count``.

    Rows whose ``text`` is empty or missing are passed through unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a ``text`` column.
    max_tokens, overlap_tokens, tokenizer_model_id, tokenizer_cache_dir, encoding
        Forwarded to :func:`split_text_by_tokens` / :func:`_get_tokenizer`.

    Returns
    -------
    pd.DataFrame
        Expanded DataFrame (one row per chunk).
    """
    if df.empty:
        return df.copy()

    model_id = tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID
    tokenizer = _get_tokenizer(model_id, cache_dir=tokenizer_cache_dir)

    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        text = row_dict.get("text")
        if not isinstance(text, str) or not text.strip():
            out_rows.append(row_dict)
            continue

        chunks = split_text_by_tokens(
            text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if len(chunks) <= 1:
            out_rows.append(row_dict)
            continue

        import copy

        for i, chunk in enumerate(chunks):
            new_row = {k: copy.deepcopy(v) if isinstance(v, (dict, list)) else v for k, v in row_dict.items()}
            new_row["text"] = chunk
            if "content" in new_row:
                new_row["content"] = chunk
            meta = new_row.get("metadata")
            if isinstance(meta, dict):
                meta["chunk_index"] = i
                meta["chunk_count"] = len(chunks)
                meta["content"] = chunk
            new_row["page_number"] = i + 1
            out_rows.append(new_row)

    if not out_rows:
        return df.iloc[:0].copy()

    return pd.DataFrame(out_rows)


def txt_file_to_chunks_df(
    path: str,
    params: TextChunkParams | None = None,
) -> pd.DataFrame:
    chunk_params = params or TextChunkParams()
    max_tokens = chunk_params.max_tokens
    overlap_tokens = chunk_params.overlap_tokens
    tokenizer_model_id = chunk_params.tokenizer_model_id
    encoding = chunk_params.encoding
    tokenizer_cache_dir = chunk_params.tokenizer_cache_dir

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
    params: TextChunkParams | None = None,
) -> pd.DataFrame:
    chunk_params = params or TextChunkParams()
    max_tokens = chunk_params.max_tokens
    overlap_tokens = chunk_params.overlap_tokens
    tokenizer_model_id = chunk_params.tokenizer_model_id
    encoding = chunk_params.encoding
    tokenizer_cache_dir = chunk_params.tokenizer_cache_dir

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
