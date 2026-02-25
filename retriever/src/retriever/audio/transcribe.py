# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio transcription via Riva/Parakeet NIM for the retriever pipeline.

Produces chunk DataFrames compatible with embed_text_from_primitives_df
and the LanceDB row builder (text, path, page_number, metadata).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import create_audio_inference_client

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_MODEL_ID = "nvidia/llama-3.2-nv-embedqa-1b-v2"


def _get_tokenizer(model_id: str, cache_dir: Optional[str] = None):  # noqa: ANN201
    """Lazy-load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def _split_text_by_tokens(
    text: str,
    *,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    """Split text into chunks by token count with optional overlap."""
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


def _transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    grpc_endpoint: str,
    auth_token: Optional[str] = None,
    function_id: Optional[str] = None,
    use_ssl: Optional[bool] = None,
    ssl_cert: Optional[str] = None,
) -> tuple:
    """Create a ParakeetClient and transcribe raw audio bytes.

    Returns (segments, transcript) where segments is a list of
    ``{"start": float, "end": float, "text": str}`` dicts and
    transcript is the full transcript string.
    """
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    client = create_audio_inference_client(
        (grpc_endpoint, ""),
        infer_protocol="grpc",
        auth_token=auth_token,
        function_id=function_id,
        use_ssl=use_ssl if use_ssl is not None else False,
        ssl_cert=ssl_cert,
    )
    segments, transcript = client.infer(
        b64_audio,
        model_name="parakeet",
        stage_name="audio_extraction",
    )
    return segments, transcript


def _build_rows(
    transcript: str,
    segments: list,
    path: str,
    *,
    segment_audio: bool = False,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = 0,
    tokenizer_model_id: Optional[str] = None,
    tokenizer_cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert transcript/segments into row dicts for the output DataFrame."""
    rows: List[Dict[str, Any]] = []

    if segment_audio:
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text.strip():
                continue
            rows.append(
                {
                    "text": text,
                    "content": text,
                    "path": path,
                    # "page_number": None,
                    "metadata": {
                        "source_path": path,
                        "chunk_index": i,
                        "content_metadata": {
                            "type": "audio",
                            "start_time": seg.get("start"),
                            "end_time": seg.get("end"),
                        },
                        "content": text,
                    },
                }
            )
    elif max_tokens is not None:
        model_id = tokenizer_model_id or DEFAULT_TOKENIZER_MODEL_ID
        tokenizer = _get_tokenizer(model_id, cache_dir=tokenizer_cache_dir)
        chunk_texts = _split_text_by_tokens(
            transcript,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for i, chunk in enumerate(chunk_texts):
            rows.append(
                {
                    "text": chunk,
                    "content": chunk,
                    "path": path,
                    # "page_number": None,
                    "metadata": {
                        "source_path": path,
                        "chunk_index": i,
                        "content_metadata": {"type": "audio"},
                        "content": chunk,
                    },
                }
            )
    else:
        rows.append(
            {
                "text": transcript,
                "content": transcript,
                "path": path,
                # "page_number": None,
                "metadata": {
                    "source_path": path,
                    "chunk_index": 0,
                    "content_metadata": {"type": "audio"},
                    "content": transcript,
                },
            }
        )

    return rows


_EMPTY_DF_COLUMNS = ["text", "content", "path", "page_number", "metadata"]


def audio_file_to_transcript_df(
    path: str,
    grpc_endpoint: str = "audio:50051",
    auth_token: Optional[str] = None,
    function_id: Optional[str] = None,
    use_ssl: Optional[bool] = None,
    ssl_cert: Optional[str] = None,
    segment_audio: bool = False,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = 0,
    tokenizer_model_id: Optional[str] = None,
    tokenizer_cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an audio file, transcribe via Parakeet/Riva, return a chunk DataFrame.

    Parameters
    ----------
    path : str
        Path to an audio file (mp3, wav, etc.).
    grpc_endpoint : str
        Riva/Parakeet gRPC endpoint.
    segment_audio : bool
        If True, each speech segment becomes its own row.
    max_tokens : int, optional
        If set (and segment_audio is False), chunk the transcript by token
        count using the same logic as txt extraction.
    overlap_tokens : int
        Token overlap between consecutive chunks (only used with max_tokens).

    Returns
    -------
    pd.DataFrame
        Columns: text, content, path, page_number, metadata.
    """
    abs_path = str(Path(path).resolve())
    audio_bytes = Path(abs_path).read_bytes()

    segments, transcript = _transcribe_audio_bytes(
        audio_bytes,
        grpc_endpoint=grpc_endpoint,
        auth_token=auth_token,
        function_id=function_id,
        use_ssl=use_ssl,
        ssl_cert=ssl_cert,
    )

    if not transcript or not transcript.strip():
        return pd.DataFrame(columns=_EMPTY_DF_COLUMNS).astype({"page_number": "int64"})

    rows = _build_rows(
        transcript,
        segments,
        abs_path,
        segment_audio=segment_audio,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )

    if not rows:
        return pd.DataFrame(columns=_EMPTY_DF_COLUMNS).astype({"page_number": "int64"})

    return pd.DataFrame(rows)


def audio_bytes_to_transcript_df(
    content_bytes: bytes,
    path: str,
    grpc_endpoint: str = "audio:50051",
    auth_token: Optional[str] = None,
    function_id: Optional[str] = None,
    use_ssl: Optional[bool] = None,
    ssl_cert: Optional[str] = None,
    segment_audio: bool = False,
    max_tokens: Optional[int] = None,
    overlap_tokens: int = 0,
    tokenizer_model_id: Optional[str] = None,
    tokenizer_cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transcribe audio from raw bytes (for Ray Data batch mode).

    Same as :func:`audio_file_to_transcript_df` but accepts bytes directly
    instead of reading from disk.
    """
    abs_path = str(Path(path).resolve())

    segments, transcript = _transcribe_audio_bytes(
        content_bytes,
        grpc_endpoint=grpc_endpoint,
        auth_token=auth_token,
        function_id=function_id,
        use_ssl=use_ssl,
        ssl_cert=ssl_cert,
    )

    if not transcript or not transcript.strip():
        return pd.DataFrame(columns=_EMPTY_DF_COLUMNS).astype({"page_number": "int64"})

    rows = _build_rows(
        transcript,
        segments,
        abs_path,
        segment_audio=segment_audio,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )

    if not rows:
        return pd.DataFrame(columns=_EMPTY_DF_COLUMNS).astype({"page_number": "int64"})

    return pd.DataFrame(rows)
