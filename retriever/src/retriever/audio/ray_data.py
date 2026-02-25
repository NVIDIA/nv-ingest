# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for audio: AudioTranscribeActor turns bytes+path batches into transcript rows.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from retriever.params import AudioExtractParams

from .transcribe import audio_bytes_to_transcript_df


class AudioTranscribeActor:
    """Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of transcript chunks.

    Each output row has: text, content, path, page_number, metadata
    (same shape as audio_file_to_transcript_df).
    """

    def __init__(self, params: AudioExtractParams | None = None) -> None:
        self._params = params or AudioExtractParams()

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=["text", "content", "path", "page_number", "metadata"])

        params = self._params
        out_dfs: List[pd.DataFrame] = []
        for _, row in batch_df.iterrows():
            raw = row.get("bytes")
            path = row.get("path")
            if raw is None or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                chunk_df = audio_bytes_to_transcript_df(
                    raw,
                    path_str,
                    grpc_endpoint=params.grpc_endpoint,
                    auth_token=params.auth_token,
                    function_id=params.function_id,
                    use_ssl=params.use_ssl,
                    ssl_cert=params.ssl_cert,
                    segment_audio=params.segment_audio,
                    max_tokens=params.max_tokens,
                    overlap_tokens=params.overlap_tokens,
                    tokenizer_model_id=params.tokenizer_model_id,
                    tokenizer_cache_dir=params.tokenizer_cache_dir,
                )
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except Exception:
                continue
        if not out_dfs:
            return pd.DataFrame(columns=["text", "content", "path", "page_number", "metadata"])
        return pd.concat(out_dfs, ignore_index=True)
