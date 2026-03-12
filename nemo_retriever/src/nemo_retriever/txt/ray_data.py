# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for .txt: TxtSplitActor turns bytes+path batches into chunk rows.
"""

from __future__ import annotations

from typing import Any, Dict, List  # noqa: F401

import pandas as pd

from nemo_retriever.params import TextChunkParams

from .split import txt_bytes_to_chunks_df


class TextChunkActor:
    """
    Ray Data map_batches callable: re-chunk existing ``text`` column by token count.

    This is the batch-mode equivalent of :func:`~nemo_retriever.txt.split.split_df`.
    Constructor takes :class:`TextChunkParams`; ``__call__`` receives a pandas batch
    and returns the split result.
    """

    def __init__(self, params: TextChunkParams | None = None) -> None:
        self._params = params or TextChunkParams()

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        from .split import split_df

        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return batch_df

        kw = self._params.model_dump(mode="python")
        kw.pop("encoding", None)
        return split_df(batch_df, **kw)


class TxtSplitActor:
    """
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of chunks.

    Each output row has: text, path, page_number, metadata (same shape as txt_file_to_chunks_df).
    """

    def __init__(self, params: TextChunkParams | None = None) -> None:
        self._params = params or TextChunkParams()

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])

        params = self._params
        out_dfs: List[pd.DataFrame] = []
        for _, row in batch_df.iterrows():
            raw = row.get("bytes")
            text = row.get("text")
            path = row.get("path")
            if (raw is None and text is None) or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                payload = raw or text.encode("utf-8")
                chunk_df = txt_bytes_to_chunks_df(payload, path_str, params=params)
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except Exception:
                continue
        if not out_dfs:
            return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])
        return pd.concat(out_dfs, ignore_index=True)
