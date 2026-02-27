# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MediaChunkActor: Ray Data map_batches callable for audio/video chunking.

Consumes rows from rd.read_binary_files (path, bytes) and produces one row
per chunk with path, source_path, duration, chunk_index, metadata.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from retriever.audio.media_interface import MediaInterface
from retriever.audio.media_interface import is_media_available
from retriever.params import AudioChunkParams

logger = logging.getLogger(__name__)

# Output columns for downstream (ASR, embed, VDB). bytes optional for in-memory pipeline.
CHUNK_COLUMNS = ["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "bytes"]


class MediaChunkActor:
    """
    Ray Data map_batches callable: DataFrame with path, bytes -> DataFrame of chunk rows.

    Each output row has: path (chunk file), source_path, duration, chunk_index,
    metadata (dict with source_path, chunk_index, duration), page_number (= chunk_index).
    """

    def __init__(self, params: AudioChunkParams | None = None) -> None:
        if not is_media_available():
            raise RuntimeError(
                "MediaChunkActor requires ffmpeg. Install with: pip install ffmpeg-python and system ffmpeg."
            )
        self._params = params or AudioChunkParams()
        self._interface = MediaInterface()

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=CHUNK_COLUMNS)

        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            path = row.get("path")
            if path is None:
                continue
            path_str = str(path)
            if not path_str.strip():
                continue
            try:
                chunk_rows = _chunk_one(path_str, self._params, self._interface)
                out_rows.extend(chunk_rows)
            except Exception as e:
                logger.exception("Error chunking %s: %s", path_str, e)
                continue

        if not out_rows:
            return pd.DataFrame(columns=CHUNK_COLUMNS)
        return pd.DataFrame(out_rows)


def _chunk_one(source_path: str, params: AudioChunkParams, interface: MediaInterface) -> List[Dict[str, Any]]:
    """Run split for one file and return list of row dicts."""
    with tempfile.TemporaryDirectory(prefix="retriever_audio_chunk_") as tmpdir:
        files = interface.split(
            source_path,
            tmpdir,
            split_interval=params.split_interval,
            split_type=params.split_type,
            video_audio_separate=params.video_audio_separate,
            audio_only=params.audio_only,
        )
        if not files:
            return []

        rows: List[Dict[str, Any]] = []
        for idx, chunk_path in enumerate(files):
            _, _, duration = interface.probe_media(
                Path(chunk_path),
                params.split_interval,
                params.split_type,
            )
            duration = duration if duration is not None else 0.0
            meta = {
                "source_path": source_path,
                "chunk_index": idx,
                "duration": duration,
            }
            chunk_bytes: bytes
            try:
                with open(chunk_path, "rb") as f:
                    chunk_bytes = f.read()
            except Exception as e:
                logger.warning("Could not read chunk %s: %s", chunk_path, e)
                continue
            rows.append(
                {
                    "path": chunk_path,
                    "source_path": source_path,
                    "duration": duration,
                    "chunk_index": idx,
                    "metadata": meta,
                    "page_number": idx,
                    "bytes": chunk_bytes,
                }
            )
        return rows


def audio_path_to_chunks_df(path: str, params: AudioChunkParams | None = None) -> pd.DataFrame:
    """
    Synchronous loader: one media file path -> DataFrame of chunk rows (path, source_path, duration, chunk_index,
    metadata, page_number, bytes).

    Used by inprocess ingest() when _pipeline_type == "audio".
    """
    if not is_media_available():
        raise RuntimeError("audio_path_to_chunks_df requires ffmpeg.")
    params = params or AudioChunkParams()
    interface = MediaInterface()
    rows = _chunk_one(path, params, interface)
    if not rows:
        return pd.DataFrame(columns=CHUNK_COLUMNS)
    return pd.DataFrame(rows)
