# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for retriever.audio: MediaChunkActor and audio_path_to_chunks_df.
"""

import wave
from pathlib import Path

import pandas as pd
import pytest

from retriever.audio.chunk_actor import CHUNK_COLUMNS
from retriever.audio.chunk_actor import MediaChunkActor
from retriever.audio.chunk_actor import audio_path_to_chunks_df
from retriever.audio.media_interface import is_media_available
from retriever.params import AudioChunkParams


def _make_small_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 8000) -> None:
    """Write a minimal WAV file (e.g. 0.5s mono 8kHz) for tests."""
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_media_chunk_actor_empty_batch():
    from retriever.audio import MediaChunkActor

    params = AudioChunkParams(split_type="size", split_interval=1000)
    actor = MediaChunkActor(params=params)
    empty = pd.DataFrame(columns=["path", "bytes"])
    out = actor(empty)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == CHUNK_COLUMNS
    assert len(out) == 0


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_media_chunk_actor_single_small_file(tmp_path: Path):
    from retriever.audio import MediaChunkActor

    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.3)
    with open(wav, "rb") as f:
        body = f.read()

    params = AudioChunkParams(split_type="size", split_interval=1_000_000)
    actor = MediaChunkActor(params=params)
    batch = pd.DataFrame([{"path": str(wav.resolve()), "bytes": body}])
    out = actor(batch)

    assert isinstance(out, pd.DataFrame)
    for col in ["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "bytes"]:
        assert col in out.columns
    assert len(out) >= 1
    assert out["source_path"].iloc[0] == str(wav.resolve())
    assert out["chunk_index"].iloc[0] == 0
    assert isinstance(out["bytes"].iloc[0], bytes)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_audio_path_to_chunks_df(tmp_path: Path):
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.4)
    params = AudioChunkParams(split_type="size", split_interval=500_000)
    df = audio_path_to_chunks_df(str(wav), params=params)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert "path" in df.columns and "source_path" in df.columns
    assert "bytes" in df.columns
    assert df["source_path"].iloc[0] == str(wav.resolve())


def test_media_chunk_actor_requires_ffmpeg():
    """Without ffmpeg, MediaChunkActor.__init__ raises."""
    pytest.importorskip("ffmpeg")
    # If ffmpeg is available, is_media_available() is True; we can't test the failure path
    # without unimporting. So we only run the raise test when ffmpeg is missing.
    if is_media_available():
        pytest.skip("ffmpeg available; cannot test missing-ffmpeg path here")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        MediaChunkActor(params=AudioChunkParams())
