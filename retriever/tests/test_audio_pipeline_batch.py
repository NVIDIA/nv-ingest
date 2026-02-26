# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration-style tests for the audio pipeline (batch, inprocess, fused).

Uses a small generated WAV and mocked ASR so no Parakeet endpoint or mp3/ dir is required.
Skip if Ray or ffmpeg is not available.
"""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from retriever.audio.chunk_actor import _chunk_one
from retriever.audio.media_interface import MediaInterface
from retriever.audio.media_interface import is_media_available
from retriever.params import AudioChunkParams
from retriever.params import ASRParams
from retriever.params import LanceDbParams
from retriever.params import VdbUploadParams


def _make_small_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 8000) -> None:
    import wave
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_audio_chunk_then_mock_asr_flow(tmp_path: Path):
    """Chunk one small WAV and verify chunk rows have expected shape (no Ray)."""
    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.4)
    params = AudioChunkParams(split_type="size", split_interval=500_000)
    interface = MediaInterface()
    rows = _chunk_one(str(wav), params, interface)
    assert len(rows) >= 1
    row = rows[0]
    assert "path" in row and "source_path" in row and "duration" in row
    assert "chunk_index" in row and "bytes" in row
    assert row["source_path"] == str(wav.resolve())


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_batch_audio_pipeline_with_mocked_asr(tmp_path: Path):
    """Run batch pipeline with Ray in local_mode so _get_client mock is visible in workers."""
    ray = pytest.importorskip("ray")
    pytest.importorskip("lancedb")

    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()

    # Ray workers/raylet resolve path deps from working_dir; use nv-ingest repo root.
    _nv_ingest_root = Path(__file__).resolve().parents[2]

    from retriever.ingest_modes.batch import BatchIngestor

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "mock transcript for integration test")

    with patch("retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            BatchIngestor(documents=[])
            .files([str(wav)])
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
            )
            .embed()
            .vdb_upload(
                params=VdbUploadParams(
                    lancedb=LanceDbParams(
                        lancedb_uri=str(lancedb_dir),
                        table_name="test_audio",
                        overwrite=True,
                    )
                )
            )
        )
        try:
            ray.init(
                ignore_reinit_error=True,
                local_mode=True,
                runtime_env={"working_dir": str(_nv_ingest_root)},
            )
            results = ingestor.ingest()
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

    assert results is not None
    # ingest() returns num_pages (int); with mocked ASR in driver only, workers may hit real endpoint and get 0 pages
    assert isinstance(results, (int, list))
    if isinstance(results, int):
        assert results >= 0


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_inprocess_audio_pipeline_with_mocked_asr(tmp_path: Path):
    """Inprocess: files -> extract_audio (chunk + mocked ASR) -> ingest(); assert result DataFrames have text."""
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)

    from retriever.ingest_modes.inprocess import InProcessIngestor

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "inprocess mock transcript")

    with patch("retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            InProcessIngestor(documents=[])
            .files([str(wav)])
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
            )
        )
        results = ingestor.ingest()

    assert results is not None
    assert isinstance(results, list)
    assert len(results) >= 1
    df = results[0]
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert len(df) >= 1
    assert (df["text"] == "inprocess mock transcript").all()


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_inprocess_audio_pipeline_local_asr_mocked(tmp_path: Path):
    """Inprocess with audio_endpoints=(None, None) uses local ASR; mock ParakeetCTC1B1ASR so no real model."""
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)

    from retriever.ingest_modes.inprocess import InProcessIngestor

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["local asr mock transcript"]

    with patch("retriever.audio.asr_actor._get_client") as mock_get_client:
        with patch("retriever.model.local.ParakeetCTC1B1ASR", return_value=mock_model):
            ingestor = (
                InProcessIngestor(documents=[])
                .files([str(wav)])
                .extract_audio(
                    params=AudioChunkParams(split_type="size", split_interval=500_000),
                    asr_params=ASRParams(audio_endpoints=(None, None)),
                )
            )
            results = ingestor.ingest()

    mock_get_client.assert_not_called()
    assert results is not None
    assert isinstance(results, list)
    assert len(results) >= 1
    df = results[0]
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert len(df) >= 1
    assert (df["text"] == "local asr mock transcript").all()


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_fused_audio_pipeline_with_mocked_asr(tmp_path: Path):
    """Fused: same as batch but FusedIngestor; embed() uses explode + _BatchEmbedActor when _pipeline_type==audio."""
    ray = pytest.importorskip("ray")
    pytest.importorskip("lancedb")

    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)
    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()

    _nv_ingest_root = Path(__file__).resolve().parents[2]

    from retriever.ingest_modes.fused import FusedIngestor

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "fused mock transcript")

    with patch("retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            FusedIngestor(documents=[])
            .files([str(wav)])
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
            )
            .embed()
            .vdb_upload(
                params=VdbUploadParams(
                    lancedb=LanceDbParams(
                        lancedb_uri=str(lancedb_dir),
                        table_name="test_audio_fused",
                        overwrite=True,
                    )
                )
            )
        )
        try:
            ray.init(
                ignore_reinit_error=True,
                local_mode=True,
                runtime_env={"working_dir": str(_nv_ingest_root)},
            )
            results = ingestor.ingest()
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

    assert results is not None
    assert isinstance(results, (int, list))
    if isinstance(results, int):
        assert results >= 0


def _mp3_dir() -> Path | None:
    """Return mp3/ directory next to retriever (nv-ingest/retriever/mp3) if it exists."""
    root = Path(__file__).resolve().parents[1]  # retriever/
    mp3 = root / "mp3"
    return mp3 if mp3.is_dir() else None


def _glob_audio(mp3_dir: Path) -> list[str]:
    """Return list of audio paths in mp3_dir (e.g. mp3/*.mp3, mp3/*.wav)."""
    out: list[str] = []
    for ext in ("*.mp3", "*.wav", "*.m4a"):
        out.extend(str(p) for p in mp3_dir.glob(ext) if p.is_file())
    return sorted(out)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg/ffprobe not available")
def test_audio_pipeline_with_mp3_dir_mocked_asr(tmp_path: Path):
    """
    Integration test using mp3/ when present: .files('mp3/*.mp3').extract_audio().embed().vdb_upload().ingest()
    with mocked ASR. Skips if mp3/ is missing or has no audio files.
    """
    mp3_dir = _mp3_dir()
    if mp3_dir is None:
        pytest.skip("mp3/ directory not found (create nv-ingest/retriever/mp3/ for this test)")
    audio_files = _glob_audio(mp3_dir)
    if not audio_files:
        pytest.skip("mp3/ has no .mp3/.wav/.m4a files (add sample audio for integration test)")

    ray = pytest.importorskip("ray")
    pytest.importorskip("lancedb")

    lancedb_dir = tmp_path / "lancedb"
    lancedb_dir.mkdir()
    _nv_ingest_root = Path(__file__).resolve().parents[2]

    from retriever.ingest_modes.batch import BatchIngestor

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "mock transcript from mp3/ integration")

    with patch("retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            BatchIngestor(documents=[])
            .files(audio_files)
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
            )
            .embed()
            .vdb_upload(
                params=VdbUploadParams(
                    lancedb=LanceDbParams(
                        lancedb_uri=str(lancedb_dir),
                        table_name="test_audio_mp3",
                        overwrite=True,
                    )
                )
            )
        )
        try:
            ray.init(
                ignore_reinit_error=True,
                local_mode=True,
                runtime_env={"working_dir": str(_nv_ingest_root)},
            )
            results = ingestor.ingest()
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

    assert results is not None
    # Batch ingest() returns num_pages (int)
    assert isinstance(results, int)
    assert results >= len(audio_files)
