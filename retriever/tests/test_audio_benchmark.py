# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal test that the audio extraction benchmark runs (mock ASR, small row count).
Skip when ffmpeg or Ray is not available.
"""

from pathlib import Path

import pytest

from retriever.audio.media_interface import is_media_available


def _make_small_wav(path: Path, duration_sec: float = 0.3, sample_rate: int = 8000) -> None:
    import wave

    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_audio_benchmark_run_mock_asr(tmp_path: Path):
    """Run benchmark with --mock-asr and small row count; assert it completes."""
    ray = pytest.importorskip("ray")
    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.3)

    from typer.testing import CliRunner

    from retriever.utils.benchmark.audio_extract_actor import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--audio-path",
            str(wav),
            "--rows",
            "2",
            "--workers",
            "1",
            "--batch-sizes",
            "2",
            "--mock-asr",
        ],
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "audio_extract" in result.stdout or "BEST" in result.stdout
