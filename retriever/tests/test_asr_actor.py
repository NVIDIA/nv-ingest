# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for retriever.audio: ASRActor (with mocked Parakeet client).
"""

import base64
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd

from retriever.audio.asr_actor import ASRActor
from retriever.audio.asr_actor import apply_asr_to_df
from retriever.params import ASRParams


def test_asr_actor_empty_batch():
    with patch("retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        empty = pd.DataFrame(columns=["path", "bytes"])
        out = actor(empty)

        assert isinstance(out, pd.DataFrame)
        assert "text" in out.columns
        assert len(out) == 0
        mock_client.infer.assert_not_called()


def test_asr_actor_mock_transcribe():
    with patch("retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "hello world transcript")
        mock_get.return_value = mock_client

        params = ASRParams(audio_endpoints=("localhost:50051", None))
        actor = ASRActor(params=params)
        raw = b"\x00\x00\x00\x00"
        batch = pd.DataFrame(
            [
                {
                    "path": "/tmp/chunk.wav",
                    "bytes": raw,
                    "source_path": "/tmp/source.wav",
                    "duration": 1.0,
                    "chunk_index": 0,
                    "metadata": {"source_path": "/tmp/source.wav", "chunk_index": 0, "duration": 1.0},
                    "page_number": 0,
                }
            ]
        )
        out = actor(batch)

        assert len(out) == 1
        assert out["text"].iloc[0] == "hello world transcript"
        assert out["path"].iloc[0] == "/tmp/chunk.wav"
        assert out["source_path"].iloc[0] == "/tmp/source.wav"
        mock_client.infer.assert_called_once()
        call_arg = mock_client.infer.call_args[0][0]
        assert call_arg == base64.b64encode(raw).decode("ascii")


def test_apply_asr_to_df():
    with patch("retriever.audio.asr_actor._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.infer.return_value = ([], "applied transcript")
        mock_get.return_value = mock_client

        batch = pd.DataFrame(
            [
                {
                    "path": "/p",
                    "bytes": b"x",
                    "source_path": "/s",
                    "duration": 0.5,
                    "chunk_index": 0,
                    "metadata": {},
                    "page_number": 0,
                }
            ]
        )
        out = apply_asr_to_df(batch, asr_params={"audio_endpoints": ("localhost:50051", None)})
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out["text"].iloc[0] == "applied transcript"


def test_local_asr_does_not_call_get_client():
    """When audio_endpoints are both null, ASRActor uses local model and does not call _get_client."""
    with patch("retriever.audio.asr_actor._get_client") as mock_get:
        with patch("retriever.model.local.ParakeetCTC1B1ASR") as mock_class:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ["mocked local transcript"]
            mock_class.return_value = mock_model

            params = ASRParams(audio_endpoints=(None, None))
            actor = ASRActor(params=params)

            mock_get.assert_not_called()
            assert actor._client is None
            assert actor._model is mock_model

            batch = pd.DataFrame(
                [
                    {
                        "path": "/tmp/chunk.wav",
                        "bytes": b"fake_audio_bytes",
                        "source_path": "/tmp/source.wav",
                        "duration": 1.0,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = actor(batch)

            assert len(out) == 1
            assert out["text"].iloc[0] == "mocked local transcript"
            mock_model.transcribe.assert_called_once()
            # One path passed (temp file or /tmp/chunk.wav)
            call_args = mock_model.transcribe.call_args[0][0]
            assert isinstance(call_args, list)
            assert len(call_args) == 1


def test_local_asr_apply_asr_to_df():
    """apply_asr_to_df with audio_endpoints=(None, None) uses local model when mocked."""
    with patch("retriever.audio.asr_actor._get_client") as mock_get:
        with patch("retriever.model.local.ParakeetCTC1B1ASR") as mock_class:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ["apply local text"]
            mock_class.return_value = mock_model

            batch = pd.DataFrame(
                [
                    {
                        "path": "/p",
                        "bytes": b"x",
                        "source_path": "/s",
                        "duration": 0.5,
                        "chunk_index": 0,
                        "metadata": {},
                        "page_number": 0,
                    }
                ]
            )
            out = apply_asr_to_df(batch, asr_params={"audio_endpoints": (None, None)})

            mock_get.assert_not_called()
            assert len(out) == 1
            assert out["text"].iloc[0] == "apply local text"
