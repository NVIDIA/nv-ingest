# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for retriever.audio.transcribe:
  _split_text_by_tokens, _build_rows, audio_file_to_transcript_df, audio_bytes_to_transcript_df.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from retriever.audio.transcribe import (
    _build_rows,
    _split_text_by_tokens,
    audio_bytes_to_transcript_df,
    audio_file_to_transcript_df,
)


# ---------------------------------------------------------------------------
# Mock tokenizer (same pattern as test_txt_split.py)
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer: encode = split on spaces, decode = join."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, (list, range)):
            return " ".join(str(i) for i in ids)
        return str(ids)


SAMPLE_TRANSCRIPT = "the quick brown fox jumped over the lazy dog"
SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": "the quick brown fox"},
    {"start": 1.5, "end": 3.0, "text": "jumped over the lazy dog"},
]


# ---------------------------------------------------------------------------
# _split_text_by_tokens
# ---------------------------------------------------------------------------


class TestSplitTextByTokens:
    def test_empty_text(self):
        tok = _MockTokenizer()
        assert _split_text_by_tokens("", tokenizer=tok, max_tokens=5) == []
        assert _split_text_by_tokens("   ", tokenizer=tok, max_tokens=5) == []

    def test_single_chunk(self):
        tok = _MockTokenizer()
        chunks = _split_text_by_tokens("a b c", tokenizer=tok, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == "a b c"

    def test_multiple_chunks_no_overlap(self):
        tok = _MockTokenizer()
        text = "a b c d e f g h i"
        chunks = _split_text_by_tokens(text, tokenizer=tok, max_tokens=3, overlap_tokens=0)
        assert len(chunks) == 3
        assert chunks[0] == "a b c"
        assert chunks[1] == "d e f"
        assert chunks[2] == "g h i"

    def test_multiple_chunks_with_overlap(self):
        tok = _MockTokenizer()
        text = "a b c d e f"
        chunks = _split_text_by_tokens(text, tokenizer=tok, max_tokens=3, overlap_tokens=1)
        assert len(chunks) >= 2
        assert chunks[0] == "a b c"
        assert chunks[1] == "c d e"

    def test_invalid_max_tokens(self):
        tok = _MockTokenizer()
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            _split_text_by_tokens("hello", tokenizer=tok, max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            _split_text_by_tokens("hello", tokenizer=tok, max_tokens=-1)


# ---------------------------------------------------------------------------
# _build_rows — default (whole transcript)
# ---------------------------------------------------------------------------


class TestBuildRowsDefault:
    def test_single_row(self):
        rows = _build_rows(SAMPLE_TRANSCRIPT, SAMPLE_SEGMENTS, "/audio/test.mp3")
        assert len(rows) == 1
        row = rows[0]
        assert row["text"] == SAMPLE_TRANSCRIPT
        assert row["content"] == SAMPLE_TRANSCRIPT
        assert row["path"] == "/audio/test.mp3"
        meta = row["metadata"]
        assert meta["source_path"] == "/audio/test.mp3"
        assert meta["chunk_index"] == 0
        assert meta["content_metadata"]["type"] == "audio"
        assert meta["content"] == SAMPLE_TRANSCRIPT

    def test_empty_transcript_returns_empty(self):
        rows = _build_rows("", [], "/audio/empty.mp3")
        assert len(rows) == 1
        assert rows[0]["text"] == ""

    def test_whitespace_only_transcript(self):
        rows = _build_rows("   ", [], "/audio/blank.mp3")
        assert len(rows) == 1
        assert rows[0]["text"] == "   "


# ---------------------------------------------------------------------------
# _build_rows — segmented
# ---------------------------------------------------------------------------


class TestBuildRowsSegmented:
    def test_segment_rows(self):
        rows = _build_rows(SAMPLE_TRANSCRIPT, SAMPLE_SEGMENTS, "/audio/test.mp3", segment_audio=True)
        assert len(rows) == 2
        assert rows[0]["text"] == "the quick brown fox"
        assert rows[0]["metadata"]["content_metadata"]["start_time"] == 0.0
        assert rows[0]["metadata"]["content_metadata"]["end_time"] == 1.5
        assert rows[1]["text"] == "jumped over the lazy dog"
        assert rows[1]["metadata"]["chunk_index"] == 1

    def test_empty_segments_skipped(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "world"},
        ]
        rows = _build_rows("hello world", segs, "/audio/test.mp3", segment_audio=True)
        assert len(rows) == 2
        assert rows[0]["text"] == "hello"
        assert rows[1]["text"] == "world"


# ---------------------------------------------------------------------------
# _build_rows — token-chunked
# ---------------------------------------------------------------------------


class TestBuildRowsTokenChunked:
    @patch("retriever.audio.transcribe._get_tokenizer", return_value=_MockTokenizer())
    def test_chunked_rows(self, _mock_tok):
        rows = _build_rows(SAMPLE_TRANSCRIPT, SAMPLE_SEGMENTS, "/audio/test.mp3", max_tokens=3)
        assert len(rows) >= 2
        for i, row in enumerate(rows):
            assert row["metadata"]["chunk_index"] == i
            assert row["metadata"]["content_metadata"]["type"] == "audio"
            assert row["text"] == row["content"]
            assert row["text"] == row["metadata"]["content"]


# ---------------------------------------------------------------------------
# audio_file_to_transcript_df
# ---------------------------------------------------------------------------


def _mock_transcribe(_audio_bytes, *, grpc_endpoint, **kw):
    return SAMPLE_SEGMENTS, SAMPLE_TRANSCRIPT


class TestAudioFileToTranscriptDf:
    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    def test_basic(self, _mock, tmp_path: Path):
        f = tmp_path / "clip.wav"
        f.write_bytes(b"\x00\x01\x02\x03")
        df = audio_file_to_transcript_df(str(f), grpc_endpoint="localhost:50051")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"text", "content", "path", "metadata"}
        assert len(df) == 1
        assert df["text"].iloc[0] == SAMPLE_TRANSCRIPT
        assert df["path"].iloc[0] == str(f.resolve())
        meta = df["metadata"].iloc[0]
        assert meta["content_metadata"]["type"] == "audio"

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", return_value=([], ""))
    def test_empty_transcript(self, _mock, tmp_path: Path):
        f = tmp_path / "silent.wav"
        f.write_bytes(b"\x00")
        df = audio_file_to_transcript_df(str(f), grpc_endpoint="localhost:50051")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["text", "content", "path", "page_number", "metadata"]

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    def test_segmented(self, _mock, tmp_path: Path):
        f = tmp_path / "clip.mp3"
        f.write_bytes(b"\xff\xfb")
        df = audio_file_to_transcript_df(str(f), grpc_endpoint="localhost:50051", segment_audio=True)
        assert len(df) == 2
        assert df["text"].iloc[0] == "the quick brown fox"
        assert df["metadata"].iloc[0]["content_metadata"]["start_time"] == 0.0

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    @patch("retriever.audio.transcribe._get_tokenizer", return_value=_MockTokenizer())
    def test_token_chunked(self, _mock_tok, _mock_transcribe, tmp_path: Path):
        f = tmp_path / "clip.wav"
        f.write_bytes(b"\x00\x01")
        df = audio_file_to_transcript_df(str(f), grpc_endpoint="localhost:50051", max_tokens=4)
        assert len(df) >= 2
        for i in range(len(df)):
            assert df["metadata"].iloc[i]["chunk_index"] == i

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    def test_extra_kwargs_ignored(self, _mock, tmp_path: Path):
        f = tmp_path / "clip.wav"
        f.write_bytes(b"\x00")
        df = audio_file_to_transcript_df(str(f), grpc_endpoint="localhost:50051", unknown_param="ignored")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# audio_bytes_to_transcript_df
# ---------------------------------------------------------------------------


class TestAudioBytesToTranscriptDf:
    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    def test_basic(self, _mock, tmp_path: Path):
        path = str(tmp_path / "virtual.wav")
        df = audio_bytes_to_transcript_df(b"\x00\x01\x02\x03", path, grpc_endpoint="localhost:50051")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["text"].iloc[0] == SAMPLE_TRANSCRIPT
        assert df["metadata"].iloc[0]["content_metadata"]["type"] == "audio"

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", return_value=([], ""))
    def test_empty_transcript(self, _mock, tmp_path: Path):
        path = str(tmp_path / "silent.wav")
        df = audio_bytes_to_transcript_df(b"\x00", path, grpc_endpoint="localhost:50051")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("retriever.audio.transcribe._transcribe_audio_bytes", side_effect=_mock_transcribe)
    def test_segmented(self, _mock, tmp_path: Path):
        path = str(tmp_path / "clip.mp3")
        df = audio_bytes_to_transcript_df(b"\xff\xfb", path, grpc_endpoint="localhost:50051", segment_audio=True)
        assert len(df) == 2
        assert df["text"].iloc[1] == "jumped over the lazy dog"
