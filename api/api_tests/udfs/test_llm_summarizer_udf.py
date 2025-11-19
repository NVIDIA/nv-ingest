#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for LLM Content Summarizer UDF.

Tests the content_summarizer UDF function and its helper functions.
"""

import os

# Standard library and third-party imports
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Import the UDF functions from API udfs location
from udfs.llm_summarizer_udf import content_summarizer, _extract_content, _store_summary

pytest.importorskip("openai")
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice


class MockIngestControlMessage:
    """Mock IngestControlMessage for testing."""

    def __init__(self, payload_df=None):
        self._payload = payload_df

    def payload(self, new_payload=None):
        if new_payload is not None:
            self._payload = new_payload
        return self._payload


# Test Data Fixtures


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with text content."""
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "This is a sample document about artificial intelligence. " * 10,
                    "custom_content": {},
                }
            }
        ]
    )


@pytest.fixture
def stats():
    """Create a mock default stats dict"""
    return {"skipped": False}


@pytest.fixture
def dataframe_with_text_metadata():
    """Create DataFrame with content in text_metadata location."""
    return pd.DataFrame(
        [{"metadata": {"text_metadata": {"text": "Content stored in text_metadata field."}, "custom_content": {}}}]
    )


@pytest.fixture
def dataframe_no_content():
    """Create DataFrame with no content."""
    return pd.DataFrame([{"metadata": {"custom_content": {}}}, {"metadata": None}])


@pytest.fixture
def mock_openai_success():
    """Mock successful OpenAI response."""
    mock_choice = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant", content="This is a comprehensive summary of the document content."
        ),
        finish_reason="stop",
    )
    mock_completion = ChatCompletion(
        id="test-completion-id",
        choices=[mock_choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=None,
    )
    return mock_completion


# Test content_summarizer main function


class TestContentSummarizer:

    def test_content_summarizer_no_api_key(self, sample_dataframe):
        """Test UDF behavior when NVIDIA_API_KEY is not set."""
        control_message = MockIngestControlMessage(sample_dataframe)

        with patch.dict(os.environ, {}, clear=True):
            result = content_summarizer(control_message)

        assert result is control_message
        # DataFrame should be unchanged
        df = result.payload()
        assert len(df) == len(sample_dataframe)
        assert "llm_summary" not in df.iloc[0]["metadata"]["custom_content"]

    def test_content_summarizer_no_payload(self):
        """Test UDF behavior when control message has no payload."""
        control_message = MockIngestControlMessage(None)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            result = content_summarizer(control_message)

        assert result is control_message

    def test_content_summarizer_empty_payload(self):
        """Test UDF behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        control_message = MockIngestControlMessage(empty_df)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            result = content_summarizer(control_message)

        assert result is control_message

    @patch("openai.OpenAI")
    def test_content_summarizer_openai_client_failure(self, mock_openai_class, sample_dataframe):
        """Test UDF behavior when OpenAI client initialization fails."""
        mock_openai_class.side_effect = Exception("Client initialization failed")
        control_message = MockIngestControlMessage(sample_dataframe)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            result = content_summarizer(control_message)

        assert result is control_message
        # DataFrame should be unchanged
        df = result.payload()
        assert "llm_summary" not in df.iloc[0]["metadata"]["custom_content"]

    @patch("openai.OpenAI")
    def test_content_summarizer_successful_processing(self, mock_openai_class, sample_dataframe, mock_openai_success):
        """Test successful processing with LLM summarization."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_success
        mock_openai_class.return_value = mock_client

        control_message = MockIngestControlMessage(sample_dataframe.copy())

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            result = content_summarizer(control_message)

        assert result is control_message
        df = result.payload()

        first_summary = df.iloc[0]["metadata"]["custom_content"]["llm_summarizer_udf"]
        assert isinstance(first_summary["summary"], str)
        assert first_summary["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"

    @patch("openai.OpenAI")
    def test_content_summarizer_with_custom_config(self, mock_openai_class, sample_dataframe, mock_openai_success):
        """Test UDF with custom configuration via environment variables."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_success
        mock_openai_class.return_value = mock_client

        control_message = MockIngestControlMessage(sample_dataframe.copy())

        env_vars = {
            "NVIDIA_API_KEY": "test-key",
            "LLM_SUMMARIZATION_MODEL": "custom/model",
            "LLM_SUMMARIZATION_BASE_URL": "https://custom.api.com/v1",
            "LLM_SUMMARIZATION_TIMEOUT": "30",
            "LLM_MIN_CONTENT_LENGTH": "10",  # Lower threshold
            "LLM_MAX_CONTENT_LENGTH": "8000",
        }

        with patch.dict(os.environ, env_vars):
            result = content_summarizer(control_message)

        df = result.payload()

        # Verify custom model is used
        first_summary = df.iloc[0]["metadata"]["custom_content"]["llm_summarizer_udf"]
        assert first_summary["model"] == "custom/model"


# Test helper functions


class TestExtractContent:
    """Test extracting content from row in payload DataFrame"""

    def test_default(self, stats):
        valid_content = "Lorem ipsum" * 50
        valid_row = pd.Series({"metadata": {"content": valid_content}})
        returned_content = _extract_content(valid_row, stats)

        assert returned_content == valid_content
        assert stats["skipped"] is False

    def test_too_short(self, stats):
        short_row = pd.Series({"metadata": {"content": "Lorem ipsum"}})
        returned_content = _extract_content(short_row, stats)

        assert returned_content == ""
        assert stats["skipped"] is True

    def test_too_long(self, stats):
        long_row = pd.Series({"metadata": {"content": "C" * 20_000}})
        returned_content = _extract_content(long_row, stats)

        assert isinstance(returned_content, str) and len(returned_content) == 12_000
        assert stats["skipped"] is False

    def test_no_content(self, stats):
        no_content_row = pd.Series({"metadata": None})
        returned_content = _extract_content(no_content_row, stats)

        assert returned_content == ""
        assert stats["skipped"] is True


def test_store_summary_to_payload():
    """Test if content is stored in dataframe's metadata"""
    df = pd.DataFrame([{"metadata": {"content": "test content", "custom_content": {"existing": "data"}}}])

    summary = "Summary to store"  # simluate LLM summary
    _store_summary(df, summary, "test-model")

    metadata = df.iloc[0]["metadata"]
    assert metadata["custom_content"]["existing"] == "data"  # Preserved
    assert metadata["custom_content"]["llm_summarizer_udf"]["summary"] == summary
    assert metadata["custom_content"]["llm_summarizer_udf"]["model"] == "test-model"


if __name__ == "__main__":
    pytest.main([__file__])
