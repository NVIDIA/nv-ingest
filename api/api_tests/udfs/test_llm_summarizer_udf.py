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
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

# Import the UDF functions from API udfs location
from udfs.llm_summarizer_udf import content_summarizer, _extract_content, _generate_summary, _add_summary


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
            },
            {"metadata": {"content": "Short content", "custom_content": {"existing": "data"}}},
            {"metadata": {"content": "A" * 15000, "custom_content": {}}},  # Long content to test truncation
        ]
    )


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
        assert len(df) == 3
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

        # First row should have summary (content long enough)
        first_summary = df.iloc[0]["metadata"]["custom_content"]["llm_summary"]
        assert first_summary["summary"] == "This is a comprehensive summary of the document content."
        assert first_summary["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"

        # Second row should be skipped (content too short with default min length)
        assert "llm_summary" not in df.iloc[1]["metadata"]["custom_content"]

        # Third row should have summary (content long enough, even if truncated)
        third_summary = df.iloc[2]["metadata"]["custom_content"]["llm_summary"]
        assert third_summary["summary"] == "This is a comprehensive summary of the document content."

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
        first_summary = df.iloc[0]["metadata"]["custom_content"]["llm_summary"]
        assert first_summary["model"] == "custom/model"

        # Verify OpenAI client was created with custom config
        mock_openai_class.assert_called_once_with(base_url="https://custom.api.com/v1", api_key="test-key", timeout=30)

        # Second row should now have summary (lower min length threshold)
        second_summary = df.iloc[1]["metadata"]["custom_content"]["llm_summary"]
        assert second_summary["summary"] == "This is a comprehensive summary of the document content."


# Test helper functions


class TestExtractContent:

    def test_extract_content_from_metadata_content(self):
        """Test extracting content from metadata.content."""
        row = pd.Series({"metadata": {"content": "Primary content location"}})

        content = _extract_content(row, Mock())
        assert content == "Primary content location"

    def test_extract_content_from_text_metadata_text(self):
        """Test extracting content from metadata.text_metadata.text."""
        row = pd.Series({"metadata": {"text_metadata": {"text": "Text metadata content"}}})

        content = _extract_content(row, Mock())
        assert content == "Text metadata content"

    def test_extract_content_from_text_metadata_content(self):
        """Test extracting content from metadata.text_metadata.content."""
        row = pd.Series({"metadata": {"text_metadata": {"content": "Text metadata content field"}}})

        content = _extract_content(row, Mock())
        assert content == "Text metadata content field"

    def test_extract_content_from_top_level_content(self):
        """Test extracting content from top-level content field."""
        row = pd.Series({"content": "Top level content", "metadata": {}})

        content = _extract_content(row, Mock())
        assert content == "Top level content"

    def test_extract_content_priority_order(self):
        """Test content extraction priority order."""
        row = pd.Series(
            {
                "content": "Top level content",
                "metadata": {"content": "Metadata content", "text_metadata": {"text": "Text metadata content"}},
            }
        )

        content = _extract_content(row, Mock())
        # metadata.content should take priority
        assert content == "Metadata content"

    def test_extract_content_no_content_found(self):
        """Test when no content is found."""
        row = pd.Series({"metadata": {}})

        content = _extract_content(row, Mock())
        assert content is None

    def test_extract_content_invalid_metadata(self):
        """Test with invalid metadata structure."""
        row = pd.Series({"metadata": "invalid_metadata_type"})

        content = _extract_content(row, Mock())
        assert content is None


class TestGenerateSummary:

    def test_generate_summary_success(self, mock_openai_success):
        """Test successful summary generation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_success

        summary = _generate_summary(mock_client, "Test content", "test-model", Mock())

        assert summary == "This is a comprehensive summary of the document content."
        mock_client.chat.completions.create.assert_called_once()

        # Verify the API call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["max_tokens"] == 400
        assert call_args[1]["temperature"] == 0.7
        assert "Test content" in call_args[1]["messages"][0]["content"]

    def test_generate_summary_empty_response(self):
        """Test handling of empty API response."""
        mock_completion = ChatCompletion(
            id="test-completion-id",
            choices=[],  # Empty choices
            created=1234567890,
            model="test-model",
            object="chat.completion",
            usage=None,
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion

        summary = _generate_summary(mock_client, "Test content", "test-model", Mock())
        assert summary is None

    def test_generate_summary_api_exception(self):
        """Test handling of API exceptions."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_logger = Mock()

        summary = _generate_summary(mock_client, "Test content", "test-model", mock_logger)

        assert summary is None
        mock_logger.error.assert_called_once_with("API call failed: API Error")

    def test_generate_summary_prompt_structure(self):
        """Test that the prompt is properly structured."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content="Summary"))])

        test_content = "Sample document content for testing"
        _generate_summary(mock_client, test_content, "test-model", Mock())

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]

        assert "comprehensive 3-4 sentence summary" in prompt
        assert test_content in prompt
        assert "main purpose, key topics, and important details" in prompt
        assert "Summary:" in prompt


class TestAddSummary:

    def test_add_summary_to_existing_metadata(self):
        """Test adding summary to existing metadata structure."""
        df = pd.DataFrame([{"metadata": {"content": "test content", "custom_content": {"existing": "data"}}}])

        row = df.iloc[0]
        _add_summary(df, 0, row, "Test summary", "test-model", Mock())

        metadata = df.iloc[0]["metadata"]
        assert metadata["custom_content"]["existing"] == "data"  # Preserved
        assert metadata["custom_content"]["llm_summary"]["summary"] == "Test summary"
        assert metadata["custom_content"]["llm_summary"]["model"] == "test-model"

    def test_add_summary_to_empty_custom_content(self):
        """Test adding summary when custom_content doesn't exist."""
        df = pd.DataFrame([{"metadata": {"content": "test content"}}])

        row = df.iloc[0]
        _add_summary(df, 0, row, "Test summary", "test-model", Mock())

        metadata = df.iloc[0]["metadata"]
        assert metadata["custom_content"]["llm_summary"]["summary"] == "Test summary"
        assert metadata["custom_content"]["llm_summary"]["model"] == "test-model"

    def test_add_summary_to_none_metadata(self):
        """Test adding summary when metadata is None."""
        df = pd.DataFrame([{"metadata": None}])

        row = df.iloc[0]
        _add_summary(df, 0, row, "Test summary", "test-model", Mock())

        metadata = df.iloc[0]["metadata"]
        assert metadata["custom_content"]["llm_summary"]["summary"] == "Test summary"
        assert metadata["custom_content"]["llm_summary"]["model"] == "test-model"

    def test_add_summary_to_invalid_metadata_type(self):
        """Test adding summary when metadata is not a dict."""
        df = pd.DataFrame([{"metadata": "invalid_type"}])

        row = df.iloc[0]
        _add_summary(df, 0, row, "Test summary", "test-model", Mock())

        metadata = df.iloc[0]["metadata"]
        assert metadata["custom_content"]["llm_summary"]["summary"] == "Test summary"
        assert metadata["custom_content"]["llm_summary"]["model"] == "test-model"

    def test_add_summary_with_none_custom_content(self):
        """Test adding summary when custom_content is None."""
        df = pd.DataFrame([{"metadata": {"content": "test content", "custom_content": None}}])

        row = df.iloc[0]
        _add_summary(df, 0, row, "Test summary", "test-model", Mock())

        metadata = df.iloc[0]["metadata"]
        assert metadata["custom_content"]["llm_summary"]["summary"] == "Test summary"
        assert metadata["custom_content"]["llm_summary"]["model"] == "test-model"

    def test_add_summary_exception_handling(self):
        """Test exception handling in _add_summary."""
        # Create a DataFrame that will cause an exception
        df = pd.DataFrame([{"metadata": {}}])

        # Mock a row that causes an exception when accessing
        row = Mock()
        row.get.side_effect = Exception("Test exception")

        mock_logger = Mock()

        # Should not raise exception, should log error
        _add_summary(df, 0, row, "Test summary", "test-model", mock_logger)

        mock_logger.error.assert_called_once()
        assert "Failed to add summary to row 0" in mock_logger.error.call_args[0][0]


# Integration Tests


class TestIntegrationScenarios:

    @patch("openai.OpenAI")
    def test_mixed_content_processing(self, mock_openai_class, mock_openai_success):
        """Test processing DataFrame with mixed content scenarios."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_success
        mock_openai_class.return_value = mock_client

        # Create DataFrame with various content scenarios
        df = pd.DataFrame(
            [
                {"metadata": {"content": "Long content " * 20, "custom_content": {}}},  # Should be summarized
                {"metadata": {"content": "Short", "custom_content": {}}},  # Too short
                {
                    "metadata": {"text_metadata": {"text": "Text metadata content " * 10}, "custom_content": {}}
                },  # Alternative location
                {"metadata": {"custom_content": {}}},  # No content
                {"metadata": None},  # Invalid metadata
            ]
        )

        control_message = MockIngestControlMessage(df)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key", "LLM_MIN_CONTENT_LENGTH": "50"}):
            result = content_summarizer(control_message)

        result_df = result.payload()

        # Row 0: Should have summary (long content)
        assert "llm_summary" in result_df.iloc[0]["metadata"]["custom_content"]

        # Row 1: Should not have summary (too short)
        assert "llm_summary" not in result_df.iloc[1]["metadata"]["custom_content"]

        # Row 2: Should have summary (content from text_metadata)
        assert "llm_summary" in result_df.iloc[2]["metadata"]["custom_content"]

        # Row 3: Should not have summary (no content)
        assert "llm_summary" not in result_df.iloc[3]["metadata"]["custom_content"]

        # Row 4: Should not have summary (invalid metadata, now corrected)
        if result_df.iloc[4]["metadata"] is not None:
            assert "llm_summary" not in result_df.iloc[4]["metadata"].get("custom_content", {})

    @patch("openai.OpenAI")
    def test_content_length_thresholds(self, mock_openai_class, mock_openai_success):
        """Test content length filtering and truncation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_success
        mock_openai_class.return_value = mock_client

        df = pd.DataFrame(
            [
                {"metadata": {"content": "A" * 15000, "custom_content": {}}},  # Should be truncated
                {"metadata": {"content": "Short content", "custom_content": {}}},  # Should be skipped
            ]
        )

        control_message = MockIngestControlMessage(df)

        env_vars = {"NVIDIA_API_KEY": "test-key", "LLM_MIN_CONTENT_LENGTH": "50", "LLM_MAX_CONTENT_LENGTH": "10000"}

        with patch.dict(os.environ, env_vars):
            result = content_summarizer(control_message)  # noqa: F841

        # Verify the API was called with truncated content
        call_args = mock_client.chat.completions.create.call_args_list[0]
        sent_content = call_args[1]["messages"][0]["content"]
        # The content in the prompt should be truncated to 10000 characters
        # (plus the additional prompt text)
        assert "A" * 10000 in sent_content
        assert len(sent_content.split("Summary:")[0].split("\n\n")[1]) == 10000


if __name__ == "__main__":
    pytest.main([__file__])
