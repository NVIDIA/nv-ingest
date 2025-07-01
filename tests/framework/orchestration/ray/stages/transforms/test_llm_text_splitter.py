# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from openai import OpenAI

from nv_ingest_api.internal.schemas.transform.transform_llm_text_splitter_schema import LLMTextSplitterSchema
from nv_ingest_api.internal.transform.llm_split_text import (
    _split_by_markdown,
    transform_text_split_llm_internal,
)
from transformers import AutoTokenizer


class TestLLMTextSplitter(unittest.TestCase):
    def setUp(self):
        """Set up common resources for tests."""
        self.config = LLMTextSplitterSchema()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.sample_markdown = (
            "Content before header.\n\n"
            "# Header 1\n"
            "This is the first section.\n\n"
            "## Header 2\n"
            "This is a subsection.\n\n"
            "# Header 3\n"
            "This is the third section."
        )
        self.long_chunk = " ".join(["word"] * 600)
        self.valid_llm_split_point = " ".join(["word"] * 15)

    def test_markdown_splitting_and_hierarchy(self):
        """Verify basic markdown splitting and correct hierarchical header generation."""
        chunks, headers = _split_by_markdown(self.sample_markdown, self.tokenizer, self.config, None)

        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(headers), 4)

        self.assertEqual(chunks[0], "Content before header.")
        self.assertEqual(headers[0], "")

        self.assertIn("This is the first section.", chunks[1])
        self.assertEqual(headers[1], "Header 1")

        self.assertIn("This is a subsection.", chunks[2])
        self.assertEqual(headers[2], "Header 1 > Header 2")

        self.assertIn("This is the third section.", chunks[3])
        self.assertEqual(headers[3], "Header 3")

    def test_no_headers_in_text(self):
        """Test that text with no headers is treated as a single chunk."""
        text = "This is a simple text document without any markdown headers."
        chunks, headers = _split_by_markdown(text, self.tokenizer, self.config, None)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
        self.assertEqual(headers[0], "")

    def test_document_starting_with_header(self):
        """Test that a document starting with a header is parsed correctly."""
        text = "# First Header\nContent of first header."
        chunks, headers = _split_by_markdown(text, self.tokenizer, self.config, None)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Content of first header.")
        self.assertEqual(headers[0], "First Header")

    def test_empty_content_between_headers(self):
        """Test that empty content between headers is handled gracefully."""
        text = "# Header 1\n\n## Header 2\nContent for H2"
        chunks, headers = _split_by_markdown(text, self.tokenizer, self.config, None)
        # The logic should not produce a chunk for Header 1 as it has no content.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Content for H2")
        self.assertEqual(headers[0], "Header 1 > Header 2")

    def test_header_with_extra_whitespace(self):
        """Test that headers with extra whitespace are parsed correctly."""
        text = "#   Extra Space Header   \nSome content."
        chunks, headers = _split_by_markdown(text, self.tokenizer, self.config, None)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(headers[0], "Extra Space Header")

    def test_complex_hierarchy(self):
        """Test correct handling of a more complex header hierarchy."""
        text = (
            "# H1\nContent 1\n"
            "## H2a\nContent 2a\n"
            "### H3\nContent 3\n"
            "## H2b\nContent 2b"
        )
        chunks, headers = _split_by_markdown(text, self.tokenizer, self.config, None)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(headers[0], "H1")
        self.assertEqual(headers[1], "H1 > H2a")
        self.assertEqual(headers[2], "H1 > H2a > H3")
        self.assertEqual(headers[3], "H1 > H2b")

    def test_full_transform_with_audio_payload(self):
        """Verify the full transform function handles audio payloads correctly."""
        audio_doc = {
            "document_type": "audio",
            "metadata": {
                "source": "test.mp3",
                "audio_metadata": {"audio_transcript": "# Title\nContent of audio."}
            }
        }
        df = pd.DataFrame([audio_doc])
        result_df = transform_text_split_llm_internal(df, {}, self.config, None)

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["document_type"], "audio")
        self.assertEqual(result_df.iloc[0]["metadata"]["custom_content"]["hierarchical_header"], "Title")
        self.assertEqual(result_df.iloc[0]["metadata"]["audio_metadata"]["audio_transcript"], "Content of audio.")

    def test_full_transform_with_mixed_payload(self):
        """Verify the full transform function handles mixed-type dataframes."""
        text_doc = {"document_type": "text", "metadata": {"content": "# Header\nText content"}}
        image_doc = {"document_type": "image", "metadata": {"path": "/path/to/image.jpg"}}
        df = pd.DataFrame([text_doc, image_doc])
        result_df = transform_text_split_llm_internal(df, {}, self.config, None)

        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df.iloc[0]["document_type"], "text")
        self.assertEqual(result_df.iloc[1]["document_type"], "image")
        # Check that the image doc is untouched
        self.assertNotIn("custom_content", result_df.iloc[1]["metadata"])

    def test_task_config_override(self):
        """Verify that task-level config overrides stage-level config."""
        text = "# H1\nContent 1\n## H2\nContent 2"
        # This task config should only split on H1
        task_config = {"markdown_headers_to_split_on": ["#"]}
        df = pd.DataFrame([{"document_type": "text", "metadata": {"content": text}}])
        result_df = transform_text_split_llm_internal(df, task_config, self.config, None)

        # Expecting only one split, as ## is ignored. The content of H2 becomes part of the H1 chunk.
        self.assertEqual(len(result_df), 1)
        self.assertIn("Content 1\n## H2\nContent 2", result_df.iloc[0]["metadata"]["content"])
        self.assertEqual(result_df.iloc[0]["metadata"]["custom_content"]["hierarchical_header"], "H1")

    @patch("nv_ingest_api.internal.transform.llm_split_text._get_llm_split_point")
    def test_llm_sub_splitting_success(self, mock_get_split_point):
        """Test successful sub-splitting of an oversized chunk using a mocked LLM."""
        # By mocking a level deeper, we avoid the client instantiation issues
        mock_get_split_point.return_value = self.valid_llm_split_point

        config = self.config.model_copy(update={"subsplit_with_llm": True, "llm_endpoint": "https://fake.endpoint"})
        chunks, headers = _split_by_markdown(self.long_chunk, self.tokenizer, config, MagicMock())

        mock_get_split_point.assert_called_once()
        self.assertEqual(len(chunks), 2)
        # Verify the split happened at the point returned by the LLM
        self.assertTrue(chunks[1].startswith(self.valid_llm_split_point))
        self.assertEqual(headers[0], headers[1])  # Header should be propagated

    @patch("nv_ingest_api.internal.transform.llm_split_text._get_llm_split_point")
    def test_llm_sub_splitting_invalid_response_fallback(self, mock_get_split_point):
        """Test fallback to hard splitting when LLM gives an invalid response."""
        # Simulate the function returning None, as it would on API failure or invalid response
        mock_get_split_point.return_value = None

        config = self.config.model_copy(
            update={"subsplit_with_llm": True, "chunk_size": 256, "chunk_overlap": 20, "llm_endpoint": "https://fake.endpoint"}
        )
        chunks, headers = _split_by_markdown(self.long_chunk, self.tokenizer, config, MagicMock())

        mock_get_split_point.assert_called_once()
        # Should fall back to hard splitting, which will create more than 2 chunks
        self.assertGreater(len(chunks), 2)

    @patch("nv_ingest_api.internal.transform.llm_split_text._get_llm_split_point")
    def test_llm_api_failure_fallback(self, mock_get_split_point):
        """Test fallback to hard splitting when the LLM API call fails."""
        mock_get_split_point.return_value = None

        config = self.config.model_copy(
            update={"subsplit_with_llm": True, "chunk_size": 256, "chunk_overlap": 20, "llm_endpoint": "https://fake.endpoint"}
        )
        chunks, headers = _split_by_markdown(self.long_chunk, self.tokenizer, config, MagicMock())

        mock_get_split_point.assert_called_once()
        self.assertGreater(len(chunks), 2)

    @patch("nv_ingest_api.internal.transform.llm_split_text._get_llm_split_point")
    def test_llm_safety_valve(self, mock_get_split_point):
        """Test that the max_llm_splits_per_document safety valve is respected."""
        mock_get_split_point.return_value = self.valid_llm_split_point

        # Create a document with 5 oversized chunks
        five_long_chunks = "\n\n# Page\n\n".join([self.long_chunk] * 5)
        config = self.config.model_copy(
            update={
                "subsplit_with_llm": True,
                "chunk_size": 256,
                "max_llm_splits_per_document": 2,
                "llm_endpoint": "https://fake.endpoint",
            }
        )

        # The internal function is what manages the counter
        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "metadata": {"content": five_long_chunks},
                }
            ]
        )

        # We need to set the API key for the client to be created
        with patch.dict(os.environ, {config.llm_api_key_env_var: "fake-key"}):
            transform_text_split_llm_internal(df, {}, config, None)

        # Assert that the LLM was only called twice, respecting the safety limit
        self.assertEqual(mock_get_split_point.call_count, 2)


if __name__ == "__main__":
    unittest.main() 