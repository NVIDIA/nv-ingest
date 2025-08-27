#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Structural Text Splitter UDF.

Tests the structural_split UDF function and its helper functions.
"""

# Add the examples/udfs directory to the path for importing the UDF
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../udfs"))

# Standard library imports
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import uuid  # noqa: E402
import base64  # noqa: E402
from unittest.mock import patch  # noqa: E402

# Import the UDF functions
from structural_split_udf import (  # noqa: E402
    structural_split,
    structural_split_coarse,
    _split_by_markdown_headers,
    _extract_header_info,
)


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
def markdown_content():
    """Sample markdown content with various header levels."""
    return """# Introduction
This is the introduction section with some content.

## Getting Started
Here's how to get started with the system.

### Prerequisites
You need these things first:
- Python 3.8+
- Required packages

### Installation
Run these commands to install.

## Advanced Usage
This section covers advanced topics.

### Configuration
Details about configuration options.

#### Database Settings
Specific database configuration.

## Conclusion
Final thoughts and summary."""


@pytest.fixture
def sample_text_dataframe():
    """Create a sample DataFrame with text content."""
    return pd.DataFrame(
        [
            {
                "document_type": "text",
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "content": "# Sample Header\nSome content here.\n## Second Header\nMore content.",
                    "source_metadata": {"source_id": "test1"},
                },
            },
            {
                "document_type": "text",
                "uuid": str(uuid.uuid4()),
                "metadata": {"content": "Plain text without headers", "source_metadata": {"source_id": "test2"}},
            },
        ]
    )


@pytest.fixture
def mixed_document_dataframe():
    """Create DataFrame with mixed document types."""
    return pd.DataFrame(
        [
            {
                "document_type": "text",
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "content": "# Text Document\nThis should be split.\n\n## Second Section\nMore content here.",
                    "source_metadata": {"source_id": "text1"},
                },
            },
            {
                "document_type": "image",
                "uuid": str(uuid.uuid4()),
                "metadata": {"content": "image data", "source_metadata": {"source_id": "img1"}},
            },
            {
                "document_type": "pdf",
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "content": "# PDF Content\nThis should not be split.\n\n## PDF Section\nMore PDF content.",
                    "source_metadata": {"source_id": "pdf1"},
                },
            },
        ]
    )


@pytest.fixture
def base64_content_dataframe():
    """Create DataFrame with base64-encoded content."""
    markdown_text = "# Encoded Header\nThis was base64 encoded.\n## Second Section\nMore content."
    encoded_content = base64.b64encode(markdown_text.encode("utf-8")).decode("ascii")

    return pd.DataFrame(
        [
            {
                "document_type": "text",
                "uuid": str(uuid.uuid4()),
                "metadata": {"content": encoded_content, "source_metadata": {"source_id": "encoded1"}},
            }
        ]
    )


# Test main UDF function


class TestStructuralSplit:

    def test_structural_split_no_payload(self):
        """Test UDF behavior when control message has no payload."""
        control_message = MockIngestControlMessage(None)

        result = structural_split(control_message)

        assert result is control_message
        assert result.payload() is None

    def test_structural_split_empty_payload(self):
        """Test UDF behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        control_message = MockIngestControlMessage(empty_df)

        result = structural_split(control_message)

        assert result is control_message
        result_df = result.payload()
        assert len(result_df) == 0

    def test_structural_split_basic_splitting(self, sample_text_dataframe):
        """Test basic structural splitting functionality."""
        control_message = MockIngestControlMessage(sample_text_dataframe.copy())

        result = structural_split(control_message)

        result_df = result.payload()

        # Should have more rows after splitting (first row should be split)
        assert len(result_df) > len(sample_text_dataframe)

        # Check that chunks were created with proper metadata
        split_chunks = result_df[
            result_df["metadata"].apply(
                lambda x: "custom_content" in x and "chunk_index" in x.get("custom_content", {})
            )
        ].copy()
        assert len(split_chunks) >= 2  # At least 2 chunks from "# Sample Header" and "## Second Header"

        # Verify chunk metadata
        first_chunk = split_chunks.iloc[0]["metadata"]["custom_content"]
        assert "chunk_index" in first_chunk
        assert "total_chunks" in first_chunk
        assert "splitting_method" in first_chunk
        assert first_chunk["splitting_method"] == "structural_markdown"

    def test_structural_split_no_headers_content(self, sample_text_dataframe):
        """Test with content that has no markdown headers."""
        # Modify the sample to have no headers
        df = sample_text_dataframe.copy()
        df.at[0, "metadata"] = {
            "content": "Plain text content with no headers at all. Just regular paragraphs.",
            "source_metadata": {"source_id": "test1"},
        }

        control_message = MockIngestControlMessage(df)

        result = structural_split(control_message)

        result_df = result.payload()

        # Should have same number of rows (no splitting)
        assert len(result_df) == len(df)

        # Original row should be preserved
        assert (
            result_df.iloc[0]["metadata"]["content"]
            == "Plain text content with no headers at all. Just regular paragraphs."
        )

    def test_structural_split_mixed_document_types(self, mixed_document_dataframe):
        """Test that only text documents are processed."""
        control_message = MockIngestControlMessage(mixed_document_dataframe.copy())

        result = structural_split(control_message)

        result_df = result.payload()

        # Should have more rows due to text document splitting
        assert len(result_df) > len(mixed_document_dataframe)

        # Non-text documents should be unchanged
        image_rows = result_df[result_df["document_type"] == "image"]
        pdf_rows = result_df[result_df["document_type"] == "pdf"]

        assert len(image_rows) == 1
        assert len(pdf_rows) == 1
        assert image_rows.iloc[0]["metadata"]["content"] == "image data"
        assert (
            pdf_rows.iloc[0]["metadata"]["content"]
            == "# PDF Content\nThis should not be split.\n\n## PDF Section\nMore PDF content."
        )

    def test_structural_split_base64_content(self, base64_content_dataframe):
        """Test handling of base64-encoded content."""
        control_message = MockIngestControlMessage(base64_content_dataframe.copy())

        result = structural_split(control_message)

        result_df = result.payload()

        # Should have been decoded and split
        assert len(result_df) > len(base64_content_dataframe)

        # Check that content was properly decoded and split
        chunks = result_df[
            result_df["metadata"].apply(
                lambda x: "custom_content" in x and "chunk_index" in x.get("custom_content", {})
            )
        ]
        assert len(chunks) >= 2  # Should be split into at least 2 chunks

        # First chunk should contain decoded header
        first_chunk_content = chunks.iloc[0]["metadata"]["content"]
        assert "# Encoded Header" in first_chunk_content
        assert "This was base64 encoded" in first_chunk_content

    def test_structural_split_empty_content(self):
        """Test with empty or whitespace-only content."""
        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {"content": "", "source_metadata": {"source_id": "empty1"}},
                },
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {"content": "   \n\t  ", "source_metadata": {"source_id": "whitespace1"}},
                },
            ]
        )

        control_message = MockIngestControlMessage(df)

        result = structural_split(control_message)

        result_df = result.payload()

        # Should preserve rows even with empty content
        assert len(result_df) == len(df)

    def test_structural_split_comprehensive_markdown(self, markdown_content):
        """Test with comprehensive markdown content containing various header levels."""
        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {"content": markdown_content, "source_metadata": {"source_id": "comprehensive1"}},
                }
            ]
        )

        control_message = MockIngestControlMessage(df)

        result = structural_split(control_message)

        result_df = result.payload()

        # Should be split into multiple chunks
        assert len(result_df) > 5  # Expecting multiple header sections

        # All chunks should have proper metadata
        chunks = result_df[result_df["metadata"].apply(lambda x: "custom_content" in x)]
        for idx, chunk in chunks.iterrows():
            custom_content = chunk["metadata"]["custom_content"]
            assert "chunk_index" in custom_content
            assert "total_chunks" in custom_content
            assert "splitting_method" in custom_content
            assert "hierarchical_header" in custom_content
            assert "header_level" in custom_content

        # Check specific headers are preserved
        chunk_contents = [chunk["metadata"]["content"] for _, chunk in chunks.iterrows()]
        header_found = any("# Introduction" in content for content in chunk_contents)
        assert header_found, "Introduction header should be found in chunks"


class TestSplitByMarkdownHeaders:

    def test_split_empty_text(self):
        """Test splitting empty or None text."""
        headers = ["#", "##", "###"]

        result = _split_by_markdown_headers("", headers)
        assert result == [""]

        result = _split_by_markdown_headers(None, headers)
        assert result == [None]

    def test_split_no_headers(self):
        """Test text with no markdown headers."""
        text = "This is plain text with no headers.\nJust regular paragraphs.\nNothing special here."
        headers = ["#", "##", "###"]

        result = _split_by_markdown_headers(text, headers)
        assert result == [text]

    def test_split_single_header(self):
        """Test text with a single header."""
        text = "# Main Title\nThis is the content under the main title.\nMore content here."
        headers = ["#", "##", "###"]

        result = _split_by_markdown_headers(text, headers)
        assert len(result) == 1
        assert result[0] == text

    def test_split_multiple_headers(self):
        """Test text with multiple headers."""
        text = """# First Header
Content for first section.

## Second Header
Content for second section.
More content here.

### Third Header
Content for third section."""

        headers = ["#", "##", "###"]

        result = _split_by_markdown_headers(text, headers)

        assert len(result) == 3
        assert "# First Header" in result[0]
        assert "## Second Header" in result[1]
        assert "### Third Header" in result[2]

        # Check that content is properly split
        assert "Content for first section" in result[0]
        assert "Content for second section" in result[1]
        assert "Content for third section" in result[2]

    def test_split_with_empty_sections(self):
        """Test splitting where some sections might be empty."""
        text = """# First Header

## Empty Section

### Another Section
Some content here.

#### Yet Another Section

# Final Section
Final content."""

        headers = ["#", "##", "###", "####"]

        result = _split_by_markdown_headers(text, headers)

        # Should skip empty sections
        non_empty_chunks = [chunk for chunk in result if chunk.strip()]
        assert len(non_empty_chunks) >= 3  # At least the sections with content

    def test_split_header_patterns(self):
        """Test various header patterns and edge cases."""
        text = """# Header with spaces
Content here.

##NoSpaceHeader
This should still work.

### Header with # symbols in content
Content with # symbols should not interfere.

#### Header with multiple    spaces
Content here."""

        headers = ["#", "##", "###", "####"]

        result = _split_by_markdown_headers(text, headers)

        assert len(result) >= 3  # Should identify most headers

        # Check that headers are preserved in chunks
        assert any("# Header with spaces" in chunk for chunk in result)
        assert any("##NoSpaceHeader" in chunk for chunk in result)


class TestExtractHeaderInfo:

    def test_extract_header_info_basic(self):
        """Test basic header information extraction."""
        lines = ["# Main Title", "Some content here", "More content"]
        headers = ["#", "##", "###"]

        result = _extract_header_info(lines, headers)

        assert result["hierarchical_header"] == "# Main Title"
        assert result["header_level"] == 1
        assert result["parent_headers"] == []

    def test_extract_header_info_different_levels(self):
        """Test header extraction for different levels."""
        test_cases = [
            (["## Second Level", "content"], 2, "## Second Level"),
            (["### Third Level Header", "content"], 3, "### Third Level Header"),
            (["#### Fourth Level", "content"], 4, "#### Fourth Level"),
        ]

        headers = ["#", "##", "###", "####", "#####", "######"]

        for lines, expected_level, expected_header in test_cases:
            result = _extract_header_info(lines, headers)
            assert result["header_level"] == expected_level
            assert result["hierarchical_header"] == expected_header

    def test_extract_header_info_no_headers(self):
        """Test when no headers are found."""
        lines = ["Just plain text", "No headers here", "More plain text"]
        headers = ["#", "##", "###"]

        result = _extract_header_info(lines, headers)

        assert result["hierarchical_header"] == "(no headers found)"
        assert result["header_level"] == 0
        assert result["parent_headers"] == []

    def test_extract_header_info_first_header_only(self):
        """Test that only the first header is extracted."""
        lines = ["## First Header", "Some content", "### Second Header", "More content"]
        headers = ["#", "##", "###"]

        result = _extract_header_info(lines, headers)

        # Should only extract the first header
        assert result["hierarchical_header"] == "## First Header"
        assert result["header_level"] == 2

    def test_extract_header_info_whitespace_handling(self):
        """Test handling of whitespace around headers."""
        lines = ["  ### Spaced Header  ", "content"]
        headers = ["#", "##", "###"]

        result = _extract_header_info(lines, headers)

        assert result["hierarchical_header"] == "### Spaced Header"
        assert result["header_level"] == 3


class TestStructuralSplitCoarse:

    def test_structural_split_coarse_calls_main(self, sample_text_dataframe):
        """Test that coarse splitting calls the main structural_split function."""
        control_message = MockIngestControlMessage(sample_text_dataframe.copy())

        # Mock the main structural_split function
        with patch("structural_split_udf.structural_split") as mock_structural_split:
            mock_structural_split.return_value = control_message

            result = structural_split_coarse(control_message)

            # Should call the main function
            mock_structural_split.assert_called_once_with(control_message)
            assert result is control_message


# Integration Tests


class TestIntegrationScenarios:

    def test_end_to_end_markdown_processing(self):
        """Test complete end-to-end processing of markdown document."""
        markdown_doc = """# User Guide

Welcome to our system!

## Installation

### Requirements
- Python 3.8+
- 4GB RAM

### Setup Steps
1. Download the package
2. Run installation

## Usage

### Basic Usage
Start with these commands.

### Advanced Features
For power users.

## Troubleshooting

Common issues and solutions."""

        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {
                        "content": markdown_doc,
                        "source_metadata": {"source_id": "user_guide", "source_type": "markdown"},
                        "content_metadata": {"type": "text"},
                    },
                }
            ]
        )

        control_message = MockIngestControlMessage(df)

        result = structural_split(control_message)

        result_df = result.payload()

        # Should create multiple chunks
        assert len(result_df) > 5

        # All chunks should have proper structure
        for idx, row in result_df.iterrows():
            metadata = row["metadata"]
            if "custom_content" in metadata:
                custom = metadata["custom_content"]
                assert "chunk_index" in custom
                assert "total_chunks" in custom
                assert "splitting_method" in custom
                assert custom["splitting_method"] == "structural_markdown"
                assert "hierarchical_header" in custom
                assert "header_level" in custom
                assert isinstance(custom["header_level"], int)
                assert custom["header_level"] >= 0

    def test_mixed_content_and_document_types(self):
        """Test processing mixed content with various document types and content scenarios."""
        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {
                        "content": "# Markdown Doc\nWith headers to split.\n\n## Second Header\nMore content.",
                        "source_metadata": {"source_id": "md1"},
                    },
                },
                {
                    "document_type": "image",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {"content": "binary image data", "source_metadata": {"source_id": "img1"}},
                },
                {
                    "document_type": "text",
                    "uuid": str(uuid.uuid4()),
                    "metadata": {"content": "Plain text without headers", "source_metadata": {"source_id": "plain1"}},
                },
                {
                    "document_type": "TEXT",  # Test case insensitive
                    "uuid": str(uuid.uuid4()),
                    "metadata": {
                        "content": "## Another Text Doc\nWith header.\n\n### Subsection\nSubsection content here.",
                        "source_metadata": {"source_id": "text2"},
                    },
                },
            ]
        )

        control_message = MockIngestControlMessage(df)

        result = structural_split(control_message)

        result_df = result.payload()

        # Should have more rows due to splitting
        assert len(result_df) > len(df)

        # Check document type preservation
        doc_types = set(result_df["document_type"].unique())
        assert "image" in doc_types
        assert "text" in doc_types or "TEXT" in doc_types

        # Image should be unchanged
        image_rows = result_df[result_df["document_type"] == "image"]
        assert len(image_rows) == 1
        assert image_rows.iloc[0]["metadata"]["content"] == "binary image data"

        # Text documents should be processed
        text_rows = result_df[result_df["document_type"].str.lower() == "text"]
        chunks_with_metadata = text_rows[text_rows["metadata"].apply(lambda x: "custom_content" in x)]
        assert len(chunks_with_metadata) >= 2  # At least some chunks created


if __name__ == "__main__":
    pytest.main([__file__])
