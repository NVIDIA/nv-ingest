# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch
import pandas as pd

# Import the module under test
import nv_ingest_api.internal.extract.html.html_extractor as module_under_test
from nv_ingest_api.internal.extract.html.html_extractor import _convert_html
from nv_ingest_api.internal.enums.common import ContentTypeEnum

# Define module path constant for patching
MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class TestHtmlExtraction(unittest.TestCase):
    """Tests for HTML extraction functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Sample HTML content
        self.sample_content = "<!DOCTYPE html><html><body><h1>The Heading</h1><p>The body.</p></body></html>"

        # Sample valid audio metadata
        self.valid_html_metadata = {
            "content": self.sample_content,
        }

        # Sample row data
        self.sample_row = pd.Series(
            {
                "content": self.sample_content,
                "source_id": "test-html",
                "filename": "test.html",
                "file_size": 12345,
            }
        )

        # Sample task config
        self.sample_task_config = {
            "method": "markitdown",
            "params": {
                "extract_text": True,
                "extract_images": False,
                "extract_tables": False,
                "extract_charts": False,
                "extract_infographics": False,
            },
        }

        # Sample trace info
        self.trace_info = {"request_id": "test-request-123", "timestamp": "2025-03-10T12:00:00Z"}

    @patch(f"{MODULE_UNDER_TEST}.validate_schema")
    def test_convert_html_valid(self, mock_validate_schema):
        """Test _convert_html with valid html content"""
        # Configure mocks
        mock_validate_schema.side_effect = lambda data, schema: Mock(model_dump=lambda: data)

        # Create a sample row with valid audio metadata
        row = pd.Series({"content": self.sample_content, "metadata": self.valid_html_metadata.copy()})

        result = _convert_html(row, self.trace_info)

        self.assertEqual(result[0][1]["content"], "# The Heading\n\nThe body.")
        self.assertEqual(result[0][0], ContentTypeEnum.TEXT)

        # Verify validate_schema was called once
        self.assertEqual(mock_validate_schema.call_count, 1)

    def test_convert_html_empty(self):
        """Test _convert_html with empty html content"""

        # Create a sample row with valid audio metadata
        empty_meta = self.valid_html_metadata.copy()
        empty_meta["content"] = ""
        row = pd.Series({"content": "", "metadata": empty_meta})

        result = _convert_html(row, self.trace_info)

        self.assertEqual(result[0][1]["content"], "")
        self.assertEqual(result[0][0], ContentTypeEnum.TEXT)


if __name__ == "__main__":
    unittest.main()
