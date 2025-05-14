# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import base64
import io

from pydantic import BaseModel

# Import the module under test
import nv_ingest_api.internal.extract.docx.docx_extractor as module_under_test
from nv_ingest_api.internal.extract.docx.docx_extractor import (
    _prepare_task_props,
    _decode_and_extract_from_docx,
    extract_primitives_from_docx_internal,
)

# Define module path constant for patching
MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class TestDocxExtraction(unittest.TestCase):
    """Tests for DOCX extraction functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Sample base64 encoded content (this is not actual DOCX content, just a placeholder)
        self.sample_base64 = base64.b64encode(b"Sample DOCX content").decode("utf-8")

        # Sample row data
        self.sample_row = pd.Series(
            {
                "content": self.sample_base64,
                "source_id": "test-doc-123",
                "filename": "test_document.docx",
                "file_size": 12345,
            }
        )

        # Sample task config
        self.sample_task_config = {
            "method": "python_docx",
            "params": {
                "extract_text": True,
                "extract_images": False,
                "extract_tables": True,
                "extract_charts": False,
                "extract_infographics": False,
                "custom_param": "value",
            },
        }

        # Sample extraction config
        self.mock_extraction_config = Mock()
        self.mock_extraction_config.docx_extraction_config = {
            "max_image_size": 1024,
            "image_quality": "high",
            "table_format": "markdown",
        }

        # Sample trace info
        self.trace_info = {"request_id": "test-request-123", "timestamp": "2025-03-10T12:00:00Z"}

    def test_prepare_task_props_with_dict(self):
        """Test _prepare_task_props with dictionary input."""
        # Call the function with a dictionary
        task_config, source_id = _prepare_task_props(self.sample_task_config, self.sample_row)

        # Verify results
        self.assertEqual(source_id, "test-doc-123")
        self.assertIn("row_data", task_config["params"])
        row_data = task_config["params"]["row_data"]

        # Verify row data doesn't contain 'content' but has other fields
        self.assertNotIn("content", row_data)
        self.assertEqual(row_data["filename"], "test_document.docx")
        self.assertEqual(row_data["file_size"], 12345)

    def test_prepare_task_props_with_model(self):
        """Test _prepare_task_props with Pydantic model input."""
        # Create a mock with proper type identification
        mock_model = Mock(spec=BaseModel)
        mock_model.model_dump.return_value = self.sample_task_config

        # Call the function with the model
        task_config, source_id = _prepare_task_props(mock_model, self.sample_row)

        # Verify results
        self.assertEqual(source_id, "test-doc-123")
        self.assertIn("row_data", task_config["params"])

        # Verify model_dump was called
        mock_model.model_dump.assert_called_once()

        # Verify results
        self.assertEqual(source_id, "test-doc-123")
        self.assertIn("row_data", task_config["params"])

    def test_prepare_task_props_without_params(self):
        """Test _prepare_task_props with a config that has no params key."""
        # Create a task config without params
        task_config_no_params = {"method": "python_docx"}

        # Call the function
        task_config, source_id = _prepare_task_props(task_config_no_params, self.sample_row)

        # Verify params was added
        self.assertIn("params", task_config)
        self.assertIn("row_data", task_config["params"])

    def test_prepare_task_props_without_source_id(self):
        """Test _prepare_task_props with a row that has no source_id."""
        # Create a row without source_id
        row_no_source = pd.Series({"content": self.sample_base64, "filename": "test_document.docx"})

        # Call the function
        task_config, source_id = _prepare_task_props(self.sample_task_config, row_no_source)

        # Verify source_id is None
        self.assertIsNone(source_id)

    @patch(f"{MODULE_UNDER_TEST}.python_docx")
    def test_decode_and_extract_from_docx_success(self, mock_python_docx):
        """Test _decode_and_extract_from_docx with successful extraction."""
        # Configure the mock with a properly structured return value
        expected_result = {
            "document_type": "docx",
            "metadata": {"text_content": "Sample extracted text", "page_count": 5},
            "uuid": "test-uuid-456",
        }
        mock_python_docx.return_value = expected_result

        # Call the function
        result = _decode_and_extract_from_docx(
            self.sample_row, self.sample_task_config, self.mock_extraction_config, self.trace_info
        )

        # Verify python_docx was called with correct params
        mock_python_docx.assert_called_once()
        call_args = mock_python_docx.call_args[1]

        # Check required arguments
        self.assertIsInstance(call_args["docx_stream"], io.BytesIO)
        self.assertTrue(call_args["extract_text"])
        self.assertFalse(call_args["extract_images"])
        self.assertTrue(call_args["extract_tables"])
        self.assertFalse(call_args["extract_charts"])
        self.assertFalse(call_args["extract_infographics"])

        # Check extraction config was passed
        self.assertIn("docx_extraction_config", call_args["extraction_config"])

        # Verify result
        self.assertEqual(result, expected_result)

    @patch(f"{MODULE_UNDER_TEST}.python_docx")
    def test_decode_and_extract_from_docx_missing_flags(self, mock_python_docx):
        """Test _decode_and_extract_from_docx with missing extraction flags."""
        # Create a task config without extraction flags but with empty params dict
        bad_task_config = {
            "method": "python_docx",
            "params": {},  # Empty params will use default values (False) for all flags
        }

        # The function will use default values (all False) instead of raising an error
        # So we expect the function to complete, but with all extraction flags set to False
        result = _decode_and_extract_from_docx(
            self.sample_row, bad_task_config, self.mock_extraction_config, self.trace_info
        )
        _ = result

        # Verify python_docx was called with all extraction flags set to False
        mock_python_docx.assert_called_once()
        call_args = mock_python_docx.call_args[1]

        self.assertFalse(call_args["extract_text"])
        self.assertFalse(call_args["extract_images"])
        self.assertFalse(call_args["extract_tables"])
        self.assertFalse(call_args["extract_charts"])
        self.assertFalse(call_args["extract_infographics"])

    @patch(f"{MODULE_UNDER_TEST}.python_docx")
    def test_decode_and_extract_from_docx_python_docx_error(self, mock_python_docx):
        """Test _decode_and_extract_from_docx when python_docx raises an error."""
        # Configure python_docx to raise an error
        mock_python_docx.side_effect = Exception("Failed to extract content")

        # Expectation: The unified_exception_handler will wrap the error
        with self.assertRaises(Exception) as context:
            _decode_and_extract_from_docx(
                self.sample_row, self.sample_task_config, self.mock_extraction_config, self.trace_info
            )

        # Verify that the error message is wrapped by the decorator
        self.assertIn("_decode_and_extract_from_docx", str(context.exception))
        self.assertIn("Failed to extract content", str(context.exception))

    @patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_docx")
    @patch("pandas.Series.explode")
    @patch("pandas.Series.to_list")
    def test_extract_primitives_from_docx_internal_success(self, mock_to_list, mock_explode, mock_decode_and_extract):
        """Test extract_primitives_from_docx_internal with successful extraction."""
        # Define a dictionary with the correct structure for DataFrame creation
        successful_extraction = {
            "document_type": "docx",
            "metadata": {"text_content": "Sample extracted text", "page_count": 5},
            "uuid": "test-uuid-456",
        }

        # Configure the mocks - simulate the entire data flow
        # 1. mock_decode_and_extract is called for each row
        mock_decode_and_extract.side_effect = [successful_extraction, successful_extraction]

        # 2. The Series is exploded (which normally happens in the function)
        mock_explode.return_value = pd.Series([successful_extraction, successful_extraction])

        # 3. to_list() is called on the Series to convert to list for DataFrame
        mock_to_list.return_value = [
            {"document_type": "docx", "metadata": {"text_content": "Sample extracted text"}, "uuid": "test-uuid-456"},
            {"document_type": "docx", "metadata": {"text_content": "Sample extracted text"}, "uuid": "test-uuid-456"},
        ]

        # Create a DataFrame with multiple rows
        df = pd.DataFrame(
            [{"content": self.sample_base64, "source_id": "doc1"}, {"content": self.sample_base64, "source_id": "doc2"}]
        )

        # Call the function
        result_df, _ = extract_primitives_from_docx_internal(
            df, self.sample_task_config, self.mock_extraction_config, self.trace_info
        )

        # Verify the mock was called for each row
        self.assertEqual(mock_decode_and_extract.call_count, 2)

        # Verify result DataFrame structure
        self.assertEqual(len(result_df), 2)
        self.assertListEqual(list(result_df.columns), ["document_type", "metadata", "uuid"])

        # Verify values in result DataFrame
        for _, row in result_df.iterrows():
            self.assertEqual(row["document_type"], "docx")
            self.assertEqual(row["metadata"]["text_content"], "Sample extracted text")
            self.assertEqual(row["uuid"], "test-uuid-456")

    @patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_docx")
    @patch("pandas.Series.explode")
    @patch("pandas.Series.to_list")
    def test_extract_primitives_from_docx_internal_with_errors(
        self, mock_to_list, mock_explode, mock_decode_and_extract
    ):
        """Test extract_primitives_from_docx_internal when some rows have extraction errors."""
        # Define a dictionary with the correct structure for DataFrame creation
        successful_extraction = {
            "document_type": "docx",
            "metadata": {"text_content": "Sample extracted text", "page_count": 5},
            "uuid": "test-uuid-456",
        }

        # Configure the mocks - simulate the entire data flow
        # 1. mock_decode_and_extract is called for each row
        mock_decode_and_extract.side_effect = [successful_extraction, None]

        # 2. The Series is exploded (which would drop None values)
        mock_explode.return_value = pd.Series([successful_extraction])

        # 3. to_list() is called on the Series to convert to list for DataFrame
        mock_to_list.return_value = [
            {"document_type": "docx", "metadata": {"text_content": "Sample extracted text"}, "uuid": "test-uuid-456"}
        ]

        # Create a DataFrame with multiple rows
        df = pd.DataFrame(
            [{"content": self.sample_base64, "source_id": "doc1"}, {"content": self.sample_base64, "source_id": "doc2"}]
        )

        # Call the function
        result_df, _ = extract_primitives_from_docx_internal(
            df, self.sample_task_config, self.mock_extraction_config, self.trace_info
        )

        # Verify the mock was called for each row
        self.assertEqual(mock_decode_and_extract.call_count, 2)

        # Verify result DataFrame only contains the successful extraction
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["document_type"], "docx")
        self.assertEqual(result_df.iloc[0]["metadata"]["text_content"], "Sample extracted text")

    @patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_docx")
    @patch("pandas.Series.explode")
    @patch("pandas.Series.to_list")
    def test_extract_primitives_from_docx_internal_empty_result(
        self, mock_to_list, mock_explode, mock_decode_and_extract
    ):
        """Test extract_primitives_from_docx_internal with no extraction results."""
        # Configure the mocks
        mock_decode_and_extract.return_value = None
        mock_explode.return_value = pd.Series([])
        mock_to_list.return_value = []

        # Create a DataFrame with a single row
        df = pd.DataFrame([{"content": self.sample_base64, "source_id": "doc1"}])

        # Call the function
        result_df, _ = extract_primitives_from_docx_internal(
            df, self.sample_task_config, self.mock_extraction_config, self.trace_info
        )

        # Verify the mock was called
        mock_decode_and_extract.assert_called_once()

        # Verify result is an empty DataFrame with correct columns
        self.assertEqual(len(result_df), 0)
        self.assertListEqual(list(result_df.columns), ["document_type", "metadata", "uuid"])

    @patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_docx")
    def test_extract_primitives_from_docx_internal_multiple_results_per_document(self, mock_decode_and_extract):
        """Test extract_primitives_from_docx_internal with multiple extraction results per document."""
        # Configure the mock to return a list of results
        multi_result = [
            {
                "document_type": "docx",
                "metadata": {"text_content": "First page content", "page_number": 1},
                "uuid": "uuid-1",
            },
            {
                "document_type": "docx",
                "metadata": {"text_content": "Second page content", "page_number": 2},
                "uuid": "uuid-2",
            },
        ]
        mock_decode_and_extract.return_value = multi_result

        # Create a DataFrame with a single row
        df = pd.DataFrame([{"content": self.sample_base64, "source_id": "doc1"}])

        # Call the function
        result_df, _ = extract_primitives_from_docx_internal(
            df, self.sample_task_config, self.mock_extraction_config, self.trace_info
        )

        # Verify the mock was called
        mock_decode_and_extract.assert_called_once()

        # Verify result DataFrame has two rows (one per extraction result)
        self.assertEqual(len(result_df), 2)

        # Verify values in result DataFrame
        self.assertEqual(result_df.iloc[0]["metadata"]["text_content"], "First page content")
        self.assertEqual(result_df.iloc[0]["metadata"]["page_number"], 1)
        self.assertEqual(result_df.iloc[0]["uuid"], "uuid-1")

        self.assertEqual(result_df.iloc[1]["metadata"]["text_content"], "Second page content")
        self.assertEqual(result_df.iloc[1]["metadata"]["page_number"], 2)
        self.assertEqual(result_df.iloc[1]["uuid"], "uuid-2")

    @patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_docx")
    @patch("pandas.Series.explode")
    @patch("pandas.Series.to_list")
    def test_extract_primitives_from_docx_internal_propagates_exceptions(
        self, mock_to_list, mock_explode, mock_decode_and_extract
    ):
        """Test that extract_primitives_from_docx_internal propagates exceptions from _decode_and_extract_from_docx."""
        # Configure the mock to raise an exception
        mock_decode_and_extract.side_effect = ValueError("Test error in extraction")

        # Create a DataFrame with a single row
        df = pd.DataFrame([{"content": self.sample_base64, "source_id": "doc1"}])

        # Call the function and expect the exception to be propagated
        with self.assertRaises(ValueError) as context:
            extract_primitives_from_docx_internal(
                df, self.sample_task_config, self.mock_extraction_config, self.trace_info
            )

        # Verify error message is wrapped by the unified_exception_handler
        self.assertIn("extract_primitives_from_docx_internal", str(context.exception))
        self.assertIn("Test error in extraction", str(context.exception))

        # Verify the mock was called
        mock_decode_and_extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
