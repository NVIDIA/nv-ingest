# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch
import pandas as pd

# Import the module under test
import nv_ingest_api.internal.extract.image.chart_extractor as module_under_test
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal

# Define module path constant for patching
MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class TestExtractChartDataFromImageInternal(unittest.TestCase):
    """Tests for extract_chart_data_from_image_internal function."""

    def setUp(self):
        """Set up common test fixtures."""
        # Sample trace info
        self.trace_info = {"request_id": "test-request-123", "timestamp": "2025-03-10T12:00:00Z"}

        # Sample chart content
        self.chart_content = {
            "chart_type": "bar",
            "title": "Sample Chart",
            "x_label": "Categories",
            "y_label": "Values",
            "data": [{"x": "A", "y": 10}, {"x": "B", "y": 20}, {"x": "C", "y": 15}],
        }

        # Mock clients
        self.mock_yolox_client = Mock()
        self.mock_ocr_client = Mock()

        # Sample task config (not used by the function but required in signature)
        self.task_config = {"param": "value"}

        # Sample extraction config
        self.extraction_config = Mock()
        endpoint_config = Mock()
        endpoint_config.yolox_endpoints = ("yolox_grpc", "yolox_http")
        endpoint_config.yolox_infer_protocol = "grpc"
        endpoint_config.ocr_endpoints = ("ocr_grpc", "ocr_http")
        endpoint_config.ocr_infer_protocol = "http"
        endpoint_config.auth_token = "test_token"
        endpoint_config.workers_per_progress_engine = 8
        self.extraction_config.endpoint_config = endpoint_config

    @patch(f"{MODULE_UNDER_TEST}._create_yolox_client")
    @patch(f"{MODULE_UNDER_TEST}._create_ocr_client")
    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_empty_dataframe(self, mock_update_chart_metadata, mock_ocr_client, mock_yolox_client):
        """Test behavior with an empty DataFrame."""
        # Create empty DataFrame
        df = pd.DataFrame()

        # Call the function
        result_df, result_trace = module_under_test.extract_chart_data_from_image_internal(
            df, self.task_config, self.extraction_config, self.trace_info
        )

        # Verify clients were not created
        mock_yolox_client.assert_not_called()
        mock_ocr_client.assert_not_called()

        # Verify _update_chart_metadata was not called
        mock_update_chart_metadata.assert_not_called()

        # Verify empty DataFrame was returned
        self.assertTrue(result_df.empty)

        # Verify trace info was returned
        self.assertEqual(result_trace, self.trace_info)

    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_no_matching_rows(self, mock_update_chart_metadata):
        """Test behavior when no rows match the extraction criteria."""
        # Create DataFrame with rows that don't meet criteria
        df = pd.DataFrame(
            [
                {"metadata": {"content": "base64_image1", "content_metadata": {"type": "text"}}},  # Not structured
                {
                    "metadata": {
                        "content": "base64_image2",
                        "content_metadata": {"type": "structured", "subtype": "table"},  # Not chart
                    }
                },
                {
                    "metadata": {
                        "content": "base64_image3",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        # Missing table_metadata
                    }
                },
                {
                    "metadata": {
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {},
                        # Missing content
                    }
                },
            ]
        )

        # Call the function
        result_df, result_trace = extract_chart_data_from_image_internal(
            df, self.task_config, self.extraction_config, self.trace_info
        )

        # Verify _update_chart_metadata was not called
        mock_update_chart_metadata.assert_not_called()

        # Verify DataFrame was returned unchanged
        pd.testing.assert_frame_equal(result_df, df)

        # Verify trace info was returned
        self.assertEqual(result_trace, {"trace_info": self.trace_info})

    @patch(f"{MODULE_UNDER_TEST}._create_yolox_client")
    @patch(f"{MODULE_UNDER_TEST}._create_ocr_client")
    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_successful_extraction(self, mock_update_chart_metadata, mock_create_ocr_client, mock_create_yolox_client):
        """Test successful chart data extraction and DataFrame update."""
        # Configure _create_clients to return our mock clients
        mock_create_yolox_client.return_value = self.mock_yolox_client
        mock_create_ocr_client.return_value = self.mock_ocr_client

        # Configure _update_chart_metadata to return chart results
        chart_results = [("base64_image1", self.chart_content), ("base64_image2", self.chart_content)]
        mock_update_chart_metadata.return_value = chart_results

        # Create DataFrame with rows that meet criteria
        df = pd.DataFrame(
            [
                {
                    "metadata": {
                        "content": "base64_image1",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {"table_type": "chart"},
                    }
                },
                {
                    "metadata": {
                        "content": "base64_image2",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {"table_type": "chart"},
                    }
                },
            ]
        )

        # Call the function
        result_df, result_trace = extract_chart_data_from_image_internal(
            df, self.task_config, self.extraction_config, self.trace_info
        )

        # Verify clients were created
        mock_create_yolox_client.assert_called_once()
        mock_create_ocr_client.assert_called_once()

        # Verify _update_chart_metadata was called with correct parameters
        mock_update_chart_metadata.assert_called_once()
        call_args = mock_update_chart_metadata.call_args[1]
        self.assertEqual(call_args["base64_images"], ["base64_image1", "base64_image2"])
        self.assertEqual(call_args["yolox_client"], self.mock_yolox_client)
        self.assertEqual(call_args["ocr_client"], self.mock_ocr_client)
        self.assertEqual(
            call_args["worker_pool_size"], self.extraction_config.endpoint_config.workers_per_progress_engine
        )
        self.assertEqual(call_args["trace_info"], self.trace_info)

        # Verify DataFrame was updated with chart content
        self.assertEqual(result_df.iloc[0]["metadata"]["table_metadata"]["table_content"], self.chart_content)
        self.assertEqual(result_df.iloc[1]["metadata"]["table_metadata"]["table_content"], self.chart_content)

        # Verify trace info was returned
        self.assertEqual(result_trace, {"trace_info": self.trace_info})

    @patch(f"{MODULE_UNDER_TEST}._create_yolox_client")
    @patch(f"{MODULE_UNDER_TEST}._create_ocr_client")
    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_mixed_rows(self, mock_update_chart_metadata, mock_create_ocr_client, mock_create_yolox_client):
        """Test behavior with a mix of matching and non-matching rows."""
        # Configure _create_clients to return our mock clients
        mock_create_yolox_client.return_value = self.mock_yolox_client
        mock_create_ocr_client.return_value = self.mock_ocr_client

        # Configure _update_chart_metadata to return chart results
        chart_results = [("base64_image1", self.chart_content)]
        mock_update_chart_metadata.return_value = chart_results

        # Create DataFrame with mixed rows
        df = pd.DataFrame(
            [
                {
                    "metadata": {
                        "content": "base64_image1",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {"table_type": "chart"},
                    }
                },
                {
                    "metadata": {
                        "content": "base64_image2",
                        "content_metadata": {"type": "text"},  # Doesn't meet criteria
                    }
                },
            ]
        )

        # Call the function
        result_df, result_trace = extract_chart_data_from_image_internal(
            df, self.task_config, self.extraction_config, self.trace_info
        )

        # Verify _update_chart_metadata was called only with the matching image
        mock_update_chart_metadata.assert_called_once()
        call_args = mock_update_chart_metadata.call_args[1]
        self.assertEqual(call_args["base64_images"], ["base64_image1"])

        # Verify only matching row was updated
        self.assertEqual(result_df.iloc[0]["metadata"]["table_metadata"]["table_content"], self.chart_content)
        # Non-matching row should be unchanged
        self.assertEqual(result_df.iloc[1]["metadata"]["content_metadata"]["type"], "text")

    @patch(f"{MODULE_UNDER_TEST}._create_yolox_client")
    @patch(f"{MODULE_UNDER_TEST}._create_ocr_client")
    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_null_trace_info(self, mock_update_chart_metadata, mock_create_ocr_client, mock_create_yolox_client):
        """Test behavior when trace_info is None."""
        # Configure _create_clients to return our mock clients
        mock_create_yolox_client.return_value = self.mock_yolox_client
        mock_create_ocr_client.return_value = self.mock_ocr_client

        # Configure _update_chart_metadata to return chart results
        chart_results = [("base64_image1", self.chart_content)]
        mock_update_chart_metadata.return_value = chart_results

        # Create DataFrame with a single matching row
        df = pd.DataFrame(
            [
                {
                    "metadata": {
                        "content": "base64_image1",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {"table_type": "chart"},
                    }
                }
            ]
        )

        # Call the function with None for trace_info
        result_df, result_trace = extract_chart_data_from_image_internal(
            df, self.task_config, self.extraction_config, None
        )

        # Verify clients were created
        mock_create_yolox_client.assert_called_once()
        mock_create_ocr_client.assert_called_once()

        # Verify _update_chart_metadata was called with empty trace_info
        mock_update_chart_metadata.assert_called_once()
        call_args = mock_update_chart_metadata.call_args[1]
        self.assertEqual(call_args["trace_info"], {})

        # Verify trace info was returned
        self.assertEqual(result_trace, {"trace_info": {}})

    @patch(f"{MODULE_UNDER_TEST}._create_yolox_client")
    @patch(f"{MODULE_UNDER_TEST}._create_ocr_client")
    @patch(f"{MODULE_UNDER_TEST}._update_chart_metadata")
    def test_extraction_error(self, mock_update_chart_metadata, mock_create_ocr_client, mock_create_yolox_client):
        """Test behavior when an error occurs during extraction."""
        # Configure _create_clients to return our mock clients
        mock_create_yolox_client.return_value.return_value = self.mock_yolox_client
        mock_create_ocr_client.return_value.return_value = self.mock_ocr_client

        # Configure _update_chart_metadata to raise an exception
        mock_update_chart_metadata.side_effect = Exception("Chart extraction failed")

        # Create DataFrame with a matching row
        df = pd.DataFrame(
            [
                {
                    "metadata": {
                        "content": "base64_image1",
                        "content_metadata": {"type": "structured", "subtype": "chart"},
                        "table_metadata": {"table_type": "chart"},
                    }
                }
            ]
        )

        # Call the function and expect exception to be propagated
        with self.assertRaises(Exception) as context:
            extract_chart_data_from_image_internal(df, self.task_config, self.extraction_config, self.trace_info)

        # Verify exception was propagated
        self.assertEqual(str(context.exception), "Chart extraction failed")


if __name__ == "__main__":
    unittest.main()
