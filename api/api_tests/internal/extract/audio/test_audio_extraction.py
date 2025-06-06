# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch
import pandas as pd

# Import the module under test
import nv_ingest_api.internal.extract.audio.audio_extraction as module_under_test
from nv_ingest_api.internal.extract.audio.audio_extraction import (
    _extract_from_audio,
    extract_text_from_audio_internal,
    ContentTypeEnum,
)

# Define module path constant for patching
MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class TestAudioExtraction(unittest.TestCase):
    """Tests for audio extraction functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create a mock audio client
        self.mock_audio_client = Mock()

        self.mock_audio_client.infer = Mock(
            return_value=(
                [
                    {"text": "This is a", "start": 0, "end": 5},
                    {"text": "transcribed audio content", "start": 5, "end": 10},
                ],
                "This is a transcribed audio content",
            )
        )

        # Common trace info
        self.trace_info = {"request_id": "test-request-123", "timestamp": "2025-03-10T12:00:00Z"}

        # Sample valid audio metadata
        self.valid_audio_metadata = {
            "content": "base64encodedaudiocontent",
            "content_metadata": {
                "type": ContentTypeEnum.AUDIO,
                "file_size": 1024,
                "duration": 120.5,
                "sample_rate": 44100,
            },
        }

        # Sample invalid metadata (non-audio type)
        self.non_audio_metadata = {
            "content": "base64encodedcontent",
            "content_metadata": {"type": ContentTypeEnum.TEXT, "file_size": 1024},
        }

        # Sample empty content metadata
        self.empty_content_metadata = {
            "content": "",
            "content_metadata": {"type": ContentTypeEnum.AUDIO, "file_size": 0},
        }

    @patch(f"{MODULE_UNDER_TEST}.validate_schema")
    def test_extract_from_audio_metadata_valid(self, mock_validate_schema):
        """Test _update_audio_metadata with valid audio content."""
        # Configure mocks
        mock_validate_schema.side_effect = lambda data, schema: Mock(model_dump=lambda: data)

        # Create a sample row with valid audio metadata
        row = pd.Series({"metadata": self.valid_audio_metadata.copy()})

        # Call the function
        result = _extract_from_audio(row, self.mock_audio_client, self.trace_info)

        # Verify results
        self.assertIn("audio_metadata", result[0][1])
        self.assertEqual(result[0][1]["audio_metadata"]["audio_transcript"], "This is a transcribed audio content")

        # Verify the mock was called correctly
        self.mock_audio_client.infer.assert_called_once_with(
            "base64encodedaudiocontent",
            model_name="parakeet",
            trace_info=self.trace_info,
            stage_name="audio_extraction",
        )

        # Verify validate_schema was called twice (for audio metadata and metadata)
        self.assertEqual(mock_validate_schema.call_count, 2)

    @patch(f"{MODULE_UNDER_TEST}.validate_schema")
    def test_extract_from_audio_metadata_valid_segment(self, mock_validate_schema):
        """Test _update_audio_metadata with valid audio content."""
        # Configure mocks
        mock_validate_schema.side_effect = lambda data, schema: Mock(model_dump=lambda: data)

        # Create a sample row with valid audio metadata
        row = pd.Series({"metadata": self.valid_audio_metadata.copy()})

        # Call the function
        result = _extract_from_audio(row, self.mock_audio_client, self.trace_info, segment_audio=True)

        # Verify results
        self.assertIn("audio_metadata", result[0][1])
        self.assertIn("audio_metadata", result[1][1])
        self.assertEqual(result[0][1]["audio_metadata"]["audio_transcript"], "This is a")
        self.assertEqual(result[1][1]["audio_metadata"]["audio_transcript"], "transcribed audio content")

        # Verify the mock was called correctly
        self.mock_audio_client.infer.assert_called_once_with(
            "base64encodedaudiocontent",
            model_name="parakeet",
            trace_info=self.trace_info,
            stage_name="audio_extraction",
        )

        # Verify validate_schema was called twice (for audio metadata and metadata)
        self.assertEqual(mock_validate_schema.call_count, 4)

    def test_extract_from_audio_non_audio(self):
        """Test _update_audio_metadata with non-audio content."""
        # Create a sample row with non-audio metadata
        row = pd.Series({"metadata": self.non_audio_metadata.copy()})

        # Call the function
        result = _extract_from_audio(row, self.mock_audio_client, self.trace_info)

        # The 'content' key is popped from metadata in the function
        # so we need to compare with that in mind
        expected = self.non_audio_metadata.copy()
        expected_content = expected.pop("content")

        # Verify that the metadata was returned appropriately
        self.assertEqual(result, [[expected]])
        self.assertEqual(expected_content, "base64encodedcontent")

        # Verify the mock was not called
        self.mock_audio_client.infer.assert_not_called()

    def test_extract_from_audio_empty_content(self):
        """Test _update_audio_metadata with empty audio content."""
        # Create a sample row with empty content metadata
        row = pd.Series({"metadata": self.empty_content_metadata.copy()})

        # Call the function
        result = _extract_from_audio(row, self.mock_audio_client, self.trace_info)

        # The 'content' key is popped from metadata in the function
        # so we need to compare with that in mind
        expected = self.empty_content_metadata.copy()
        expected_content = expected.pop("content")

        # Verify that the metadata was returned as expected
        self.assertEqual(result, [[expected]])
        self.assertEqual(expected_content, "")

        # Verify the mock was not called
        self.mock_audio_client.infer.assert_not_called()

    def test_extract_from_audio_missing_metadata(self):
        """Test _update_audio_metadata with missing metadata."""
        # Create a sample row without metadata
        row = pd.Series({"other_field": "value"})

        # Call the function and check for exception
        with self.assertRaises(ValueError) as context:
            _extract_from_audio(row, self.mock_audio_client, self.trace_info)

        # The unified_exception_handler decorator appears to modify the error message format
        self.assertEqual(str(context.exception), "_extract_from_audio: error: Row does not contain 'metadata'.")

    @patch(f"{MODULE_UNDER_TEST}.create_audio_inference_client")
    @patch(f"{MODULE_UNDER_TEST}._extract_from_audio")
    @patch("pandas.DataFrame.apply")
    def test_extract_text_from_audio_internal(self, mock_df_apply, mock_extract_from_audio, mock_create_client):
        """Test extract_text_from_audio_internal with various inputs."""
        # Setup test data
        mock_create_client.return_value = self.mock_audio_client

        # Configure the update_audio_metadata mock to modify metadata
        def update_metadata_side_effect(row, client, trace_info):
            if isinstance(row, pd.Series) and "metadata" in row:
                metadata = row["metadata"]
                if metadata.get("content_metadata", {}).get("type") == ContentTypeEnum.AUDIO:
                    return [[ContentTypeEnum.AUDIO, {"audio_metadata": {"audio_transcript": "Test transcript"}}, 12345]]
                return metadata
            return []

        mock_extract_from_audio.side_effect = update_metadata_side_effect

        # Create test DataFrame
        df = pd.DataFrame(
            [
                {"id": 1, "metadata": self.valid_audio_metadata.copy()},
                {"id": 2, "metadata": self.non_audio_metadata.copy()},
                {"id": 3, "metadata": self.empty_content_metadata.copy()},
            ]
        )

        # Create audio extractor config with default values
        audio_extraction_config = Mock()
        audio_extraction_config.audio_endpoints = ["grpc://localhost:50051", "http://localhost:8080"]
        audio_extraction_config.audio_infer_protocol = "grpc"
        audio_extraction_config.auth_token = "test-token"
        audio_extraction_config.function_id = "audio-function"
        audio_extraction_config.use_ssl = False
        audio_extraction_config.ssl_cert = None
        audio_extraction_config.segment_audio = False

        extraction_config = Mock()
        extraction_config.audio_extraction_config = audio_extraction_config

        # Create task config
        task_config = {"params": {"extract_audio_params": {}}}

        # Configure the DataFrame.apply mock to return a Series of updated metadata
        mock_df_apply.return_value = pd.Series(
            [
                {"audio_metadata": {"audio_transcript": "Test transcript"}},
                self.non_audio_metadata,
                self.empty_content_metadata,
            ]
        )

        # Configure the DataFrame.apply mock
        mock_df_apply.return_value = pd.Series(
            [[[ContentTypeEnum.AUDIO, {"audio_metadata": {"audio_transcript": "Test transcript"}}, 12345]]]
        )

        # Call the function
        result_df, trace_info = extract_text_from_audio_internal(
            df.copy(), task_config, extraction_config, self.trace_info
        )

        # Verify results
        self.assertEqual(len(result_df), 1)  # Should have same number of rows

        # Verify create_audio_inference_client was called with correct params
        mock_create_client.assert_called_once_with(
            ("grpc://localhost:50051", "http://localhost:8080"),
            infer_protocol="grpc",
            auth_token="test-token",
            function_id="audio-function",
            use_ssl=False,
            ssl_cert=None,
        )

    @patch(f"{MODULE_UNDER_TEST}.create_audio_inference_client")
    @patch("pandas.DataFrame.apply")
    def test_extract_text_from_audio_internal_with_custom_params(self, mock_df_apply, mock_create_client):
        """Test extract_text_from_audio_internal with custom parameters."""
        # Setup test data
        mock_create_client.return_value = self.mock_audio_client

        # Create test DataFrame with a single row
        df = pd.DataFrame([{"id": 1, "metadata": self.valid_audio_metadata.copy()}])

        # Create audio extractor config with default values
        audio_extraction_config = Mock()
        audio_extraction_config.audio_endpoints = ["grpc://default:50051", "http://default:8080"]
        audio_extraction_config.audio_infer_protocol = "grpc"
        audio_extraction_config.auth_token = "default-token"
        audio_extraction_config.function_id = "default-function"
        audio_extraction_config.use_ssl = False
        audio_extraction_config.ssl_cert = None
        audio_extraction_config.segment_audio = False

        extraction_config = Mock()
        extraction_config.audio_extraction_config = audio_extraction_config

        # Create task config with custom parameters that should override defaults
        task_config = {
            "params": {
                "extract_audio_params": {
                    "grpc_endpoint": "grpc://custom:50051",
                    "http_endpoint": "http://custom:8080",
                    "infer_protocol": "http",
                    "auth_token": "custom-token",
                    "function_id": "custom-function",
                    "use_ssl": True,
                    "ssl_cert": "custom-cert",
                    "segment_audio": True,
                }
            }
        }

        # Call the function
        result_df, trace_info = extract_text_from_audio_internal(
            df.copy(), task_config, extraction_config, self.trace_info
        )

        # Verify create_audio_inference_client was called with custom params
        mock_create_client.assert_called_once_with(
            ("grpc://custom:50051", "http://custom:8080"),
            infer_protocol="http",
            auth_token="custom-token",
            function_id="custom-function",
            use_ssl=True,
            ssl_cert="custom-cert",
        )

    @patch(f"{MODULE_UNDER_TEST}.create_audio_inference_client")
    @patch("pandas.DataFrame.apply")
    def test_extract_text_from_audio_internal_exception_handling(self, mock_df_apply, mock_create_client):
        """Test exception handling in extract_text_from_audio_internal."""
        # Setup mock to raise an exception
        mock_create_client.return_value = self.mock_audio_client
        self.mock_audio_client.infer.side_effect = Exception("Audio inference failed")

        # Create test DataFrame
        df = pd.DataFrame([{"id": 1, "metadata": self.valid_audio_metadata.copy()}])

        # Create configs
        audio_extraction_config = Mock()
        audio_extraction_config.audio_endpoints = ["grpc://localhost:50051", "http://localhost:8080"]
        audio_extraction_config.audio_infer_protocol = "grpc"
        audio_extraction_config.auth_token = "test-token"
        audio_extraction_config.function_id = "audio-function"
        audio_extraction_config.use_ssl = False
        audio_extraction_config.ssl_cert = None

        extraction_config = Mock()
        extraction_config.audio_extraction_config = audio_extraction_config

        task_config = {"params": {"extract_audio_params": {}}}

        # Configure the DataFrame.apply mock to raise an exception
        mock_df_apply.side_effect = Exception("Audio inference failed")

        # Test exception propagation
        with self.assertRaises(Exception) as context:
            _ = context
            extract_text_from_audio_internal(df.copy(), task_config, extraction_config, self.trace_info)

    @patch(f"{MODULE_UNDER_TEST}.create_audio_inference_client")
    @patch("pandas.DataFrame.apply")
    def test_extract_text_from_audio_internal_no_trace_info(self, mock_df_apply, mock_create_client):
        """Test extract_text_from_audio_internal with no trace info provided."""
        # Setup test data
        mock_create_client.return_value = self.mock_audio_client

        # Create test DataFrame with a single row
        df = pd.DataFrame([{"id": 1, "metadata": self.valid_audio_metadata.copy()}])

        # Create configs
        audio_extraction_config = Mock()
        audio_extraction_config.audio_endpoints = ["grpc://localhost:50051", "http://localhost:8080"]
        audio_extraction_config.audio_infer_protocol = "grpc"
        audio_extraction_config.auth_token = "test-token"
        audio_extraction_config.function_id = "audio-function"
        audio_extraction_config.use_ssl = False
        audio_extraction_config.ssl_cert = None

        extraction_config = Mock()
        extraction_config.audio_extraction_config = audio_extraction_config

        task_config = {"params": {"extract_audio_params": {}}}

        # Configure the DataFrame.apply mock
        mock_df_apply.return_value = pd.Series(
            [[[ContentTypeEnum.AUDIO, {"audio_metadata": {"audio_transcript": "Test transcript"}}, 12345]]]
        )

        # Call the function with no trace_info
        result_df, trace_info = extract_text_from_audio_internal(df.copy(), task_config, extraction_config, None)

        # Verify an empty dict was created for trace_info
        self.assertIsInstance(trace_info, dict)
        self.assertEqual(len(trace_info), 0)


class TestAudioInferenceClient(unittest.TestCase):
    """Tests for the audio inference client creation and usage."""

    @patch(f"{MODULE_UNDER_TEST}.create_audio_inference_client")
    def test_client_creation(self, mock_create_client):
        """Test that the audio inference client is created with correct parameters."""
        # Setup mock return value
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        # Test client creation with different parameters
        endpoints = ("grpc://test:50051", "http://test:8080")
        infer_protocol = "grpc"
        auth_token = "test-token"
        function_id = "test-function"
        use_ssl = True
        ssl_cert = "test-cert"

        client = module_under_test.create_audio_inference_client(
            endpoints,
            infer_protocol=infer_protocol,
            auth_token=auth_token,
            function_id=function_id,
            use_ssl=use_ssl,
            ssl_cert=ssl_cert,
        )

        # Verify the client was created with correct parameters
        mock_create_client.assert_called_once_with(
            endpoints,
            infer_protocol=infer_protocol,
            auth_token=auth_token,
            function_id=function_id,
            use_ssl=use_ssl,
            ssl_cert=ssl_cert,
        )

        # Verify the client was returned
        self.assertEqual(client, mock_client)


if __name__ == "__main__":
    unittest.main()
