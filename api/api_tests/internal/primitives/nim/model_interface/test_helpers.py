# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import requests
from functools import wraps

import nv_ingest_api.internal.primitives.nim.model_interface.helpers as module_under_test

from nv_ingest_api.internal.primitives.nim.model_interface.helpers import (
    _query_metadata,
    get_version,
    get_model_name,
    is_ready,
    preprocess_image_for_ocr,
)

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# Mock the multiprocessing_cache decorator since we want to test the functions directly
def mock_multiprocessing_cache(max_calls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Mock the backoff decorator
def mock_backoff_on_predicate(*args, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TestPreprocessImageForPaddle(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing
        self.sample_image = np.zeros((100, 200, 3), dtype=np.uint8)
        # Add some content to the image
        self.sample_image[30:70, 50:150] = 255

        # Mock the pad_image function
        self.pad_patcher = patch(f"{MODULE_UNDER_TEST}.pad_image")
        self.mock_pad = self.pad_patcher.start()

        # Make pad_image return the input with simulated padding info
        def mock_pad_effect(img, target_height, target_width, background_color, dtype, how):
            padded = np.zeros((target_height, target_width, img.shape[2]), dtype=dtype)
            h, w = img.shape[:2]
            padded[:h, :w] = img
            return padded, (target_width - w, target_height - h)

        self.mock_pad.side_effect = mock_pad_effect

    def tearDown(self):
        self.pad_patcher.stop()

    def test_preprocess_image_default_dimension(self):
        """Test image preprocessing with default max dimension."""
        result, metadata = preprocess_image_for_ocr(self.sample_image, channel_first=True)

        # Check that the result has the correct shape (channel, height, width)
        self.assertEqual(result.shape[0], 3)  # 3 channels

        # Check that pad_image was called
        self.mock_pad.assert_called_once()

        # Check metadata
        self.assertEqual(metadata["original_height"], 100)
        self.assertEqual(metadata["original_width"], 200)
        self.assertEqual(metadata["new_height"], result.shape[1])
        self.assertEqual(metadata["new_width"], result.shape[2])
        self.assertTrue("pad_height" in metadata)
        self.assertTrue("pad_width" in metadata)

    def test_preprocess_image_square_image(self):
        """Test preprocessing with a square image."""
        square_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result, metadata = preprocess_image_for_ocr(square_image)

        # Check metadata
        self.assertEqual(metadata["original_height"], 100)
        self.assertEqual(metadata["original_width"], 100)

    def test_preprocess_image_tall_image(self):
        """Test preprocessing with a tall image (height > width)."""
        tall_image = np.zeros((300, 100, 3), dtype=np.uint8)
        result, metadata = preprocess_image_for_ocr(tall_image)

        # Check metadata
        self.assertEqual(metadata["original_height"], 300)
        self.assertEqual(metadata["original_width"], 100)


class TestIsReady(unittest.TestCase):

    def setUp(self):
        # Mock the logger to prevent actual logging during tests
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the URL utility functions
        self.generate_url_patcher = patch(f"{MODULE_UNDER_TEST}.generate_url")
        self.mock_generate_url = self.generate_url_patcher.start()
        self.mock_generate_url.side_effect = lambda url: url  # Simply return the input URL

        self.remove_url_endpoints_patcher = patch(f"{MODULE_UNDER_TEST}.remove_url_endpoints")
        self.mock_remove_url_endpoints = self.remove_url_endpoints_patcher.start()
        self.mock_remove_url_endpoints.side_effect = lambda url: url  # Simply return the input URL

    def tearDown(self):
        self.logger_patcher.stop()
        self.generate_url_patcher.stop()
        self.remove_url_endpoints_patcher.stop()

    def test_is_ready_null_endpoint(self):
        """Test is_ready with null/empty endpoint."""
        # Test with None endpoint
        result = is_ready(None, "/ready")
        self.assertTrue(result)

        # Test with empty endpoint
        result = is_ready("", "/ready")
        self.assertTrue(result)

    def test_is_ready_nvidia_endpoint(self):
        """Test is_ready with NVIDIA endpoint."""
        result = is_ready("https://ai.api.nvidia.com/endpoint", "/ready")
        self.assertTrue(result)

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_success_response(self, mock_get):
        """Test is_ready with successful 200 response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = is_ready("http://example.com", "/ready")

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://example.com/ready", timeout=5)
        self.mock_logger.warning.assert_not_called()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_not_ready_response(self, mock_get):
        """Test is_ready with 503 (not ready) response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_not_called()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_other_status_code(self, mock_get):
        """Test is_ready with other status code."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_response

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_http_error(self, mock_get):
        """Test is_ready with HTTP error."""
        # Setup mock to raise HTTPError
        mock_get.side_effect = requests.HTTPError("HTTP Error")

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_timeout(self, mock_get):
        """Test is_ready with timeout."""
        # Setup mock to raise Timeout
        mock_get.side_effect = requests.Timeout("Request timed out")

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_connection_error(self, mock_get):
        """Test is_ready with connection error."""
        # Setup mock to raise ConnectionError
        mock_get.side_effect = ConnectionError("Connection Error")

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_request_exception(self, mock_get):
        """Test is_ready with request exception."""
        # Setup mock to raise RequestException
        mock_get.side_effect = requests.RequestException("Request Exception")

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_general_exception(self, mock_get):
        """Test is_ready with general exception."""
        # Setup mock to raise general Exception
        mock_get.side_effect = Exception("General Exception")

        result = is_ready("http://example.com", "/ready")

        self.assertFalse(result)
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_is_ready_url_path_joining(self, mock_get):
        """Test is_ready properly joins URL paths."""
        # Setup successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Test with URL ending with slash and endpoint starting with slash
        _ = is_ready("http://example.com/", "/ready")
        mock_get.assert_called_with("http://example.com//ready", timeout=5)
        mock_get.reset_mock()

        # Test with URL not ending with slash and endpoint not starting with slash
        _ = is_ready("http://example.com", "ready")
        mock_get.assert_called_with("http://example.com/ready", timeout=5)
        mock_get.reset_mock()

        # Test with URL ending with slash and endpoint not starting with slash
        _ = is_ready("http://example.com/", "ready")
        mock_get.assert_called_with("http://example.com/ready", timeout=5)


class TestMetadataQueryFunctions(unittest.TestCase):

    def setUp(self):
        # Patch the decorators
        self.cache_patcher = patch(f"{MODULE_UNDER_TEST}.multiprocessing_cache", mock_multiprocessing_cache)
        self.backoff_patcher = patch(f"{MODULE_UNDER_TEST}.backoff.on_predicate", mock_backoff_on_predicate)

        # Start the patchers
        self.mock_cache = self.cache_patcher.start()
        self.mock_backoff = self.backoff_patcher.start()

        # Mock the logger to prevent actual logging during tests
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the URL utility functions
        self.generate_url_patcher = patch(f"{MODULE_UNDER_TEST}.generate_url")
        self.mock_generate_url = self.generate_url_patcher.start()
        self.mock_generate_url.side_effect = lambda url: url  # Simply return the input URL

        self.remove_url_endpoints_patcher = patch(f"{MODULE_UNDER_TEST}.remove_url_endpoints")
        self.mock_remove_url_endpoints = self.remove_url_endpoints_patcher.start()
        self.mock_remove_url_endpoints.side_effect = lambda url: url  # Simply return the input URL

    def tearDown(self):
        # Stop all patchers
        self.cache_patcher.stop()
        self.backoff_patcher.stop()
        self.logger_patcher.stop()
        self.generate_url_patcher.stop()
        self.remove_url_endpoints_patcher.stop()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_null_endpoint(self, mock_get):
        """Test _query_metadata with null/empty endpoint."""
        # Test with None endpoint
        result = _query_metadata(None, "field", "default")
        self.assertEqual(result, "default")
        mock_get.assert_not_called()

        # Test with empty endpoint
        result = _query_metadata("", "field", "default")
        self.assertEqual(result, "default")
        mock_get.assert_not_called()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_success(self, mock_get):
        """Test _query_metadata with successful response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"field_name": "field_value"}
        mock_get.return_value = mock_response

        result = _query_metadata("http://example.com", "field_name", "default")

        self.assertEqual(result, "field_value")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_not_called()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_empty_field(self, mock_get):
        """Test _query_metadata with empty field in response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"other_field": "value"}
        mock_get.return_value = mock_response

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_non_200_status(self, mock_get):
        """Test _query_metadata with non-200 status code."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_http_error(self, mock_get):
        """Test _query_metadata with HTTP error."""
        # Setup mock to raise HTTPError
        mock_get.side_effect = requests.HTTPError("HTTP Error")

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_timeout(self, mock_get):
        """Test _query_metadata with timeout."""
        # Setup mock to raise Timeout
        mock_get.side_effect = requests.Timeout("Request timed out")

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_connection_error(self, mock_get):
        """Test _query_metadata with connection error."""
        # Setup mock to raise ConnectionError
        mock_get.side_effect = ConnectionError("Connection Error")

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_request_exception(self, mock_get):
        """Test _query_metadata with generic request exception."""
        # Setup mock to raise RequestException
        mock_get.side_effect = requests.RequestException("Request Exception")

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.requests.get")
    def test_query_metadata_general_exception(self, mock_get):
        """Test _query_metadata with general exception."""
        # Setup mock to raise general Exception
        mock_get.side_effect = Exception("General Exception")

        result = _query_metadata("http://example.com", "field_name", "default", "retry")

        self.assertEqual(result, "retry")
        mock_get.assert_called_once()
        self.mock_logger.warning.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}._query_metadata")
    def test_get_version_nvidia_endpoint(self, mock_query):
        """Test get_version with NVIDIA endpoint."""
        result = get_version("https://ai.api.nvidia.com/endpoint")

        self.assertEqual(result, "1.0.0")
        mock_query.assert_not_called()

        result = get_version("https://api.nvcf.nvidia.com/endpoint")

        self.assertEqual(result, "1.0.0")
        mock_query.assert_not_called()

    @patch(f"{MODULE_UNDER_TEST}._query_metadata")
    def test_get_version_normal_endpoint(self, mock_query):
        """Test get_version with normal endpoint."""
        mock_query.return_value = "2.0.0"

        result = get_version("http://example.com")

        self.assertEqual(result, "2.0.0")
        mock_query.assert_called_once_with("http://example.com", field_name="version", default_value="1.0.0")

    @patch(f"{MODULE_UNDER_TEST}._query_metadata")
    def test_get_model_name_nvidia_endpoint(self, mock_query):
        """Test get_model_name with NVIDIA endpoint."""
        result = get_model_name("https://ai.api.nvidia.com/model1", "default_model")

        self.assertEqual(result, "model1")
        mock_query.assert_not_called()

        result = get_model_name(
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/12345678-1234-5678-1234-567812345678", "model2"
        )

        self.assertEqual(result, "model2")
        mock_query.assert_not_called()
