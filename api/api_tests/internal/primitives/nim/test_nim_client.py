# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import requests

from nv_ingest_api.internal.primitives.nim.nim_client import NimClient


class MockModelInterface:
    def name(self):
        return "mock_model"


class TestNimClientGrpc(unittest.TestCase):
    def setUp(self):
        self.model_interface = MockModelInterface()
        self.endpoints = ("localhost:8001", "http://localhost:8000")

    @patch("tritonclient.grpc.InferenceServerClient")
    def test_grpc_infer_output_format(self, mock_grpc_client):
        # Setup
        client = NimClient(self.model_interface, "grpc", self.endpoints)
        mock_response = MagicMock()
        mock_response.as_numpy = MagicMock(return_value=np.array([1, 2, 3]).astype(np.float32))

        # Mock the client's infer method
        client.client.infer = MagicMock(return_value=mock_response)

        # Test data
        test_input = np.array([1, 2, 3]).astype(np.float32)

        # Test single output
        result = client._grpc_infer(test_input, "test_model", output_names=["output1"])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, test_input)

        # Verify that InferRequestedOutput was called correctly for single output
        last_call_args = client.client.infer.call_args
        self.assertEqual(len(last_call_args[1]["outputs"]), 1)
        self.assertEqual(last_call_args[1]["outputs"][0].name(), "output1")

        # Test multiple outputs
        result = client._grpc_infer(test_input, "test_model", output_names=["output1", "output2"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], test_input)
        np.testing.assert_array_equal(result[1], test_input)

        # Verify that InferRequestedOutput was called correctly for multiple outputs
        last_call_args = client.client.infer.call_args
        self.assertEqual(len(last_call_args[1]["outputs"]), 2)
        self.assertEqual(last_call_args[1]["outputs"][0].name(), "output1")
        self.assertEqual(last_call_args[1]["outputs"][1].name(), "output2")


class TestNimClientHttp(unittest.TestCase):
    def setUp(self):
        self.model_interface = MockModelInterface()
        self.endpoints = ("", "http://fake-url/v1/infer")
        self.client = NimClient(self.model_interface, "http", self.endpoints, max_retries=3, max_429_retries=4)

    @patch("requests.post")
    @patch("time.sleep", return_value=None)  # Mock time.sleep to speed up tests
    def test_http_infer_handles_429_retries_and_succeeds(self, mock_sleep, mock_post):
        # Arrange: Simulate three 429 responses, then a success
        mock_429_response = MagicMock()
        mock_429_response.status_code = 429
        mock_429_response.reason = "Too Many Requests"

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"status": "success"}

        mock_post.side_effect = [mock_429_response, mock_429_response, mock_429_response, mock_success_response]

        # Act
        result = self.client._http_infer({})

        # Assert
        self.assertEqual(result, {"status": "success"})
        self.assertEqual(mock_post.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_has_calls([call(4.0), call(8.0), call(16.0)])

    @patch("requests.post")
    @patch("time.sleep", return_value=None)
    def test_http_infer_fails_after_max_429_retries(self, mock_sleep, mock_post):
        # Arrange: Simulate continuous 429 responses
        mock_429_response = MagicMock()
        mock_429_response.status_code = 429
        mock_429_response.reason = "Too Many Requests"
        # Make raise_for_status raise an HTTPError
        mock_429_response.raise_for_status.side_effect = requests.exceptions.HTTPError

        mock_post.return_value = mock_429_response

        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._http_infer({})

        self.assertEqual(mock_post.call_count, 4)  # max_429_retries is 4
        self.assertEqual(mock_sleep.call_count, 3)  # Sleeps before retries 2, 3, 4

    @patch("requests.post")
    @patch("time.sleep", return_value=None)
    def test_http_infer_handles_503_retries_and_succeeds(self, mock_sleep, mock_post):
        # Arrange: Simulate two 503 responses, then a success
        mock_503_response = MagicMock()
        mock_503_response.status_code = 503
        mock_503_response.reason = "Service Unavailable"

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"status": "success"}

        mock_post.side_effect = [mock_503_response, mock_503_response, mock_success_response]

        # Act
        result = self.client._http_infer({})

        # Assert
        self.assertEqual(result, {"status": "success"})
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_has_calls([call(2.0), call(4.0)])

    @patch("requests.post")
    @patch("time.sleep", return_value=None)
    def test_http_infer_fails_after_max_retries_for_503(self, mock_sleep, mock_post):
        # Arrange: Simulate continuous 503 responses
        mock_503_response = MagicMock()
        mock_503_response.status_code = 503
        mock_503_response.reason = "Service Unavailable"
        mock_503_response.raise_for_status.side_effect = requests.exceptions.HTTPError

        mock_post.return_value = mock_503_response

        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._http_infer({})

        self.assertEqual(mock_post.call_count, 3)  # max_retries is 3
        self.assertEqual(mock_sleep.call_count, 2)  # Sleeps before retries 2, 3


if __name__ == "__main__":
    unittest.main()
