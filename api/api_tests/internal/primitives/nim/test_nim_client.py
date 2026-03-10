# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import requests

from tritonclient.grpc import InferenceServerException

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


class TestNimClientGrpcRetry(unittest.TestCase):
    def setUp(self):
        self.model_interface = MockModelInterface()
        self.endpoints = ("localhost:8001", "http://localhost:8000")
        # Configure client with specific retry counts for testing
        self.client = NimClient(self.model_interface, "grpc", self.endpoints, max_429_retries=4)

    @patch("tritonclient.grpc.InferenceServerClient")
    @patch("time.sleep", return_value=None)  # Mock time.sleep to speed up tests
    def test_grpc_infer_retries_and_succeeds(self, mock_sleep, mock_grpc_client):
        """
        Tests that _grpc_infer retries on 'queue full' error and eventually succeeds.
        """
        retryable_error = InferenceServerException(msg="Exceeds maximum queue size", status="StatusCode.UNAVAILABLE")

        mock_success_response = MagicMock()
        test_output = np.array([1.0, 2.0], dtype=np.float32)
        mock_success_response.as_numpy.return_value = test_output

        self.client.client.infer = MagicMock(side_effect=[retryable_error, retryable_error, mock_success_response])

        test_input = np.array([1.0, 2.0], dtype=np.float32)
        result = self.client._grpc_infer(test_input, "test_model")

        np.testing.assert_array_equal(result, test_output)
        self.assertEqual(self.client.client.infer.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_has_calls([call(4.0), call(8.0)])

    @patch("tritonclient.grpc.InferenceServerClient")
    @patch("time.sleep", return_value=None)
    def test_grpc_infer_fails_after_max_retries(self, mock_sleep, mock_grpc_client):
        """
        Tests that _grpc_infer fails after exhausting all retries for 'queue full' errors.
        """
        retryable_error = InferenceServerException(msg="Exceeds maximum queue size", status="StatusCode.UNAVAILABLE")

        self.client.client.infer = MagicMock(side_effect=retryable_error)

        with self.assertRaises(InferenceServerException) as context:
            test_input = np.array([1.0], dtype=np.float32)
            self.client._grpc_infer(test_input, "test_model")

        self.assertIn("Exceeds maximum queue size", context.exception.message())
        self.assertEqual(context.exception.status(), "StatusCode.UNAVAILABLE")

        self.assertEqual(self.client.client.infer.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("tritonclient.grpc.InferenceServerClient")
    @patch("time.sleep", return_value=None)
    def test_grpc_infer_fails_immediately_on_non_retryable_error(self, mock_sleep, mock_grpc_client):
        """
        Tests that _grpc_infer fails immediately for a gRPC error that is not 'queue full'.
        """
        non_retryable_error = InferenceServerException(msg="Invalid argument", status="StatusCode.INVALID_ARGUMENT")

        self.client.client.infer = MagicMock(side_effect=non_retryable_error)

        with self.assertRaises(InferenceServerException) as context:
            test_input = np.array([1.0], dtype=np.float32)
            self.client._grpc_infer(test_input, "test_model")

        self.assertEqual(context.exception.status(), "StatusCode.INVALID_ARGUMENT")

        self.client.client.infer.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("nv_ingest_api.internal.primitives.nim.nim_client.reload_models", return_value=True)
    @patch("time.sleep", return_value=None)
    def test_grpc_infer_cuda_internal_retries_and_succeeds(self, mock_sleep, mock_reload_models):
        """
        _grpc_infer should retry on INTERNAL + CUDA-like message, call reload_models,
        and succeed if a subsequent call returns a response.
        """
        # 1) First two attempts raise INTERNAL with CUDA-ish message; third succeeds
        cuda_msg = "TritonModelException: failed to copy data: cuMemcpy HtoD failed: illegal memory access"
        cuda_err1 = InferenceServerException(msg=cuda_msg, status="StatusCode.INTERNAL")
        cuda_err2 = InferenceServerException(msg=cuda_msg, status="StatusCode.INTERNAL")

        mock_success_response = MagicMock()
        test_output = np.array([3.14], dtype=np.float32)
        mock_success_response.as_numpy.return_value = test_output

        self.client.client.infer = MagicMock(side_effect=[cuda_err1, cuda_err2, mock_success_response])

        # 2) Run
        test_input = np.array([42.0], dtype=np.float32)
        result = self.client._grpc_infer(test_input, "test_model")

        # 3) Assert
        np.testing.assert_array_equal(result, test_output)
        self.assertEqual(self.client.client.infer.call_count, 3)

        # reload_models should be called once per retry attempt before success
        self.assertEqual(mock_reload_models.call_count, 2)
        # time.sleep called for the two backoffs
        self.assertEqual(mock_sleep.call_count, 2)

        # Ensure reload_models was called with the right client/timeout
        for call_args in mock_reload_models.call_args_list:
            self.assertIs(call_args.kwargs["client"], self.client.client)
            self.assertEqual(call_args.kwargs["client_timeout"], self.client.timeout)

    @patch("nv_ingest_api.internal.primitives.nim.nim_client.reload_models", return_value=False)
    @patch("time.sleep", return_value=None)
    def test_grpc_infer_cuda_internal_fails_after_max_retries(self, mock_sleep, mock_reload_models):
        """
        _grpc_infer should raise after exhausting max_retries for INTERNAL + CUDA-like errors.
        """
        cuda_msg = "failed to copy data: cudaMemcpyAsync invalid argument"
        # Create one exception instance per attempt (safer for side_effect)
        errors = []
        for _ in range(self.client.max_retries):
            e = InferenceServerException(msg=cuda_msg, status="StatusCode.INTERNAL")
            errors.append(e)

        self.client.client.infer = MagicMock(side_effect=errors)

        with self.assertRaises(InferenceServerException) as ctx:
            test_input = np.array([0.0], dtype=np.float32)
            self.client._grpc_infer(test_input, "test_model")

        # Confirm it is indeed the INTERNAL/CUDA path we exercised
        self.assertIn("cuda", ctx.exception.message().lower())
        self.assertEqual(ctx.exception.status(), "StatusCode.INTERNAL")

        # infer called once per attempt
        self.assertEqual(self.client.client.infer.call_count, self.client.max_retries)
        # reload_models & sleep called for each retry before the final raise
        self.assertEqual(mock_reload_models.call_count, self.client.max_retries - 1)
        self.assertEqual(mock_sleep.call_count, self.client.max_retries - 1)


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
