import unittest
from unittest.mock import MagicMock, patch
import numpy as np

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
        result = client._grpc_infer(test_input, "test_model", outputs=["output1"])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, test_input)

        # Verify that InferRequestedOutput was called correctly for single output
        last_call_args = client.client.infer.call_args
        self.assertEqual(len(last_call_args[1]["outputs"]), 1)
        self.assertEqual(last_call_args[1]["outputs"][0].name(), "output1")

        # Test multiple outputs
        result = client._grpc_infer(test_input, "test_model", outputs=["output1", "output2"])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], test_input)
        np.testing.assert_array_equal(result[1], test_input)

        # Verify that InferRequestedOutput was called correctly for multiple outputs
        last_call_args = client.client.infer.call_args
        self.assertEqual(len(last_call_args[1]["outputs"]), 2)
        self.assertEqual(last_call_args[1]["outputs"][0].name(), "output1")
        self.assertEqual(last_call_args[1]["outputs"][1].name(), "output2")


if __name__ == "__main__":
    unittest.main()
