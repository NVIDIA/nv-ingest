# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import deserialize_bytes_tensor
from unittest.mock import Mock

# Import the module under test
import nv_ingest_api.internal.primitives.nim.model_interface.text_embedding as model_interface_module
from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.text_embedding import EmbeddingModelInterface

MODULE_UNDER_TEST = f"{model_interface_module.__name__}"

# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors


class TestEmbeddingModelInterface(unittest.TestCase):

    def setUp(self):
        # Create an instance of the model interface
        self.model_interface = EmbeddingModelInterface()

        # Test data
        self.single_prompt = "This is a test prompt."
        self.multiple_prompts = ["First prompt.", "Second prompt.", "Third prompt."]

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "Embedding")

    def test_inheritance(self):
        """Test that EmbeddingModelInterface inherits from ModelInterface."""
        self.assertIsInstance(self.model_interface, ModelInterface)

    def test_prepare_data_single_prompt(self):
        """Test prepare_data_for_inference with a single prompt."""
        test_data = {"prompts": self.single_prompt}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that prompt was converted to a list
        self.assertIsInstance(result["prompts"], list)
        self.assertEqual(result["prompts"], [self.single_prompt])

    def test_prepare_data_multiple_prompts(self):
        """Test prepare_data_for_inference with multiple prompts."""
        test_data = {"prompts": self.multiple_prompts}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that prompt list remains the same
        self.assertIsInstance(result["prompts"], list)
        self.assertEqual(result["prompts"], self.multiple_prompts)

    def test_prepare_data_missing_prompts(self):
        """Test prepare_data_for_inference with missing prompts."""
        test_data = {"some_other_key": "value"}

        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("prompts" in str(context.exception))

    def test_format_input_single_prompt(self):
        """Test format_input with a single prompt."""
        test_data = {"prompts": [self.single_prompt]}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=10, model_name=model_name
        )

        # Check payloads
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["model"], model_name)
        self.assertEqual(payloads[0]["input"], [self.single_prompt])
        self.assertEqual(payloads[0]["encoding_format"], "float")
        self.assertEqual(payloads[0]["input_type"], "passage")
        self.assertEqual(payloads[0]["truncate"], "NONE")

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["prompts"], [self.single_prompt])

    def test_format_input_multiple_prompts(self):
        """Test format_input with multiple prompts."""
        test_data = {"prompts": self.multiple_prompts}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=10, model_name=model_name
        )

        # Check payloads
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["model"], model_name)
        self.assertEqual(payloads[0]["input"], self.multiple_prompts)

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["prompts"], self.multiple_prompts)

    def test_format_input_batching(self):
        """Test format_input with batching (max_batch_size)."""
        # Create a list of prompts that exceeds the max_batch_size
        many_prompts = [f"Prompt {i}" for i in range(10)]
        test_data = {"prompts": many_prompts}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data,
            protocol="http",
            max_batch_size=3,  # Small batch size to force multiple batches
            model_name=model_name,
        )

        # Check number of batches
        self.assertEqual(len(payloads), 4)  # 10 prompts with batch size 3 should yield 4 batches

        # Check first batch
        self.assertEqual(payloads[0]["input"], many_prompts[0:3])
        self.assertEqual(batch_data[0]["prompts"], many_prompts[0:3])

        # Check last batch
        self.assertEqual(payloads[3]["input"], many_prompts[9:])
        self.assertEqual(batch_data[3]["prompts"], many_prompts[9:])

    def test_format_input_custom_params(self):
        """Test format_input with custom parameters."""
        test_data = {"prompts": [self.single_prompt]}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data,
            protocol="http",
            max_batch_size=10,
            model_name=model_name,
            encoding_format="binary",
            input_type="document",
            truncate="END",
        )

        # Check custom parameters
        self.assertEqual(payloads[0]["encoding_format"], "binary")
        self.assertEqual(payloads[0]["input_type"], "document")
        self.assertEqual(payloads[0]["truncate"], "END")

    def test_parse_output_success(self):
        """Test parse_output with a successful response."""
        # Mock response with embeddings
        mock_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that embeddings were extracted correctly
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])

    def test_parse_output_missing_data(self):
        """Test parse_output with missing data in response."""
        # Mock response with missing data
        mock_response = {"some_other_key": "value"}

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'data' key is missing" in str(context.exception))

    def test_parse_output_empty_data(self):
        """Test parse_output with empty data in response."""
        # Mock response with empty data
        mock_response = {"data": []}

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'data' key is missing or empty" in str(context.exception))

    def test_parse_output_non_dict_response(self):
        """Test parse_output with non-dict response."""
        # Mock non-dict response
        mock_response = "Some raw text response"

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that response is wrapped in a list as a string
        self.assertEqual(result, ["Some raw text response"])

    def test_parse_output_missing_embedding(self):
        """Test parse_output with missing 'embedding' field in data items."""
        # Mock response with missing embedding field
        mock_response = {
            "data": [
                {"some_field": "value"},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that None is returned for missing embedding
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])

    def test_process_inference_results(self):
        """Test process_inference_results."""
        # Sample embeddings
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        result = self.model_interface.process_inference_results(embeddings, protocol="http")

        # Check that result is the same as input (pass-through)
        self.assertEqual(result, embeddings)

    def test_format_input_grpc_single_prompt(self):
        """Test format_input with a single prompt using GRPC protocol."""
        test_data = {"prompts": [self.single_prompt]}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="grpc", max_batch_size=10, model_name=model_name
        )

        # Check that text was properly converted to numpy array with bytes
        self.assertIsInstance(payloads[0], np.ndarray)
        self.assertEqual(len(payloads[0]), 1)  # One input tensor

        text_input = payloads[0]
        self.assertIsInstance(text_input, np.ndarray)

        # Verify the numpy array shape and content
        self.assertEqual(text_input.shape, (1, 1))  # Single prompt
        self.assertEqual(text_input[0][0], self.single_prompt.encode("utf-8"))

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["prompts"], [self.single_prompt])

    def test_format_input_grpc_multiple_prompts(self):
        """Test format_input with multiple prompts using GRPC protocol."""
        test_data = {"prompts": self.multiple_prompts}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="grpc", max_batch_size=10, model_name=model_name
        )

        # Check input tensor
        text_input = payloads[0]
        self.assertIsInstance(text_input, np.ndarray)

        self.assertEqual(text_input.shape, (len(self.multiple_prompts), 1))
        for i, prompt in enumerate(self.multiple_prompts):
            self.assertEqual(text_input[i][0], prompt.encode("utf-8"))

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["prompts"], self.multiple_prompts)

    def test_format_input_grpc_batching(self):
        """Test format_input with batching using GRPC protocol."""
        many_prompts = [f"Prompt {i}" for i in range(10)]
        test_data = {"prompts": many_prompts}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data,
            protocol="grpc",
            max_batch_size=3,  # Small batch size to force multiple batches
            model_name=model_name,
        )

        # Check number of batches
        self.assertEqual(len(payloads), 4)  # 10 prompts with batch size 3 should yield 4 batches

        # Check first batch
        first_batch_input = payloads[0]
        self.assertEqual(first_batch_input.shape, (3, 1))
        for i in range(3):
            self.assertEqual(first_batch_input[i], many_prompts[i].encode("utf-8"))

        # Check last batch
        last_batch_input = payloads[3]
        self.assertEqual(last_batch_input.shape, (1, 1))  # Last batch should have 1 prompt
        self.assertEqual(last_batch_input[0], many_prompts[9].encode("utf-8"))

    def test_parse_output_grpc(self):
        """Test parse_output with GRPC response."""
        # Mock GRPC response with embeddings and token counts
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        result = self.model_interface.parse_output(mock_embeddings, protocol="grpc")

        # Check that embeddings were extracted correctly
        self.assertEqual(len(result), 2)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])

    def test_process_inference_results_grpc(self):
        """Test process_inference_results with GRPC output."""
        # Sample embeddings as numpy arrays
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        result = self.model_interface.process_inference_results(embeddings, protocol="grpc")

        # Check that result is converted to list format
        self.assertEqual(len(result), 2)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()
