# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch
import json
import numpy as np

import nv_ingest_api.internal.primitives.nim.model_interface.nemoretriever_parse as module_under_test
from nv_ingest_api.internal.primitives.nim.model_interface.nemoretriever_parse import (
    NemotronParseModelInterface,
    ACCEPTED_TEXT_CLASSES,
    ACCEPTED_TABLE_CLASSES,
    ACCEPTED_CLASSES,
    ACCEPTED_IMAGE_CLASSES,
)

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


class TestNemotronParseModelInterface(unittest.TestCase):

    def setUp(self):
        # Create an instance of the model interface
        self.model_interface = NemotronParseModelInterface()

        # Mock the logger to prevent actual logging during tests
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the numpy_to_base64 function
        self.base64_patcher = patch(f"{MODULE_UNDER_TEST}.numpy_to_base64")
        self.mock_base64 = self.base64_patcher.start()
        # Make it return a predictable value for testing
        self.mock_base64.side_effect = lambda img: f"base64_encoded_{id(img)}"

        # Create sample test data
        self.sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.sample_images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    def tearDown(self):
        # Stop all patchers
        self.logger_patcher.stop()
        self.base64_patcher.stop()

    def test_initialization(self):
        """Test the initialization of the model interface."""
        # Test default initialization
        self.assertEqual(self.model_interface.model_name, "nvidia/nemotron-parse")

        # Test custom model name
        custom_model = NemotronParseModelInterface(model_name="custom/model")
        self.assertEqual(custom_model.model_name, "custom/model")

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "nemotron_parse")

    def test_prepare_data_for_inference(self):
        """Test prepare_data_for_inference method."""
        # This method simply returns the input data unchanged
        test_data = {"images": self.sample_images}
        result = self.model_interface.prepare_data_for_inference(test_data)
        self.assertEqual(result, test_data)

    def test_format_input_http_single_image(self):
        """Test format_input method with HTTP protocol and a single image."""
        test_data = {"image": self.sample_image}
        formatted_batches, formatted_batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=1
        )

        # Check that numpy_to_base64 was called once with the sample image
        self.mock_base64.assert_called_once_with(self.sample_image)

        # Check the format of the payload
        self.assertEqual(len(formatted_batches), 1)
        self.assertEqual(len(formatted_batch_data), 1)

        payload = formatted_batches[0]
        self.assertEqual(payload["model"], "nvidia/nemotron-parse")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")

        # Check the content structure
        content = payload["messages"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue("base64_encoded" in content[0]["image_url"]["url"])

    def test_format_input_http_multiple_images(self):
        """Test format_input method with HTTP protocol and multiple images."""
        test_data = {"images": self.sample_images}
        formatted_batches, formatted_batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=2
        )

        # Check that numpy_to_base64 was called for each image
        self.assertEqual(self.mock_base64.call_count, len(self.sample_images))

        # Check that batches were created correctly based on max_batch_size
        self.assertEqual(len(formatted_batches), 2)  # 3 images with max_batch_size=2 should give 2 batches
        self.assertEqual(len(formatted_batch_data), 2)

        # Check first batch
        self.assertEqual(len(formatted_batches[0]["messages"]), 2)

        # Check second batch
        self.assertEqual(len(formatted_batches[1]["messages"]), 1)

    def test_format_input_grpc_protocol(self):
        """Test format_input method with gRPC protocol (should raise ValueError)."""
        test_data = {"image": self.sample_image}
        with self.assertRaises(ValueError) as context:
            self.model_interface.format_input(test_data, protocol="grpc", max_batch_size=1)

        self.assertTrue("gRPC protocol is not supported" in str(context.exception))

    def test_format_input_invalid_protocol(self):
        """Test format_input method with an invalid protocol."""
        test_data = {"image": self.sample_image}
        with self.assertRaises(ValueError) as context:
            self.model_interface.format_input(test_data, protocol="invalid", max_batch_size=1)

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_parse_output_http(self):
        """Test parse_output method with HTTP protocol."""
        # Create a mock response that mimics the expected structure
        mock_response = {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps({"key": "value"})}}]}}]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that the function correctly extracted and parsed the JSON
        self.assertEqual(result, {"key": "value"})

    def test_parse_output_missing_choices(self):
        """Test parse_output method with a response missing the 'choices' key."""
        mock_response = {}  # Missing 'choices' key

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'choices' key is missing or empty" in str(context.exception))

    def test_parse_output_empty_choices(self):
        """Test parse_output method with an empty 'choices' list."""
        mock_response = {"choices": []}  # Empty 'choices' list

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'choices' key is missing or empty" in str(context.exception))

    def test_parse_output_grpc_protocol(self):
        """Test parse_output method with gRPC protocol (should raise ValueError)."""
        mock_response = {}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="grpc")

        self.assertTrue("gRPC protocol is not supported" in str(context.exception))

    def test_parse_output_invalid_protocol(self):
        """Test parse_output method with an invalid protocol."""
        mock_response = {}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="invalid")

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_process_inference_results(self):
        """Test process_inference_results method."""
        # This method simply returns the input unchanged
        test_output = {"key": "value"}
        result = self.model_interface.process_inference_results(test_output)
        self.assertEqual(result, test_output)

    def test_prepare_nemotron_parse_payload(self):
        """Test _prepare_nemotron_parse_payload method."""
        base64_list = ["base64_data1", "base64_data2"]

        payload = self.model_interface._prepare_nemotron_parse_payload(base64_list)

        # Check the payload structure
        self.assertEqual(payload["model"], "nvidia/nemotron-parse")
        self.assertEqual(len(payload["messages"]), 2)

        # Check the first message
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image_url")
        self.assertEqual(payload["messages"][0]["content"][0]["image_url"]["url"], "data:image/png;base64,base64_data1")

        # Check the second message
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "image_url")
        self.assertEqual(payload["messages"][1]["content"][0]["image_url"]["url"], "data:image/png;base64,base64_data2")

    def test_extract_content_from_nemotron_parse_response(self):
        """Test _extract_content_from_nemotron_parse_response method."""
        # Create a mock response
        expected_content = {"parsed": "content", "items": [1, 2, 3]}
        mock_response = {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps(expected_content)}}]}}]
        }

        result = self.model_interface._extract_content_from_nemotron_parse_response(mock_response)

        # Check that the content was correctly extracted and parsed
        self.assertEqual(result, expected_content)

    def test_accepted_classes_consistency(self):
        """Test that the ACCEPTED_CLASSES set is correctly defined."""
        # Verify that ACCEPTED_CLASSES is the union of the individual class sets
        expected_union = set().union(ACCEPTED_TEXT_CLASSES, ACCEPTED_TABLE_CLASSES, ACCEPTED_IMAGE_CLASSES)
        self.assertEqual(ACCEPTED_CLASSES, expected_union)

        # Verify specific classes are in the correct sets
        self.assertIn("Text", ACCEPTED_TEXT_CLASSES)
        self.assertIn("Table", ACCEPTED_TABLE_CLASSES)
        self.assertIn("Picture", ACCEPTED_IMAGE_CLASSES)


if __name__ == "__main__":
    unittest.main()
