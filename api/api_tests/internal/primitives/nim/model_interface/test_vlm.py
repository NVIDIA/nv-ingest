# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

# Import the module under test
import nv_ingest_api.internal.primitives.nim.model_interface.vlm as module_under_test
from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.vlm import VLMModelInterface

MODULE_UNDER_TEST = f"{module_under_test.__name__}"

# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors


class TestVLMModelInterface(unittest.TestCase):

    def setUp(self):
        # Create an instance of the model interface
        self.model_interface = VLMModelInterface()

        # Test data
        self.sample_prompt = "Describe this image."
        self.sample_image = "base64encodedimagedata1"
        self.sample_images = ["base64encodedimagedata1", "base64encodedimagedata2", "base64encodedimagedata3"]

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "VLM")

    def test_inheritance(self):
        """Test that VLMModelInterface inherits from ModelInterface."""
        self.assertIsInstance(self.model_interface, ModelInterface)

    def test_prepare_data_single_image(self):
        """Test prepare_data_for_inference with a single image."""
        test_data = {"base64_image": self.sample_image, "prompt": self.sample_prompt}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that the single image was converted to a list
        self.assertIn("base64_images", result)
        self.assertIsInstance(result["base64_images"], list)
        self.assertEqual(result["base64_images"], [self.sample_image])
        self.assertEqual(result["prompt"], self.sample_prompt)

    def test_prepare_data_multiple_images(self):
        """Test prepare_data_for_inference with multiple images."""
        test_data = {"base64_images": self.sample_images, "prompt": self.sample_prompt}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that the image list remains the same
        self.assertIn("base64_images", result)
        self.assertIsInstance(result["base64_images"], list)
        self.assertEqual(result["base64_images"], self.sample_images)
        self.assertEqual(result["prompt"], self.sample_prompt)

    def test_prepare_data_missing_images(self):
        """Test prepare_data_for_inference with missing images."""
        test_data = {"prompt": self.sample_prompt}

        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("base64_image" in str(context.exception))

    def test_prepare_data_missing_prompt(self):
        """Test prepare_data_for_inference with missing prompt."""
        test_data = {"base64_image": self.sample_image}

        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("prompt" in str(context.exception))

    def test_prepare_data_invalid_images_type(self):
        """Test prepare_data_for_inference with invalid base64_images type."""
        test_data = {"base64_images": "not_a_list", "prompt": self.sample_prompt}

        with self.assertRaises(ValueError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("must contain a list" in str(context.exception))

    def test_format_input_single_image(self):
        """Test format_input with a single image."""
        test_data = {"base64_images": [self.sample_image], "prompt": self.sample_prompt}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=10, model_name=model_name
        )

        # Check payloads
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["model"], model_name)
        self.assertEqual(len(payloads[0]["messages"]), 1)

        # Check message format
        message = payloads[0]["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertIsInstance(message["content"], list)
        self.assertEqual(len(message["content"]), 2)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], self.sample_prompt)
        self.assertEqual(message["content"][1]["type"], "image_url")
        self.assertEqual(message["content"][1]["image_url"]["url"], f"data:image/png;base64,{self.sample_image}")

        # Check default parameters
        self.assertEqual(payloads[0]["max_tokens"], 512)
        self.assertEqual(payloads[0]["temperature"], 1.0)
        self.assertEqual(payloads[0]["top_p"], 1.0)
        self.assertEqual(payloads[0]["stream"], False)

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["base64_images"], [self.sample_image])
        self.assertEqual(batch_data[0]["prompt"], self.sample_prompt)

    def test_format_input_multiple_images(self):
        """Test format_input with multiple images."""
        test_data = {"base64_images": self.sample_images, "prompt": self.sample_prompt}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data, protocol="http", max_batch_size=10, model_name=model_name
        )

        # Check payloads
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["model"], model_name)

        # Check messages
        messages = payloads[0]["messages"]
        self.assertEqual(len(messages), len(self.sample_images))

        # Check each message
        for i, message in enumerate(messages):
            self.assertEqual(message["role"], "user")
            self.assertIsInstance(message["content"], list)
            self.assertEqual(len(message["content"]), 2)
            self.assertEqual(message["content"][0]["type"], "text")
            self.assertEqual(message["content"][0]["text"], self.sample_prompt)
            self.assertEqual(message["content"][1]["type"], "image_url")
            self.assertEqual(
                message["content"][1]["image_url"]["url"], f"data:image/png;base64,{self.sample_images[i]}"
            )

        # Check batch data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(batch_data[0]["base64_images"], self.sample_images)
        self.assertEqual(batch_data[0]["prompt"], self.sample_prompt)

    def test_format_input_batching(self):
        """Test format_input with batching (max_batch_size)."""
        # Create a list of images that exceeds the max_batch_size
        many_images = [f"base64image{i}" for i in range(10)]
        test_data = {"base64_images": many_images, "prompt": self.sample_prompt}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data,
            protocol="http",
            max_batch_size=3,  # Small batch size to force multiple batches
            model_name=model_name,
        )

        # Check number of batches
        self.assertEqual(len(payloads), 4)  # 10 images with batch size 3 should yield 4 batches

        # Check first batch
        self.assertEqual(len(payloads[0]["messages"]), 3)
        self.assertEqual(batch_data[0]["base64_images"], many_images[0:3])

        # Check last batch
        self.assertEqual(len(payloads[3]["messages"]), 1)
        self.assertEqual(batch_data[3]["base64_images"], many_images[9:])

    def test_format_input_custom_params(self):
        """Test format_input with custom parameters."""
        test_data = {"base64_images": [self.sample_image], "prompt": self.sample_prompt}
        model_name = "test-model"

        payloads, batch_data = self.model_interface.format_input(
            test_data,
            protocol="http",
            max_batch_size=10,
            model_name=model_name,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        # Check custom parameters
        self.assertEqual(payloads[0]["max_tokens"], 1024)
        self.assertEqual(payloads[0]["temperature"], 0.7)
        self.assertEqual(payloads[0]["top_p"], 0.9)
        self.assertEqual(payloads[0]["stream"], True)

    def test_format_input_invalid_protocol(self):
        """Test format_input with invalid protocol."""
        test_data = {"base64_images": [self.sample_image], "prompt": self.sample_prompt}

        with self.assertRaises(ValueError) as context:
            self.model_interface.format_input(test_data, protocol="grpc", max_batch_size=10)  # Only http is supported

        self.assertTrue("only supports HTTP" in str(context.exception))

    def test_parse_output_success(self):
        """Test parse_output with a successful response."""
        # Mock response with choices
        mock_response = {
            "choices": [
                {"message": {"content": "This is a cat on a couch."}},
                {"message": {"content": "A dog playing in the park."}},
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that captions were extracted correctly
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "This is a cat on a couch.")
        self.assertEqual(result[1], "A dog playing in the park.")

    def test_parse_output_missing_choices(self):
        """Test parse_output with missing choices in response."""
        # Mock response with missing choices
        mock_response = {"some_other_key": "value"}

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'choices' key is missing" in str(context.exception))

    def test_parse_output_empty_choices(self):
        """Test parse_output with empty choices in response."""
        # Mock response with empty choices
        mock_response = {"choices": []}

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http")

        self.assertTrue("'choices' key is missing or empty" in str(context.exception))

    def test_parse_output_missing_message(self):
        """Test parse_output with missing message in choices."""
        # Mock response with missing message field
        mock_response = {
            "choices": [
                {"some_field": "value"},
                {"message": {"content": "A dog playing in the park."}},
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that default value is returned for missing message
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "No caption returned")
        self.assertEqual(result[1], "A dog playing in the park.")

    def test_parse_output_missing_content(self):
        """Test parse_output with missing content in message."""
        # Mock response with missing content field
        mock_response = {
            "choices": [
                {"message": {"some_field": "value"}},
                {"message": {"content": "A dog playing in the park."}},
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that default value is returned for missing content
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "No caption returned")
        self.assertEqual(result[1], "A dog playing in the park.")

    def test_parse_output_non_dict_response(self):
        """Test parse_output with non-dict response."""
        # Mock non-dict response
        mock_response = "Some raw text response"

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check that response is wrapped in a list as a string
        self.assertEqual(result, ["Some raw text response"])

    def test_parse_output_invalid_protocol(self):
        """Test parse_output with invalid protocol."""
        mock_response = {"choices": [{"message": {"content": "This is a caption."}}]}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="grpc")

        self.assertTrue("only supports HTTP" in str(context.exception))

    def test_process_inference_results(self):
        """Test process_inference_results."""
        # Sample captions
        captions = ["This is a cat on a couch.", "A dog playing in the park.", "A mountain landscape at sunset."]

        result = self.model_interface.process_inference_results(captions, protocol="http")

        # Check that result is the same as input (pass-through)
        self.assertEqual(result, captions)


if __name__ == "__main__":
    unittest.main()
