# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# noqa
import unittest
from unittest.mock import patch
import json
import numpy as np

# Import using the specified pattern
import nv_ingest_api.internal.primitives.nim.model_interface.ocr as model_interface_module
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface

MODULE_UNDER_TEST = f"{model_interface_module.__name__}"


class TestPaddleOCRModelInterface(unittest.TestCase):

    def setUp(self):
        # Create an instance of the model interface
        self.model_interface = PaddleOCRModelInterface()

        # Mock the logger to prevent actual logging during tests
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the base64_to_numpy function
        self.base64_to_numpy_patcher = patch(f"{MODULE_UNDER_TEST}.base64_to_numpy")
        self.mock_base64_to_numpy = self.base64_to_numpy_patcher.start()
        # Make it return a predictable image array
        self.mock_base64_to_numpy.side_effect = lambda b64: np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock the preprocess_image_for_paddle function
        self.preprocess_patcher = patch(f"{MODULE_UNDER_TEST}.preprocess_image_for_paddle")
        self.mock_preprocess = self.preprocess_patcher.start()
        # Make it return a predictable processed array and metadata
        self.mock_preprocess.side_effect = lambda img, **kwargs: (
            np.zeros((3, 32, 64), dtype=np.float32),  # Processed image with channels first
            {
                "original_height": img.shape[0],
                "original_width": img.shape[1],
                "new_height": 32,
                "new_width": 64,
                "pad_height": 2,
                "pad_width": 4,
            },
        )

        # Create sample test data
        self.sample_base64 = "base64_encoded_image_data"
        self.sample_base64_list = ["base64_image_1", "base64_image_2", "base64_image_3"]

    def tearDown(self):
        # Stop all patchers
        self.logger_patcher.stop()
        self.base64_to_numpy_patcher.stop()
        self.preprocess_patcher.stop()

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "PaddleOCR")

    def test_prepare_data_for_inference_single_image(self):
        """Test prepare_data_for_inference method with a single image."""
        test_data = {"base64_image": self.sample_base64}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that base64_to_numpy was called once with the base64 string
        self.mock_base64_to_numpy.assert_called_once_with(self.sample_base64)

        # Check that image_arrays was added to the result
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertTrue(isinstance(result["images"][0], np.ndarray))

    def test_prepare_data_for_inference_multiple_images(self):
        """Test prepare_data_for_inference method with multiple images."""
        test_data = {"base64_images": self.sample_base64_list}
        result = self.model_interface.prepare_data_for_inference(test_data)

        # Check that base64_to_numpy was called for each image
        self.assertEqual(self.mock_base64_to_numpy.call_count, len(self.sample_base64_list))

        # Check that image_arrays was added to the result
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), len(self.sample_base64_list))
        for img in result["images"]:
            self.assertTrue(isinstance(img, np.ndarray))

    def test_prepare_data_for_inference_missing_data(self):
        """Test prepare_data_for_inference method with missing data."""
        test_data = {"some_other_key": "value"}
        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("must include 'base64_image' or 'base64_images'" in str(context.exception))

    def test_prepare_data_for_inference_invalid_base64_images(self):
        """Test prepare_data_for_inference method with invalid base64_images."""
        test_data = {"base64_images": "not_a_list"}  # Should be a list
        with self.assertRaises(ValueError) as context:
            self.model_interface.prepare_data_for_inference(test_data)

        self.assertTrue("must contain a list" in str(context.exception))

    def test_format_input_grpc_single_image(self):
        """Test format_input method with gRPC protocol and a single image."""
        # Set up test data with image_arrays and empty image_dims
        img_array = np.zeros((100, 200, 3), dtype=np.uint8)
        test_data = {"images": [img_array], "image_dims": []}

        batches, batch_data = self.model_interface.format_input(test_data, protocol="grpc", max_batch_size=1)

        # Check that preprocess_image_for_ocr was called
        self.mock_preprocess.assert_called_once_with(img_array)

        # Check the format of the output
        self.assertEqual(len(batches), 1)  # Should have 1 batch
        self.assertEqual(len(batch_data), 1)  # Should have 1 batch data dict

        # The batch should be a numpy array with shape (1, 3, 32, 64)
        # where 1 is the batch size, 3 is channels, and 32x64 is the image dimensions
        self.assertTrue(isinstance(batches[0][0], np.ndarray))

        # Check that image_dims was updated
        self.assertEqual(len(test_data["image_dims"]), 1)

    def test_format_input_grpc_multiple_images(self):
        """Test format_input method with gRPC protocol and multiple images."""
        # Set up test data with multiple image arrays and empty image_dims
        img_arrays = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
        test_data = {"images": img_arrays, "image_dims": []}

        batches, batch_data = self.model_interface.format_input(test_data, protocol="grpc", max_batch_size=2)

        # Check that preprocess_image_for_ocr was called for each image
        self.assertEqual(self.mock_preprocess.call_count, len(img_arrays))

        # Check the format of the output
        self.assertEqual(len(batches), 2)  # Should have 2 batches (2 images in first, 1 in second)
        self.assertEqual(len(batch_data), 2)  # Should have 2 batch data dicts

        # Check that image_dims was updated
        self.assertEqual(len(test_data["image_dims"]), 3)

    def test_format_input_http_single_image(self):
        """Test format_input method with HTTP protocol and a single image."""
        # Set up test data with image_arrays, empty image_dims, and base64_image
        img_array = np.zeros((100, 200, 3), dtype=np.uint8)
        test_data = {
            "images": [img_array],
            "image_dims": [],
            "base64_images": self.sample_base64,
        }

        batches, batch_data = self.model_interface.format_input(test_data, protocol="http", max_batch_size=1)

        # Check the format of the output
        self.assertEqual(len(batches), 1)  # Should have 1 batch
        self.assertEqual(len(batch_data), 1)  # Should have 1 batch data dict

        # The batch should be a dict with 'input' key containing a list of image objects
        self.assertTrue(isinstance(batches[0], dict))
        self.assertIn("input", batches[0])
        self.assertEqual(len(batches[0]["input"]), 1)

        # Check the format of the image object
        img_obj = batches[0]["input"][0]
        self.assertEqual(img_obj["type"], "image_url")
        self.assertTrue(img_obj["url"].startswith("data:image/png;base64,"))

        # Check that image_dims was updated
        self.assertEqual(len(test_data["image_dims"]), 1)

    def test_format_input_http_multiple_images(self):
        """Test format_input method with HTTP protocol and multiple images."""
        # Set up test data with multiple image arrays, empty image_dims, and base64_images
        img_arrays = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
        test_data = {
            "images": img_arrays,
            "image_dims": [],
            "base64_images": self.sample_base64_list,
        }

        batches, batch_data = self.model_interface.format_input(test_data, protocol="http", max_batch_size=2)

        # Check the format of the output
        self.assertEqual(len(batches), 2)  # Should have 2 batches (2 images in first, 1 in second)
        self.assertEqual(len(batch_data), 2)  # Should have 2 batch data dicts

        # Check the first batch
        self.assertTrue(isinstance(batches[0], dict))
        self.assertIn("input", batches[0])
        self.assertEqual(len(batches[0]["input"]), 2)

        # Check the second batch
        self.assertTrue(isinstance(batches[1], dict))
        self.assertIn("input", batches[1])
        self.assertEqual(len(batches[1]["input"]), 1)

        # Check that image_dims was updated
        self.assertEqual(len(test_data["image_dims"]), 3)

    def test_format_input_missing_data(self):
        """Test format_input method with missing data."""
        test_data = {"some_other_key": "value"}
        with self.assertRaises(KeyError) as context:
            self.model_interface.format_input(test_data, protocol="http", max_batch_size=1)

        self.assertTrue("'images'" in str(context.exception))

    def test_format_input_invalid_protocol(self):
        """Test format_input method with an invalid protocol."""
        test_data = {
            "images": [np.zeros((100, 200, 3), dtype=np.uint8)],
            "image_dims": [],
        }
        with self.assertRaises(ValueError) as context:
            self.model_interface.format_input(test_data, protocol="invalid", max_batch_size=1)

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_parse_output_http(self):
        """Test parse_output method with HTTP protocol."""
        # Create a mock HTTP response
        mock_response = {
            "data": [
                {
                    "text_detections": [
                        {
                            "text_prediction": {"text": "Hello", "confidence": 0.9},
                            "bounding_box": {
                                "points": [
                                    {"x": 10, "y": 20},
                                    {"x": 50, "y": 20},
                                    {"x": 50, "y": 40},
                                    {"x": 10, "y": 40},
                                ]
                            },
                        },
                        {
                            "text_prediction": {"text": "World", "confidence": 0.8},
                            "bounding_box": {
                                "points": [
                                    {"x": 60, "y": 20},
                                    {"x": 100, "y": 20},
                                    {"x": 100, "y": 40},
                                    {"x": 60, "y": 40},
                                ]
                            },
                        },
                    ]
                }
            ]
        }

        data = {"image_dims": [{"new_width": 100, "new_height": 200}]}
        result = self.model_interface.parse_output(mock_response, protocol="http", data=data)

        # Check the format of the result
        self.assertEqual(len(result), 1)  # Should have 1 result (for 1 image)

        # Each result should be a list with two elements: bounding boxes and text predictions
        self.assertEqual(len(result[0]), 3)
        bboxes, texts, conf_scores = result[0]

        # Check bounding boxes and text predictions
        self.assertEqual(len(bboxes), 2)  # Should have 2 bounding boxes
        self.assertEqual(len(texts), 2)  # Should have 2 text predictions
        self.assertEqual(texts, ["Hello", "World"])
        self.assertEqual(len(conf_scores), 2)

    def test_parse_output_http_missing_data(self):
        """Test parse_output method with HTTP protocol and missing data."""
        mock_response = {}  # Missing 'data' key
        data = {"image_dims": [{}]}

        with self.assertRaises(RuntimeError) as context:
            self.model_interface.parse_output(mock_response, protocol="http", data=data)

        self.assertTrue("'data' key is missing" in str(context.exception))

    def test_parse_output_grpc(self):
        """Test parse_output method with gRPC protocol."""
        # Create a mock gRPC response
        # Shape (3, 1) for a single image batch
        mock_response = np.array(
            [
                # Bounding boxes
                [json.dumps([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]).encode("utf8")],
                # Text predictions
                [json.dumps(["Sample Text"]).encode("utf8")],
                # Extra data
                [b"extra data"],
            ]
        )

        data = {"image_dims": [{"new_width": 100, "new_height": 200, "pad_width": 4, "pad_height": 2}]}

        # Mock the _extract_content_from_ocr_grpc_response method to isolate the test
        with patch.object(self.model_interface, "_extract_content_from_ocr_grpc_response") as mock_extract:
            # Set a return value that matches the expected format
            mock_extract.return_value = [([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]], ["Sample Text"])]

            result = self.model_interface.parse_output(mock_response, protocol="grpc", data=data)

            # Verify that the method was called with the correct arguments
            mock_extract.assert_called_once_with(
                mock_response, data.get("image_dims"), model_name=model_interface_module.DEFAULT_OCR_MODEL_NAME
            )

            # Check the format of the result
            self.assertEqual(len(result), 1)  # Should have 1 result (for 1 image)

            # Each result should be a list with two elements: bounding boxes and text predictions
            bboxes, texts = result[0]

            # Check bounding boxes and text predictions
            self.assertEqual(bboxes, [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]])
            self.assertEqual(texts, ["Sample Text"])

    def test_parse_output_grpc_multiple_images(self):
        """Test parse_output method with gRPC protocol and multiple images."""
        # Create a mock gRPC response for multiple images
        # Shape (3, 2) for a batch of 2 images
        mock_response = np.array(
            [
                # Bounding boxes for 2 images - must match the format expected by _postprocess_ocr_response
                [
                    json.dumps([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]).encode("utf8"),
                    json.dumps([[0.5, 0.6], [0.7, 0.6], [0.7, 0.8], [0.5, 0.8]]).encode("utf8"),
                ],
                # Text predictions for 2 images
                [
                    json.dumps(["Image 1 Text"]).encode("utf8"),
                    json.dumps(["Image 2 Text"]).encode("utf8"),
                ],
                # Extra data for 2 images
                [b"extra data 1", b"extra data 2"],
            ]
        )

        # Mock the _extract_content_from_ocr_grpc_response method to isolate the test
        with patch.object(self.model_interface, "_extract_content_from_ocr_grpc_response") as mock_extract:
            # Set a return value that matches the expected format
            mock_extract.return_value = [
                ([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]], ["Image 1 Text"]),
                ([[0.5, 0.6], [0.7, 0.6], [0.7, 0.8], [0.5, 0.8]], ["Image 2 Text"]),
            ]

            data = {
                "image_dims": [
                    {
                        "new_width": 100,
                        "new_height": 200,
                        "pad_width": 4,
                        "pad_height": 2,
                    },
                    {
                        "new_width": 120,
                        "new_height": 240,
                        "pad_width": 6,
                        "pad_height": 3,
                    },
                ]
            }

            result = self.model_interface.parse_output(mock_response, protocol="grpc", data=data)

            # Verify that the method was called with the correct arguments
            mock_extract.assert_called_once_with(
                mock_response, data.get("image_dims"), model_name=model_interface_module.DEFAULT_OCR_MODEL_NAME
            )

            # Check the format of the result
            self.assertEqual(len(result), 2)  # Should have 2 results (for 2 images)

            # Check first image result
            bboxes1, texts1 = result[0]
            self.assertEqual(bboxes1, [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]])
            self.assertEqual(texts1, ["Image 1 Text"])

            # Check second image result
            bboxes2, texts2 = result[1]
            self.assertEqual(bboxes2, [[0.5, 0.6], [0.7, 0.6], [0.7, 0.8], [0.5, 0.8]])
            self.assertEqual(texts2, ["Image 2 Text"])

    def test_parse_output_grpc_single_item_response(self):
        """Test parse_output method with gRPC protocol and a response with shape (3,)."""
        # Create a mock gRPC response with shape (3,) for a single image
        mock_response = np.array(
            [
                json.dumps([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]).encode("utf8"),
                json.dumps(["Single Image Text"]).encode("utf8"),
                b"extra data",
            ]
        )

        data = {"image_dims": [{"new_width": 100, "new_height": 200, "pad_width": 4, "pad_height": 2}]}

        # Mock the _extract_content_from_ocr_grpc_response method to isolate the test
        with patch.object(self.model_interface, "_extract_content_from_ocr_grpc_response") as mock_extract:
            # Set a return value that matches the expected format
            mock_extract.return_value = [
                (
                    [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
                    ["Single Image Text"],
                )
            ]

            result = self.model_interface.parse_output(mock_response, protocol="grpc", data=data)

            # Verify that the method was called with the correct arguments
            mock_extract.assert_called_once_with(
                mock_response, data.get("image_dims"), model_name=model_interface_module.DEFAULT_OCR_MODEL_NAME
            )

            # Check the format of the result
            self.assertEqual(len(result), 1)  # Should have 1 result (for 1 image)

            # Each result should be a list with two elements: bounding boxes and text predictions
            bboxes, texts = result[0]

            # Check bounding boxes and text predictions
            self.assertEqual(bboxes, [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]])
            self.assertEqual(texts, ["Single Image Text"])

    def test_parse_output_grpc_invalid_response_type(self):
        """Test parse_output method with gRPC protocol and an invalid response type."""
        mock_response = "not a numpy array"
        data = {"image_dims": [{}]}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="grpc", data=data)

        self.assertTrue("not a NumPy array" in str(context.exception))

    def test_parse_output_invalid_protocol(self):
        """Test parse_output method with an invalid protocol."""
        mock_response = {}
        data = {"image_dims": [{}]}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="invalid", data=data)

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_process_inference_results(self):
        """Test process_inference_results method."""
        # This method simply returns the input unchanged
        test_output = [["bounding_boxes"], ["text_predictions"]]
        result = self.model_interface.process_inference_results(test_output)
        self.assertEqual(result, test_output)

    def test_prepare_ocr_payload(self):
        """Test _prepare_ocr_payload method."""
        test_base64 = "test_base64_string"
        payload = self.model_interface._prepare_ocr_payload(test_base64)

        # Check payload structure
        self.assertIn("input", payload)
        self.assertEqual(len(payload["input"]), 1)
        self.assertEqual(payload["input"][0]["type"], "image_url")
        self.assertEqual(payload["input"][0]["url"], f"data:image/png;base64,{test_base64}")

    def test_postprocess_ocr_response(self):
        """Test _postprocess_ocr_response static method."""
        # Set up test data
        bounding_boxes = [
            [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
            [[0.5, 0.6], [0.7, 0.6], [0.7, 0.8], [0.5, 0.8]],
        ]
        text_predictions = ["Text 1", "Text 2"]
        conf_scores = [0.9, 0.8]
        dims = [
            {
                "new_width": 100,
                "new_height": 200,
                "pad_width": 10,
                "pad_height": 20,
                "scale_factor": 0.5,
            }
        ]

        bboxes, texts, scores = PaddleOCRModelInterface._postprocess_ocr_response(
            bounding_boxes, text_predictions, conf_scores, dims, img_index=0
        )

        # Check that bounding boxes were properly scaled
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(len(texts), 2)

        x_expected = ((0.1 * 100) - 10) / 0.5
        y_expected = ((0.2 * 200) - 20) / 0.5
        self.assertAlmostEqual(bboxes[0][0][0], x_expected)
        self.assertAlmostEqual(bboxes[0][0][1], y_expected)
        self.assertAlmostEqual(conf_scores[0], 0.9)

    def test_postprocess_ocr_response_nan_box(self):
        """Test _postprocess_ocr_response static method with 'nan' box."""
        # Set up test data with one valid box and one 'nan' box
        bounding_boxes = [
            [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
            "nan",
        ]  # This should be skipped
        text_predictions = ["Text 1", "Text 2"]
        conf_scores = [0.9, 0.8]
        dims = [{"new_width": 100, "new_height": 200, "pad_width": 10, "pad_height": 20}]

        bboxes, texts, scores = PaddleOCRModelInterface._postprocess_ocr_response(
            bounding_boxes, text_predictions, conf_scores, dims, img_index=0
        )

        # Check that only the valid box was processed
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], "Text 1")
        self.assertEqual(conf_scores[0], 0.9)

    def test_postprocess_ocr_response_no_dims(self):
        """Test _postprocess_ocr_response static method with no dims."""
        bounding_boxes = [[[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]]
        text_predictions = ["Text 1"]
        conf_scores = [0.9]

        with self.assertRaises(ValueError) as context:
            PaddleOCRModelInterface._postprocess_ocr_response(bounding_boxes, text_predictions, conf_scores, None)

        self.assertTrue("No image_dims provided" in str(context.exception))

    def test_postprocess_ocr_response_index_out_of_range(self):
        """Test _postprocess_ocr_response with img_index out of range."""
        bounding_boxes = [[[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]]
        text_predictions = ["Text 1"]
        conf_scores = [0.9]
        dims = [{"new_width": 100, "new_height": 200, "pad_width": 10, "pad_height": 20}]

        # Mock the logger to check warning
        with patch(f"{MODULE_UNDER_TEST}.logger") as mock_logger:
            bboxes, texts, scores = PaddleOCRModelInterface._postprocess_ocr_response(
                bounding_boxes,
                text_predictions,
                conf_scores,
                dims,
                img_index=1,  # Out of range
            )

            # Should use index 0 as fallback and log a warning
            mock_logger.warning.assert_called_once()
            self.assertEqual(len(bboxes), 1)
            self.assertEqual(len(texts), 1)
            self.assertEqual(len(conf_scores), 1)


if __name__ == "__main__":
    unittest.main()
