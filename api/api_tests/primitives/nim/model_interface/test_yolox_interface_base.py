# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the module under test
import nv_ingest_api.internal.primitives.nim.model_interface.yolox as module_under_test
from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxModelInterfaceBase

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors


# Create a concrete implementation of the abstract class for testing
class TestableYoloxModelInterface(YoloxModelInterfaceBase):
    """
    Concrete implementation of YoloxModelInterfaceBase for testing.
    """

    def name(self):
        return "Yolox"

    def postprocess_annotations(self, annotation_dicts, **kwargs):
        # Simple pass-through implementation for testing
        return annotation_dicts


class TestYoloxModelInterface(unittest.TestCase):

    def setUp(self):
        # Create dummy test images
        self.image1 = np.ones((100, 200, 3), dtype=np.float32)  # 100x200 white image
        self.image2 = np.zeros((150, 250, 3), dtype=np.float32)  # 150x250 black image
        self.image3 = np.ones((120, 180, 3), dtype=np.float32) * 0.5  # 120x180 gray image

        # Test parameters for the model
        self.test_params = {
            "image_preproc_width": 1024,
            "image_preproc_height": 1024,
            "nim_max_image_size": 1000000,
            "num_classes": 80,
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "min_score": 0.3,
            "final_score": 0.3,
            "class_labels": ["person", "car", "dog", "cat"],
        }

        # Create a mockable instance of our testable implementation
        self.model_interface = TestableYoloxModelInterface(**self.test_params)

        # Mock common dependencies
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        self.resize_patcher = patch(f"{MODULE_UNDER_TEST}.resize_image")
        self.mock_resize = self.resize_patcher.start()
        self.mock_resize.side_effect = lambda img, size: np.zeros((*size[::-1], 3), dtype=np.float32)

        self.scale_patcher = patch(f"{MODULE_UNDER_TEST}.scale_image_to_encoding_size")
        self.mock_scale = self.scale_patcher.start()
        self.mock_scale.side_effect = lambda img, max_base64_size: (img, (1024, 1024))

        self.postprocess_model_patcher = patch(f"{MODULE_UNDER_TEST}.postprocess_model_prediction")
        self.mock_postprocess_model = self.postprocess_model_patcher.start()

        self.postprocess_results_patcher = patch(f"{MODULE_UNDER_TEST}.postprocess_results")
        self.mock_postprocess_results = self.postprocess_results_patcher.start()

        self.image_patcher = patch(f"{MODULE_UNDER_TEST}.Image")
        self.mock_image = self.image_patcher.start()
        self.mock_pil_image = MagicMock()
        self.mock_image.fromarray.return_value = self.mock_pil_image
        self.mock_pil_image.size = (200, 100)

        # Mock BytesIO and base64
        self.bytesio_patcher = patch(f"{MODULE_UNDER_TEST}.io.BytesIO")
        self.mock_bytesio = self.bytesio_patcher.start()
        self.mock_buffer = MagicMock()
        self.mock_bytesio.return_value = self.mock_buffer
        self.mock_buffer.getvalue.return_value = b"test_image_data"

        self.base64_patcher = patch(f"{MODULE_UNDER_TEST}.base64.b64encode")
        self.mock_base64 = self.base64_patcher.start()
        self.mock_base64.return_value = b"base64_encoded_data"

        # Mock log function
        self.log_patcher = patch(f"{MODULE_UNDER_TEST}.log")
        self.mock_log = self.log_patcher.start()
        self.mock_log.return_value = 1.0  # log2(2) = 1.0

    def tearDown(self):
        # Stop all patchers
        self.logger_patcher.stop()
        self.resize_patcher.stop()
        self.scale_patcher.stop()
        self.postprocess_model_patcher.stop()
        self.postprocess_results_patcher.stop()
        self.image_patcher.stop()
        self.bytesio_patcher.stop()
        self.base64_patcher.stop()
        self.log_patcher.stop()

    def test_initialization(self):
        """Test initialization with parameters."""
        model = TestableYoloxModelInterface(**self.test_params)

        # Check that parameters are correctly stored
        self.assertEqual(model.image_preproc_width, self.test_params["image_preproc_width"])
        self.assertEqual(model.image_preproc_height, self.test_params["image_preproc_height"])
        self.assertEqual(model.nim_max_image_size, self.test_params["nim_max_image_size"])
        self.assertEqual(model.num_classes, self.test_params["num_classes"])
        self.assertEqual(model.conf_threshold, self.test_params["conf_threshold"])
        self.assertEqual(model.iou_threshold, self.test_params["iou_threshold"])
        self.assertEqual(model.min_score, self.test_params["min_score"])
        self.assertEqual(model.final_score, self.test_params["final_score"])
        self.assertEqual(model.class_labels, self.test_params["class_labels"])

    def test_inheritance(self):
        """Test that YoloxModelInterfaceBase inherits from ModelInterface."""
        self.assertIsInstance(self.model_interface, ModelInterface)

    def test_prepare_data_valid_input(self):
        """Test prepare_data_for_inference with valid input."""
        # Create input data with images
        input_data = {"images": [self.image1, self.image2, self.image3]}

        result = self.model_interface.prepare_data_for_inference(input_data)

        # Check that original shapes are stored correctly
        self.assertIn("original_image_shapes", result)
        self.assertEqual(len(result["original_image_shapes"]), 3)
        self.assertEqual(result["original_image_shapes"][0], self.image1.shape)
        self.assertEqual(result["original_image_shapes"][1], self.image2.shape)
        self.assertEqual(result["original_image_shapes"][2], self.image3.shape)

    def test_prepare_data_missing_images_key(self):
        """Test prepare_data_for_inference with missing 'images' key."""
        # Create invalid input data without 'images' key
        input_data = {"other_key": "value"}

        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(input_data)

        self.assertTrue("'images' key" in str(context.exception))

    def test_prepare_data_not_dict(self):
        """Test prepare_data_for_inference with non-dictionary input."""
        # Pass a non-dict value
        input_data = ["image1", "image2"]

        with self.assertRaises(KeyError) as context:
            self.model_interface.prepare_data_for_inference(input_data)

        self.assertTrue("must be a dictionary" in str(context.exception))

    def test_prepare_data_non_numpy_images(self):
        """Test prepare_data_for_inference with non-numpy array images."""
        # Create input data with non-numpy images
        input_data = {"images": ["image1", "image2"]}

        with self.assertRaises(ValueError) as context:
            self.model_interface.prepare_data_for_inference(input_data)

        self.assertTrue("must be numpy.ndarray" in str(context.exception))

    def test_format_input_grpc(self):
        """Test format_input with gRPC protocol."""
        # Create input data with images and shapes
        input_data = {"images": [self.image1, self.image2], "original_image_shapes": [(100, 200, 3), (150, 250, 3)]}

        batches, batch_data = self.model_interface.format_input(input_data, protocol="grpc", max_batch_size=2)

        # Check that resize was called for each image
        self.assertEqual(self.mock_resize.call_count, 2)

        # Check batches
        self.assertEqual(len(batches), 1)  # Should be 1 batch with 2 images
        self.assertIsInstance(batches[0], np.ndarray)

        # Check batch_data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(len(batch_data[0]["images"]), 2)
        self.assertEqual(len(batch_data[0]["original_image_shapes"]), 2)

    def test_format_input_grpc_batching(self):
        """Test format_input with gRPC protocol and batching."""
        # Create input data with more images than max_batch_size
        input_data = {
            "images": [self.image1, self.image2, self.image3],
            "original_image_shapes": [(100, 200, 3), (150, 250, 3), (120, 180, 3)],
        }

        batches, batch_data = self.model_interface.format_input(input_data, protocol="grpc", max_batch_size=2)

        # Check that resize was called for each image
        self.assertEqual(self.mock_resize.call_count, 3)

        # Check batches (should be 2 batches: 2 images + 1 image)
        self.assertEqual(len(batches), 2)

        # Check batch_data
        self.assertEqual(len(batch_data), 2)
        self.assertEqual(len(batch_data[0]["images"]), 2)
        self.assertEqual(len(batch_data[1]["images"]), 1)

    def test_format_input_http(self):
        """Test format_input with HTTP protocol."""
        # Create input data with images and shapes
        input_data = {"images": [self.image1, self.image2], "original_image_shapes": [(100, 200, 3), (150, 250, 3)]}

        batches, batch_data = self.model_interface.format_input(input_data, protocol="http", max_batch_size=2)

        # Check that Image.fromarray was called for each image
        self.assertEqual(self.mock_image.fromarray.call_count, 2)

        # Check batches
        self.assertEqual(len(batches), 1)  # Should be 1 batch with 2 images
        self.assertIsInstance(batches[0], dict)
        self.assertIn("input", batches[0])
        self.assertEqual(len(batches[0]["input"]), 2)

        # Check batch_data
        self.assertEqual(len(batch_data), 1)
        self.assertEqual(len(batch_data[0]["images"]), 2)
        self.assertEqual(len(batch_data[0]["original_image_shapes"]), 2)

        # Check the format of the content
        content_item = batches[0]["input"][0]
        self.assertEqual(content_item["type"], "image_url")
        self.assertTrue(content_item["url"].startswith("data:image/png;base64,"))

    def test_format_input_http_batching(self):
        """Test format_input with HTTP protocol and batching."""
        # Create input data with more images than max_batch_size
        input_data = {
            "images": [self.image1, self.image2, self.image3],
            "original_image_shapes": [(100, 200, 3), (150, 250, 3), (120, 180, 3)],
        }

        batches, batch_data = self.model_interface.format_input(input_data, protocol="http", max_batch_size=2)

        # Check batches (should be 2 batches: 2 images + 1 image)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]["input"]), 2)
        self.assertEqual(len(batches[1]["input"]), 1)

        # Check batch_data
        self.assertEqual(len(batch_data), 2)
        self.assertEqual(len(batch_data[0]["images"]), 2)
        self.assertEqual(len(batch_data[1]["images"]), 1)

    def test_format_input_invalid_protocol(self):
        """Test format_input with invalid protocol."""
        input_data = {"images": [self.image1], "original_image_shapes": [(100, 200, 3)]}

        with self.assertRaises(ValueError) as context:
            self.model_interface.format_input(input_data, protocol="invalid", max_batch_size=2)

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_parse_output_grpc(self):
        """Test parse_output with gRPC protocol."""
        # Create a mock gRPC response (numpy array)
        mock_response = np.array([[[1.0, 2.0, 3.0]]])

        result = self.model_interface.parse_output(mock_response, protocol="grpc")

        # Check that the response is returned as-is
        self.assertIs(result, mock_response)

    def test_parse_output_http(self):
        """Test parse_output with HTTP protocol."""
        # Create a mock HTTP response
        mock_response = {
            "data": [
                {
                    "bounding_boxes": {
                        "person": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, "confidence": 0.9}],
                        "car": [{"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8, "confidence": 0.85}],
                    }
                }
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check the formatted output
        self.assertEqual(len(result), 1)  # One item for one image
        self.assertIn("person", result[0])
        self.assertIn("car", result[0])

        # Check person detection
        self.assertEqual(len(result[0]["person"]), 1)
        person_detection = result[0]["person"][0]
        self.assertEqual(person_detection[0], 0.1)  # x_min
        self.assertEqual(person_detection[1], 0.2)  # y_min
        self.assertEqual(person_detection[2], 0.3)  # x_max
        self.assertEqual(person_detection[3], 0.4)  # y_max
        self.assertEqual(person_detection[4], 0.9)  # confidence

        # Check car detection
        self.assertEqual(len(result[0]["car"]), 1)
        car_detection = result[0]["car"][0]
        self.assertEqual(car_detection[0], 0.5)  # x_min
        self.assertEqual(car_detection[1], 0.6)  # y_min
        self.assertEqual(car_detection[2], 0.7)  # x_max
        self.assertEqual(car_detection[3], 0.8)  # y_max
        self.assertEqual(car_detection[4], 0.85)  # confidence

    def test_parse_output_http_multiple_images(self):
        """Test parse_output with HTTP protocol and multiple images."""
        # Create a mock HTTP response with multiple images
        mock_response = {
            "data": [
                {
                    "bounding_boxes": {
                        "person": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4, "confidence": 0.9}]
                    }
                },
                {
                    "bounding_boxes": {
                        "car": [{"x_min": 0.5, "y_min": 0.6, "x_max": 0.7, "y_max": 0.8, "confidence": 0.85}]
                    }
                },
            ]
        }

        result = self.model_interface.parse_output(mock_response, protocol="http")

        # Check the number of results
        self.assertEqual(len(result), 2)

        # Check first result (person)
        self.assertIn("person", result[0])
        self.assertEqual(len(result[0]["person"]), 1)

        # Check second result (car)
        self.assertIn("car", result[1])
        self.assertEqual(len(result[1]["car"]), 1)

    def test_parse_output_invalid_protocol(self):
        """Test parse_output with invalid protocol."""
        mock_response = {}

        with self.assertRaises(ValueError) as context:
            self.model_interface.parse_output(mock_response, protocol="invalid")

        self.assertTrue("Invalid protocol" in str(context.exception))

    def test_process_inference_results_http(self):
        """Test process_inference_results with HTTP protocol."""
        # For HTTP, the output should already be in the correct format
        mock_output = [{"person": [[0.1, 0.2, 0.3, 0.4, 0.9]], "car": [[0.5, 0.6, 0.7, 0.8, 0.85]]}]

        # Mock postprocess_annotations to return output unchanged
        self.model_interface.postprocess_annotations = MagicMock(return_value=mock_output)

        result = self.model_interface.process_inference_results(mock_output, protocol="http")

        # Check that postprocess_annotations was called
        self.model_interface.postprocess_annotations.assert_called_once_with(mock_output)

        # Check that the result matches the mock output
        self.assertEqual(result, mock_output)

    def test_process_inference_results_grpc(self):
        """Test process_inference_results with gRPC protocol."""
        # For gRPC, further processing is needed
        mock_output = np.array([[[1.0, 2.0, 3.0]]])
        mock_shapes = [(100, 200, 3)]

        # Set up mock returns for postprocessing functions
        self.mock_postprocess_model.return_value = "processed_pred"
        self.mock_postprocess_results.return_value = "processed_results"

        # Mock postprocess_annotations to return output unchanged
        self.model_interface.postprocess_annotations = MagicMock(return_value="final_results")

        result = self.model_interface.process_inference_results(
            mock_output, protocol="grpc", original_image_shapes=mock_shapes
        )

        # Check that postprocessing functions were called correctly
        self.mock_postprocess_model.assert_called_once_with(
            mock_output,
            self.model_interface.num_classes,
            self.model_interface.conf_threshold,
            self.model_interface.iou_threshold,
            class_agnostic=False,
        )

        self.mock_postprocess_results.assert_called_once_with(
            "processed_pred",
            mock_shapes,
            self.model_interface.image_preproc_width,
            self.model_interface.image_preproc_height,
            self.model_interface.class_labels,
            min_score=self.model_interface.min_score,
        )

        # Check that postprocess_annotations was called with processed results
        # Use ANY for kwargs since they might vary
        self.model_interface.postprocess_annotations.assert_called_once()
        args, kwargs = self.model_interface.postprocess_annotations.call_args
        self.assertEqual(args[0], "processed_results")
        self.assertEqual(kwargs.get("original_image_shapes"), mock_shapes)

    def test_transform_normalized_coordinates(self):
        """Test transform_normalized_coordinates_to_original."""
        # Create mock results with normalized coordinates
        mock_results = [{"person": [[0.1, 0.2, 0.3, 0.4, 0.9]], "car": [[0.5, 0.6, 0.7, 0.8, 0.85]]}]

        # Create original shapes
        original_shapes = [(100, 200, 3)]  # height, width, channels

        transformed_results = self.model_interface.transform_normalized_coordinates_to_original(
            mock_results, original_shapes
        )

        # Check transformed coordinates for person
        person_bbox = transformed_results[0]["person"][0]
        self.assertEqual(person_bbox[0], 0.1 * 200)  # x_min * width
        self.assertEqual(person_bbox[1], 0.2 * 100)  # y_min * height
        self.assertEqual(person_bbox[2], 0.3 * 200)  # x_max * width
        self.assertEqual(person_bbox[3], 0.4 * 100)  # y_max * height
        self.assertEqual(person_bbox[4], 0.9)  # confidence (unchanged)

        # Check transformed coordinates for car
        car_bbox = transformed_results[0]["car"][0]
        self.assertEqual(car_bbox[0], 0.5 * 200)  # x_min * width
        self.assertEqual(car_bbox[1], 0.6 * 100)  # y_min * height
        self.assertEqual(car_bbox[2], 0.7 * 200)  # x_max * width
        self.assertEqual(car_bbox[3], 0.8 * 100)  # y_max * height
        self.assertEqual(car_bbox[4], 0.85)  # confidence (unchanged)

    def test_transform_normalized_coordinates_multiple_images(self):
        """Test transform_normalized_coordinates_to_original with multiple images."""
        # Create mock results with normalized coordinates for multiple images
        mock_results = [
            {"person": [[0.1, 0.2, 0.3, 0.4, 0.9]]},  # First image
            {"car": [[0.5, 0.6, 0.7, 0.8, 0.85]]},  # Second image
        ]

        # Create original shapes for multiple images
        original_shapes = [(100, 200, 3), (150, 250, 3)]  # height, width, channels

        transformed_results = self.model_interface.transform_normalized_coordinates_to_original(
            mock_results, original_shapes
        )

        # Check transformed coordinates for first image (person)
        person_bbox = transformed_results[0]["person"][0]
        self.assertEqual(person_bbox[0], 0.1 * 200)  # x_min * width
        self.assertEqual(person_bbox[1], 0.2 * 100)  # y_min * height

        # Check transformed coordinates for second image (car)
        car_bbox = transformed_results[1]["car"][0]
        self.assertEqual(car_bbox[0], 0.5 * 250)  # x_min * width
        self.assertEqual(car_bbox[1], 0.6 * 150)  # y_min * height


if __name__ == "__main__":
    unittest.main()
