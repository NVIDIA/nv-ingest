# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Import the module under test
import nv_ingest_api.internal.primitives.nim.model_interface.yolox as module_under_test
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YoloxPageElementsModelInterface,
    YoloxModelInterfaceBase,
)

MODULE_UNDER_TEST = f"{module_under_test.__name__}"

# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors


class TestYoloxPageElementsModelInterface(unittest.TestCase):

    def setUp(self):
        # Mock constants
        self.constants_patcher = patch.multiple(
            MODULE_UNDER_TEST,
            YOLOX_PAGE_NIM_MAX_IMAGE_SIZE=1000000,
            YOLOX_PAGE_CONF_THRESHOLD=0.5,
            YOLOX_PAGE_IOU_THRESHOLD=0.45,
            YOLOX_PAGE_MIN_SCORE=0.3,
            YOLOX_PAGE_CLASS_LABELS=["table", "chart", "infographic", "title"],
        )
        self.mocked_constants = self.constants_patcher.start()

        # Mock the logger
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the expansion functions
        self.expand_table_patcher = patch(f"{MODULE_UNDER_TEST}.expand_table_bboxes")
        self.mock_expand_table = self.expand_table_patcher.start()
        self.mock_expand_table.side_effect = lambda x: x  # Identity function for testing

        self.expand_chart_patcher = patch(f"{MODULE_UNDER_TEST}.expand_chart_bboxes")
        self.mock_expand_chart = self.expand_chart_patcher.start()
        self.mock_expand_chart.side_effect = lambda x: x  # Identity function for testing

        # Create an instance of YoloxPageElementsModelInterface
        self.model_interface = YoloxPageElementsModelInterface()

    def tearDown(self):
        # Stop all patchers
        self.constants_patcher.stop()
        self.logger_patcher.stop()
        self.expand_table_patcher.stop()
        self.expand_chart_patcher.stop()

    def test_default_initialization(self):
        """Test default initialization (should use v2 parameters)."""
        model = YoloxPageElementsModelInterface()

        # Check that v2 parameters are used by default
        self.assertEqual(model.class_labels, ["table", "chart", "infographic", "title"])

    def test_inheritance(self):
        """Test that YoloxPageElementsModelInterface inherits from YoloxModelInterfaceBase."""
        self.assertIsInstance(self.model_interface, YoloxModelInterfaceBase)

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "yolox-page-elements")

    def test_postprocess_annotations_table_filtering(self):
        """Test postprocess_annotations with table filtering."""
        # Create mock annotation dictionaries
        annotations = [
            {
                "table": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # Above threshold (0.4)
                    [0.5, 0.6, 0.7, 0.8, 0.3],  # Below threshold (0.4)
                ],
                "chart": [],
            }
        ]

        original_shapes = [(100, 200, 3)]

        # Save original method and create a proper mock
        original_transform = self.model_interface.transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[{"table": [[0.1, 0.2, 0.3, 0.4, 0.5]], "chart": []}]
        )

        try:
            result = self.model_interface.postprocess_annotations(annotations, original_image_shapes=original_shapes)

            # Check that both expansion functions were called
            self.mock_expand_table.assert_called_once_with(annotations[0])
            self.mock_expand_chart.assert_called_once()

            # Check that transform_normalized_coordinates_to_original was called
            self.model_interface.transform_normalized_coordinates_to_original.assert_called_once()

            # Check the result (should have only one table entry with confidence >= 0.4)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["table"]), 1)
            self.assertEqual(result[0]["table"][0][4], 0.5)  # The entry with confidence 0.5
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_chart_filtering(self):
        """Test postprocess_annotations with chart filtering."""
        # Create mock annotation dictionaries
        annotations = [
            {
                "table": [],
                "chart": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # Above threshold (0.4)
                    [0.5, 0.6, 0.7, 0.8, 0.3],  # Below threshold (0.4)
                ],
            }
        ]

        original_shapes = [(100, 200, 3)]

        # Save original method and create a proper mock
        original_transform = self.model_interface.transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[{"table": [], "chart": [[0.1, 0.2, 0.3, 0.4, 0.5]]}]
        )

        try:
            result = self.model_interface.postprocess_annotations(annotations, original_image_shapes=original_shapes)

            # Check that both expansion functions were called
            self.mock_expand_table.assert_called_once()
            self.mock_expand_chart.assert_called_once()

            # Check the result (should have only one chart entry with confidence >= 0.4)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["chart"]), 1)
            self.assertEqual(result[0]["chart"][0][4], 0.5)  # The entry with confidence 0.5
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_infographic_filtering(self):
        """Test postprocess_annotations with infographic filtering."""
        # Create mock annotation dictionaries
        annotations = [
            {
                "table": [],
                "chart": [],
                "infographic": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # Above threshold (0.4)
                    [0.5, 0.6, 0.7, 0.8, 0.3],  # Below threshold (0.4)
                ],
            }
        ]

        original_shapes = [(100, 200, 3)]

        # Save original method and create a proper mock
        original_transform = self.model_interface.transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[{"table": [], "chart": [], "infographic": [[0.1, 0.2, 0.3, 0.4, 0.5]]}]
        )

        try:
            result = self.model_interface.postprocess_annotations(annotations, original_image_shapes=original_shapes)

            # Check the result (should have only one infographic entry with confidence >= 0.4)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["infographic"]), 1)
            self.assertEqual(result[0]["infographic"][0][4], 0.5)  # The entry with confidence 0.5
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_title_preservation(self):
        """Test postprocess_annotations preserves title entries without filtering."""
        # Create mock annotation dictionaries
        annotations = [
            {
                "table": [],
                "chart": [],
                "title": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],  # Should be preserved regardless of confidence
                    [0.5, 0.6, 0.7, 0.8, 0.3],  # Should be preserved regardless of confidence
                ],
            }
        ]

        original_shapes = [(100, 200, 3)]

        # Save original method and create a proper mock
        original_transform = self.model_interface.transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[{"table": [], "chart": [], "title": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.3]]}]
        )

        try:
            result = self.model_interface.postprocess_annotations(annotations, original_image_shapes=original_shapes)

            # Check that title entries are preserved without filtering
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]["title"]), 2)
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_multiple_images(self):
        """Test postprocess_annotations with multiple images."""
        # Create mock annotation dictionaries for multiple images
        annotations = [
            {"table": [[0.1, 0.2, 0.3, 0.4, 0.5]]},  # First image
            {"chart": [[0.1, 0.2, 0.3, 0.4, 0.5]]},  # Second image
        ]

        original_shapes = [(100, 200, 3), (150, 250, 3)]

        # Mock transform_normalized_coordinates_to_original to return transformed values
        transformed_results = [
            {"table": [[20, 10, 60, 40, 0.5]]},  # First image, coordinates transformed
            {"chart": [[25, 15, 75, 60, 0.5]]},  # Second image, coordinates transformed
        ]

        # Create a proper mock for the method
        original_transform = self.model_interface.transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(return_value=transformed_results)

        try:
            result = self.model_interface.postprocess_annotations(annotations, original_image_shapes=original_shapes)

            # Check that transform_normalized_coordinates_to_original was called with both images
            self.model_interface.transform_normalized_coordinates_to_original.assert_called_once_with(
                [{"table": [[0.1, 0.2, 0.3, 0.4, 0.5]]}, {"chart": [[0.1, 0.2, 0.3, 0.4, 0.5]]}], original_shapes
            )

            # Check the result (should match the transformed results)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["table"], [[20, 10, 60, 40, 0.5]])
            self.assertEqual(result[1]["chart"], [[25, 15, 75, 60, 0.5]])
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_invalid_final_score(self):
        """Test postprocess_annotations with invalid final_score."""
        # Create an instance with an invalid final_score (not a dictionary)
        model = YoloxPageElementsModelInterface()

        # Create mock annotations
        annotations = [{"table": [[0.1, 0.2, 0.3, 0.4, 0.5]]}]

        # Should raise ValueError due to invalid final_score
        with self.assertRaises(ValueError) as context:
            model.postprocess_annotations(annotations, final_score=0.5)

        self.assertTrue("requires a dictionary of thresholds" in str(context.exception))

    def test_postprocess_annotations_missing_class_in_final_score(self):
        """Test postprocess_annotations with final_score missing a required class."""
        # Create an instance with final_score missing a required class
        model = YoloxPageElementsModelInterface()

        # Create mock annotations
        annotations = [{"table": [[0.1, 0.2, 0.3, 0.4, 0.5]]}]

        # Should raise ValueError due to missing class in final_score
        with self.assertRaises(ValueError) as context:
            model.postprocess_annotations(annotations, final_score={"table": 0.5, "chart": 0.5})

        self.assertTrue("requires a dictionary of thresholds" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
