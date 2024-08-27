# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import PIL
import pytest

from nv_ingest.extraction_workflows.pdf.eclair_utils import convert_mmd_to_plain_text_ours
from nv_ingest.extraction_workflows.pdf.eclair_utils import crop_image
from nv_ingest.extraction_workflows.pdf.eclair_utils import extract_classes_bboxes
from nv_ingest.extraction_workflows.pdf.eclair_utils import pad_image
from nv_ingest.extraction_workflows.pdf.eclair_utils import reverse_transform_bbox


def test_pad_image_same_size():
    array = np.ones((100, 100, 3), dtype=np.uint8)
    padded_array, (pad_width, pad_height) = pad_image(array, 100, 100)

    assert np.array_equal(padded_array, array)
    assert pad_width == 0
    assert pad_height == 0


def test_pad_image_smaller_size():
    array = np.ones((50, 50, 3), dtype=np.uint8)
    padded_array, (pad_width, pad_height) = pad_image(array, 100, 100)

    assert padded_array.shape == (100, 100, 3)
    assert pad_width == (100 - 50) // 2
    assert pad_height == (100 - 50) // 2
    assert np.array_equal(padded_array[pad_height : pad_height + 50, pad_width : pad_width + 50], array)  # noqa: E203


def test_pad_image_width_exceeds_target():
    array = np.ones((50, 150, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image array is too large"):
        pad_image(array, 100, 100)


def test_pad_image_height_exceeds_target():
    array = np.ones((150, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image array is too large"):
        pad_image(array, 100, 100)


def test_pad_image_with_non_default_target():
    array = np.ones((60, 60, 3), dtype=np.uint8)
    target_width = 80
    target_height = 80
    padded_array, (pad_width, pad_height) = pad_image(array, target_width, target_height)

    assert padded_array.shape == (target_height, target_width, 3)
    assert pad_width == (target_width - 60) // 2
    assert pad_height == (target_height - 60) // 2
    assert np.array_equal(padded_array[pad_height : pad_height + 60, pad_width : pad_width + 60], array)  # noqa: E203


def test_extract_classes_bboxes_simple():
    text = "<x_10><y_20>text1<x_30><y_40><class_A>"
    expected_classes = ["A"]
    expected_bboxes = [(10, 20, 30, 40)]
    expected_texts = ["text1"]
    classes, bboxes, texts = extract_classes_bboxes(text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


def test_extract_classes_bboxes_multiple():
    text = "<x_10><y_20>text1<x_30><y_40><class_A>\n<x_50><y_60>text2<x_70><y_80><class_B>"
    expected_classes = ["A", "B"]
    expected_bboxes = [(10, 20, 30, 40), (50, 60, 70, 80)]
    expected_texts = ["text1", "text2"]
    classes, bboxes, texts = extract_classes_bboxes(text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


def test_extract_classes_bboxes_no_match():
    text = "This text does not match the pattern"
    expected_classes = []
    expected_bboxes = []
    expected_texts = []
    classes, bboxes, texts = extract_classes_bboxes(text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


def test_extract_classes_bboxes_different_format():
    text = "<x_1><y_2>sample<x_3><y_4><class_test>"
    expected_classes = ["test"]
    expected_bboxes = [(1, 2, 3, 4)]
    expected_texts = ["sample"]
    classes, bboxes, texts = extract_classes_bboxes(text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


def test_extract_classes_bboxes_empty_input():
    text = ""
    expected_classes = []
    expected_bboxes = []
    expected_texts = []
    classes, bboxes, texts = extract_classes_bboxes(text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


def test_convert_mmd_to_plain_text_ours_headers():
    mmd_text = "## Header"
    expected = "Header"
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def test_convert_mmd_to_plain_text_ours_bold_italic():
    mmd_text = "This is **bold** and *italic* text."
    expected = "This is bold and italic text."
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def test_convert_mmd_to_plain_text_ours_inline_math():
    mmd_text = "This is a formula: \\(E=mc^2\\)"
    expected_with_math = "This is a formula: E=mc^2"
    expected_without_math = "This is a formula:"
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected_with_math
    assert convert_mmd_to_plain_text_ours(mmd_text, remove_inline_math=True) == expected_without_math


def test_convert_mmd_to_plain_text_ours_lists_tables():
    mmd_text = "* List item\n\\begin{table}content\\end{table}"
    expected = "List item"
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def test_convert_mmd_to_plain_text_ours_code_blocks_equations():
    mmd_text = "```\ncode block\n```\n\\[ equation \\]"
    expected = ""
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def test_convert_mmd_to_plain_text_ours_special_chars():
    mmd_text = "Backslash \\ should be removed."
    expected = "Backslash  should be removed."
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def test_convert_mmd_to_plain_text_ours_mixed_content():
    mmd_text = """
    ## Header

    This is **bold** and *italic* text with a formula: \\(E=mc^2\\).

    \\\[ equation \\\]

    \\begin{table}content\\end{table}
    """  # noqa: W605
    expected = """
    Header

    This is bold and italic text with a formula: E=mc^2.

    """.strip()  # noqa: W605
    assert convert_mmd_to_plain_text_ours(mmd_text) == expected


def create_test_image(width: int, height: int, color: tuple = (255, 0, 0)) -> PIL.Image:
    """Helper function to create a solid color image for testing."""
    image = PIL.Image.new("RGB", (width, height), color)
    return image


# def test_pymupdf_page_to_numpy_array_simple():
#    mock_page = Mock(spec=fitz.Page)
#    mock_pixmap = Mock()
#    mock_pixmap.pil_tobytes.return_value = b"fake image data"
#    mock_page.get_pixmap.return_value = mock_pixmap
#
#    with patch("PIL.Image.open", return_value=PIL.Image.new("RGB", (100, 100))):
#        image, offset = pymupdf_page_to_numpy_array(mock_page, target_width=100, target_height=100)
#
#    assert isinstance(image, np.ndarray)
#    assert image.shape == (1280, 1024, 3)
#    assert isinstance(offset, tuple)
#    assert len(offset) == 2
#    assert all(isinstance(x, int) for x in offset)
#    mock_page.get_pixmap.assert_called_once_with(dpi=300)
#    mock_pixmap.pil_tobytes.assert_called_once_with(format="PNG")


# def test_pymupdf_page_to_numpy_array_different_dpi():
#     mock_page = Mock(spec=fitz.Page)
#     mock_pixmap = Mock()
#     mock_pixmap.pil_tobytes.return_value = b"fake image data"
#     mock_page.get_pixmap.return_value = mock_pixmap


def test_crop_image_valid_bbox():
    array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = (10, 10, 50, 50)
    result = crop_image(array, bbox)
    assert result is not None
    assert isinstance(result, str)


def test_crop_image_partial_outside_bbox():
    array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = (90, 90, 110, 110)
    result = crop_image(array, bbox)
    assert result is not None
    assert isinstance(result, str)


def test_crop_image_completely_outside_bbox():
    array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = (110, 110, 120, 120)
    result = crop_image(array, bbox)
    assert result is None


def test_crop_image_zero_area_bbox():
    array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = (50, 50, 50, 50)
    result = crop_image(array, bbox)
    assert result is None


def test_crop_image_different_format():
    array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = (10, 10, 50, 50)
    result = crop_image(array, bbox, format="JPEG")
    assert result is not None
    assert isinstance(result, str)


def test_reverse_transform_bbox_no_offset():
    bbox = (10, 20, 30, 40)
    bbox_offset = (0, 0)
    expected_bbox = (10, 20, 30, 40)
    transformed_bbox = reverse_transform_bbox(bbox, bbox_offset, 100, 100)

    assert transformed_bbox == expected_bbox


def test_reverse_transform_bbox_with_offset():
    bbox = (20, 30, 40, 50)
    bbox_offset = (10, 10)
    expected_bbox = (12, 25, 37, 50)
    transformed_bbox = reverse_transform_bbox(bbox, bbox_offset, 100, 100)

    assert transformed_bbox == expected_bbox


def test_reverse_transform_bbox_with_large_offset():
    bbox = (60, 80, 90, 100)
    bbox_offset = (20, 30)
    width_ratio = (100 - 2 * bbox_offset[0]) / 100
    height_ratio = (100 - 2 * bbox_offset[1]) / 100
    expected_bbox = (
        int((60 - bbox_offset[0]) / width_ratio),
        int((80 - bbox_offset[1]) / height_ratio),
        int((90 - bbox_offset[0]) / width_ratio),
        int((100 - bbox_offset[1]) / height_ratio),
    )
    transformed_bbox = reverse_transform_bbox(bbox, bbox_offset, 100, 100)

    assert transformed_bbox == expected_bbox


def test_reverse_transform_bbox_custom_dimensions():
    bbox = (15, 25, 35, 45)
    bbox_offset = (5, 5)
    original_width = 200
    original_height = 200
    width_ratio = (original_width - 2 * bbox_offset[0]) / original_width
    height_ratio = (original_height - 2 * bbox_offset[1]) / original_height
    expected_bbox = (
        int((15 - bbox_offset[0]) / width_ratio),
        int((25 - bbox_offset[1]) / height_ratio),
        int((35 - bbox_offset[0]) / width_ratio),
        int((45 - bbox_offset[1]) / height_ratio),
    )
    transformed_bbox = reverse_transform_bbox(bbox, bbox_offset, original_width, original_height)

    assert transformed_bbox == expected_bbox


def test_reverse_transform_bbox_zero_dimension():
    bbox = (10, 10, 20, 20)
    bbox_offset = (0, 0)
    original_width = 0
    original_height = 0
    with pytest.raises(ZeroDivisionError):
        reverse_transform_bbox(bbox, bbox_offset, original_width, original_height)
