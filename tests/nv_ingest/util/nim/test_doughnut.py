# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest.util.nim.doughnut import extract_classes_bboxes
from nv_ingest.util.nim.doughnut import postprocess_text
from nv_ingest.util.nim.doughnut import reverse_transform_bbox
from nv_ingest.util.nim.doughnut import strip_markdown_formatting


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


def test_postprocess_text_with_unaccepted_class():
    # Input text that should not be processed
    txt = "This text should not be processed"
    cls = "InvalidClass"  # Not in ACCEPTED_CLASSES

    result = postprocess_text(txt, cls)

    assert result == ""


def test_postprocess_text_removes_tbc_and_processes_text():
    # Input text containing "<tbc>"
    txt = "<tbc>Some text"
    cls = "Title"  # An accepted class

    expected_output = "Some text"

    result = postprocess_text(txt, cls)

    assert result == expected_output


def test_postprocess_text_no_tbc_but_accepted_class():
    # Input text without "<tbc>"
    txt = "This is a test **without** tbc"
    cls = "Section-header"  # An accepted class

    expected_output = "This is a test without tbc"

    result = postprocess_text(txt, cls)

    assert result == expected_output


@pytest.mark.parametrize(
    "input_text, expected_classes, expected_bboxes, expected_texts",
    [
        ("<x_10><y_20>Sample text<x_30><y_40><class_Text>", ["Text"], [(10, 20, 30, 40)], ["Sample text"]),
        (
            "Invalid text <x_10><y_20><x_30><y_40><class_Invalid>",
            ["Bad-box", "Bad-box"],
            [(0, 0, 0, 0), (10, 20, 30, 40)],
            ["Invalid text", ""],
        ),
        ("<x_15><y_25>Header content<x_35><y_45><class_Title>", ["Title"], [(15, 25, 35, 45)], ["Header content"]),
        ("<x_5><y_10>Overlapping box<x_5><y_10><class_Text>", ["Bad-box"], [(5, 10, 5, 10)], ["Overlapping box"]),
    ],
)
def test_extract_classes_bboxes(input_text, expected_classes, expected_bboxes, expected_texts):
    classes, bboxes, texts = extract_classes_bboxes(input_text)
    assert classes == expected_classes
    assert bboxes == expected_bboxes
    assert texts == expected_texts


# Test cases for strip_markdown_formatting
@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("# Header\n**Bold text**\n*Italic*", "Header\nBold text\nItalic"),
        ("~~Strikethrough~~", "Strikethrough"),
        ("[Link](http://example.com)", "Link"),
        ("`inline code`", "inline code"),
        ("> Blockquote", "Blockquote"),
        ("Normal text\n\n\nMultiple newlines", "Normal text\n\nMultiple newlines"),
        ("Dot sequence...... more text", "Dot sequence..... more text"),
    ],
)
def test_strip_markdown_formatting(input_text, expected_output):
    assert strip_markdown_formatting(input_text) == expected_output
