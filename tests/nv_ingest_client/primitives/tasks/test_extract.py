# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.extract import ExtractTask


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_charts, extract_infographics",
    [
        ("pdf", "tika", True, False, True, True, True),
        (None, "pdfium", False, True, None, False, False),
        ("txt", None, None, None, False, False, False),
    ],
)
def test_extract_task_str_representation(
    document_type,
    extract_method,
    extract_text,
    extract_images,
    extract_tables,
    extract_charts,
    extract_infographics,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
    )

    task_str = str(task)

    expected_parts = [
        "Extract Task:",
        f"document type: {document_type}",
        f"extract method: {extract_method}",
        f"extract text: {extract_text}",
        f"extract images: {extract_images}",
        f"extract tables: {extract_tables}",
        f"extract charts: {extract_charts}",  # If extract_charts is not specified,
        # it defaults to the same value as extract_tables.
        f"extract infographics: {extract_infographics}",  # If extract_infographics is not specified,
        # it defaults to the same value as extract_tables.
        "text depth: document",  # Assuming this is a fixed value for all instances
    ]

    for part in expected_parts:
        assert part in task_str, f"Expected part '{part}' not found in task string representation"


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_charts, extract_infographics",
    [
        ("pdf", "tika", True, False, True, False, False),
        (None, "pdfium", False, True, None, False, False),
        ("txt", None, None, None, False, False, False),
    ],
)
def test_extract_task_str_representation_extract_charts_false(
    document_type,
    extract_method,
    extract_text,
    extract_images,
    extract_tables,
    extract_charts,
    extract_infographics,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
    )

    task_str = str(task)

    expected_parts = [
        "Extract Task:",
        f"document type: {document_type}",
        f"extract method: {extract_method}",
        f"extract text: {extract_text}",
        f"extract images: {extract_images}",
        f"extract tables: {extract_tables}",
        f"extract charts: {extract_charts}",
        f"extract infographics: {extract_infographics}",
        "text depth: document",  # Assuming this is a fixed value for all instances
    ]

    for part in expected_parts:
        assert part in task_str, f"Expected part '{part}' not found in task string representation"


# Initialization and Property Setting


@pytest.mark.parametrize(
    "extract_method, extract_text, extract_images, extract_tables",
    [
        ("pdfium", True, False, False),
        ("haystack", False, True, True),
        ("unstructured_local", True, True, True),
    ],
)
def test_extract_task_initialization(extract_method, extract_text, extract_images, extract_tables):
    task = ExtractTask(
        document_type="pdf",
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
    )
    assert task._document_type == "pdf"
    assert task._extract_method == extract_method
    assert task._extract_text == extract_text
    assert task._extract_images == extract_images
    assert task._extract_tables == extract_tables


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables,"
    "extract_images_method, extract_tables_method, paddle_output_format",
    [
        ("pdf", "tika", True, False, False, "merged", "yolox", "pseudo_markdown"),
        ("docx", "haystack", False, True, True, "simple", "yolox", "simple"),
        ("txt", "llama_parse", True, True, False, "simple", "yolox", "simple"),
    ],
)
def test_extract_task_to_dict_basic(
    document_type,
    extract_method,
    extract_text,
    extract_images,
    extract_tables,
    extract_images_method,
    extract_tables_method,
    paddle_output_format,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_images_method=extract_images_method,
        extract_tables=extract_tables,
        extract_tables_method=extract_tables_method,
        paddle_output_format=paddle_output_format,
    )
    expected_dict = {
        "type": "extract",
        "task_properties": {
            "method": extract_method,
            "document_type": document_type,
            "params": {
                "extract_text": extract_text,
                "extract_images": extract_images,
                "extract_tables": extract_tables,
                "extract_images_method": extract_images_method,
                "extract_tables_method": extract_tables_method,
                "extract_charts": extract_tables,  # If extract_charts is not specified,
                # it defaults to the same value as extract_tables.
                "extract_infographics": False,  # extract_infographics is False by default
                "text_depth": "document",
                "paddle_output_format": paddle_output_format,
            },
        },
    }

    assert task.to_dict() == expected_dict, "ExtractTask.to_dict() did not return the expected dictionary"


@pytest.mark.parametrize(
    (
        "document_type, extract_method, extract_text, extract_images, extract_tables,"
        "extract_images_method, extract_tables_method, extract_charts, paddle_output_format"
    ),
    [
        ("pdf", "tika", True, False, False, "merged", "yolox", False, "pseudo_markdown"),
        ("docx", "haystack", False, True, True, "simple", "yolox", False, "simple"),
        ("txt", "llama_parse", True, True, False, "simple", "yolox", False, "simple"),
    ],
)
def test_extract_task_to_dict_extract_charts_false(
    document_type,
    extract_method,
    extract_text,
    extract_images,
    extract_tables,
    extract_images_method,
    extract_tables_method,
    extract_charts,
    paddle_output_format,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_images_method=extract_images_method,
        extract_tables=extract_tables,
        extract_tables_method=extract_tables_method,
        extract_charts=extract_charts,
        paddle_output_format=paddle_output_format,
    )
    expected_dict = {
        "type": "extract",
        "task_properties": {
            "method": extract_method,
            "document_type": document_type,
            "params": {
                "extract_text": extract_text,
                "extract_images": extract_images,
                "extract_tables": extract_tables,
                "extract_images_method": extract_images_method,
                "extract_tables_method": extract_tables_method,
                "extract_charts": extract_charts,
                "extract_infographics": False,
                "text_depth": "document",
                "paddle_output_format": paddle_output_format,
            },
        },
    }

    assert task.to_dict() == expected_dict, "ExtractTask.to_dict() did not return the expected dictionary"


# Method-Specific Properties Test


@pytest.mark.parametrize(
    "extract_method, has_method_specific",
    [
        ("unstructured_local", True),
        ("pdfium", False),
        ("tika", False),
    ],
)
def test_extract_task_to_dict_method_specific(extract_method, has_method_specific):
    task = ExtractTask(extract_method=extract_method, document_type="pdf", extract_text=True)
    task_desc = task.to_dict()
    params = task_desc["task_properties"]["params"]

    if has_method_specific:
        assert "api_key" in params, f"api_key should be in params for {extract_method}"
        assert "unstructured_url" in params, f"unstructured_url should be in params for {extract_method}"
    else:
        assert "api_key" not in params, f"api_key should not be in params for {extract_method}"
        assert "unstructured_url" not in params, f"unstructured_url should not be in params for {extract_method}"
