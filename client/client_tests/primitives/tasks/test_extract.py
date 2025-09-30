# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# noqa
# flake8: noqa

import pytest
from nv_ingest_api.internal.enums.common import DocumentTypeEnum
from nv_ingest_client.primitives.tasks.extract import ExtractTask


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_charts, extract_infographics",
    [
        (DocumentTypeEnum.PDF, "pdfium", True, False, True, True, True),
        (DocumentTypeEnum.PDF, "pdfium", False, True, None, False, False),
        (DocumentTypeEnum.TXT, None, None, None, False, False, False),
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
        f"document_type: {task.document_type}",
        f"extract_method: {task._extract_method}",
        f"extract_text: {extract_text}",
        f"extract_images: {extract_images}",
        f"extract_tables: {extract_tables}",
        f"extract_charts: {task._extract_charts}",
        f"extract_infographics: {extract_infographics}",
        "text_depth: document",
    ]

    for part in expected_parts:
        assert part in task_str


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_charts, extract_infographics",
    [
        (DocumentTypeEnum.PDF, "pdfium", True, False, True, False, True),
        (DocumentTypeEnum.PDF, "pdfium", False, True, True, False, False),
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
        f"document_type: {task.document_type}",
        f"extract_method: {task._extract_method}",
        f"extract_text: {extract_text}",
        f"extract_images: {extract_images}",
        f"extract_tables: {extract_tables}",
        f"extract_charts: {extract_charts}",
        f"extract_infographics: {extract_infographics}",
        "text_depth: document",
    ]

    for part in expected_parts:
        assert part in task_str


# Initialization and Property Setting


@pytest.mark.parametrize(
    "extract_method, extract_text, extract_images, extract_tables",
    [
        ("pdfium", True, False, True),
        ("pdfium", False, True, False),
        ("pdfium", True, True, True),
    ],
)
def test_extract_task_initialization(extract_method, extract_text, extract_images, extract_tables):
    task = ExtractTask(
        document_type=DocumentTypeEnum.PDF,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
    )

    assert task.document_type == "pdf"
    assert task._extract_method == extract_method
    assert task._extract_text == extract_text
    assert task._extract_images == extract_images
    assert task._extract_tables == extract_tables


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_images_method, "
    "extract_tables_method, table_output_format",
    [
        (DocumentTypeEnum.PDF, "pdfium", True, False, True, "group", "yolox", "pseudo_markdown"),
        (DocumentTypeEnum.PDF, "pdfium", False, True, False, "group", "yolox", "pseudo_markdown"),
        (DocumentTypeEnum.PDF, "pdfium", True, True, True, "group", "yolox", "pseudo_markdown"),
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
    table_output_format,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_images_method=extract_images_method,
        extract_tables_method=extract_tables_method,
        table_output_format=table_output_format,
    )

    task_dict = task.to_dict()

    assert task_dict["type"] == "extract"
    assert task_dict["task_properties"]["document_type"] == document_type.value
    assert task_dict["task_properties"]["method"] == extract_method

    params = task_dict["task_properties"]["params"]
    assert params["extract_text"] == extract_text
    assert params["extract_images"] == extract_images
    assert params["extract_tables"] == extract_tables
    assert params["extract_images_method"] == extract_images_method
    assert params["extract_tables_method"] == extract_tables_method
    assert params["table_output_format"] == table_output_format


@pytest.mark.parametrize(
    "document_type, extract_method, extract_text, extract_images, extract_tables, extract_images_method,"
    " extract_tables_method, extract_charts, table_output_format",
    [
        (DocumentTypeEnum.PDF, "pdfium", True, False, True, "group", "yolox", False, "pseudo_markdown"),
        (DocumentTypeEnum.PDF, "pdfium", False, True, True, "group", "yolox", False, "pseudo_markdown"),
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
    table_output_format,
):
    task = ExtractTask(
        document_type=document_type,
        extract_method=extract_method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_images_method=extract_images_method,
        extract_tables_method=extract_tables_method,
        extract_charts=extract_charts,
        table_output_format=table_output_format,
    )

    task_dict = task.to_dict()

    assert task_dict["type"] == "extract"
    assert task_dict["task_properties"]["document_type"] == document_type.value
    assert task_dict["task_properties"]["method"] == extract_method

    params = task_dict["task_properties"]["params"]
    assert params["extract_text"] == extract_text
    assert params["extract_images"] == extract_images
    assert params["extract_tables"] == extract_tables
    assert params["extract_charts"] == extract_charts
    assert params["extract_images_method"] == extract_images_method
    assert params["extract_tables_method"] == extract_tables_method
    assert params["table_output_format"] == table_output_format


# Method-Specific Properties Test


@pytest.mark.parametrize(
    "extract_method, has_method_specific",
    [
        ("pdfium", False),
        ("unstructured_io", True),
        ("unstructured_local", True),
        ("adobe", True),
    ],
)
def test_extract_task_to_dict_method_specific(extract_method, has_method_specific):
    task = ExtractTask(document_type=DocumentTypeEnum.PDF, extract_method=extract_method)
    task_dict = task.to_dict()

    if has_method_specific:
        # Check that method-specific properties are added to params
        params = task_dict["task_properties"]["params"]
        if extract_method == "unstructured_io":
            assert "unstructured_api_key" in params
            assert "unstructured_url" in params
        elif extract_method == "unstructured_local":
            assert "api_key" in params
            assert "unstructured_url" in params
        elif extract_method == "adobe":
            assert "adobe_client_id" in params
            assert "adobe_client_secrect" in params


def test_extract_task_api_schema_validation():
    """Test that ExtractTask uses API schema for validation"""
    # Valid task should work
    task = ExtractTask(
        document_type=DocumentTypeEnum.PDF,
        extract_method="pdfium",
        extract_text=True,
        extract_tables=True,
    )
    assert task.document_type == "pdf"
    assert task._extract_method == "pdfium"

    # Invalid document type should raise error
    with pytest.raises(ValueError, match="is not a valid DocumentTypeEnum value"):
        ExtractTask(
            document_type="invalid_type",
            extract_method="pdfium",
        )


def test_extract_task_default_method_selection():
    """Test that ExtractTask selects appropriate default methods"""
    # PDF should default to pdfium
    task = ExtractTask(document_type=DocumentTypeEnum.PDF)
    assert task._extract_method == "pdfium"

    # Text should default to txt
    task = ExtractTask(document_type=DocumentTypeEnum.TXT)
    assert task._extract_method == "txt"

    # Image should default to image
    task = ExtractTask(document_type=DocumentTypeEnum.JPEG)
    assert task._extract_method == "image"


def test_extract_task_charts_default_behavior():
    """Test that extract_charts defaults to extract_tables value"""
    # When extract_charts is None, it should default to extract_tables value
    task = ExtractTask(
        document_type=DocumentTypeEnum.PDF,
        extract_tables=True,
        extract_charts=None,
    )
    assert task._extract_charts == True

    task = ExtractTask(
        document_type=DocumentTypeEnum.PDF,
        extract_tables=False,
        extract_charts=None,
    )
    assert task._extract_charts == False

    # When extract_charts is explicitly set, it should use that value
    task = ExtractTask(
        document_type=DocumentTypeEnum.PDF,
        extract_tables=True,
        extract_charts=False,
    )
    assert task._extract_charts == False
