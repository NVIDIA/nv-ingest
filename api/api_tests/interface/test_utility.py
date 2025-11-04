# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from datetime import datetime
from typing import List

import pandas as pd
import pytest

from .. import get_project_root, find_root_by_pattern
from nv_ingest_api.interface.utility import (
    read_file_as_base64,
    create_source_metadata,
    build_dataframe_from_files,
    DOCUMENT_TO_CONTENT_MAPPING,
)
from nv_ingest_api.internal.enums.common import DocumentTypeEnum, ContentTypeEnum


# ------------------------------------------------------------------------------
# Test create_source_metadata
# ------------------------------------------------------------------------------
def test_create_source_metadata():
    source_name = "test_file.pdf"
    source_id = "test_file.pdf"
    document_type = DocumentTypeEnum.PDF
    meta = create_source_metadata(source_name, source_id, document_type)
    expected_keys = {
        "source_name",
        "source_id",
        "source_location",
        "source_type",
        "collection_id",
        "date_created",
        "last_modified",
        "summary",
        "partition_id",
        "access_level",
    }
    assert expected_keys.issubset(meta.keys())
    # Check that date_created and last_modified are ISO 8601 formatted.
    try:
        datetime.fromisoformat(meta["date_created"])
        datetime.fromisoformat(meta["last_modified"])
    except ValueError:
        pytest.fail("date_created or last_modified is not in ISO8601 format.")


# ------------------------------------------------------------------------------
# Test create_content_metadata for various document types.
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "file_key,rel_path",
    [
        ("png", "./data/multimodal_test.png"),
        ("docx", "./data/multimodal_test.docx"),
        ("pptx", "./data/multimodal_test.pptx"),
        ("jpeg", "./data/multimodal_test.jpeg"),
        ("pdf", "./data/multimodal_test.pdf"),
        ("tiff", "./data/multimodal_test.tiff"),
        ("bmp", "./data/multimodal_test.bmp"),
    ],
)
def test_read_file_as_base64(file_key, rel_path):
    # Try to locate the git repository root based on the current file.
    project_root = get_project_root(__file__)
    if project_root:
        full_path = os.path.join(project_root, rel_path.lstrip("./"))
    else:
        # Fallback: use find_root_by_pattern to locate a directory containing the file.
        found_root = find_root_by_pattern(rel_path)
        if found_root:
            full_path = os.path.join(found_root, rel_path.lstrip("./"))
        else:
            pytest.skip(f"Could not locate a directory containing {rel_path}.")

    if not os.path.exists(full_path):
        pytest.skip(f"{full_path} not found for {file_key} test.")

    b64_str = read_file_as_base64(full_path)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0
    # Verify it decodes properly.
    decoded = base64.b64decode(b64_str)
    assert len(decoded) > 0


# ------------------------------------------------------------------------------
# Test build_dataframe_from_files
# ------------------------------------------------------------------------------
@pytest.mark.integration
def test_build_dataframe_from_files():
    # Supported files (skip those marked as "skip")
    supported_files = [
        (DocumentTypeEnum.PNG, "./data/multimodal_test.png"),
        (DocumentTypeEnum.WAV, "./data/harvard.wav"),
        (DocumentTypeEnum.DOCX, "./data/multimodal_test.docx"),
        (DocumentTypeEnum.PPTX, "./data/multimodal_test.pptx"),
        (DocumentTypeEnum.JPEG, "./data/multimodal_test.jpeg"),
        (DocumentTypeEnum.PDF, "./data/multimodal_test.pdf"),
        (DocumentTypeEnum.TIFF, "./data/multimodal_test.tiff"),
        (DocumentTypeEnum.BMP, "./data/multimodal_test.bmp"),
    ]
    file_paths: List[str] = []
    source_names: List[str] = []
    source_ids: List[str] = []
    document_types: List[str] = []

    # Try to determine the project root.
    project_root = get_project_root(__file__)

    for doc_type, rel_path in supported_files:
        # Determine the full file path based on the project root or fallback.
        if project_root:
            full_path = os.path.join(project_root, rel_path.lstrip("./"))
        else:
            found_root = find_root_by_pattern(rel_path)
            if found_root:
                full_path = os.path.join(found_root, rel_path.lstrip("./"))
            else:
                pytest.skip(f"Could not locate a directory containing {rel_path}.")

        if not os.path.exists(full_path):
            pytest.skip(f"{full_path} not found for build_dataframe_from_files test.")

        file_paths.append(full_path)
        source_names.append(full_path)
        source_ids.append(full_path)
        document_types.append(doc_type)

    df = build_dataframe_from_files(file_paths, source_names, source_ids, document_types)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(supported_files)
    for idx, row in df.iterrows():
        meta = row.get("metadata", {})
        for key in ["content", "source_metadata", "content_metadata"]:
            assert key in meta, f"Row {idx} missing key {key} in metadata."
        doc_type = row["document_type"]
        expected_type = DOCUMENT_TO_CONTENT_MAPPING.get(doc_type, ContentTypeEnum.UNKNOWN)
        actual_type = meta["content_metadata"].get("type")
        assert actual_type == expected_type, f"Row {idx} expected type {expected_type} but got {actual_type}."


# ------------------------------------------------------------------------------
# Test get_project_root
# ------------------------------------------------------------------------------
def test_get_project_root():
    project_root = get_project_root(__file__)
    if project_root is None:
        pytest.skip("Not in a project repository; get_project_root returned None.")
    assert isinstance(project_root, str)
    # Check that the .git directory exists at the root.


# ------------------------------------------------------------------------------
# Test find_root_by_pattern
# ------------------------------------------------------------------------------
def test_find_root_by_pattern():
    # We'll use a pattern we expect exists: "data/multimodal_test.pdf"
    pattern = os.path.join("data", "multimodal_test.pdf")
    project_root = get_project_root(__file__)
    start_dir = project_root if project_root is not None else os.getcwd()
    found_root = find_root_by_pattern(pattern, start_dir=start_dir)
    assert found_root is not None, "find_root_by_pattern returned None."
    candidate = os.path.join(found_root, pattern)
    assert os.path.exists(candidate), f"Candidate file {candidate} does not exist."
