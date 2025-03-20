# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
from typing import Dict, Union

from nv_ingest_api.internal.mutate.filter import (
    filter_images_internal,
)


# Dummy enumerations and helper functions for testing.
class DummyContentTypeEnum:
    IMAGE = "IMAGE"
    INFO_MSG = "INFO_MSG"


class DummyTaskTypeEnum:
    FILTER = type("DummyEnum", (), {"value": "FILTER"})


class DummyStatusEnum:
    SUCCESS = type("DummyEnum", (), {"value": "SUCCESS"})


class DummySchema:
    def model_dump(self):
        return {"dummy": "info"}


def dummy_validate_schema(info_msg: dict, schema) -> DummySchema:
    return DummySchema()


# Patch dependencies in the module under test.
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    import nv_ingest_api.internal.mutate.filter as module_under_test

    monkeypatch.setattr(module_under_test, "ContentTypeEnum", DummyContentTypeEnum)
    monkeypatch.setattr(module_under_test, "TaskTypeEnum", DummyTaskTypeEnum)
    monkeypatch.setattr(module_under_test, "StatusEnum", DummyStatusEnum)
    monkeypatch.setattr(module_under_test, "validate_schema", dummy_validate_schema)
    # Dummy value for InfoMessageMetadataSchema; its value is not used by dummy_validate_schema.
    monkeypatch.setattr(module_under_test, "InfoMessageMetadataSchema", object)


# Test when required columns are missing.
def test_missing_required_columns():
    df = pd.DataFrame({"wrong_column": ["IMAGE"], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]})
    task_params: Dict[str, Union[int, float, bool]] = {
        "min_size": 100,
        "max_aspect_ratio": 4,
        "min_aspect_ratio": 1,
        "filter": False,
    }
    with pytest.raises(ValueError):
        filter_images_internal(df, task_params)


# Test invalid min_size (negative value).
def test_invalid_min_size():
    df = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]}
    )
    task_params = {"min_size": -1, "max_aspect_ratio": 4, "min_aspect_ratio": 1, "filter": False}
    with pytest.raises(ValueError):
        filter_images_internal(df, task_params)


# Test invalid max_aspect_ratio (non-positive).
def test_invalid_max_aspect_ratio():
    df = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]}
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 0, "min_aspect_ratio": 1, "filter": False}
    with pytest.raises(ValueError):
        filter_images_internal(df, task_params)


# Test invalid min_aspect_ratio (non-positive).
def test_invalid_min_aspect_ratio():
    df = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]}
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 4, "min_aspect_ratio": 0, "filter": False}
    with pytest.raises(ValueError):
        filter_images_internal(df, task_params)


# Test min_aspect_ratio greater than max_aspect_ratio.
def test_min_aspect_ratio_greater_than_max():
    df = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]}
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 2, "min_aspect_ratio": 3, "filter": False}
    with pytest.raises(ValueError):
        filter_images_internal(df, task_params)


# Test DataFrame with no image rows.
def test_no_image_rows():
    df = pd.DataFrame(
        {
            "document_type": ["TEXT", "TEXT"],
            "metadata": [
                {"image_metadata": {"width": 200, "height": 100}},
                {"image_metadata": {"width": 150, "height": 150}},
            ],
        }
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 4, "min_aspect_ratio": 1, "filter": False}
    result = filter_images_internal(df, task_params)
    pd.testing.assert_frame_equal(result, df)


# Test image row that meets criteria (should remain unchanged).
def test_image_row_valid():
    df = pd.DataFrame(
        {"document_type": [DummyContentTypeEnum.IMAGE], "metadata": [{"image_metadata": {"width": 200, "height": 100}}]}
    )
    # Average = (200+100)/2 = 150 > 100; aspect ratio = 200/100 = 2, which is >1 and <4.
    task_params = {"min_size": 100, "max_aspect_ratio": 4, "min_aspect_ratio": 1, "filter": False}
    result = filter_images_internal(df.copy(), task_params)
    pd.testing.assert_frame_equal(result, df)


# Test image row failing criteria with filter flag True (row should be dropped).
def test_image_row_fail_filter_true():
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE, DummyContentTypeEnum.IMAGE],
            "metadata": [
                {"image_metadata": {"width": 50, "height": 50}},  # Average = 50, fails min_size=100.
                {"image_metadata": {"width": 200, "height": 100}},  # Valid row.
            ],
        }
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 4, "min_aspect_ratio": 1, "filter": True}
    result = filter_images_internal(df.copy(), task_params)
    expected = df.iloc[[1]].copy()
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


# Test image row failing criteria with filter flag False (row should be flagged).
def test_image_row_fail_filter_false():
    df = pd.DataFrame(
        {
            "document_type": [DummyContentTypeEnum.IMAGE],
            "metadata": [{"image_metadata": {"width": 50, "height": 50}}],  # Fails average size.
        }
    )
    task_params = {"min_size": 100, "max_aspect_ratio": 4, "min_aspect_ratio": 1, "filter": False}
    result = filter_images_internal(df.copy(), task_params)
    # The row should be flagged: document_type becomes INFO_MSG.
    assert result.loc[0, "document_type"] == DummyContentTypeEnum.INFO_MSG
    metadata = result.loc[0, "metadata"]
    # The _add_info_message function adds/overwrites the "info_message_metadata" key.
    expected_info_msg = {
        "task": DummyTaskTypeEnum.FILTER.value,
        "status": DummyStatusEnum.SUCCESS.value,
        "message": "Filtered due to image size or aspect ratio.",
        "filter": True,
    }
    assert metadata.get("info_message_metadata") == expected_info_msg
