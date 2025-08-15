# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.dedup import DedupTask
from nv_ingest_api.internal.enums.common import ContentTypeEnum


def test_dedup_task_initialization():
    task = DedupTask(
        content_type="image",
        filter=True,
    )
    assert task._content_type == ContentTypeEnum.IMAGE
    assert task._filter is True


def test_dedup_task_default_initialization():
    """Test DedupTask with default parameters."""
    task = DedupTask()
    assert task._content_type == ContentTypeEnum.IMAGE
    assert task._filter is False


# String Representation Tests


def test_dedup_task_str_representation():
    task = DedupTask(content_type="image", filter=True)
    expected_str = "Dedup Task:\n" "  content_type: image\n" "  filter: True\n"
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "content_type, filter_val, expected_content_type",
    [
        ("image", True, "image"),
        ("image", False, "image"),
        (None, True, "image"),  # Test default content_type
        (None, False, "image"),  # Test default content_type
    ],
)
def test_dedup_task_to_dict(content_type, filter_val, expected_content_type):
    kwargs = {}
    if content_type is not None:
        kwargs["content_type"] = content_type
    kwargs["filter"] = filter_val

    task = DedupTask(**kwargs)
    task_dict = task.to_dict()

    expected_dict = {
        "type": "dedup",
        "task_properties": {
            "content_type": expected_content_type,
            "params": {"filter": filter_val},
        },
    }

    assert task_dict == expected_dict


# Default Parameter Handling


def test_dedup_task_default_params():
    """Test DedupTask with all default parameters."""
    task = DedupTask()

    assert task._content_type == ContentTypeEnum.IMAGE
    assert task._filter is False

    task_dict = task.to_dict()
    expected_dict = {
        "type": "dedup",
        "task_properties": {
            "content_type": "image",
            "params": {"filter": False},
        },
    }
    assert task_dict == expected_dict


# Schema Consolidation Tests


def test_dedup_task_schema_consolidation():
    """Test that DedupTask validates against API schema constraints."""
    # Test that API schema validation is enforced
    task = DedupTask(content_type="image", filter=True)

    # Verify the task was created with proper validation
    assert task._content_type == ContentTypeEnum.IMAGE
    assert task._filter is True

    # Test serialization matches API schema expectations
    task_dict = task.to_dict()
    assert task_dict["type"] == "dedup"
    assert task_dict["task_properties"]["content_type"] == "image"
    assert task_dict["task_properties"]["params"]["filter"] is True


def test_dedup_task_invalid_content_type():
    """Test that DedupTask rejects invalid content_type values."""
    with pytest.raises(ValueError):
        DedupTask(content_type="invalid_type")
