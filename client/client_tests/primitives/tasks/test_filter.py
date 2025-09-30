# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.filter import FilterTask

# Initialization and Property Setting


def test_filter_task_initialization():
    task = FilterTask(
        content_type="image",
        min_size=128,
        max_aspect_ratio=5.0,
        min_aspect_ratio=0.2,
        filter=True,
    )
    assert task._content_type == "image"
    assert task._min_size == 128
    assert task._max_aspect_ratio == 5.0
    assert task._min_aspect_ratio == 0.2
    assert task._filter is True


# String Representation Tests


def test_filter_task_str_representation():
    task = FilterTask(
        content_type="image",
        min_size=128,
        max_aspect_ratio=5.0,
        min_aspect_ratio=0.2,
        filter=True,
    )
    expected_str = (
        "Filter Task:\n"
        "  content_type: image\n"
        "  min_size: 128\n"
        "  max_aspect_ratio: 5.0\n"
        "  min_aspect_ratio: 0.2\n"
        "  filter: True\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "content_type, min_size, max_aspect_ratio, min_aspect_ratio, filter",
    [
        ("image", 128, 5.0, 0.2, True),
        ("image", 256, 3.0, 0.5, False),
        ("image", 64, 10.0, 0.1, True),
    ],
)
def test_filter_task_to_dict(
    content_type,
    min_size,
    max_aspect_ratio,
    min_aspect_ratio,
    filter,
):
    task = FilterTask(
        content_type=content_type,
        min_size=min_size,
        max_aspect_ratio=max_aspect_ratio,
        min_aspect_ratio=min_aspect_ratio,
        filter=filter,
    )

    expected_dict = {
        "type": "filter",
        "task_properties": {
            "content_type": content_type,
            "params": {
                "min_size": min_size,
                "max_aspect_ratio": max_aspect_ratio,
                "min_aspect_ratio": min_aspect_ratio,
                "filter": filter,
            },
        },
    }

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_filter_task_default_params():
    task = FilterTask()
    expected_str_contains = [
        "content_type: image",
        "min_size: 128",
        "max_aspect_ratio: 5.0",
        "min_aspect_ratio: 0.2",
        "filter: False",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "filter",
        "task_properties": {
            "content_type": "image",
            "params": {"min_size": 128, "max_aspect_ratio": 5.0, "min_aspect_ratio": 0.2, "filter": False},
        },
    }
    assert task.to_dict() == expected_dict


# Schema Consolidation Tests


def test_filter_task_schema_consolidation():
    """Test that FilterTask uses API schema for validation."""
    # Test that invalid content_type raises validation error
    with pytest.raises(ValueError):
        FilterTask(content_type="invalid_type")

    # Test that valid parameters work
    task = FilterTask(
        content_type="image",
        min_size=200,
        max_aspect_ratio=8.0,
        min_aspect_ratio=0.1,
        filter=True,
    )
    assert task._content_type == "image"
    assert task._min_size == 200
    assert task._max_aspect_ratio == 8.0
    assert task._min_aspect_ratio == 0.1
    assert task._filter is True


def test_filter_task_edge_cases():
    """Test FilterTask edge cases and boundary conditions."""
    # Test with minimum valid values
    task = FilterTask(
        content_type="image",
        min_size=1,
        max_aspect_ratio=0.1,
        min_aspect_ratio=0.01,
        filter=False,
    )
    assert task._min_size == 1
    assert task._max_aspect_ratio == 0.1
    assert task._min_aspect_ratio == 0.01
    assert task._filter is False

    # Test serialization of edge case values
    result_dict = task.to_dict()
    assert result_dict["task_properties"]["params"]["min_size"] == 1
    assert result_dict["task_properties"]["params"]["max_aspect_ratio"] == 0.1
    assert result_dict["task_properties"]["params"]["min_aspect_ratio"] == 0.01
    assert result_dict["task_properties"]["params"]["filter"] is False
