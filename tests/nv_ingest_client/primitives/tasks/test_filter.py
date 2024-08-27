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
        (None, 128, 5.0, 0.2, True),
        ("image", None, 5.0, 0.2, True),
        ("image", 128, None, 0.2, True),
        ("image", 128, 5.0, None, True),
        ("image", 128, 5.0, 0.2, None),
    ],
)
def test_filter_task_to_dict(
    content_type,
    min_size,
    max_aspect_ratio,
    min_aspect_ratio,
    filter,
):
    kwargs = {}
    if content_type is not None:
        kwargs["content_type"] = content_type
    if min_size is not None:
        kwargs["min_size"] = min_size
    if max_aspect_ratio is not None:
        kwargs["max_aspect_ratio"] = max_aspect_ratio
    if min_aspect_ratio is not None:
        kwargs["min_aspect_ratio"] = min_aspect_ratio
    if filter is not None:
        kwargs["filter"] = filter

    task = FilterTask(**kwargs)

    expected_dict = {
        "type": "filter",
        "task_properties": {
            "content_type": "image",
            "params": {"min_size": 128, "max_aspect_ratio": 5.0, "min_aspect_ratio": 0.2, "filter": False},
        },
    }

    # Only add properties to expected_dict if they are not None
    if content_type is not None:
        expected_dict["task_properties"]["content_type"] = content_type
    if min_size is not None:
        expected_dict["task_properties"]["params"]["min_size"] = min_size
    if max_aspect_ratio is not None:
        expected_dict["task_properties"]["params"]["max_aspect_ratio"] = max_aspect_ratio
    if min_aspect_ratio is not None:
        expected_dict["task_properties"]["params"]["min_aspect_ratio"] = min_aspect_ratio
    if filter is not None:
        expected_dict["task_properties"]["params"]["filter"] = filter

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
