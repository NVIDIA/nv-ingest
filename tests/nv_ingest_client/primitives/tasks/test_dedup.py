# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.dedup import DedupTask

# Initialization and Property Setting


def test_dedup_task_initialization():
    task = DedupTask(
        content_type="image",
        filter=True,
    )
    assert task._content_type == "image"
    assert task._filter is True


# String Representation Tests


def test_dedup_task_str_representation():
    task = DedupTask(content_type="image", filter=True)
    expected_str = "Dedup Task:\n" "  content_type: image\n" "  filter: True\n"
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "content_type, filter",
    [
        ("image", True),
        ("image", False),
        (None, True),  # Test default parameters
        (None, None),  # Test default parameters
    ],
)
def test_dedup_task_to_dict(
    content_type,
    filter,
):
    kwargs = {}
    if content_type is not None:
        kwargs["content_type"] = content_type
    if filter is not None:
        kwargs["filter"] = filter

    task = DedupTask(**kwargs)

    expected_dict = {
        "type": "dedup",
        "task_properties": {
            "content_type": "image",
            "params": {"filter": False},
        },
    }

    # Only add properties to expected_dict if they are not None
    if content_type is not None:
        expected_dict["task_properties"]["content_type"] = content_type
    if filter is not None:
        expected_dict["task_properties"]["params"]["filter"] = filter

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_dedup_task_default_params():
    task = DedupTask()
    expected_str_contains = [
        "content_type: image",
        "filter: False",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "dedup",
        "task_properties": {
            "content_type": "image",
            "params": {"filter": False},
        },
    }

    assert task.to_dict() == expected_dict
