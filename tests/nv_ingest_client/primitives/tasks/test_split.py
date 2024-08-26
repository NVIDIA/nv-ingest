# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.split import SplitTask

# Initialization and Property Setting


def test_split_task_initialization():
    task = SplitTask(
        split_by="word",
        split_length=100,
        split_overlap=10,
        max_character_length=1000,
        sentence_window_size=5,
    )
    assert task._split_by == "word"
    assert task._split_length == 100
    assert task._split_overlap == 10
    assert task._max_character_length == 1000
    assert task._sentence_window_size == 5


# String Representation Tests


def test_split_task_str_representation():
    task = SplitTask(split_by="sentence", split_length=50, split_overlap=5)
    expected_str = (
        "Split Task:\n"
        "  split_by: sentence\n"
        "  split_length: 50\n"
        "  split_overlap: 5\n"
        "  split_max_character_length: None\n"
        "  split_sentence_window_size: None\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "split_by, split_length, split_overlap, max_character_length, sentence_window_size",
    [
        ("word", 100, 10, 1000, 5),
        ("sentence", 50, 5, None, None),
        ("passage", None, None, 1500, 3),
        (None, None, None, None, None),  # Test default parameters
    ],
)
def test_split_task_to_dict(
    split_by,
    split_length,
    split_overlap,
    max_character_length,
    sentence_window_size,
):
    task = SplitTask(
        split_by=split_by,
        split_length=split_length,
        split_overlap=split_overlap,
        max_character_length=max_character_length,
        sentence_window_size=sentence_window_size,
    )

    expected_dict = {"type": "split", "task_properties": {}}

    # Only add properties to expected_dict if they are not None
    if split_by is not None:
        expected_dict["task_properties"]["split_by"] = split_by
    if split_length is not None:
        expected_dict["task_properties"]["split_length"] = split_length
    if split_overlap is not None:
        expected_dict["task_properties"]["split_overlap"] = split_overlap
    if max_character_length is not None:
        expected_dict["task_properties"]["max_character_length"] = max_character_length
    if sentence_window_size is not None:
        expected_dict["task_properties"]["sentence_window_size"] = sentence_window_size

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_split_task_default_params():
    task = SplitTask()
    expected_str_contains = [
        "split_by: None",
        "split_length: None",
        "split_overlap: None",
        "split_max_character_length: None",
        "split_sentence_window_size: None",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {"type": "split", "task_properties": {}}
    assert task.to_dict() == expected_dict
