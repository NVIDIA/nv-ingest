# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.split import SplitTask

# Initialization and Property Setting


def test_split_task_initialization():
    task = SplitTask(
        tokenizer="meta-llama/Llama-3.2-1B",
        chunk_size=1024,
        chunk_overlap=0,
        params={},
    )
    assert task._tokenizer == "meta-llama/Llama-3.2-1B"
    assert task._chunk_size == 1024
    assert task._chunk_overlap == 0
    assert task._params == {}


# String Representation Tests


def test_split_task_str_representation():
    task = SplitTask(tokenizer="intfloat/e5-large-unsupervised", chunk_size=50, chunk_overlap=5)
    expected_str = (
        "Split Task:\n" "  tokenizer: intfloat/e5-large-unsupervised\n" "  chunk_size: 50\n" "  chunk_overlap: 5\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "tokenizer, chunk_size, chunk_overlap, params",
    [
        ("intfloat/e5-large-unsupervised", 100, 10, {}),
        ("microsoft/deberta-large", 50, 5, None),
        ("meta-llama/Llama-3.2-1B", 1024, 0, {"hf_access_token": "TOKEN"}),
    ],
)
def test_split_task_to_dict(
    tokenizer,
    chunk_size,
    chunk_overlap,
    params,
):
    task = SplitTask(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        params=params,
    )

    expected_dict = {"type": "split", "task_properties": {}}

    # Only add properties to expected_dict if they are not None
    if tokenizer is not None:
        expected_dict["task_properties"]["tokenizer"] = tokenizer
    if chunk_size is not None:
        expected_dict["task_properties"]["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        expected_dict["task_properties"]["chunk_overlap"] = chunk_overlap
    if params is not None:
        expected_dict["task_properties"]["params"] = params

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_split_task_default_params():
    task = SplitTask()
    expected_str_contains = [
        "chunk_size: 1024",
        "chunk_overlap: 150",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "split",
        "task_properties": {
            "chunk_size": 1024,
            "chunk_overlap": 150,
            "params": {},
        },
    }
    assert task.to_dict() == expected_dict
