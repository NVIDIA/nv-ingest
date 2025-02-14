# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.embed import EmbedTask


# Initialization and Property Setting


def test_embed_task_initialization():
    task = EmbedTask(
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        endpoint_url="http://embedding:8000/v1",
    )
    assert task._model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert task._endpoint_url == "http://embedding:8000/v1"

# String Representation Tests


def test_embed_task_str_representation():
    task = EmbedTask(model_name="nvidia/nv-embedqa-e5-v5", endpoint_url="http://localhost:8024/v1", filter_errors=True)
    expected_str = (
        "Embed Task:\n"
        "  model_name: nvidia/nv-embedqa-e5-v5\n"
        "  endpoint_url: http://localhost:8024/v1\n"
        "  filter_errors: True\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "model_name, endpoint_url, filter_errors",
    [
        ("meta-llama/Llama-3.2-1B", "http://embedding:8012/v1", False),
        ("nvidia/nv-embedqa-mistral-7b-v2", "http://localhost:8000/v1", True),
        ("nvidia/nv-embedqa-e5-v5", "http://embedding:8000/v1", True),
        (None, None, False),  # Test default parameters
    ],
)
def test_embed_task_to_dict(
    model_name,
    endpoint_url,
    filter_errors,
):
    task = EmbedTask(
        model_name=model_name,
        endpoint_url=endpoint_url,
        filter_errors=filter_errors,
    )

    expected_dict = {"type": "embed", "task_properties": {"model_name": model_name, "endpoint_url": endpoint_url, "filter_errors": filter_errors}}
    print(expected_dict)
    print(task.to_dict())

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_embed_task_default_params():
    task = EmbedTask()
    expected_str_contains = [
        "model_name: None",
        "endpoint_url: None",
        "filter_errors: False"
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {"type": "embed", "task_properties": {"model_name": None, "endpoint_url": None, "filter_errors": False}}
    assert task.to_dict() == expected_dict
