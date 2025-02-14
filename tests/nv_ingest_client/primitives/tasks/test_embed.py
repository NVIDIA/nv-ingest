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
        api_key="API_KEY",
    )
    assert task._model_name == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert task._endpoint_url == "http://embedding:8000/v1"
    assert task._api_key == "API_KEY"


# String Representation Tests


def test_embed_task_str_representation():
    task = EmbedTask(model_name="nvidia/nv-embedqa-e5-v5", endpoint_url="http://localhost:8024/v1", api_key="API_KEY", filter_errors=True)
    expected_str = (
        "Embed Task:\n"
        "  model_name: nvidia/nv-embedqa-e5-v5\n"
        "  endpoint_url: http://localhost:8024/v1\n"
        "  api_key: [redacted]\n"
        "  filter_errors: True\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "model_name, endpoint_url, api_key, filter_errors",
    [
        ("meta-llama/Llama-3.2-1B", "http://embedding:8012/v1", "TEST", False),
        ("nvidia/nv-embedqa-mistral-7b-v2", "http://localhost:8000/v1", "12345", True),
        ("nvidia/nv-embedqa-e5-v5", "http://embedding:8000/v1", "key", True),
        (None, None, None, False),  # Test default parameters
    ],
)
def test_embed_task_to_dict(
    model_name,
    endpoint_url,
    api_key,
    filter_errors,
):
    task = EmbedTask(
        model_name=model_name,
        endpoint_url=endpoint_url,
        api_key=api_key,
        filter_errors=filter_errors,
    )

    expected_dict = {"type": "embed", "task_properties": {}}

    # Only add properties to expected_dict if they are not None
    if model_name is not None:
        expected_dict["task_properties"]["model_name"] = model_name
    if endpoint_url is not None:
        expected_dict["task_properties"]["endpoint_url"] = endpoint_url
    if api_key is not None:
        expected_dict["task_properties"]["api_key"] = api_key
    expected_dict["task_properties"]["filter_errors"] = filter_errors
    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_embed_task_default_params():
    task = EmbedTask()
    assert "Embed Task:" in str(task)
    assert "filter_errors: False" in str(task)

    task_dict = task.to_dict()
    assert task_dict == {"type": "embed", "task_properties": {"filter_errors": False}}
