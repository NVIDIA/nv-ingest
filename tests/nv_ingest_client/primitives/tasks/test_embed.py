# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.embed import EmbedTask

# Initialization and Property Setting


def test_embed_task_initialization():

    task = EmbedTask(
        endpoint_url="http://embedding-ms:8000/v1",
        model_name="nvidia/test-model",
        api_key="api-key",
        filter_errors=True,
    )

    assert task._endpoint_url == "http://embedding-ms:8000/v1"
    assert task._model_name == "nvidia/test-model"
    assert task._api_key == "api-key"
    assert task._filter_errors


# String Representation Tests


def test_embed_task_str_representation():
    task = EmbedTask(
        endpoint_url="http://embedding-ms:8000/v1",
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key="api-key",
        filter_errors=False,
    )
    expected_str = (
        "Embed Task:\n"
        "  endpoint_url: http://embedding-ms:8000/v1\n"
        "  model_name: nvidia/llama-3.2-nv-embedqa-1b-v2\n"
        "  api_key: [redacted]\n"
        "  filter_errors: False\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "endpoint_url, model_name, api_key, filter_errors",
    [
        ("https://integrate.api.nvidia.com/v1", "nvidia/embedding-model", "", True),
        ("http://embedding-ms:8000/v1", "nvidia/llama-3.2-nv-embedqa-1b-v2", "test-key", False),
        ("", "nvidia/nv-embedqa-e5-v5", "42", True),
        (None, None, None, False),
    ],
)
def test_embed_task_to_dict(
    endpoint_url,
    model_name,
    api_key,
    filter_errors,
):

    task = EmbedTask(endpoint_url=endpoint_url, model_name=model_name, api_key=api_key, filter_errors=filter_errors)

    expected_dict = {"type": "embed", "task_properties": {"filter_errors": filter_errors}}

    # Only add properties to expected_dict if they are not None
    if endpoint_url:
        expected_dict["task_properties"]["endpoint_url"] = endpoint_url
    if model_name:
        expected_dict["task_properties"]["model_name"] = model_name
    if api_key:
        expected_dict["task_properties"]["api_key"] = api_key

    print(expected_dict)
    print(task.to_dict())

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_embed_task_default_params():
    task = EmbedTask()
    expected_str_contains = [
        "filter_errors: False",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "embed",
        "task_properties": {
            "filter_errors": False,
        },
    }
    assert task.to_dict() == expected_dict
