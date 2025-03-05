# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from nv_ingest_client.primitives.tasks.embed import EmbedTask

# Initialization and Property Setting


def test_embed_task_initialization():
    os.environ["EMBEDDING_NIM_ENDPOINT"] = "embedding-ms:8000"
    os.environ["EMBEDDING_NIM_MODEL_NAME"] = "nvidia/test-model"
    os.environ["NVIDIA_BUILD_API_KEY"] = "api-key"

    task = EmbedTask(
        filter_errors=True,
    )

    del os.environ['EMBEDDING_NIM_ENDPOINT']
    del os.environ['EMBEDDING_NIM_MODEL_NAME']
    del os.environ["NVIDIA_BUILD_API_KEY"]

    assert task._embedding_nim_endpoint == "http://embedding-ms:8000/v1"
    assert task._embedding_nim_model_name == "nvidia/test-model"
    assert task._nvidia_build_api_key == "api-key"
    assert task._filter_errors == True


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "embedding_nim_endpoint, embedding_nim_model_name, nvidia_build_api_key, filter_errors",
    [
        ("localhost:8000", "nvidia/embedding-model", "", True),
        ("http://embedding-ms:8000/v1", "nvidia/llama-3.2-nv-embedqa-1b-v2", "test-key", False),
        ("", "nvidia/nv-embedqa-e5-v5", "42", True),
    ],
)
def test_embed_task_to_dict(
    embedding_nim_endpoint,
    embedding_nim_model_name,
    nvidia_build_api_key,
    filter_errors,
):
    os.environ["EMBEDDING_NIM_ENDPOINT"] = embedding_nim_endpoint
    os.environ["EMBEDDING_NIM_MODEL_NAME"] = embedding_nim_model_name
    os.environ["NVIDIA_BUILD_API_KEY"] = nvidia_build_api_key

    task = EmbedTask(
        filter_errors = filter_errors
    )

    del os.environ['EMBEDDING_NIM_ENDPOINT']
    del os.environ['EMBEDDING_NIM_MODEL_NAME']
    del os.environ["NVIDIA_BUILD_API_KEY"]

    expected_dict = {"type": "embed", "task_properties": {
        "embedding_nim_model_name": embedding_nim_model_name,
        "nvidia_build_api_key": nvidia_build_api_key,
        "filter_errors": filter_errors,
    }}

    if embedding_nim_endpoint == "":
        expected_dict["task_properties"]["embedding_nim_endpoint"] = "https://integrate.api.nvidia.com/v1"
    elif embedding_nim_endpoint[:7] != "http://" and embedding_nim_endpoint[:8] != "https://":
        expected_dict["task_properties"]["embedding_nim_endpoint"] = "http://" + embedding_nim_endpoint + "/v1"
    else:
        expected_dict["task_properties"]["embedding_nim_endpoint"] = embedding_nim_endpoint
    
    print(expected_dict)
    print(task.to_dict())

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_embed_task_default_params():
    task = EmbedTask()
    expected_str_contains = [
        "embedding_nim_endpoint: https://integrate.api.nvidia.com/v1",
        "embedding_nim_model_name: nvidia/llama-3.2-nv-embedqa-1b-v2",
        "nvidia_build_api_key: [redacted]",
        "filter_errors: False"
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "embed",
        "task_properties": {
            "embedding_nim_endpoint": "https://integrate.api.nvidia.com/v1",
            "embedding_nim_model_name": "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "nvidia_build_api_key": "",
            "filter_errors": False,
        },
    }
    assert task.to_dict() == expected_dict
