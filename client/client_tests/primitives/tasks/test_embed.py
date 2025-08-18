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


def test_embed_task_initialization_with_modalities():
    """Test EmbedTask initialization with modality parameters."""
    task = EmbedTask(
        endpoint_url="http://embedding-ms:8000/v1",
        model_name="nvidia/test-model",
        api_key="api-key",
        filter_errors=True,
        text_elements_modality="text",
        image_elements_modality="image",
        structured_elements_modality="text_image",
        audio_elements_modality="text",
    )

    assert task._endpoint_url == "http://embedding-ms:8000/v1"
    assert task._model_name == "nvidia/test-model"
    assert task._api_key == "api-key"
    assert task._filter_errors
    assert task._text_elements_modality == "text"
    assert task._image_elements_modality == "image"
    assert task._structured_elements_modality == "text_image"
    assert task._audio_elements_modality == "text"


def test_embed_task_deprecated_parameters():
    """Test EmbedTask handles deprecated parameters with warnings."""
    from unittest.mock import patch

    with patch("nv_ingest_client.primitives.tasks.embed.logger") as mock_logger:
        task = EmbedTask(
            endpoint_url="http://embedding-ms:8000/v1",
            model_name="nvidia/test-model",
            api_key="api-key",
            text=True,  # Deprecated
            tables=False,  # Deprecated
        )

        # Check that warnings were issued via logger
        assert mock_logger.warning.call_count == 2

        # Check the warning messages
        warning_calls = mock_logger.warning.call_args_list
        assert "deprecated" in warning_calls[0][0][0]
        assert "text" in warning_calls[0][0][0]
        assert "deprecated" in warning_calls[1][0][0]
        assert "tables" in warning_calls[1][0][0]

        # Check that task was still created successfully
        assert task._endpoint_url == "http://embedding-ms:8000/v1"
        assert task._model_name == "nvidia/test-model"
        assert task._api_key == "api-key"


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


# Schema Consolidation Tests


def test_embed_task_schema_consolidation():
    """Test that EmbedTask uses API schema for validation."""
    # Test that valid parameters work
    task = EmbedTask(
        endpoint_url="http://embedding-ms:8000/v1",
        model_name="nvidia/test-model",
        api_key="api-key",
        filter_errors=True,
        text_elements_modality="text",
        image_elements_modality="image",
        structured_elements_modality="text_image",
        audio_elements_modality="text",
    )

    assert task._endpoint_url == "http://embedding-ms:8000/v1"
    assert task._model_name == "nvidia/test-model"
    assert task._api_key == "api-key"
    assert task._filter_errors
    assert task._text_elements_modality == "text"
    assert task._image_elements_modality == "image"
    assert task._structured_elements_modality == "text_image"
    assert task._audio_elements_modality == "text"


def test_embed_task_api_schema_validation():
    """Test that EmbedTask validates against API schema constraints."""
    # Test that None values are handled correctly
    task = EmbedTask()

    assert task._endpoint_url is None
    assert task._model_name is None
    assert task._api_key is None
    assert task._filter_errors is False
    assert task._text_elements_modality is None
    assert task._image_elements_modality is None
    assert task._structured_elements_modality is None
    assert task._audio_elements_modality is None


def test_embed_task_serialization_with_api_schema():
    """Test EmbedTask serialization works correctly with API schema."""
    task = EmbedTask(
        endpoint_url="http://embedding-ms:8000/v1",
        model_name="nvidia/test-model",
        api_key="api-key",
        filter_errors=True,
        text_elements_modality="text",
    )

    task_dict = task.to_dict()

    assert task_dict["type"] == "embed"
    assert task_dict["task_properties"]["endpoint_url"] == "http://embedding-ms:8000/v1"
    assert task_dict["task_properties"]["model_name"] == "nvidia/test-model"
    assert task_dict["task_properties"]["api_key"] == "api-key"
    assert task_dict["task_properties"]["filter_errors"] is True
    assert task_dict["task_properties"]["text_elements_modality"] == "text"
