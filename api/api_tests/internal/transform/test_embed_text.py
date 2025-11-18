# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import nv_ingest_api.internal.transform.embed_text as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# -----------------------------------------
# Fixtures
# -----------------------------------------
@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "Example text content",
                    "content_metadata": {"type": "text"},
                    "custom_content": {
                        "custom_content_field_1": "custom_text",
                        "custom_content_field_2": {"nested_content_field": "custom_text_2"},
                    },
                },
                "document_type": "TEXT",
            },
            {
                "metadata": {
                    "content": "Example table content",
                    "content_metadata": {"type": "structured"},
                    "table_metadata": {"table_content": "Table content here"},
                },
                "document_type": "STRUCTURED",
            },
        ]
    )


@pytest.fixture
def dummy_task_config():
    return {
        "api_key": "dummy_key",
        "endpoint_url": "http://dummy-endpoint",
        "model_name": "dummy_model",
    }


@pytest.fixture
def dummy_task_config_custom_embedding():
    return {
        "api_key": "dummy_key",
        "endpoint_url": "http://dummy-endpoint",
        "model_name": "dummy_model",
        "custom_content_field": "custom_content_field_1",
    }


@pytest.fixture
def dummy_task_config_custom_embedding_nested():
    return {
        "api_key": "dummy_key",
        "endpoint_url": "http://dummy-endpoint",
        "model_name": "dummy_model",
        "custom_content_field": "custom_content_field_2.nested_content_field",
    }


# -----------------------------------------
# Tests
# -----------------------------------------
@patch(f"{MODULE_UNDER_TEST}._async_runner")
def test_transform_create_text_embeddings_internal_happy_path(mock_async_runner, dummy_df, dummy_task_config):
    # Mock embeddings response
    mock_async_runner.side_effect = lambda *args, **kwargs: {
        "embeddings": [[0.1, 0.2, 0.3]] * len(args[0]),
        "info_msgs": [None] * len(args[0]),
    }

    result_df, trace = module_under_test.transform_create_text_embeddings_internal(
        dummy_df.copy(),
        dummy_task_config,
    )

    # Validate async_runner call
    mock_async_runner.assert_called()

    # Validate TEXT row has expected embedding
    assert result_df.iloc[0]["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert bool(result_df.iloc[0]["_contains_embeddings"])

    # Validate STRUCTURED row also has expected embedding (since it's supported)
    assert result_df.iloc[1]["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert bool(result_df.iloc[1]["_contains_embeddings"])

    # Validate trace structure
    assert "trace_info" in trace


@patch(f"{MODULE_UNDER_TEST}._async_runner")
def test_transform_create_text_embeddings_internal_custom_embedding(
    mock_async_runner, dummy_df, dummy_task_config_custom_embedding
):
    # Mock embeddings response
    mock_async_runner.side_effect = lambda *args, **kwargs: {
        "embeddings": [[0.1, 0.2, 0.3]] * len(args[0]),
        "info_msgs": [None] * len(args[0]),
    }

    result_df, trace = module_under_test.transform_create_text_embeddings_internal(
        dummy_df.copy(),
        dummy_task_config_custom_embedding,
    )

    # Validate async_runner call
    mock_async_runner.assert_called()

    # Validate TEXT row has expected embedding
    assert result_df.iloc[0]["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert bool(result_df.iloc[0]["_contains_embeddings"])

    # Validate STRUCTURED row also has expected embedding (since it's supported)
    assert result_df.iloc[1]["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert bool(result_df.iloc[1]["_contains_embeddings"])

    # Validate trace structure
    assert "trace_info" in trace

    # Validate custom embedding in custom content
    assert result_df.iloc[0]["metadata"]["custom_content"]["custom_content_field_1_embedding"] == [0.1, 0.2, 0.3]


@patch(f"{MODULE_UNDER_TEST}._async_runner")
def test_transform_create_text_embeddings_internal_custom_embedding_nested(
    mock_async_runner, dummy_df, dummy_task_config_custom_embedding_nested
):
    # Mock embeddings response
    mock_async_runner.side_effect = lambda *args, **kwargs: {
        "embeddings": [[0.1, 0.2, 0.3]] * len(args[0]),
        "info_msgs": [None] * len(args[0]),
    }

    result_df, trace = module_under_test.transform_create_text_embeddings_internal(
        dummy_df.copy(),
        dummy_task_config_custom_embedding_nested,
    )

    # Validate async_runner call
    mock_async_runner.assert_called()

    # Validate nested custom embedding in custom content
    assert result_df.iloc[0]["metadata"]["custom_content"]["custom_content_field_2"][
        "nested_content_field_embedding"
    ] == [0.1, 0.2, 0.3]


@patch(f"{MODULE_UNDER_TEST}._async_runner")
def test_transform_create_text_embeddings_internal_handles_empty_df(mock_async_runner):
    df = pd.DataFrame()
    result_df, trace = module_under_test.transform_create_text_embeddings_internal(
        df, {}, module_under_test.TextEmbeddingSchema()
    )
    mock_async_runner.assert_not_called()
    assert result_df.empty
    assert "trace_info" in trace


@patch(f"{MODULE_UNDER_TEST}._async_runner")
def test_transform_create_text_embeddings_internal_with_only_empty_text(mock_async_runner, dummy_task_config):
    df = pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "   ",
                    "content_metadata": {"type": "text"},
                },
                "document_type": "TEXT",
            },
        ]
    )

    result_df, trace = module_under_test.transform_create_text_embeddings_internal(
        df,
        dummy_task_config,  # Now it's correctly injected as dict
    )

    # Assert embedding was still added but is None
    assert "embedding" in result_df.iloc[0]["metadata"]
    assert result_df.iloc[0]["metadata"]["embedding"] is None
    # Assert no async call was made (since all content is empty)
    mock_async_runner.assert_not_called()


@patch(f"{MODULE_UNDER_TEST}._async_request_handler")
def test_async_runner_happy_path(mock_handler):
    # Arrange
    mock_handler.return_value = [
        {"embedding": [[1, 2, 3]], "info_msg": None},
        {"embedding": [[4, 5, 6]], "info_msg": {"msg": "partial success"}},
    ]

    prompts = ["prompt1", "prompt2", "prompt3"]

    # Act
    result = module_under_test._async_runner(
        prompts,
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )

    # Assert
    mock_handler.assert_called_once_with(
        prompts,
        "dummy_key",
        "http://dummy-endpoint",
        "dummy_model",
        "float",
        "passage",
        "END",
        False,
        modalities=None,
        dimensions=None,
    )
    assert result["embeddings"] == [[1, 2, 3], [4, 5, 6]]
    assert result["info_msgs"] == [None, {"msg": "partial success"}]


@patch(f"{MODULE_UNDER_TEST}._async_request_handler")
def test_async_runner_handles_none_embeddings(mock_handler):
    # Arrange
    mock_handler.return_value = [
        {"embedding": [None], "info_msg": {"msg": "error"}},
    ]

    prompts = ["prompt1"]

    # Act
    result = module_under_test._async_runner(
        prompts,
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )

    # Assert
    assert result["embeddings"] == [None]
    assert result["info_msgs"] == [{"msg": "error"}]


@patch(f"{MODULE_UNDER_TEST}._async_request_handler")
def test_async_runner_handles_embedding_object(mock_handler):
    # Arrange: simulate non-list embedding object with `.embedding` attribute
    dummy_embedding = MagicMock()
    dummy_embedding.embedding = [42, 43]

    mock_handler.return_value = [
        {"embedding": [dummy_embedding], "info_msg": None},
    ]

    prompts = ["prompt1"]

    # Act
    result = module_under_test._async_runner(
        prompts,
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )

    # Assert
    assert result["embeddings"] == [[42, 43]]
    assert result["info_msgs"] == [None]


@patch(f"{MODULE_UNDER_TEST}._make_async_request")
def test_async_request_handler_happy_path(mock_make_async_request):
    # Arrange: mock returns per batch call
    mock_make_async_request.side_effect = [
        {"embedding": [[1, 2, 3]], "info_msg": None},
        {"embedding": [[4, 5, 6]], "info_msg": {"note": "ok"}},
    ]

    prompts = [["batch1_prompt"], ["batch2_prompt"]]

    # Act
    result = module_under_test._async_request_handler(
        prompts,
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )

    # Assert: called once per batch
    assert mock_make_async_request.call_count == 2
    mock_make_async_request.assert_any_call(
        prompts=["batch1_prompt"],
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        modalities=None,
        dimensions=None,
    )
    mock_make_async_request.assert_any_call(
        prompts=["batch2_prompt"],
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        modalities=None,
        dimensions=None,
    )

    # And we get both results combined in order
    assert result == [
        {"embedding": [[1, 2, 3]], "info_msg": None},
        {"embedding": [[4, 5, 6]], "info_msg": {"note": "ok"}},
    ]


@patch(f"{MODULE_UNDER_TEST}._make_async_request")
def test_async_request_handler_empty_prompts_list(mock_make_async_request):
    # Act
    result = module_under_test._async_request_handler(
        [],
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )

    # Assert
    mock_make_async_request.assert_not_called()
    assert result == []


@patch("nv_ingest_api.internal.transform.embed_text.infer_microservice")
def test_make_async_request_happy_path(im_mock):
    # Assign
    im_mock.return_value = [[0.1, 0.2, 0.3]]
    # Act
    result = module_under_test._make_async_request(
        prompts=["Hello world"],
        api_key="dummy_key",
        embedding_nim_endpoint="http://dummy-endpoint",
        embedding_model="dummy_model",
        encoding_format="float",
        input_type="passage",
        truncate="END",
        filter_errors=False,
        dimensions=None,
    )
    # Assert: client called as expected
    im_mock.assert_called_once_with(
        ["Hello world"],
        "dummy_model",
        embedding_endpoint="http://dummy-endpoint",
        nvidia_api_key="dummy_key",
        input_type="passage",
        truncate="END",
        batch_size=8191,
        grpc=False,
        input_names=["text"],
        output_names=["embeddings"],
        dtypes=["BYTES"],
    )

    assert result == {
        "embedding": [[0.1, 0.2, 0.3]],
        "info_msg": None
    }


@patch("nv_ingest_api.internal.transform.embed_text.infer_microservice")
def test_make_async_request_failure_returns_none_embedding_and_info_message(im_mock):
    # Arrange
    im_mock.side_effect = RuntimeError("Simulated client failure")

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        module_under_test._make_async_request(
            prompts=["Test prompt"],
            api_key="dummy_key",
            embedding_nim_endpoint="http://dummy-endpoint",
            embedding_model="dummy_model",
            encoding_format="float",
            input_type="passage",
            truncate="END",
            filter_errors=True,
            dimensions=None,
        )

    # The raised error should have our validated info message attached in text
    assert "Embedding error occurred: Simulated client failure" in str(excinfo.value)
