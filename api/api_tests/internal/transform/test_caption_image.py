# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
import nv_ingest_api.internal.transform.caption_image as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def dummy_df_with_images():
    return pd.DataFrame(
        [
            {"metadata": {"content": "base64_image_1", "content_metadata": {"type": "image"}, "image_metadata": {}}},
            {"metadata": {"content": "base64_image_2", "content_metadata": {"type": "image"}, "image_metadata": {}}},
            {"metadata": {"content": "non_image_content", "content_metadata": {"type": "text"}, "image_metadata": {}}},
        ]
    )


@pytest.fixture
def dummy_task_config():
    return {
        "api_key": "api_key",
        "prompt": "Describe the image",
        "system_prompt": "You are a helpful assistant.",
        "endpoint_url": "https://fake.endpoint",
        "model_name": "fake_model",
    }


@pytest.fixture
def dummy_transform_config():
    # Simple mock config object with attribute fallback
    class Dummy:
        api_key = "default_api_key"
        prompt = "default_prompt"
        system_prompt = "/no_think"  # Default system prompt for Nemotron Nano 12B v2 VL (reasoning off)
        endpoint_url = "https://default.endpoint"
        model_name = "default_model"

    return Dummy()


# --- Tests ---


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_image_create_vlm_caption_internal_happy_path(
    mock_generate, dummy_df_with_images, dummy_task_config, dummy_transform_config
):
    # Setup mock captions
    mock_generate.return_value = ["caption1", "caption2"]

    result = transform_image_create_vlm_caption_internal(
        dummy_df_with_images.copy(),
        dummy_task_config,
        dummy_transform_config,
    )

    # Assert _generate_captions was called with the correct arguments
    mock_generate.assert_called_once_with(
        ["base64_image_1", "base64_image_2"],
        dummy_task_config["prompt"],
        dummy_task_config["system_prompt"],
        dummy_task_config["api_key"],
        dummy_task_config["endpoint_url"],
        dummy_task_config["model_name"],
        temperature=1.0,
    )

    # Assert captions updated correctly in the DataFrame
    assert result.iloc[0]["metadata"]["image_metadata"]["caption"] == "caption1"
    assert result.iloc[1]["metadata"]["image_metadata"]["caption"] == "caption2"
    # Ensure non-image row was untouched
    assert "caption" not in result.iloc[2]["metadata"]["image_metadata"]


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_image_create_vlm_caption_internal_no_image_rows(mock_generate, dummy_transform_config):
    # DF with only non-image rows
    df = pd.DataFrame(
        [
            {"metadata": {"content": "text1", "content_metadata": {"type": "text"}}},
            {"metadata": {"content": "text2", "content_metadata": {"type": "text"}}},
        ]
    )

    result = transform_image_create_vlm_caption_internal(
        df.copy(),
        {},
        dummy_transform_config,
    )

    # _generate_captions should not be called
    mock_generate.assert_not_called()

    # DataFrame should be unchanged
    pd.testing.assert_frame_equal(result, df)


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_image_create_vlm_caption_internal_uses_fallback_config(
    mock_generate, dummy_df_with_images, dummy_transform_config
):
    # Provide empty task_config to force fallback to transform_config
    mock_generate.return_value = ["caption1", "caption2"]

    result = transform_image_create_vlm_caption_internal(
        dummy_df_with_images.copy(),
        {},
        dummy_transform_config,
    )

    # _generate_captions should be called with fallback config values
    mock_generate.assert_called_once_with(
        ["base64_image_1", "base64_image_2"],
        dummy_transform_config.prompt,
        dummy_transform_config.system_prompt,
        dummy_transform_config.api_key,
        dummy_transform_config.endpoint_url,
        dummy_transform_config.model_name,
        temperature=1.0,
    )

    # Assert captions updated correctly
    assert result.iloc[0]["metadata"]["image_metadata"]["caption"] == "caption1"
    assert result.iloc[1]["metadata"]["image_metadata"]["caption"] == "caption2"


def test_prepare_dataframes_mod_happy_path():
    df = pd.DataFrame(
        [
            {"metadata": {}, "document_type": ContentTypeEnum.IMAGE},
            {"metadata": {}, "document_type": "TEXT"},
            {"metadata": {}, "document_type": ContentTypeEnum.IMAGE},
        ]
    )

    df_original, df_matched, bool_index = module_under_test._prepare_dataframes_mod(df)

    # Assert original unchanged
    pd.testing.assert_frame_equal(df_original, df)

    # Assert correct rows selected
    assert len(df_matched) == 2
    assert all(df_matched["document_type"] == ContentTypeEnum.IMAGE)

    # Assert boolean mask correctness
    assert bool_index.tolist() == [True, False, True]


def test_prepare_dataframes_mod_empty_df():
    df = pd.DataFrame()

    df_original, df_matched, bool_index = module_under_test._prepare_dataframes_mod(df)

    # All should be empty
    pd.testing.assert_frame_equal(df_original, df)
    assert df_matched.empty
    assert bool_index.empty


def test_prepare_dataframes_mod_missing_document_type():
    df = pd.DataFrame([{"metadata": {}}, {"metadata": {}}])

    df_original, df_matched, bool_index = module_under_test._prepare_dataframes_mod(df)

    pd.testing.assert_frame_equal(df_original, df)
    assert df_matched.empty
    assert bool_index.empty


def test_prepare_dataframes_mod_no_image_rows():
    df = pd.DataFrame([{"metadata": {}, "document_type": "TEXT"}, {"metadata": {}, "document_type": "STRUCTURED"}])

    df_original, df_matched, bool_index = module_under_test._prepare_dataframes_mod(df)

    pd.testing.assert_frame_equal(df_original, df)
    assert df_matched.empty
    assert bool_index.tolist() == [False, False]


def test_prepare_dataframes_mod_handles_invalid_column_type():
    df = pd.DataFrame(
        [
            {"metadata": {}, "document_type": ["not", "a", "string"]},
            {"metadata": {}, "document_type": {"unexpected": "dict"}},
        ]
    )

    # Should still not match anything and not raise error
    df_original, df_matched, bool_index = module_under_test._prepare_dataframes_mod(df)

    pd.testing.assert_frame_equal(df_original, df)
    assert df_matched.empty
    assert bool_index.tolist() == [False, False]


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}.scale_image_to_encoding_size")
def test_generate_captions_happy_path(mock_scale, mock_create_client):
    # Arrange mocks
    mock_scale.side_effect = lambda b64: (f"scaled_{b64}", None)
    mock_client = MagicMock()
    mock_client.infer.return_value = ["Caption 1", "Caption 2"]
    mock_create_client.return_value = mock_client

    # Input images
    base64_images = ["b64img1", "b64img2"]

    # Act
    result = module_under_test._generate_captions(
        base64_images,
        prompt="describe this",
        system_prompt=None,
        api_key="test_api_key",
        endpoint_url="https://fake.endpoint",
        model_name="test_model",
    )

    # Assert scaling applied correctly
    mock_scale.assert_any_call("b64img1")
    mock_scale.assert_any_call("b64img2")

    # Assert client created correctly
    mock_create_client.assert_called_once()
    client_call_args = mock_create_client.call_args[1]
    assert client_call_args["auth_token"] == "test_api_key"
    assert client_call_args["endpoints"] == (None, "https://fake.endpoint")
    assert client_call_args["infer_protocol"] == "http"

    # Assert infer called with correct data
    expected_payload = {"base64_images": ["scaled_b64img1", "scaled_b64img2"], "prompt": "describe this"}
    mock_client.infer.assert_called_once_with(expected_payload, model_name="test_model", temperature=1.0)

    # Result matches mock captions
    assert result == ["Caption 1", "Caption 2"]


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}.scale_image_to_encoding_size")
def test_generate_captions_raises_on_client_error(mock_scale, mock_create_client):
    mock_scale.side_effect = lambda b64: (f"scaled_{b64}", None)
    mock_client = MagicMock()
    mock_client.infer.side_effect = RuntimeError("client error")
    mock_create_client.return_value = mock_client

    with pytest.raises(RuntimeError) as excinfo:
        module_under_test._generate_captions(
            ["b64img1"],
            prompt="describe this",
            system_prompt=None,
            api_key="test_api_key",
            endpoint_url="https://fake.endpoint",
            model_name="test_model",
        )

    assert "_generate_captions: Error generating captions:" in str(excinfo.value)


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}.scale_image_to_encoding_size")
def test_generate_captions_empty_images_returns_empty_list(mock_scale, mock_create_client):
    # Should still try to infer with empty images
    mock_client = MagicMock()
    mock_client.infer.return_value = []
    mock_create_client.return_value = mock_client

    result = module_under_test._generate_captions(
        [],
        prompt="describe this",
        system_prompt=None,
        api_key="test_api_key",
        endpoint_url="https://fake.endpoint",
        model_name="test_model",
    )

    mock_client.infer.assert_called_once_with(
        {"base64_images": [], "prompt": "describe this"}, model_name="test_model", temperature=1.0
    )

    assert result == []


# --- _gather_context_text_for_image tests ---


def test_gather_context_text_page_match():
    """Page text is returned when page_number matches."""
    image_meta = {
        "content_metadata": {
            "type": "image",
            "page_number": 3,
        },
    }
    page_text_map = {3: ["page three text", "more text"]}
    result = module_under_test._gather_context_text_for_image(image_meta, page_text_map, 200)
    assert result == "page three text more text"


def test_gather_context_text_truncation():
    """Text is truncated to max_chars."""
    image_meta = {
        "content_metadata": {
            "type": "image",
            "page_number": 0,
        },
    }
    page_text_map = {0: ["a" * 500]}
    result = module_under_test._gather_context_text_for_image(image_meta, page_text_map, 10)
    assert len(result) == 10


def test_gather_context_text_safety_cap():
    """Text is capped at _MAX_CONTEXT_TEXT_CHARS even if max_chars is larger."""
    image_meta = {
        "content_metadata": {
            "type": "image",
            "page_number": 0,
        },
    }
    big_text = "x" * 10000
    page_text_map = {0: [big_text]}
    result = module_under_test._gather_context_text_for_image(image_meta, page_text_map, 99999)
    assert len(result) == module_under_test._MAX_CONTEXT_TEXT_CHARS


def test_gather_context_text_no_text():
    """Returns empty string when no text is available."""
    image_meta = {
        "content_metadata": {
            "type": "image",
            "page_number": 5,
        },
    }
    result = module_under_test._gather_context_text_for_image(image_meta, {}, 200)
    assert result == ""


def test_gather_context_text_wrong_page():
    """Returns empty string when page number doesn't match any text."""
    image_meta = {
        "content_metadata": {
            "type": "image",
            "page_number": 99,
        },
    }
    page_text_map = {0: ["some text"]}
    result = module_under_test._gather_context_text_for_image(image_meta, page_text_map, 200)
    assert result == ""


# --- _build_prompt_with_context tests ---


def test_build_prompt_with_context():
    result = module_under_test._build_prompt_with_context("Caption this:", "nearby text")
    assert result == "Text near this image:\n---\nnearby text\n---\n\nCaption this:"


def test_build_prompt_with_empty_context():
    result = module_under_test._build_prompt_with_context("Caption this:", "")
    assert result == "Caption this:"


# --- _build_page_text_map tests ---


def test_build_page_text_map():
    df = pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "text on page 0",
                    "content_metadata": {"type": "text", "page_number": 0},
                }
            },
            {
                "metadata": {
                    "content": "more on page 0",
                    "content_metadata": {"type": "text", "page_number": 0},
                }
            },
            {
                "metadata": {
                    "content": "image content",
                    "content_metadata": {"type": "image", "page_number": 0},
                }
            },
            {
                "metadata": {
                    "content": "page 1 text",
                    "content_metadata": {"type": "text", "page_number": 1},
                }
            },
        ]
    )
    result = module_under_test._build_page_text_map(df)
    assert result == {0: ["text on page 0", "more on page 0"], 1: ["page 1 text"]}


# --- Context-enabled integration tests ---


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_context_enabled_per_image_calls(mock_generate, dummy_transform_config):
    """With context enabled, each image gets its own VLM call with enriched prompt."""
    df = pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "b64_img1",
                    "content_metadata": {"type": "image", "page_number": 0},
                    "image_metadata": {},
                }
            },
            {
                "metadata": {
                    "content": "page zero text",
                    "content_metadata": {"type": "text", "page_number": 0},
                    "image_metadata": {},
                }
            },
        ]
    )
    mock_generate.return_value = ["caption_with_context"]

    task_config = {
        "api_key": "key",
        "prompt": "Caption this:",
        "system_prompt": "sys",
        "endpoint_url": "https://url",
        "model_name": "model",
        "context_text_max_chars": 500,
    }

    result = transform_image_create_vlm_caption_internal(df.copy(), task_config, dummy_transform_config)

    # Should be called once (one image)
    assert mock_generate.call_count == 1
    call_args = mock_generate.call_args
    # The prompt should be enriched with context
    assert "Text near this image:" in call_args[0][1]
    assert "page zero text" in call_args[0][1]
    assert "Caption this:" in call_args[0][1]
    # The image should be passed individually
    assert call_args[0][0] == ["b64_img1"]
    # Caption should be set
    assert result.iloc[0]["metadata"]["image_metadata"]["caption"] == "caption_with_context"


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_temperature_forwarded(mock_generate, dummy_df_with_images, dummy_transform_config):
    """Temperature from task_config is forwarded to _generate_captions."""
    mock_generate.return_value = ["c1", "c2"]

    task_config = {
        "api_key": "key",
        "prompt": "Describe",
        "system_prompt": "sys",
        "endpoint_url": "https://url",
        "model_name": "model",
        "temperature": 0.7,
    }

    transform_image_create_vlm_caption_internal(dummy_df_with_images.copy(), task_config, dummy_transform_config)

    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs["temperature"] == 0.7


@patch(f"{MODULE_UNDER_TEST}._generate_captions")
def test_transform_context_disabled_batch_preserved(
    mock_generate, dummy_df_with_images, dummy_task_config, dummy_transform_config
):
    """With context disabled (default), batch behavior is unchanged."""
    mock_generate.return_value = ["c1", "c2"]

    _ = transform_image_create_vlm_caption_internal(
        dummy_df_with_images.copy(), dummy_task_config, dummy_transform_config
    )

    # Should be called once in batch mode
    mock_generate.assert_called_once()
    call_args = mock_generate.call_args
    # All images passed at once
    assert call_args[0][0] == ["base64_image_1", "base64_image_2"]
    # Prompt should NOT be enriched
    assert "Text near this image:" not in call_args[0][1]
