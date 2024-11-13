# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io

import requests
from PIL import Image
from unittest.mock import MagicMock, patch

import pytest

MODULE_UNDER_TEST = 'nv_ingest.stages.transforms.image_caption_extraction'

import pandas as pd

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.stages.transforms.image_caption_extraction import _prepare_dataframes_mod
from nv_ingest.stages.transforms.image_caption_extraction import _generate_captions
from nv_ingest.stages.transforms.image_caption_extraction import caption_extract_stage


def generate_base64_png_image() -> str:
    """Helper function to generate a base64-encoded PNG image."""
    img = Image.new("RGB", (10, 10), color="blue")  # Create a simple blue image
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_prepare_dataframes_empty_dataframe():
    # Test with an empty DataFrame
    df = pd.DataFrame()

    df_out, df_matched, bool_index = _prepare_dataframes_mod(df)

    assert df_out.equals(df)
    assert df_matched.empty
    assert bool_index.empty
    assert bool_index.dtype == bool


def test_prepare_dataframes_missing_document_type_column():
    # Test with a DataFrame missing the 'document_type' column
    df = pd.DataFrame({
        "other_column": [1, 2, 3]
    })

    df_out, df_matched, bool_index = _prepare_dataframes_mod(df)

    assert df_out.equals(df)
    assert df_matched.empty
    assert bool_index.empty
    assert bool_index.dtype == bool


def test_prepare_dataframes_no_matches():
    # Test with a DataFrame where no 'document_type' matches ContentTypeEnum.IMAGE
    df = pd.DataFrame({
        "document_type": [ContentTypeEnum.TEXT, ContentTypeEnum.STRUCTURED, ContentTypeEnum.UNSTRUCTURED]
    })

    df_out, df_matched, bool_index = _prepare_dataframes_mod(df)

    assert df_out.equals(df)
    assert df_matched.empty
    assert bool_index.equals(pd.Series([False, False, False]))
    assert bool_index.dtype == bool


def test_prepare_dataframes_partial_matches():
    # Test with a DataFrame where some rows match ContentTypeEnum.IMAGE
    df = pd.DataFrame({
        "document_type": [ContentTypeEnum.IMAGE, ContentTypeEnum.TEXT, ContentTypeEnum.IMAGE]
    })

    df_out, df_matched, bool_index = _prepare_dataframes_mod(df)

    assert df_out.equals(df)
    assert not df_matched.empty
    assert df_matched.equals(df[df["document_type"] == ContentTypeEnum.IMAGE])
    assert bool_index.equals(pd.Series([True, False, True]))
    assert bool_index.dtype == bool


def test_prepare_dataframes_all_matches():
    # Test with a DataFrame where all rows match ContentTypeEnum.IMAGE
    df = pd.DataFrame({
        "document_type": [ContentTypeEnum.IMAGE, ContentTypeEnum.IMAGE, ContentTypeEnum.IMAGE]
    })

    df_out, df_matched, bool_index = _prepare_dataframes_mod(df)

    assert df_out.equals(df)
    assert df_matched.equals(df)
    assert bool_index.equals(pd.Series([True, True, True]))
    assert bool_index.dtype == bool


@patch(f'{MODULE_UNDER_TEST}._generate_captions')
def test_caption_extract_no_image_content(mock_generate_captions):
    # DataFrame with no image content
    df = pd.DataFrame({
        "metadata": [{"content_metadata": {"type": "text"}}, {"content_metadata": {"type": "pdf"}}]
    })
    task_props = {"api_key": "test_api_key", "prompt": "Describe the image", "endpoint_url": "https://api.example.com"}
    validated_config = MagicMock()
    trace_info = {}

    # Call the function
    result_df = caption_extract_stage(df, task_props, validated_config, trace_info)

    # Check that _generate_captions was not called and df is unchanged
    mock_generate_captions.assert_not_called()
    assert result_df.equals(df)


@patch(f'{MODULE_UNDER_TEST}._generate_captions')
def test_caption_extract_with_image_content(mock_generate_captions):
    # Mock caption generation
    mock_generate_captions.return_value = "A description of the image."

    # DataFrame with image content
    df = pd.DataFrame({
        "metadata": [{"content_metadata": {"type": "image"}, "content": "base64_encoded_image_data"}]
    })
    task_props = {"api_key": "test_api_key", "prompt": "Describe the image", "endpoint_url": "https://api.example.com"}
    validated_config = MagicMock()
    trace_info = {}

    # Call the function
    result_df = caption_extract_stage(df, task_props, validated_config, trace_info)

    # Check that _generate_captions was called once
    mock_generate_captions.assert_called_once_with("base64_encoded_image_data", "Describe the image", "test_api_key",
                                                   "https://api.example.com")

    # Verify that the caption was added to image_metadata
    assert result_df.loc[0, "metadata"]["image_metadata"]["caption"] == "A description of the image."


@patch(f'{MODULE_UNDER_TEST}._generate_captions')
def test_caption_extract_mixed_content(mock_generate_captions):
    # Mock caption generation
    mock_generate_captions.return_value = "A description of the image."

    # DataFrame with mixed content types
    df = pd.DataFrame({
        "metadata": [
            {"content_metadata": {"type": "image"}, "content": "image_data_1"},
            {"content_metadata": {"type": "text"}, "content": "text_data"},
            {"content_metadata": {"type": "image"}, "content": "image_data_2"}
        ]
    })
    task_props = {"api_key": "test_api_key", "prompt": "Describe the image", "endpoint_url": "https://api.example.com"}
    validated_config = MagicMock()
    trace_info = {}

    # Call the function
    result_df = caption_extract_stage(df, task_props, validated_config, trace_info)

    # Check that _generate_captions was called twice for images only
    assert mock_generate_captions.call_count == 2
    mock_generate_captions.assert_any_call("image_data_1", "Describe the image", "test_api_key",
                                           "https://api.example.com")
    mock_generate_captions.assert_any_call("image_data_2", "Describe the image", "test_api_key",
                                           "https://api.example.com")

    # Verify that captions were added only for image rows
    assert result_df.loc[0, "metadata"]["image_metadata"]["caption"] == "A description of the image."
    assert "caption" not in result_df.loc[1, "metadata"].get("image_metadata", {})
    assert result_df.loc[2, "metadata"]["image_metadata"]["caption"] == "A description of the image."


@patch(f'{MODULE_UNDER_TEST}._generate_captions')
def test_caption_extract_empty_dataframe(mock_generate_captions):
    # Empty DataFrame
    df = pd.DataFrame(columns=["metadata"])
    task_props = {"api_key": "test_api_key", "prompt": "Describe the image", "endpoint_url": "https://api.example.com"}
    validated_config = MagicMock()
    trace_info = {}

    # Call the function
    result_df = caption_extract_stage(df, task_props, validated_config, trace_info)

    # Check that _generate_captions was not called and df is still empty
    mock_generate_captions.assert_not_called()
    assert result_df.empty


@patch(f'{MODULE_UNDER_TEST}._generate_captions')
def test_caption_extract_malformed_metadata(mock_generate_captions):
    # Mock caption generation
    mock_generate_captions.return_value = "A description of the image."

    # DataFrame with malformed metadata (missing 'content' key in one row)
    df = pd.DataFrame({
        "metadata": [{"unexpected_key": "value"}, {"content_metadata": {"type": "image"}}]
    })
    task_props = {"api_key": "test_api_key", "prompt": "Describe the image", "endpoint_url": "https://api.example.com"}
    validated_config = MagicMock()
    trace_info = {}

    # Expecting KeyError for missing 'content' in the second row
    with pytest.raises(KeyError, match="'content'"):
        caption_extract_stage(df, task_props, validated_config, trace_info)


@patch(f"{MODULE_UNDER_TEST}.requests.post")
def test_generate_captions_successful(mock_post):
    # Mock the successful API response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "A beautiful sunset over the mountains."}}
        ]
    }
    mock_post.return_value = mock_response

    # Parameters
    base64_image = generate_base64_png_image()
    prompt = "Describe the image"
    api_key = "test_api_key"
    endpoint_url = "https://api.example.com"

    # Call the function
    result = _generate_captions(base64_image, prompt, api_key, endpoint_url)

    # Verify that the correct caption was returned
    assert result == "A beautiful sunset over the mountains."
    mock_post.assert_called_once_with(
        endpoint_url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json={
            "model": 'meta/llama-3.2-90b-vision-instruct',
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/png;base64,{base64_image}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 1.00,
            "top_p": 1.00,
            "stream": False
        }
    )


@patch(f"{MODULE_UNDER_TEST}.requests.post")
def test_generate_captions_api_error(mock_post):
    # Mock a 500 Internal Server Error response
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    mock_post.return_value = mock_response

    # Parameters
    base64_image = generate_base64_png_image()
    prompt = "Describe the image"
    api_key = "test_api_key"
    endpoint_url = "https://api.example.com"

    # Expect an exception due to the server error
    with pytest.raises(requests.exceptions.RequestException, match="500 Server Error"):
        _generate_captions(base64_image, prompt, api_key, endpoint_url)


@patch(f"{MODULE_UNDER_TEST}.requests.post")
def test_generate_captions_malformed_json(mock_post):
    # Mock a response with an unexpected JSON structure
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"unexpected_key": "unexpected_value"}
    mock_post.return_value = mock_response

    # Parameters
    base64_image = generate_base64_png_image()
    prompt = "Describe the image"
    api_key = "test_api_key"
    endpoint_url = "https://api.example.com"

    # Call the function
    result = _generate_captions(base64_image, prompt, api_key, endpoint_url)

    # Verify fallback response when JSON is malformed
    assert result == "No caption returned"


@patch(f"{MODULE_UNDER_TEST}.requests.post")
def test_generate_captions_empty_caption_content(mock_post):
    # Mock a response with empty caption content
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": ""}}
        ]
    }
    mock_post.return_value = mock_response

    # Parameters
    base64_image = generate_base64_png_image()
    prompt = "Describe the image"
    api_key = "test_api_key"
    endpoint_url = "https://api.example.com"

    # Call the function
    result = _generate_captions(base64_image, prompt, api_key, endpoint_url)

    # Verify that the fallback response is returned
    assert result == ""
