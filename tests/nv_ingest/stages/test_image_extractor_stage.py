# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from nv_ingest.stages.extractors.image_extractor_stage import process_image
from nv_ingest.stages.extractors.image_extractor_stage import decode_and_extract

MODULE_UNDER_TEST = 'nv_ingest.stages.extractors.image_extractor_stage'


# Define the test function using pytest
@patch(f'{MODULE_UNDER_TEST}.decode_and_extract')
def test_process_image_single_row(mock_decode_and_extract):
    mock_decode_and_extract.return_value = [
        {"document_type": "type1", "metadata": {"key": "value"}, "uuid": "1234"}
    ]

    input_df = pd.DataFrame({
        "source_id": [1],
        "content": ["base64encodedstring"]
    })

    task_props = {"method": "some_method"}
    validated_config = MagicMock()
    trace_info = {}

    processed_df, trace_info_output = process_image(input_df, task_props, validated_config, trace_info)

    assert len(processed_df) == 1
    assert "document_type" in processed_df.columns
    assert "metadata" in processed_df.columns
    assert "uuid" in processed_df.columns
    assert processed_df.iloc[0]["document_type"] == "type1"
    assert processed_df.iloc[0]["metadata"] == {"key": "value"}
    assert processed_df.iloc[0]["uuid"] == "1234"
    assert trace_info_output["trace_info"] == trace_info


@patch(f'{MODULE_UNDER_TEST}.decode_and_extract')
def test_process_image_empty_dataframe(mock_decode_and_extract):
    mock_decode_and_extract.return_value = []

    input_df = pd.DataFrame(columns=["source_id", "content"])
    task_props = {"method": "some_method"}
    validated_config = MagicMock()
    trace_info = {}

    processed_df, trace_info_output = process_image(input_df, task_props, validated_config, trace_info)

    assert processed_df.empty
    assert "document_type" in processed_df.columns
    assert "metadata" in processed_df.columns
    assert "uuid" in processed_df.columns
    assert trace_info_output["trace_info"] == trace_info


@patch(f'{MODULE_UNDER_TEST}.decode_and_extract')
def test_process_image_multiple_rows(mock_decode_and_extract):
    mock_decode_and_extract.side_effect = [
        [{"document_type": "type1", "metadata": {"key": "value1"}, "uuid": "1234"}],
        [{"document_type": "type2", "metadata": {"key": "value2"}, "uuid": "5678"}]
    ]

    input_df = pd.DataFrame({
        "source_id": [1, 2],
        "content": ["base64encodedstring1", "base64encodedstring2"]
    })

    task_props = {"method": "some_method"}
    validated_config = MagicMock()
    trace_info = {}

    processed_df, trace_info_output = process_image(input_df, task_props, validated_config, trace_info)

    assert len(processed_df) == 2
    assert processed_df.iloc[0]["document_type"] == "type1"
    assert processed_df.iloc[0]["metadata"] == {"key": "value1"}
    assert processed_df.iloc[0]["uuid"] == "1234"
    assert processed_df.iloc[1]["document_type"] == "type2"
    assert processed_df.iloc[1]["metadata"] == {"key": "value2"}
    assert processed_df.iloc[1]["uuid"] == "5678"
    assert trace_info_output["trace_info"] == trace_info


@patch(f'{MODULE_UNDER_TEST}.decode_and_extract')
def test_process_image_with_exception(mock_decode_and_extract):
    mock_decode_and_extract.side_effect = Exception("Decoding error")

    input_df = pd.DataFrame({
        "source_id": [1],
        "content": ["base64encodedstring"]
    })

    task_props = {"method": "some_method"}
    validated_config = MagicMock()
    trace_info = {}

    with pytest.raises(Exception) as excinfo:
        process_image(input_df, task_props, validated_config, trace_info)

    assert "Decoding error" in str(excinfo.value)


@patch(f'{MODULE_UNDER_TEST}.image_helpers')
def test_decode_and_extract_valid_method(mock_image_helpers):
    # Mock the extraction function inside image_helpers
    mock_func = MagicMock(return_value="extracted_data")
    mock_image_helpers.image = mock_func  # Default extraction method

    # Sample inputs as a pandas Series (row)
    base64_content = base64.b64encode(b"dummy_image_data").decode('utf-8')
    base64_row = pd.Series({
        "content": base64_content,
        "document_type": "image",
        "source_id": 1
    })
    task_props = {"method": "image", "params": {}}
    validated_config = MagicMock()
    trace_info = []

    # Call the function
    result = decode_and_extract(base64_row, task_props, validated_config, default="image", trace_info=trace_info)

    # Assert that the mocked function was called correctly
    assert result == "extracted_data"
    mock_func.assert_called_once()


@patch(f'{MODULE_UNDER_TEST}.image_helpers')
def test_decode_and_extract_missing_content_key(mock_image_helpers):
    # Sample inputs with missing 'content' key as a pandas Series (row)
    base64_row = pd.Series({
        "document_type": "image",
        "source_id": 1
    })
    task_props = {"method": "image", "params": {}}
    validated_config = MagicMock()
    trace_info = []

    # Expecting a KeyError
    with pytest.raises(KeyError):
        decode_and_extract(base64_row, task_props, validated_config, trace_info=trace_info)


@patch(f'{MODULE_UNDER_TEST}.image_helpers')
def test_decode_and_extract_fallback_to_default_method(mock_image_helpers):
    # Mock only the default method; other methods will appear as non-existent
    mock_default_func = MagicMock(return_value="default_extracted_data")
    setattr(mock_image_helpers, 'default', mock_default_func)

    # Ensure that non_existing_method does not exist on mock_image_helpers
    if hasattr(mock_image_helpers, "non_existing_method"):
        delattr(mock_image_helpers, "non_existing_method")

    # Input with a non-existing extraction method as a pandas Series (row)
    base64_content = base64.b64encode(b"dummy_image_data").decode('utf-8')
    base64_row = pd.Series({
        "content": base64_content,
        "document_type": "image",
        "source_id": 1
    })
    task_props = {"method": "non_existing_method", "params": {}}
    validated_config = MagicMock()
    trace_info = []

    # Call the function
    result = decode_and_extract(base64_row, task_props, validated_config, default="default", trace_info=trace_info)

    # Assert that the default function was called instead of the missing one
    assert result == "default_extracted_data"
    mock_default_func.assert_called_once()


@patch(f'{MODULE_UNDER_TEST}.image_helpers')
def test_decode_and_extract_with_trace_info(mock_image_helpers):
    # Mock the extraction function with trace_info usage
    mock_func = MagicMock(return_value="extracted_data_with_trace")
    mock_image_helpers.image = mock_func  # Default extraction method

    # Sample inputs with trace_info as a pandas Series (row)
    base64_content = base64.b64encode(b"dummy_image_data").decode('utf-8')
    base64_row = pd.Series({
        "content": base64_content,
        "document_type": "image",
        "source_id": 1
    })
    task_props = {"method": "image", "params": {}}
    validated_config = MagicMock()
    trace_info = [{"some": "trace_info"}]

    # Call the function
    result = decode_and_extract(base64_row, task_props, validated_config, trace_info=trace_info)

    # Assert that the mocked function was called with trace_info in params
    assert result == "extracted_data_with_trace"
    mock_func.assert_called_once()
    _, _, kwargs = mock_func.mock_calls[0]
    assert "trace_info" in kwargs
    assert kwargs["trace_info"] == trace_info


@patch(f'{MODULE_UNDER_TEST}.image_helpers')
def test_decode_and_extract_handles_exception_in_extraction(mock_image_helpers):
    # Mock the extraction function (using a valid method) to raise an exception
    mock_func = MagicMock(side_effect=Exception("Extraction error"))
    mock_image_helpers.image = mock_func  # Use the default method or a valid method

    # Sample inputs as a pandas Series (row)
    base64_content = base64.b64encode(b"dummy_image_data").decode('utf-8')
    base64_row = pd.Series({
        "content": base64_content,
        "document_type": "image",
        "source_id": 1
    })
    task_props = {"method": "image", "params": {}}  # Use a valid method name
    validated_config = MagicMock()
    trace_info = []

    # Expecting an exception during extraction
    with pytest.raises(Exception) as excinfo:
        decode_and_extract(base64_row, task_props, validated_config, trace_info=trace_info)

    # Verify the exception message
    assert str(excinfo.value) == "Extraction error"
