# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import pandas as pd
from unittest.mock import patch, MagicMock

import pytest
from pydantic import BaseModel, Field

import nv_ingest_api.internal.extract.pptx.pptx_extractor as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def dummy_row():
    pptx_content = base64.b64encode(b"dummy_pptx_content").decode()
    return pd.Series(
        {
            "content": pptx_content,
            "source_id": "pptx123",
            "extra_field": "extra_value",
        }
    )


@pytest.fixture
def dummy_task_props():
    return {
        "params": {
            "extract_text": True,
            "extract_images": True,
            "extract_tables": False,
            "extract_charts": True,
            "extract_infographics": False,
        }
    }


@pytest.fixture
def dummy_extraction_config():
    return MagicMock(pptx_extraction_config={"some": "config"})


@pytest.fixture
def dummy_trace_info():
    return {"trace": "info"}


@patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_pptx")
def test_extract_primitives_from_pptx_internal_happy_path(mock_decode_extract):
    # Arrange dummy input DataFrame with 2 rows
    df_input = pd.DataFrame(
        [
            {"content": "pptx_base64_1", "source_id": "src_1"},
            {"content": "pptx_base64_2", "source_id": "src_2"},
        ]
    )

    # Mock return: list of tuples per row
    mock_decode_extract.side_effect = [
        [("TEXT", {"text": "slide1"}, "uuid-1"), ("IMAGE", {"image": "img1"}, "uuid-2")],
        [("TEXT", {"text": "slide2"}, "uuid-3")],
    ]

    dummy_task_config = {"dummy_key": "dummy_val"}
    dummy_extraction_config = {"param": "value"}

    # Act
    result_df, trace = module_under_test.extract_primitives_from_pptx_internal(
        df_input,
        dummy_task_config,
        dummy_extraction_config,
    )

    # Assert function called per row
    assert mock_decode_extract.call_count == 2

    # Validate result DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape[0] == 3  # 2 + 1
    assert set(result_df.columns) == {"document_type", "metadata", "uuid"}

    # Validate contents
    assert (result_df["uuid"] == ["uuid-1", "uuid-2", "uuid-3"]).all()


@patch(f"{MODULE_UNDER_TEST}._decode_and_extract_from_pptx")
def test_extract_primitives_from_pptx_internal_empty_result(mock_decode_extract):
    # Arrange dummy input DataFrame
    df_input = pd.DataFrame(
        [
            {"content": "pptx_base64_1", "source_id": "src_1"},
        ]
    )

    # Mock returns empty list
    mock_decode_extract.return_value = []

    dummy_task_config = {"dummy_key": "dummy_val"}
    dummy_extraction_config = {"param": "value"}

    # Act
    result_df, trace = module_under_test.extract_primitives_from_pptx_internal(
        df_input,
        dummy_task_config,
        dummy_extraction_config,
    )

    # Assert
    assert mock_decode_extract.call_count == 1
    assert result_df.empty
    assert set(result_df.columns) == {"document_type", "metadata", "uuid"}


def test_prepare_task_properties_dict_input():
    row = pd.Series(
        {
            "content": "base64string",
            "source_id": "abc123",
            "extra_field": "extra_value",
        }
    )

    task_props = {"params": {"existing": "yes"}}

    result_task_props, source_id = module_under_test._prepare_task_properties(row, task_props)

    # Validate 'content' was dropped
    assert "row_data" in result_task_props["params"]
    assert "content" not in result_task_props["params"]["row_data"]
    assert result_task_props["params"]["row_data"]["extra_field"] == "extra_value"
    # Validate existing keys preserved
    assert result_task_props["params"]["existing"] == "yes"
    # Validate source_id
    assert source_id == "abc123"


def test_prepare_task_properties_pydantic_input():
    class DummyModel(BaseModel):
        params: dict = Field(default_factory=dict)

    row = pd.Series({"content": "base64string", "other": "data"})

    task_props = DummyModel()

    result_task_props, source_id = module_under_test._prepare_task_properties(row, task_props)

    # Validate 'content' was dropped
    assert "row_data" in result_task_props["params"]
    assert "content" not in result_task_props["params"]["row_data"]
    assert result_task_props["params"]["row_data"]["other"] == "data"
    # Validate no source_id
    assert source_id is None


def test_prepare_task_properties_missing_params():
    row = pd.Series({"content": "base64string", "field": "data"})

    task_props = {}  # Missing 'params' key entirely

    result_task_props, source_id = module_under_test._prepare_task_properties(row, task_props)

    assert "params" in result_task_props
    assert "row_data" in result_task_props["params"]
    assert result_task_props["params"]["row_data"]["field"] == "data"
    assert source_id is None


@patch(f"{MODULE_UNDER_TEST}._prepare_task_properties")
@patch(f"{MODULE_UNDER_TEST}.python_pptx")
def test_decode_and_extract_from_pptx_happy_path(
    mock_python_pptx, mock_prepare, dummy_row, dummy_task_props, dummy_extraction_config, dummy_trace_info
):
    # Setup mocks
    mock_prepare.return_value = (dummy_task_props, "pptx123")
    mock_python_pptx.return_value = [("TEXT", {"some": "meta"}, "uuid-1")]

    # Call the function
    result = module_under_test._decode_and_extract_from_pptx(
        dummy_row,
        dummy_task_props,
        dummy_extraction_config,
        dummy_trace_info,
    )

    # Validate call to _prepare_task_properties
    mock_prepare.assert_called_once_with(dummy_row, dummy_task_props)

    # Validate python_pptx call params
    args, kwargs = mock_python_pptx.call_args
    pptx_stream_arg = kwargs["pptx_stream"]
    assert isinstance(pptx_stream_arg, io.BytesIO)
    assert pptx_stream_arg.getvalue() == b"dummy_pptx_content"

    # Validate other params passed correctly after pop
    assert kwargs["extract_text"] is True
    assert kwargs["extract_images"] is True
    assert kwargs["extract_tables"] is False
    assert kwargs["extract_charts"] is True
    assert kwargs["extract_infographics"] is False

    # Ensure extracted config and trace_info were injected
    assert kwargs["extraction_config"]["pptx_extraction_config"] == dummy_extraction_config.pptx_extraction_config
    assert kwargs["extraction_config"]["trace_info"] == dummy_trace_info

    # Validate result propagation
    assert result == [("TEXT", {"some": "meta"}, "uuid-1")]
