# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from unittest.mock import patch

import nv_ingest_api.internal.extract.pdf.pdf_extractor as module_under_test
from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@patch(f"{MODULE_UNDER_TEST}._orchestrate_row_extraction")
def test_extract_primitives_from_pdf_internal_empty_dataframe(mock_orchestrate):
    df_input = pd.DataFrame(columns=["content"])

    result_df, trace = extract_primitives_from_pdf_internal(
        df_input,
        {"task_properties": {}},
        {"param": "value"},
    )

    # It will be called once with a row containing NaN
    assert mock_orchestrate.call_count == 1
    assert mock_orchestrate.call_args[0][0].isna().all()
    assert result_df.empty


@patch(f"{MODULE_UNDER_TEST}._orchestrate_row_extraction")
def test_extract_primitives_from_pdf_internal_error_bubbling(mock_orchestrate):
    df_input = pd.DataFrame(
        [
            {"content": "dummy_base64_pdf_1"},
        ]
    )

    mock_orchestrate.side_effect = RuntimeError("Mock failure")

    with pytest.raises(RuntimeError, match="extract_primitives_from_pdf: Error processing PDF bytes: Mock failure"):
        extract_primitives_from_pdf_internal(
            df_input,
            {"task_properties": {}},
            {"param": "value"},
        )


@patch(f"{MODULE_UNDER_TEST}._orchestrate_row_extraction")
def test_extract_primitives_from_pdf_internal_happy_path(mock_orchestrate):
    df_input = pd.DataFrame(
        [
            {"content": "dummy_base64_pdf_1"},
            {"content": "dummy_base64_pdf_2"},
        ]
    ).reset_index(drop=True)

    mock_orchestrate.side_effect = [
        [("TEXT", {"text": "doc1"}, "uuid-1"), ("IMAGE", {"image": "img1"}, "uuid-2")],
        [("TEXT", {"text": "doc2"}, "uuid-3")],
    ]

    dummy_task_config = {"task_properties": {}, "validated_config": {}}
    dummy_extractor_config = {"param": "value"}

    result_df, trace = extract_primitives_from_pdf_internal(
        df_input,
        dummy_task_config,
        dummy_extractor_config,
    )

    assert mock_orchestrate.call_count == 2

    expected = pd.DataFrame(
        [
            {"document_type": "TEXT", "metadata": {"text": "doc1"}, "uuid": "uuid-1"},
            {"document_type": "IMAGE", "metadata": {"image": "img1"}, "uuid": "uuid-2"},
            {"document_type": "TEXT", "metadata": {"text": "doc2"}, "uuid": "uuid-3"},
        ]
    )

    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected)

    assert "execution_trace_log" in trace
