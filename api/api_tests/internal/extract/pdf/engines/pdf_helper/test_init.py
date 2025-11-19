# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from unittest.mock import MagicMock
from unittest.mock import patch

import nv_ingest_api.internal.extract.pdf.engines.pdf_helpers as module_under_test
import pandas as pd
import pytest
from nv_ingest_api.internal.extract.pdf.engines.pdf_helpers import _orchestrate_row_extraction

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def dummy_pdf_stream():
    return io.BytesIO(b"%PDF-1.4 dummy content")


@pytest.fixture
def dummy_pdf_base64():
    # Encode dummy PDF content
    return base64.b64encode(b"%PDF-1.4 dummy content").decode()


@pytest.fixture
def dummy_row(dummy_pdf_base64):
    return pd.Series({"content": dummy_pdf_base64, "source_id": "abc123", "extra_column": "value"})


@pytest.fixture
def dummy_task_config():
    return {
        "params": {
            "extract_text": True,
            "extract_images": False,
            "extract_tables": True,
            "extract_charts": False,
            "extract_infographics": False,
            "extract_page_as_image": False,
            "extract_method": "pdfium",
        },
        "method": "pdfium",
    }


@pytest.fixture
def dummy_extractor_config():
    return {"pdfium_config": {"option": "value"}}


@patch(f"{MODULE_UNDER_TEST}._work_extract_pdf")
def test_orchestrate_row_extraction_happy_path(
    mock_work_extract_pdf, dummy_row, dummy_task_config, dummy_extractor_config
):
    # Mock the internal extraction call
    mock_work_extract_pdf.return_value = [("TEXT", {"text": "example"}, "uuid-1")]

    result = _orchestrate_row_extraction(
        dummy_row,
        dummy_task_config,
        dummy_extractor_config,
    )

    mock_work_extract_pdf.assert_called_once()

    # Check that PDF stream is properly decoded
    _, kwargs = mock_work_extract_pdf.call_args
    pdf_stream_arg = kwargs["pdf_stream"]
    assert isinstance(pdf_stream_arg, io.BytesIO)
    assert pdf_stream_arg.getvalue().startswith(b"%PDF-1.4")

    # Check parameters passed include method-specific config and row metadata
    assert kwargs["extract_text"] is True
    assert kwargs["extract_images"] is False
    assert kwargs["extract_tables"] is True
    assert kwargs["extract_charts"] is False
    assert kwargs["extract_infographics"] is False

    # Check extractor_config includes row_data and pdfium_config
    passed_config = kwargs["extractor_config"]
    assert passed_config["row_data"]["source_id"] == "abc123"
    assert passed_config["row_data"]["extra_column"] == "value"
    assert passed_config["extract_method"] == "pdfium"
    assert passed_config["pdfium_config"]["option"] == "value"

    # Validate the result
    assert result == [("TEXT", {"text": "example"}, "uuid-1")]


def test_orchestrate_row_extraction_missing_content_raises(dummy_task_config, dummy_extractor_config):
    row = pd.Series({"other_key": "value"})
    with pytest.raises(KeyError):
        _orchestrate_row_extraction(row, dummy_task_config, dummy_extractor_config)


@patch(f"{MODULE_UNDER_TEST}._work_extract_pdf")
def test_orchestrate_row_extraction_invalid_base64_raises(
    mock_work_extract_pdf, dummy_task_config, dummy_extractor_config
):
    row = pd.Series({"content": "invalid_base64"})
    with pytest.raises(Exception, match="Error decoding base64 content"):
        _orchestrate_row_extraction(row, dummy_task_config, dummy_extractor_config)


@patch(f"{MODULE_UNDER_TEST}._work_extract_pdf")
def test_orchestrate_row_extraction_missing_extractor_config_key(mock_work_extract_pdf, dummy_row, dummy_task_config):
    dummy_extractor_config = {}  # Intentionally empty to trigger warning

    mock_work_extract_pdf.return_value = []

    result = _orchestrate_row_extraction(
        dummy_row,
        dummy_task_config,
        dummy_extractor_config,
    )

    # Should still call _work_extract_pdf and include extractor_config with row_data and extract_method only
    args, kwargs = mock_work_extract_pdf.call_args
    passed_config = kwargs["extractor_config"]
    assert "pdfium_config" not in passed_config
    assert passed_config["extract_method"] == "pdfium"
    assert "row_data" in passed_config

    assert result == []


@pytest.mark.parametrize(
    "method_name",
    [
        "adobe",
        "llama",
        "nemoretriever_parse",
        "pdfium",
        "tika",
        "unstructured_io",
    ],
)
def test_work_extract_pdf_dispatches_to_correct_extractor(dummy_pdf_stream, method_name):
    # Arrange a MagicMock extractor
    mock_extractor = MagicMock(return_value="mock_result")

    # Patch the entire lookup dict explicitly, using clear=True
    with patch.dict(f"{MODULE_UNDER_TEST}.EXTRACTOR_LOOKUP", {method_name: mock_extractor}, clear=True):
        extractor_config = {"extract_method": method_name, "custom_param": "value"}

        # Act
        result = module_under_test._work_extract_pdf(
            pdf_stream=dummy_pdf_stream,
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config=extractor_config,
            execution_trace_log={"step": "test"},
        )

        # Assert
        mock_extractor.assert_called_once_with(
            pdf_stream=dummy_pdf_stream,
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extractor_config=extractor_config,
            execution_trace_log={"step": "test"},
        )
        assert result == "mock_result"


@patch(f"{MODULE_UNDER_TEST}.pdfium_extractor", autospec=True)
def test_work_extract_pdf_defaults_to_pdfium(mock_pdfium_extractor, dummy_pdf_stream):
    # Arrange
    mock_pdfium_extractor.return_value = "default_pdfium_result"

    # No valid method triggers fallback to pdfium_extractor
    extractor_config = {"extract_method": "unknown_method"}

    # Act
    result = module_under_test._work_extract_pdf(
        pdf_stream=dummy_pdf_stream,
        extract_text=False,
        extract_images=False,
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extract_page_as_image=False,
        extractor_config=extractor_config,
        execution_trace_log=None,
    )

    # Assert
    mock_pdfium_extractor.assert_called_once_with(
        pdf_stream=dummy_pdf_stream,
        extract_text=False,
        extract_images=False,
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extract_page_as_image=False,
        extractor_config=extractor_config,
        execution_trace_log=None,
    )
    assert result == "default_pdfium_result"
