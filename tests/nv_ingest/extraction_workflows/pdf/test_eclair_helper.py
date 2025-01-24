from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nv_ingest.extraction_workflows.pdf.eclair_helper import _construct_table_metadata
from nv_ingest.extraction_workflows.pdf.eclair_helper import eclair
from nv_ingest.extraction_workflows.pdf.eclair_helper import preprocess_and_send_requests
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.util.nim import eclair as eclair_utils
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import LatexTable

_MODULE_UNDER_TEST = "nv_ingest.extraction_workflows.pdf.eclair_helper"


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
        }
    )


@pytest.fixture
def sample_pdf_stream():
    with open("data/test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_eclair_text_extraction(mock_client, sample_pdf_stream, document_df):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = "<x_0><y_1>testing<x_10><y_20><class_Text>"

    result = eclair(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        eclair_config=MagicMock(eclair_batch_size=1),
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_eclair_table_extraction(mock_client, sample_pdf_stream, document_df):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = "<x_17><y_0>table text<x_1007><y_1280><class_Table>"

    result = eclair(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        row_data=document_df.iloc[0],
        text_depth="page",
        eclair_config=MagicMock(eclair_batch_size=1),
    )

    assert len(result) == 2
    assert result[0][0].value == "structured"
    assert result[0][1]["content"] == "table text"
    assert result[0][1]["table_metadata"]["table_location"] == (0, 0, 1024, 1280)
    assert result[0][1]["table_metadata"]["table_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"
    assert result[1][1]["content"] == ""


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_eclair_image_extraction(mock_client, sample_pdf_stream, document_df):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = "<x_17><y_0><x_1007><y_1280><class_Picture>"

    result = eclair(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        eclair_config=MagicMock(eclair_batch_size=1),
    )

    assert len(result) == 2
    assert result[0][0].value == "image"
    assert result[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
    assert result[0][1]["image_metadata"]["image_location"] == (0, 0, 1024, 1280)
    assert result[0][1]["image_metadata"]["image_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"
    assert result[1][1]["content"] == ""


@patch(f"{_MODULE_UNDER_TEST}.pdfium_pages_to_numpy")
def test_preprocess_and_send_requests(mock_pdfium_pages_to_numpy):
    mock_pdfium_pages_to_numpy.return_value = (np.array([[1], [2], [3]]), [0, 1, 2])

    mock_client = MagicMock()
    batch = [MagicMock()] * 3
    batch_offset = 0

    result = preprocess_and_send_requests(mock_client, batch, batch_offset)

    assert len(result) == 3, "Result should have 3 entries"
    assert all(
        isinstance(item, tuple) and len(item) == 3 for item in result
    ), "Each entry should be a tuple with 3 items"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_eclair_text_extraction_bboxes(mock_client, sample_pdf_stream, document_df):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = (
        "<x_0><y_1>testing0<x_10><y_20><class_Title><x_30><y_40>testing1<x_50><y_60><class_Text>"
    )

    result = eclair(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        eclair_config=MagicMock(eclair_batch_size=1),
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing0\n\ntesting1"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"

    blocks = result[0][1]["content_metadata"]["hierarchy"]["nearby_objects"]
    assert blocks["text"]["content"] == ["testing0", "testing1"]
    assert blocks["text"]["type"] == ["Title", "Text"]
