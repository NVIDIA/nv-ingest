from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nv_ingest_api.internal.extract.pdf.engines.nemoretriever import nemoretriever_parse_extractor


_MODULE_UNDER_TEST = "nv_ingest_api.util.nim"


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


@pytest.fixture
def mock_parser_config():
    return {
        "nemoretriever_parse_endpoints": ("parser:8001", "http://parser:8000"),
    }


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_nemoretriever_parse_text_extraction(mock_client, sample_pdf_stream, document_df, mock_parser_config):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = [
        [
            {
                "bbox": {"xmin": 0.16633729456384325, "ymin": 0.0969, "xmax": 0.3097820480404551, "ymax": 0.1102},
                "text": "testing",
                "type": "Text",
            }
        ]
    ]

    result = nemoretriever_parse_extractor(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        extract_tables_method="nemoretriever_parse",
        nemoretriever_parse_config=mock_parser_config,
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_nemoretriever_parse_table_extraction(mock_client, sample_pdf_stream, document_df, mock_parser_config):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = [
        [
            {
                "bbox": {"xmin": 1 / 1024, "ymin": 2 / 1280, "xmax": 101 / 1024, "ymax": 102 / 1280},
                "text": "table text",
                "type": "Table",
            }
        ]
    ]

    result = nemoretriever_parse_extractor(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        extract_tables_method="nemoretriever_parse",
        nemoretriever_parse_config=mock_parser_config,
    )

    assert len(result) == 2
    assert result[0][0].value == "structured"
    assert result[0][1]["table_metadata"]["table_content"] == "table text"
    assert result[0][1]["table_metadata"]["table_location"] == (1, 2, 101, 102)
    assert result[0][1]["table_metadata"]["table_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_nemoretriever_parse_image_extraction(mock_client, sample_pdf_stream, document_df, mock_parser_config):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = [
        [
            {
                "bbox": {"xmin": 1 / 1024, "ymin": 2 / 1280, "xmax": 101 / 1024, "ymax": 102 / 1280},
                "text": "",
                "type": "Picture",
            }
        ]
    ]

    result = nemoretriever_parse_extractor(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        extract_tables_method="nemoretriever_parse",
        nemoretriever_parse_config=mock_parser_config,
    )

    assert len(result) == 2
    assert result[0][0].value == "image"
    assert result[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
    assert result[0][1]["image_metadata"]["image_location"] == (1, 2, 101, 102)
    assert result[0][1]["image_metadata"]["image_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
def test_nemoretriever_parse_text_extraction_bboxes(mock_client, sample_pdf_stream, document_df, mock_parser_config):
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.infer.return_value = [
        [
            {
                "bbox": {"xmin": 0.16633729456384325, "ymin": 0.0969, "xmax": 0.3097820480404551, "ymax": 0.1102},
                "text": "testing0",
                "type": "Title",
            },
            {
                "bbox": {"xmin": 0.16633729456384325, "ymin": 0.0969, "xmax": 0.3097820480404551, "ymax": 0.1102},
                "text": "testing1",
                "type": "Text",
            },
        ]
    ]

    result = nemoretriever_parse_extractor(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        extract_tables_method="nemoretriever_parse",
        nemoretriever_parse_config=mock_parser_config,
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing0\n\ntesting1"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"

    blocks = result[0][1]["content_metadata"]["hierarchy"]["nearby_objects"]
    assert blocks["text"]["content"] == ["testing0", "testing1"]
    assert blocks["text"]["type"] == ["Title", "Text"]
