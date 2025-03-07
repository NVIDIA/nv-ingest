import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nv_ingest_api.internal.extract.pdf.engines import nemoretriever_parse_extractor
from tests.utilities_for_test import get_git_root, find_root_by_pattern

import nv_ingest_api.internal.extract.pdf.engines.nemoretriever as module_under_test

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame({"source_id": ["source1"]})


@pytest.fixture
def sample_pdf_stream():
    # Attempt to get the git root based on the current file location.
    git_root = get_git_root(__file__)
    if git_root is None:
        git_root = find_root_by_pattern("data/test.pdf")

    if git_root is None:
        # Fallback to the directory of the current file if not in a git repo.
        git_root = os.path.dirname(os.path.abspath(__file__))

    # Build the path to the PDF file using the git root.
    pdf_path = os.path.join(git_root, "data", "test.pdf")
    with open(pdf_path, "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@pytest.fixture
def mock_parser_config():
    return {
        "nemoretriever_parse_endpoints": ("parser:8001", "http://parser:8000"),
    }


@patch(f"{MODULE_UNDER_TEST}._create_clients")
def test_nemoretriever_parse_text_extraction(mock_create_clients, sample_pdf_stream, document_df, mock_parser_config):
    mock_client_instance = MagicMock()
    mock_create_clients.return_value = mock_client_instance
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
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extractor_config={
            "row_data": document_df.iloc[0],
            "text_depth": "page",
            "extract_tables_method": "nemoretriever_parse",
            "nemoretriever_parse_config": mock_parser_config,
        },
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
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
        extract_infographics=False,
        extract_tables=True,
        extract_charts=False,
        extractor_config={
            "row_data": document_df.iloc[0],
            "text_depth": "page",
            "extract_tables_method": "nemoretriever_parse",
            "nemoretriever_parse_config": mock_parser_config,
        },
    )

    assert len(result) == 2
    assert result[0][0].value == "structured"
    assert result[0][1]["table_metadata"]["table_content"] == "table text"
    assert result[0][1]["table_metadata"]["table_location"] == (1, 2, 101, 102)
    assert result[0][1]["table_metadata"]["table_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
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
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extractor_config={
            "row_data": document_df.iloc[0],
            "text_depth": "page",
            "extract_tables_method": "nemoretriever_parse",
            "nemoretriever_parse_config": mock_parser_config,
        },
    )

    assert len(result) == 2
    assert result[0][0].value == "image"
    # Check for PNG header in base64 (first 10 characters)
    assert result[0][1]["content"][:10] == "iVBORw0KGg"
    assert result[0][1]["image_metadata"]["image_location"] == (1, 2, 101, 102)
    assert result[0][1]["image_metadata"]["image_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
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
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extractor_config={
            "row_data": document_df.iloc[0],
            "text_depth": "page",
            "extract_tables_method": "nemoretriever_parse",
            "nemoretriever_parse_config": mock_parser_config,
            "identify_nearby_objects": True,
        },
    )

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing0\n\ntesting1"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"

    blocks = result[0][1]["content_metadata"]["hierarchy"]["nearby_objects"]
    assert blocks["text"]["content"] == ["testing0", "testing1"]
    assert blocks["text"]["type"] == ["Title", "Text"]
