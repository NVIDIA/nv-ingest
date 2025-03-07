# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import logging
import os

import pandas as pd
import pytest

from api.tests.utilities_for_test import find_root_by_pattern, get_git_root
from nv_ingest_api.interface.extract import (
    extract_infographic_data_from_image,
    extract_table_data_from_image,
    extract_chart_data_from_image,
)
from nv_ingest_api.internal.enums.common import ContentTypeEnum, DocumentTypeEnum, TableFormatEnum

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_extract_infographic_data_from_image_integration():
    """
    Integration test for the extract_infographic_data_from_image function.

    This test verifies that the infographic extraction pipeline correctly processes
    image data and augments the DataFrame with extracted information.
    """
    # Build a sample ledger DataFrame with the required columns for image processing
    df_ledger = pd.DataFrame(
        {
            "source_name": ["./data/infographic_test.png"],
            "source_id": ["./data/infographic_test.png"],
            "content": ["ZmFrZV9pbWFnZV9jb250ZW50"],  # dummy base64 encoded image
            "document_type": [DocumentTypeEnum.PNG],  # Using PNG for the image document type
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": None,
                    "content_metadata": {"type": ContentTypeEnum.INFOGRAPHIC},  # Using INFOGRAPHIC content type
                    "error_metadata": None,
                    "image_metadata": {
                        "height": 800,
                        "width": 600,
                        "format": "png",
                    },
                    "source_metadata": {
                        "source_id": "./data/infographic_test.png",
                        "source_name": "./data/infographic_test.png",
                        "source_type": "png",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment first
    _PADDLE_GRPC_ENDPOINT = os.getenv("INGEST_PADDLE_GRPC_ENDPOINT", None)
    _PADDLE_HTTP_ENDPOINT = os.getenv("INGEST_PADDLE_HTTP_ENDPOINT", "http://127.0.0.1:8010")
    _PADDLE_PROTOCOL = os.getenv("INGEST_PADDLE_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the paddle endpoints tuple
    paddle_endpoints = (_PADDLE_GRPC_ENDPOINT, _PADDLE_HTTP_ENDPOINT)

    # Explicitly map the schema values to the function's expected arguments
    integration_args = {
        "paddle_endpoints": paddle_endpoints,
        "paddle_protocol": _PADDLE_PROTOCOL,
        "auth_token": _AUTH_TOKEN,
    }

    # Call the function under test with the constructed parameters
    df_result = extract_infographic_data_from_image(df_ledger=df_ledger, **integration_args)

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty."

    # Check that we have the same structure but with updated metadata
    assert list(df_result.columns) == [
        "source_name",
        "source_id",
        "content",
        "document_type",
        "metadata",
    ], "DataFrame column structure should remain unchanged"

    # Verify that each row's metadata contains the expected infographic extraction results
    for idx, row in df_result.iterrows():
        metadata = row.get("metadata", {})

        # Verify infographic data was processed and added to metadata
        # This is a generic test that can be adjusted based on the actual implementation
        assert metadata is not None, f"Row {idx} has None metadata"

        # Check that there's some evidence of infographic processing in the metadata
        # The specific structure depends on the implementation
        content_metadata = metadata.get("content_metadata", {})
        assert content_metadata is not None, f"Row {idx} missing content_metadata"

        # Check that the content type is still infographic (or was updated appropriately)
        assert content_metadata.get("type") is not None, f"Row {idx} content_metadata missing 'type' field"


@pytest.mark.integration
def test_extract_table_data_from_image_integration():
    """
    Integration test for the extract_table_data_from_image function.

    This test verifies that the table extraction pipeline correctly processes
    image data and updates the DataFrame with extracted table content.
    """
    # Get the test file path from environment or find it using helper functions
    test_file_rel_path = "./data/table.png"
    test_file_path = os.getenv("INGEST_TABLE_TEST_FILE")

    if not test_file_path:
        # Try to find the file using git root first
        git_root = get_git_root(__file__)
        if git_root:
            test_file_path = os.path.join(git_root, test_file_rel_path)

        # If not found via git, try heuristic approach
        if not test_file_path or not os.path.exists(test_file_path):
            root_dir = find_root_by_pattern(test_file_rel_path, os.path.dirname(__file__))
            if root_dir:
                test_file_path = os.path.join(root_dir, test_file_rel_path)
            else:
                # Fallback to relative path if all else fails
                test_file_path = test_file_rel_path

    # Ensure the file exists
    assert os.path.exists(test_file_path), f"Test file not found at {test_file_path}"

    # Read the file and encode it as base64
    with open(test_file_path, "rb") as f:
        file_content = f.read()
        base64_content = base64.b64encode(file_content).decode("utf-8")

    # Build a sample ledger DataFrame with the required columns for table extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],  # Actual base64 encoded image
            "document_type": [DocumentTypeEnum.PNG],  # Using PNG for the image document type
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": base64_content,  # Need content in metadata for table extraction
                    "content_metadata": {
                        "type": "structured",  # Must be structured
                        "subtype": "table",  # Must have table subtype
                    },
                    "error_metadata": None,
                    "image_metadata": {
                        "height": 800,
                        "width": 600,
                        "format": "png",
                    },
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "png",
                    },
                    "table_metadata": {  # Must have table_metadata
                        "table_content_format": TableFormatEnum.PSEUDO_MARKDOWN  # Using PSEUDO_MARKDOWN format
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8001/v1/infer")
    _YOLOX_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")

    _PADDLE_GRPC_ENDPOINT = os.getenv("INGEST_PADDLE_GRPC_ENDPOINT", None)
    _PADDLE_HTTP_ENDPOINT = os.getenv("INGEST_PADDLE_HTTP_ENDPOINT", "http://127.0.0.1:8009/v1/infer")
    _PADDLE_PROTOCOL = os.getenv("INGEST_PADDLE_PROTOCOL", "http")

    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuples
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)
    paddle_endpoints = (_PADDLE_GRPC_ENDPOINT, _PADDLE_HTTP_ENDPOINT)

    # Explicitly map the values to the function's expected arguments
    integration_args = {
        "yolox_endpoints": yolox_endpoints,
        "paddle_endpoints": paddle_endpoints,
        "yolox_protocol": _YOLOX_PROTOCOL,
        "paddle_protocol": _PADDLE_PROTOCOL,
        "auth_token": _AUTH_TOKEN,
    }

    # Call the function under test with the constructed parameters
    df_result = extract_table_data_from_image(df_ledger=df_ledger, **integration_args)

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty."

    # Check that we have the same DataFrame structure
    assert list(df_result.columns) == [
        "source_name",
        "source_id",
        "content",
        "document_type",
        "metadata",
    ], "DataFrame column structure should remain unchanged"

    # Verify that each row's metadata contains the expected table extraction results
    for idx, row in df_result.iterrows():
        metadata = row.get("metadata", {})

        # Verify metadata exists
        assert metadata is not None, f"Row {idx} has None metadata"

        # Check for table_metadata
        table_metadata = metadata.get("table_metadata", {})
        assert table_metadata is not None, f"Row {idx} missing table_metadata"

        # Check that table_content was extracted and added to table_metadata
        assert "table_content" in table_metadata, f"Row {idx} table_metadata missing 'table_content' field"

        # Verify the table_content_format matches what we expected
        assert (
            table_metadata.get("table_content_format") == TableFormatEnum.PSEUDO_MARKDOWN
        ), f"Row {idx} table_content_format does not match expected format"

        # Check that table_content is not empty
        assert table_metadata.get("table_content").strip(), f"Row {idx} table_content should not be empty"


@pytest.mark.integration
def test_extract_chart_data_from_image_integration():
    """
    Integration test for the extract_chart_data_from_image function.

    This test verifies that the chart extraction pipeline correctly processes
    image data and updates the DataFrame with extracted chart content.
    """
    # At the beginning of the test, import necessary modules

    # Get the test file path from environment or find it using helper functions
    test_file_rel_path = "./data/chart.png"
    test_file_path = os.getenv("INGEST_CHART_TEST_FILE")

    if not test_file_path:
        # Try to find the file using git root first
        git_root = get_git_root(__file__)
        if git_root:
            test_file_path = os.path.join(git_root, test_file_rel_path)

        # If not found via git, try heuristic approach
        if not test_file_path or not os.path.exists(test_file_path):
            root_dir = find_root_by_pattern(test_file_rel_path, os.path.dirname(__file__))
            if root_dir:
                test_file_path = os.path.join(root_dir, test_file_rel_path)
            else:
                # Fallback to relative path if all else fails
                test_file_path = test_file_rel_path

    # Ensure the file exists
    assert os.path.exists(test_file_path), f"Test file not found at {test_file_path}"

    # Read the file and encode it as base64
    with open(test_file_path, "rb") as f:
        file_content = f.read()
        base64_content = base64.b64encode(file_content).decode("utf-8")

    # Build a sample ledger DataFrame with the required columns for chart extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],  # Actual base64 encoded image
            "document_type": [DocumentTypeEnum.PNG],  # Using PNG for the image document type
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": base64_content,  # Need content in metadata for chart extraction
                    "content_metadata": {
                        "type": "structured",  # Must be structured
                        "subtype": "chart",  # Subtype is chart for chart extraction
                    },
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "png",
                    },
                    "table_metadata": {  # Use table_metadata even for charts
                        "table_content_format": TableFormatEnum.PSEUDO_MARKDOWN  # Format for extracted data
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8003/v1/infer")
    _YOLOX_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")

    _PADDLE_GRPC_ENDPOINT = os.getenv("INGEST_PADDLE_GRPC_ENDPOINT", None)
    _PADDLE_HTTP_ENDPOINT = os.getenv("INGEST_PADDLE_HTTP_ENDPOINT", "http://127.0.0.1:8009/v1/infer")
    _PADDLE_PROTOCOL = os.getenv("INGEST_PADDLE_PROTOCOL", "http")

    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuples
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)
    paddle_endpoints = (_PADDLE_GRPC_ENDPOINT, _PADDLE_HTTP_ENDPOINT)

    # Explicitly map the values to the function's expected arguments
    integration_args = {
        "yolox_endpoints": yolox_endpoints,
        "paddle_endpoints": paddle_endpoints,
        "yolox_protocol": _YOLOX_PROTOCOL,
        "paddle_protocol": _PADDLE_PROTOCOL,
        "auth_token": _AUTH_TOKEN,
    }

    # Call the function under test with the constructed parameters
    df_result = extract_chart_data_from_image(df_ledger=df_ledger, **integration_args)

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty."

    # Check that we have the same DataFrame structure
    assert list(df_result.columns) == [
        "source_name",
        "source_id",
        "content",
        "document_type",
        "metadata",
    ], "DataFrame column structure should remain unchanged"

    df_result.to_json("chart_extraction_results.json")
    # Verify that each row's metadata contains the expected chart extraction results
    for idx, row in df_result.iterrows():
        metadata = row.get("metadata", {})

        # Verify metadata exists
        assert metadata is not None, f"Row {idx} has None metadata"

        # Check for chart_metadata
        chart_metadata = metadata.get("table_metadata", {})
        assert chart_metadata is not None, f"Row {idx} missing table_metadata"

        # Check that chart data was extracted
        # The specific structure depends on the implementation, but at minimum
        # we expect some data to be extracted and added to chart_metadata
        assert len(chart_metadata) > 1, f"Row {idx} chart_metadata should contain extraction results"
