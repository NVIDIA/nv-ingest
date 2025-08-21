# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import logging
import os

import pandas as pd
import pytest

from .. import get_project_root, find_root_by_pattern
from nv_ingest_api.interface.extract import (
    extract_infographic_data_from_image,
    extract_table_data_from_image,
    extract_chart_data_from_image,
    extract_primitives_from_image,
    extract_primitives_from_docx,
    extract_primitives_from_pptx,
    extract_primitives_from_audio,
    extract_primitives_from_pdf,
    extract_primitives_from_pdf_pdfium,
    extract_primitives_from_pdf_nemoretriever_parse,
)
from nv_ingest_api.internal.enums.common import ContentTypeEnum, DocumentTypeEnum, TableFormatEnum

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.parametrize(
    "extract_method",
    [
        "pdfium",
        "nemoretriever_parse",
        pytest.param("adobe", marks=pytest.mark.xfail(reason="Adobe extraction not configured in test environment")),
        pytest.param("llama", marks=pytest.mark.xfail(reason="Llama extraction not configured in test environment")),
        pytest.param(
            "unstructured_io",
            marks=pytest.mark.xfail(reason="Unstructured.io extraction not configured in test environment"),
        ),
        pytest.param("tika", marks=pytest.mark.xfail(reason="Tika extraction not configured in test environment")),
    ],
)
def test_extract_primitives_from_pdf_integration(extract_method):
    """
    Integration test for the extract_primitives_from_pdf function.

    This test verifies that the PDF primitive extraction pipeline correctly processes
    a test PDF document and returns a DataFrame with extracted primitives. The test
    is parameterized to test multiple extraction methods, with some methods marked as
    expected to fail in standard test environments.

    Parameters
    ----------
    extract_method : str
        The PDF extraction method to test (e.g., "pdfium", "adobe", "llama")
    """
    # Get the test file path using helper functions
    test_file_rel_path = "./data/multimodal_test.pdf"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame matching the expected structure for PDF extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.PDF],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "document", "subtype": "pdf"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "pdf",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Retrieve environment variables for all possible extraction methods
    # YOLOX parameters (used by multiple methods)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_INFER_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # NemoRetriever Parse parameters
    _NEMO_RETRIEVER_PARSE_HTTP_ENDPOINT = os.getenv(
        "INGEST_NEMO_RETRIEVER_PARSE_HTTP_ENDPOINT", "http://localhost:8000/v1/chat/completions:8015"
    )
    _NEMO_RETRIEVER_PARSE_GRPC_ENDPOINT = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_GRPC_ENDPOINT", None)
    _NEMO_RETRIEVER_PARSE_PROTOCOL = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_PROTOCOL", "http")
    _NEMO_RETRIEVER_PARSE_MODEL_NAME = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    # Method-specific configuration parameters
    extraction_params = {
        "df_extraction_ledger": df_ledger,
        "extract_method": extract_method,
        "extract_text": True,
        "extract_images": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_infographics": True,
        "text_depth": "page",
    }

    # Add method-specific parameters
    if extract_method == "pdfium" or extract_method == "adobe" or extract_method == "tika":
        extraction_params.update(
            {
                "yolox_endpoints": (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT),
                "yolox_infer_protocol": _YOLOX_INFER_PROTOCOL,
                "yolox_auth_token": _AUTH_TOKEN,
            }
        )
    elif extract_method == "llama":
        extraction_params.update(
            {
                "llama_api_key": os.getenv("INGEST_LLAMA_API_KEY", "dummy-api-key"),
                "yolox_endpoints": (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT),
                "yolox_infer_protocol": _YOLOX_INFER_PROTOCOL,
                "yolox_auth_token": _AUTH_TOKEN,
            }
        )
    elif extract_method == "unstructured_io":
        extraction_params.update(
            {
                "unstructured_io_api_key": os.getenv("INGEST_UNSTRUCTURED_IO_API_KEY", "dummy-api-key"),
                "yolox_endpoints": (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT),
                "yolox_infer_protocol": _YOLOX_INFER_PROTOCOL,
                "yolox_auth_token": _AUTH_TOKEN,
            }
        )
    elif extract_method == "nemoretriever_parse":
        extraction_params.update(
            {
                # NemoRetriever Parse specific parameters
                "nemoretriever_parse_endpoints": (
                    _NEMO_RETRIEVER_PARSE_GRPC_ENDPOINT,
                    _NEMO_RETRIEVER_PARSE_HTTP_ENDPOINT,
                ),
                "nemoretriever_parse_protocol": _NEMO_RETRIEVER_PARSE_PROTOCOL,
                "nemoretriever_parse_model_name": _NEMO_RETRIEVER_PARSE_MODEL_NAME,
                # Also include YOLOX parameters for image processing capability
                "yolox_endpoints": (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT),
                "yolox_infer_protocol": _YOLOX_INFER_PROTOCOL,
                "yolox_auth_token": _AUTH_TOKEN,
            }
        )

    # Call the high-level function with appropriate parameters
    df_result = extract_primitives_from_pdf(**extraction_params)

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check that we have the expected DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Basic validation of results
    # Check that we have at least some rows in the result
    assert len(df_result) > 0, "Expected at least some rows in the result"

    # Verify that each row has the required fields
    for idx, row in df_result.iterrows():
        # Verify document_type is present
        assert row["document_type"] is not None, f"Row {idx} has None document_type"

        # Verify metadata exists
        assert row["metadata"] is not None, f"Row {idx} has None metadata"

        # Verify UUID is present
        assert row["uuid"] is not None, f"Row {idx} has None UUID"
        assert isinstance(row["uuid"], str), f"Row {idx} UUID should be a string"

    # Verify that at least one expected document type is present
    expected_doc_types = ["text", "structured", "image"]
    found_expected_type = False

    for doc_type in df_result["document_type"].unique():
        if doc_type in expected_doc_types:
            found_expected_type = True
            break

    assert found_expected_type, f"Expected at least one of these document types: {expected_doc_types}"


@pytest.mark.integration
def test_extract_pdf_with_pdfium_integration():
    """
    Integration test for the extract_pdf_with_pdfium function.

    Verifies that the PDFium-based extraction correctly processes a test PDF document
    and returns a DataFrame with the expected structure and content.
    """
    # Get the test file path
    test_file_rel_path = "./data/multimodal_test.pdf"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame with the expected structure
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.PDF],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "document", "subtype": "pdf"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "pdf",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Get PDFium-specific environment variables
    yolox_http_endpoint = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    yolox_grpc_endpoint = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    yolox_protocol = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    auth_token = os.getenv("INGEST_AUTH_TOKEN", None)

    # Call the specialized PDFium extraction function
    df_result = extract_primitives_from_pdf_pdfium(
        df_extraction_ledger=df_ledger,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        text_depth="page",
        yolox_auth_token=auth_token,
        yolox_endpoints=(yolox_grpc_endpoint, yolox_http_endpoint),
        yolox_infer_protocol=yolox_protocol,
    )

    # Validate results
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Check for expected content types
    doc_types = df_result["document_type"].unique()
    expected_types = ["text", "structured", "image"]
    found_expected_type = any(doc_type in expected_types for doc_type in doc_types)
    assert found_expected_type, f"Expected at least one of these document types: {expected_types}"

    # Check if text extraction worked
    text_entries = df_result[df_result["document_type"] == "text"]
    assert not text_entries.empty, "PDFium extraction should have extracted text content"

    # Validate individual entries
    for idx, row in df_result.iterrows():
        assert row["document_type"] is not None, f"Row {idx} has None document_type"
        assert row["metadata"] is not None, f"Row {idx} has None metadata"
        assert isinstance(row["uuid"], str), f"Row {idx} UUID should be a string"


@pytest.mark.integration
def test_extract_pdf_with_nemoretriever_integration():
    """
    Integration test for the extract_pdf_with_nemoretriever function.

    Verifies that the NemoRetriever Parse extraction correctly processes a test PDF document
    and returns a DataFrame with the expected structure and content.
    """
    # Get the test file path
    test_file_rel_path = "./data/multimodal_test.pdf"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame with the expected structure
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.PDF],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "document", "subtype": "pdf"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "pdf",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Get NemoRetriever Parse environment variables
    nemo_http_endpoint = os.getenv(
        "INGEST_NEMO_RETRIEVER_PARSE_HTTP_ENDPOINT", "http://localhost:8000/v1/chat/completions:8015"
    )
    nemo_grpc_endpoint = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_GRPC_ENDPOINT", None)
    nemo_protocol = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_PROTOCOL", "http")
    nemo_model_name = os.getenv("INGEST_NEMO_RETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    # Also get YOLOX parameters (needed for image processing)
    yolox_http_endpoint = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    yolox_grpc_endpoint = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    yolox_protocol = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    auth_token = os.getenv("INGEST_AUTH_TOKEN", None)

    # Call the specialized NemoRetriever extraction function
    df_result = extract_primitives_from_pdf_nemoretriever_parse(
        df_extraction_ledger=df_ledger,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        text_depth="page",
        yolox_endpoints=(yolox_grpc_endpoint, yolox_http_endpoint),
        yolox_infer_protocol=yolox_protocol,
        yolox_auth_token=auth_token,
        nemoretriever_parse_endpoints=(nemo_grpc_endpoint, nemo_http_endpoint),
        nemoretriever_parse_protocol=nemo_protocol,
        nemoretriever_parse_model_name=nemo_model_name,
    )

    # Validate results
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Check for expected content types
    doc_types = df_result["document_type"].unique()
    expected_types = ["text", "structured", "image"]
    found_expected_type = any(doc_type in expected_types for doc_type in doc_types)
    assert found_expected_type, f"Expected at least one of these document types: {expected_types}"

    # Check for structured data (tables)
    if "extract_tables" in df_result["document_type"].values:
        table_entries = df_result[df_result["document_type"] == "structured"]
        for _, row in table_entries.iterrows():
            assert "table_data" in row["metadata"], "Table metadata should contain table_data field"

    # Validate individual entries
    for idx, row in df_result.iterrows():
        assert row["document_type"] is not None, f"Row {idx} has None document_type"
        assert row["metadata"] is not None, f"Row {idx} has None metadata"
        assert isinstance(row["uuid"], str), f"Row {idx} UUID should be a string"


@pytest.mark.integration
def test_extract_primitives_from_audio_integration():
    """
    Integration test for the extract_primitives_from_audio function.

    This test verifies that the audio primitive extraction pipeline correctly processes
    an audio file and returns a DataFrame with the extracted transcript. The test checks
    that the result contains an AUDIO document type and that the transcript meets minimum
    length requirements.
    """
    # Get the test file path using helper functions
    test_file_rel_path = "./data/multimodal_test.wav"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame matching the expected structure for audio extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.WAV],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "audio", "subtype": "wav"},
                    "error_metadata": None,
                    "audio_metadata": {"audio_type": "wav"},
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "wav",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    _AUDIO_GRPC_ENDPOINT = os.getenv("INGEST_AUDIO_GRPC_ENDPOINT", "127.0.0.1:8021")
    _AUDIO_HTTP_ENDPOINT = os.getenv("INGEST_AUDIO_HTTP_ENDPOINT", None)
    _AUDIO_INFER_PROTOCOL = os.getenv("INGEST_AUDIO_PROTOCOL", "grpc")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)
    _USE_SSL = os.getenv("INGEST_USE_SSL", False)
    _SSL_CERT = os.getenv("INGEST_SSL_CERT", None)

    # Construct the endpoint tuple
    audio_endpoints = (_AUDIO_GRPC_ENDPOINT, _AUDIO_HTTP_ENDPOINT)

    # Call the function under test with the constructed parameters
    df_result = extract_primitives_from_audio(
        df_ledger=df_ledger,
        audio_endpoints=audio_endpoints,
        audio_infer_protocol=_AUDIO_INFER_PROTOCOL,
        auth_token=_AUTH_TOKEN,
        use_ssl=_USE_SSL,
        ssl_cert=_SSL_CERT,
    )

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check that we have the expected DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        # "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Verify that the result contains an AUDIO document type
    assert "audio" in df_result["document_type"].values, "Expected AUDIO document type in results"

    # Get the rows with AUDIO document type
    audio_rows = df_result[df_result["document_type"] == "audio"]
    assert not audio_rows.empty, "Expected at least one row with AUDIO document type"

    # Verify audio metadata and transcript for each audio row
    for idx, row in audio_rows.iterrows():
        metadata = row["metadata"]

        # Verify metadata exists
        assert metadata is not None, f"Row {idx} has None metadata"

        # Verify audio_metadata exists
        assert "audio_metadata" in metadata, f"Row {idx} missing audio_metadata"
        audio_metadata = metadata["audio_metadata"]
        assert audio_metadata is not None, f"Row {idx} has None audio_metadata"

        # Verify audio_transcript exists and has sufficient length
        assert "audio_transcript" in audio_metadata, f"Row {idx} missing audio_transcript in audio_metadata"
        transcript = audio_metadata["audio_transcript"]
        assert transcript is not None, f"Row {idx} has None audio_transcript"
        assert isinstance(transcript, str), f"Row {idx} audio_transcript should be a string"

        # Check that transcript is at least 100 words
        word_count = len(transcript.split())
        assert word_count >= 25, f"Row {idx} audio_transcript has only {word_count} words, expected at least 25"


@pytest.mark.integration
def test_extract_primitives_from_pptx_integration():
    """
    Integration test for the extract_primitives_from_pptx function.

    This test verifies that the PPTX primitive extraction pipeline correctly processes
    a multimodal test presentation and returns a DataFrame with the extracted primitives.
    The test locates the file, prepares the input DataFrame with the required structure,
    and validates the basic structure of the output.
    """
    # Get the test file path using helper functions
    test_file_rel_path = "./data/multimodal_test.pptx"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame matching the expected structure for PPTX extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.PPTX],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "document", "subtype": "pptx"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "pptx",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    _YOLOX_INFER_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuple
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)

    # Call the function under test with the constructed parameters
    df_result = extract_primitives_from_pptx(
        df_ledger=df_ledger,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        yolox_endpoints=yolox_endpoints,
        yolox_infer_protocol=_YOLOX_INFER_PROTOCOL,
        auth_token=_AUTH_TOKEN,
    )

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check that we have the expected DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Basic validation of results
    # Check that we have at least some rows in the result
    assert len(df_result) > 0, "Expected at least some rows in the result"

    # Verify that each row has the required fields
    for idx, row in df_result.iterrows():
        # Verify document_type is present
        assert row["document_type"] is not None, f"Row {idx} has None document_type"

        # Verify metadata exists
        assert row["metadata"] is not None, f"Row {idx} has None metadata"

        # Verify UUID is present
        assert row["uuid"] is not None, f"Row {idx} has None UUID"
        assert isinstance(row["uuid"], str), f"Row {idx} UUID should be a string"


@pytest.mark.integration
def test_extract_primitives_from_docx_integration():
    """
    Integration test for the extract_primitives_from_docx function.

    This test verifies that the DOCX primitive extraction pipeline correctly processes
    multimodal test documents and returns a DataFrame with at least one structured
    element (table or chart) or other document primitives.
    """
    # Get the test file path using helper functions
    test_file_rel_path = "./data/multimodal_test.docx"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
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

    # Build input DataFrame matching the expected structure for DOCX extraction
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [DocumentTypeEnum.DOCX],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "document", "subtype": "docx"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": "docx",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    # Note: For posterity, if you get the 'port' wrong here or issue the request to the wrong page-xxx service, you may
    #  get back something like an 'invalid' data type response. Double check your ports.
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    _YOLOX_INFER_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuple
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)

    # Call the function under test with the constructed parameters
    df_result = extract_primitives_from_docx(
        df_ledger=df_ledger,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        yolox_endpoints=yolox_endpoints,
        yolox_infer_protocol=_YOLOX_INFER_PROTOCOL,
        auth_token=_AUTH_TOKEN,
    )

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check that we have the expected DataFrame structure
    assert set(df_result.columns) >= {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain at least the expected columns"

    # Basic validation of results
    # Check that we have at least some rows in the result
    assert len(df_result) > 0, "Expected at least some rows in the result"

    # Verify that each row has the required fields
    for idx, row in df_result.iterrows():
        # Verify document_type is present
        assert row["document_type"] is not None, f"Row {idx} has None document_type"

        # Verify metadata exists
        assert row["metadata"] is not None, f"Row {idx} has None metadata"

        # Verify UUID is present
        assert row["uuid"] is not None, f"Row {idx} has None UUID"
        assert isinstance(row["uuid"], str), f"Row {idx} UUID should be a string"


@pytest.mark.integration
@pytest.mark.parametrize("file_extension", ["png", "tiff", "jpeg", "bmp"])
def test_extract_primitives_from_image_integration(file_extension):
    """
    Integration test for the extract_primitives_from_image function.

    This test verifies that the image primitive extraction pipeline correctly processes
    multimodal test images in various formats and returns a DataFrame with at least
    one structured element (table or chart).

    Parameters
    ----------
    file_extension : str
        The file extension to test (png, bmp, tiff, jpeg, or jpg)
    """
    # Get the test file path using helper functions
    test_file_rel_path = f"./data/multimodal_test.{file_extension}"

    # Try to find the file using project root first
    project_root = get_project_root(__file__)
    if project_root:
        test_file_path = os.path.join(project_root, test_file_rel_path)

    # If not found via project root, try heuristic approach
    if not project_root or not os.path.exists(test_file_path):
        root_dir = find_root_by_pattern(test_file_rel_path, os.path.dirname(__file__))
        if root_dir:
            test_file_path = os.path.join(root_dir, test_file_rel_path)
        else:
            # Fallback to relative path if all else fails
            test_file_path = test_file_rel_path

    # Ensure the file exists
    if not os.path.exists(test_file_path):
        pytest.skip(f"Test file not found at {test_file_path}")

    # Read the file and encode it as base64
    with open(test_file_path, "rb") as f:
        file_content = f.read()
        base64_content = base64.b64encode(file_content).decode("utf-8")

    # Map file extension to document type enum
    extension_to_doctype = {
        "png": DocumentTypeEnum.PNG,
        "bmp": DocumentTypeEnum.BMP,
        "tiff": DocumentTypeEnum.TIFF,
        "jpeg": DocumentTypeEnum.JPEG,
        "jpg": DocumentTypeEnum.JPEG,
    }

    document_type = extension_to_doctype.get(file_extension.lower(), DocumentTypeEnum.PNG)

    # Build input DataFrame matching the expected structure
    df_ledger = pd.DataFrame(
        {
            "source_name": [test_file_path],
            "source_id": [test_file_path],
            "content": [base64_content],
            "document_type": [document_type],
            "metadata": [
                {
                    "content": base64_content,
                    "content_metadata": {"type": "image"},
                    "error_metadata": None,
                    "audio_metadata": None,
                    "image_metadata": {"image_type": file_extension.lower()},
                    "source_metadata": {
                        "source_id": test_file_path,
                        "source_name": test_file_path,
                        "source_type": file_extension.lower(),
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Pull configuration values from the environment
    _YOLOX_GRPC_ENDPOINT = os.getenv("INGEST_YOLOX_GRPC_ENDPOINT", None)
    _YOLOX_HTTP_ENDPOINT = os.getenv("INGEST_YOLOX_HTTP_ENDPOINT", "http://127.0.0.1:8000/v1/infer")
    _YOLOX_INFER_PROTOCOL = os.getenv("INGEST_YOLOX_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuple
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)

    # Call the function under test with the constructed parameters
    df_result = extract_primitives_from_image(
        df_ledger=df_ledger,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        yolox_endpoints=yolox_endpoints,
        yolox_infer_protocol=_YOLOX_INFER_PROTOCOL,
        auth_token=_AUTH_TOKEN,
    )

    # Assert that the returned DataFrame is not empty
    assert not df_result.empty, "Resulting DataFrame should not be empty"

    # Check that we have the expected DataFrame structure
    assert set(df_result.columns) == {
        "document_type",
        "metadata",
        "uuid",
    }, "DataFrame should contain the expected columns"

    # Validate that at least some primitives were extracted
    # The test image should have at least one structured element (table or chart)
    document_types = df_result["document_type"].unique()
    assert "structured" in document_types, "Expected at least one 'structured' document type"

    # Check for specific structured subtypes in the metadata
    structured_rows = df_result[df_result["document_type"] == "structured"]
    assert not structured_rows.empty, "Expected at least one structured row"

    # Check for either a table or chart in the structured content
    found_structured_element = False

    for _, row in structured_rows.iterrows():
        metadata = row["metadata"]
        content_metadata = metadata.get("content_metadata", {})
        subtype = content_metadata.get("subtype", "")

        if subtype in ["table", "chart"]:
            found_structured_element = True

            if subtype == "table":
                # Verify table metadata exists
                assert "table_metadata" in metadata, "Expected table_metadata in table extraction"

                # Verify table location data exists
                table_metadata = metadata["table_metadata"]
                assert "table_location" in table_metadata, "Expected table_location in table_metadata"
                assert isinstance(table_metadata["table_location"], list), "table_location should be a list"
                assert len(table_metadata["table_location"]) == 4, "table_location should contain 4 coordinates"

            elif subtype == "chart":
                # Verify basic chart info exists
                assert "content_metadata" in metadata, "Expected content_metadata in chart extraction"
                assert content_metadata["description"].lower().find("chart") != -1, "Expected chart description"

    # Assert that we found at least one structured element (table or chart)
    assert found_structured_element, "No table or chart was extracted from the test image"

    # Verify UUID format for all rows
    for _, row in df_result.iterrows():
        uuid_val = row["uuid"]
        assert isinstance(uuid_val, str), "UUID should be a string"
        assert len(uuid_val) > 0, "UUID should not be empty"

        # Check UUID format (standard UUID has 36 characters including hyphens)
        assert len(uuid_val) == 36, f"UUID {uuid_val} doesn't match expected format"


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
    _OCR_GRPC_ENDPOINT = os.getenv("INGEST_OCR_GRPC_ENDPOINT", None)
    _OCR_HTTP_ENDPOINT = os.getenv("INGEST_OCR_HTTP_ENDPOINT", "http://127.0.0.1:8010")
    _OCR_PROTOCOL = os.getenv("INGEST_OCR_PROTOCOL", "http")
    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the ocr endpoints tuple
    ocr_endpoints = (_OCR_GRPC_ENDPOINT, _OCR_HTTP_ENDPOINT)

    # Explicitly map the schema values to the function's expected arguments
    integration_args = {
        "ocr_endpoints": ocr_endpoints,
        "ocr_protocol": _OCR_PROTOCOL,
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
        # Try to find the file using project root first
        project_root = get_project_root(__file__)
        if project_root:
            test_file_path = os.path.join(project_root, test_file_rel_path)

        # If not found via project root, try heuristic approach
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

    _OCR_GRPC_ENDPOINT = os.getenv("INGEST_OCR_GRPC_ENDPOINT", None)
    _OCR_HTTP_ENDPOINT = os.getenv("INGEST_OCR_HTTP_ENDPOINT", "http://127.0.0.1:8009/v1/infer")
    _OCR_PROTOCOL = os.getenv("INGEST_OCR_PROTOCOL", "http")

    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuples
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)
    ocr_endpoints = (_OCR_GRPC_ENDPOINT, _OCR_HTTP_ENDPOINT)

    # Explicitly map the values to the function's expected arguments
    integration_args = {
        "yolox_endpoints": yolox_endpoints,
        "ocr_endpoints": ocr_endpoints,
        "yolox_protocol": _YOLOX_PROTOCOL,
        "ocr_protocol": _OCR_PROTOCOL,
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
        # Try to find the file using project root first
        project_root = get_project_root(__file__)
        if project_root:
            test_file_path = os.path.join(project_root, test_file_rel_path)

        # If not found via project root, try heuristic approach
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

    _OCR_GRPC_ENDPOINT = os.getenv("INGEST_OCR_GRPC_ENDPOINT", None)
    _OCR_HTTP_ENDPOINT = os.getenv("INGEST_OCR_HTTP_ENDPOINT", "http://127.0.0.1:8009/v1/infer")
    _OCR_PROTOCOL = os.getenv("INGEST_OCR_PROTOCOL", "http")

    _AUTH_TOKEN = os.getenv("INGEST_AUTH_TOKEN", None)

    # Construct the endpoint tuples
    yolox_endpoints = (_YOLOX_GRPC_ENDPOINT, _YOLOX_HTTP_ENDPOINT)
    ocr_endpoints = (_OCR_GRPC_ENDPOINT, _OCR_HTTP_ENDPOINT)

    # Explicitly map the values to the function's expected arguments
    integration_args = {
        "yolox_endpoints": yolox_endpoints,
        "ocr_endpoints": ocr_endpoints,
        "yolox_protocol": _YOLOX_PROTOCOL,
        "ocr_protocol": _OCR_PROTOCOL,
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
