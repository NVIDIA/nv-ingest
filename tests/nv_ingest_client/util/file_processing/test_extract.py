# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from unittest.mock import mock_open
from unittest.mock import patch

import pytest
from nv_ingest_client.util.file_processing.extract import DocumentTypeEnum
from nv_ingest_client.util.file_processing.extract import detect_encoding_and_read_text_file
from nv_ingest_client.util.file_processing.extract import extract_file_content
from nv_ingest_client.util.file_processing.extract import get_or_infer_file_type
from nv_ingest_client.util.file_processing.extract import serialize_to_base64

_MODULE_UNDER_TEST = "nv_ingest_client.util.file_processing.extract"


# Test read_pdf_file with a sample PDF binary content
def test_read_pdf_file():
    sample_content = b"dummy pdf content"
    file_stream = BytesIO(sample_content)
    encoded_content = serialize_to_base64(file_stream)
    assert isinstance(encoded_content, str)
    # Further assert on the base64 encoded content if necessary


# Test detect_and_read_text_file with a sample text content
@patch(f"{_MODULE_UNDER_TEST}.charset_normalizer.detect")
def test_detect_and_read_text_file(mock_detect):
    sample_text = "This is a test"
    file_stream = BytesIO(sample_text.encode("utf-8"))
    mock_detect.return_value = {"encoding": "utf-8"}
    content = detect_encoding_and_read_text_file(file_stream)
    assert content == sample_text


# Test extract_file_content with a mocked file read for a PDF
@patch("builtins.open", new_callable=mock_open, read_data=b"PDF binary data")
def test_extract_file_content_pdf(mock_file):
    path = "dummy_path.pdf"
    content, doc_type = extract_file_content(path)
    assert doc_type == DocumentTypeEnum.pdf
    assert isinstance(content, str)  # Assuming the content is correctly base64 encoded


# Test extract_file_content with an unsupported file type
@pytest.mark.skip("Disabled while we figure out why libmagic is missing on the CI system")
@patch(f"{_MODULE_UNDER_TEST}.magic.from_file", return_value="unknown")  # Mock magic.from_file to return 'text/plain'
def test_extract_file_content_unsupported(mock_magic):
    with pytest.raises(ValueError):
        extract_file_content("unsupported_file.xyz")


# Test extract_file_content for a text file by mocking the open function and charset_normalizer.detect
@patch(f"{_MODULE_UNDER_TEST}.charset_normalizer.detect")
@patch("builtins.open", new_callable=mock_open, read_data=b"Simple text")
def test_extract_file_content_text(mock_open, mock_detect):
    mock_detect.return_value = {"encoding": "utf-8"}
    content, doc_type = extract_file_content("dummy_path.txt")
    assert doc_type == DocumentTypeEnum.txt
    assert content == "Simple text"


@pytest.mark.skip("Disabled while we figure out why libmagic is missing on the CI system")
@patch(f"{_MODULE_UNDER_TEST}.charset_normalizer.detect")
@patch("builtins.open", new_callable=mock_open, read_data=b"Simple text")
@patch(f"{_MODULE_UNDER_TEST}.magic.from_file", return_value="unknown")  # Mock magic.from_file to return 'text/plain'
def test_extract_file_content_text_bad(mock_magic, mock_open, mock_detect):
    mock_detect.return_value = {"encoding": "utf-8"}

    with pytest.raises(ValueError):
        _, _ = extract_file_content("dummy_path.not_a_file_type")


@pytest.mark.skip("Disabled while we figure out why libmagic is missing on the CI system")
@pytest.mark.parametrize(
    "file_path,expected_type",
    [
        ("/fake/path/to/document.pdf", DocumentTypeEnum.pdf),
        ("/fake/path/to/document.txt", DocumentTypeEnum.txt),
        ("/fake/path/to/document.docx", DocumentTypeEnum.docx),
        ("/fake/path/to/image.jpg", DocumentTypeEnum.jpeg),
        (
            "/fake/path/unknown.extension",
            DocumentTypeEnum.txt,
        ),  # This line simulates 'text/plain'
    ],
)
@patch(
    f"{_MODULE_UNDER_TEST}.magic.from_file", return_value="text/plain"
)  # Mock magic.from_file to return 'text/plain'
def test_get_or_infer_file_type_known_extension(mock_magic, file_path, expected_type):
    """
    Test that known file extensions correctly determine the file type, with special handling for unknown extensions.
    """
    assert get_or_infer_file_type(file_path) == expected_type


@pytest.mark.skip("Disabled while we figure out why libmagic is missing on the CI system")
@patch(f"{_MODULE_UNDER_TEST}.magic.from_file")
def test_get_or_infer_file_type_fallback_mime(mock_magic):
    """
    Test that the function falls back on MIME type detection for unrecognized extensions.
    """
    mock_magic.return_value = "text/plain"
    file_path = "/fake/path/to/unknown"

    get_or_infer_file_type(file_path)


@pytest.mark.skip("Disabled while we figure out why libmagic is missing on the CI system")
@patch(f"{_MODULE_UNDER_TEST}.magic.from_file")
def test_get_or_infer_file_type_unrecognized_extension_and_mime(mock_magic):
    """
    Test that the function raises a ValueError for unrecognized extensions and undetectable MIME types.
    """
    mock_magic.return_value = "application/octet-stream"
    file_path = "/fake/path/to/unknown.filetype"
    with pytest.raises(ValueError):
        get_or_infer_file_type(file_path)
