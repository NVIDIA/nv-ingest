# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO
import pypdfium2 as pdfium

import nv_ingest.framework.orchestration.ray.util.pipeline.scatterers.pdf_fragmenter as module_under_test
from api.api_tests import find_root_by_pattern, get_git_root

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


def _create_pdf_with_pages(source_pdf_path: str, target_pages: int) -> bytes:
    """
    Create a PDF with the specified number of pages by copying pages from source.

    Parameters:
        source_pdf_path (str): Path to source PDF
        target_pages (int): Number of pages needed in the output

    Returns:
        bytes: PDF bytes with the target number of pages
    """
    # Load source PDF
    with open(source_pdf_path, "rb") as f:
        source_bytes = f.read()

    source_pdf = pdfium.PdfDocument(source_bytes)
    source_page_count = len(source_pdf)

    # If source has exactly the pages we need, return it as-is
    if source_page_count == target_pages:
        source_pdf.close()
        return source_bytes

    # Create new PDF with exact number of pages
    new_pdf = pdfium.PdfDocument.new()

    # Add pages until we reach target
    pages_added = 0
    while pages_added < target_pages:
        # Copy pages from source, cycling through them if needed
        page_to_copy = pages_added % source_page_count
        new_pdf.import_pages(source_pdf, [page_to_copy])
        pages_added += 1

    # Save to bytes
    buffer = BytesIO()
    new_pdf.save(buffer)
    result_bytes = buffer.getvalue()

    # Cleanup
    new_pdf.close()
    source_pdf.close()

    return result_bytes


class TestFragmentPdf:
    """Black box tests for fragment_pdf function."""

    @pytest.fixture
    def mock_message(self):
        """Create a mock IngestControlMessage."""
        message = Mock()
        message.metadata.return_value = {}
        return message

    @pytest.fixture
    def sample_pdf_base64(self):
        """Provide a base64 encoded PDF for testing."""
        # This would be replaced with actual PDF bytes in real tests
        return base64.b64encode(b"fake_pdf_content").decode("utf-8")

    @pytest.fixture
    def pdf_dataframe(self, sample_pdf_base64):
        """Create a DataFrame with PDF content."""
        metadata = {
            "content": sample_pdf_base64,
            "filename": "test.pdf",
            "file_size": 1024,
            "other_field": "preserved_value",
        }

        return pd.DataFrame(
            {"document_type": ["pdf"], "metadata": [json.dumps(metadata)], "other_column": ["should_be_preserved"]}
        )

    def test_non_pdf_returns_original_message(self, mock_message):
        """Test that non-PDF documents return unchanged."""
        # Arrange
        df = pd.DataFrame({"document_type": ["text"], "metadata": ['{"content": "some text"}']})
        mock_message.payload.return_value = df

        # Act
        result = module_under_test.fragment_pdf(mock_message)

        # Assert
        assert len(result) == 1
        assert result[0] == mock_message

    def test_multiple_rows_raises_error(self, mock_message):
        """Test that multiple row DataFrames raise ValueError."""
        # Arrange
        df = pd.DataFrame({"document_type": ["pdf", "pdf"], "metadata": ["{}", "{}"]})
        mock_message.payload.return_value = df

        # Act & Assert
        with pytest.raises(ValueError, match="Expected single row DataFrame, got 2 rows"):
            module_under_test.fragment_pdf(mock_message)

    def test_missing_content_raises_error(self, mock_message):
        """Test that missing content field raises ValueError."""
        # Arrange
        df = pd.DataFrame({"document_type": ["pdf"], "metadata": ['{"no_content": "here"}']})
        mock_message.payload.return_value = df

        # Act & Assert
        with pytest.raises(ValueError, match="No 'content' field found in metadata"):
            module_under_test.fragment_pdf(mock_message)

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_single_page_pdf_not_fragmented(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test that PDFs smaller than fragment size aren't fragmented."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 5  # 5 pages, less than default 10
        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        assert len(result) == 1
        assert result[0] == mock_message
        mock_pdf.close.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_pdf_fragmentation_basic(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test basic PDF fragmentation into multiple parts."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 25  # 25 pages, should create 3 fragments with 10 pages each
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        assert len(result) == 3
        assert mock_pdfium.PdfDocument.new.call_count == 3
        assert mock_new_pdf.import_pages.call_count == 3
        assert mock_new_pdf.save.call_count == 3
        assert mock_new_pdf.close.call_count == 3
        mock_pdf.close.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_fragment_metadata_structure(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test that fragment metadata is properly structured."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 15  # 15 pages, 2 fragments
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_buffer = BytesIO()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment_content")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        assert len(result) == 2

        # Check first fragment
        first_payload = result[0].payload.call_args[0][0]
        first_metadata = json.loads(first_payload.iloc[0]["metadata"])

        assert "fragment_info" in first_metadata
        assert first_metadata["fragment_info"]["fragment_index"] == 0
        assert first_metadata["fragment_info"]["total_fragments"] == 2
        assert first_metadata["fragment_info"]["start_page"] == 0
        assert first_metadata["fragment_info"]["end_page"] == 10
        assert first_metadata["fragment_info"]["total_pages"] == 15
        assert first_metadata["fragment_info"]["pages_in_fragment"] == 10
        assert first_metadata["fragment_info"]["source_document_type"] == "pdf"
        assert first_metadata["fragment_info"]["is_fragment"] is True
        assert first_metadata["fragment_info"]["fragment_id"] == "1_of_2"

        # Check second fragment
        second_payload = result[1].payload.call_args[0][0]
        second_metadata = json.loads(second_payload.iloc[0]["metadata"])

        assert second_metadata["fragment_info"]["fragment_index"] == 1
        assert second_metadata["fragment_info"]["start_page"] == 10
        assert second_metadata["fragment_info"]["end_page"] == 15
        assert second_metadata["fragment_info"]["pages_in_fragment"] == 5
        assert second_metadata["fragment_info"]["fragment_id"] == "2_of_2"

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_pdf_metadata_extraction_and_preservation(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test that PDF internal metadata is extracted and preserved."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 20

        # Mock PDF metadata extraction
        def get_metadata_side_effect(key):
            metadata_map = {
                "Title": "Original Document",
                "Author": "Test Author",
                "Subject": "Test Subject",
                "Keywords": "test, document",
                "Creator": "Test Creator",
                "Producer": "Test Producer",
                "CreationDate": "2024-01-01",
                "ModDate": "2024-01-02",
            }
            return metadata_map.get(key)

        mock_pdf.get_metadata_value.side_effect = get_metadata_side_effect

        mock_new_pdf = MagicMock()
        mock_buffer = BytesIO()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment_content")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        assert len(result) == 2

        # PDF metadata should NOT be copied to new PDFs (pypdfium2 doesn't support it)
        assert mock_new_pdf.set_metadata_value.call_count == 0

        # Instead, verify that original metadata is preserved in fragment_info
        for i in range(2):
            payload = result[i].payload.call_args[0][0]
            metadata = json.loads(payload.iloc[0]["metadata"])

            original_pdf_metadata = metadata["fragment_info"]["original_pdf_metadata"]
            assert original_pdf_metadata["Title"] == "Original Document"
            assert original_pdf_metadata["Author"] == "Test Author"
            assert original_pdf_metadata["Subject"] == "Test Subject"
            assert original_pdf_metadata["Keywords"] == "test, document"
            assert original_pdf_metadata["Creator"] == "Test Creator"
            assert original_pdf_metadata["Producer"] == "Test Producer"
            assert original_pdf_metadata["CreationDate"] == "2024-01-01"
            assert original_pdf_metadata["ModDate"] == "2024-01-02"

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_fragment_preserves_original_metadata(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test that all original DataFrame metadata is preserved in fragments."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 20
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_buffer = BytesIO()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment_content")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        first_payload = result[0].payload.call_args[0][0]
        first_metadata = json.loads(first_payload.iloc[0]["metadata"])

        # Original metadata fields should be preserved
        assert first_metadata["filename"] == "test.pdf"
        assert first_metadata["file_size"] == 1024
        assert first_metadata["other_field"] == "preserved_value"

        # Other DataFrame columns should be preserved
        assert first_payload.iloc[0]["other_column"] == "should_be_preserved"

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_existing_fragment_info_preserved(self, mock_pdfium, mock_message):
        """Test that existing fragment_info is preserved as original_fragment_info."""
        # Arrange
        metadata = {
            "content": base64.b64encode(b"fake_pdf").decode("utf-8"),
            "fragment_info": {"fragment_index": 2, "total_fragments": 5, "from_previous_operation": True},
        }

        df = pd.DataFrame({"document_type": ["pdf"], "metadata": [json.dumps(metadata)]})

        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 20
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = df

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        first_payload = result[0].payload.call_args[0][0]
        first_metadata = json.loads(first_payload.iloc[0]["metadata"])

        assert "original_fragment_info" in first_metadata
        assert first_metadata["original_fragment_info"]["from_previous_operation"] is True
        assert first_metadata["original_fragment_info"]["fragment_index"] == 2

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_message_metadata_updated(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test that message-level metadata is updated with fragment info."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 20
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe
        mock_message.metadata.return_value = {"existing": "data"}

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=10)

        # Assert
        assert len(result) == 2

        # Check that message metadata was updated
        for i, msg in enumerate(result):
            expected_metadata = {
                "existing": "data",
                "fragment_index": i,
                "total_fragments": 2,
                "source_document_type": "pdf",
            }
            msg.metadata.assert_called_with(expected_metadata)

    @patch(f"{MODULE_UNDER_TEST}.pdfium")
    def test_custom_pages_per_fragment(self, mock_pdfium, mock_message, pdf_dataframe):
        """Test fragmentation with custom pages per fragment."""
        # Arrange
        mock_pdf = MagicMock()
        mock_pdf.__len__.return_value = 50
        mock_pdf.get_metadata_value.return_value = None

        mock_new_pdf = MagicMock()
        mock_new_pdf.save.side_effect = lambda b: b.write(b"fragment")

        mock_pdfium.PdfDocument.return_value = mock_pdf
        mock_pdfium.PdfDocument.new.return_value = mock_new_pdf

        mock_message.payload.return_value = pdf_dataframe

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=15)

        # Assert
        assert len(result) == 4  # 50 pages / 15 = 3.33, so 4 fragments

        # Verify page ranges
        import_calls = mock_new_pdf.import_pages.call_args_list
        assert import_calls[0][0][1] == list(range(0, 15))
        assert import_calls[1][0][1] == list(range(15, 30))
        assert import_calls[2][0][1] == list(range(30, 45))
        assert import_calls[3][0][1] == list(range(45, 50))


class TestFragmentPdfWithRealData:
    """Integration tests using real PDF files."""

    @pytest.fixture
    def test_pdf_path(self):
        """Find the test PDF file."""
        # Try different possible paths where test data might be stored
        pdfs = ["./data/multimodal_test.pdf"]

        # First try from git root
        git_root = get_git_root(__file__)
        if git_root:
            for pattern in pdfs:
                candidate = os.path.join(git_root, pattern)
                if os.path.exists(candidate):
                    return candidate

        # Then try backtracking from current directory
        for pattern in pdfs:
            root = find_root_by_pattern(pattern)
            if root:
                return os.path.join(root, pattern)

        pytest.skip("No test PDF found in expected locations")

    @pytest.fixture
    def mock_message(self):
        """Create a mock IngestControlMessage."""
        message = Mock()
        message.metadata.return_value = {}
        return message

    @pytest.mark.parametrize(
        "total_pages,pages_per_fragment,expected_fragments",
        [
            (10, 5, 2),  # Even split
            (15, 5, 3),  # Even split with more fragments
            (17, 5, 4),  # Uneven split - last fragment has 2 pages
            (25, 10, 3),  # Standard case
            (100, 25, 4),  # Large document
            (7, 10, 1),  # No fragmentation needed
            (1, 1, 1),  # Single page
            (50, 1, 50),  # One page per fragment
            (1000, 200, 5),  # One page per fragment
        ],
    )
    def test_pdf_fragmentation_with_real_pdf(
        self, test_pdf_path, mock_message, total_pages, pages_per_fragment, expected_fragments
    ):
        """Test PDF fragmentation with various page counts and fragment sizes."""
        # Arrange
        # Create PDF with required number of pages
        pdf_bytes = _create_pdf_with_pages(test_pdf_path, total_pages)
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        metadata = {
            "content": pdf_base64,
            "filename": f"test_{total_pages}pages.pdf",
            "original_page_count": total_pages,
            "test_metadata": "should_be_preserved",
        }

        df = pd.DataFrame({"document_type": ["pdf"], "metadata": [json.dumps(metadata)], "test_column": ["test_value"]})

        mock_message.payload.return_value = df

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=pages_per_fragment)

        # Assert
        assert len(result) == expected_fragments

        # If no fragmentation expected, verify original message returned
        if expected_fragments == 1 and total_pages <= pages_per_fragment:
            assert result[0] == mock_message
            return

        # Verify each fragment
        total_pages_verified = 0
        for i, fragment_message in enumerate(result):
            # Get the payload DataFrame
            payload_df = fragment_message.payload.call_args[0][0]

            # Verify DataFrame structure preserved
            assert len(payload_df) == 1
            assert payload_df.iloc[0]["document_type"] == "pdf"
            assert payload_df.iloc[0]["test_column"] == "test_value"

            # Parse metadata
            fragment_metadata = json.loads(payload_df.iloc[0]["metadata"])

            # Verify original metadata preserved
            assert fragment_metadata["filename"] == f"test_{total_pages}pages.pdf"
            assert fragment_metadata["original_page_count"] == total_pages
            assert fragment_metadata["test_metadata"] == "should_be_preserved"

            # Verify fragment info
            fragment_info = fragment_metadata["fragment_info"]
            assert fragment_info["fragment_index"] == i
            assert fragment_info["total_fragments"] == expected_fragments
            assert fragment_info["total_pages"] == total_pages
            assert fragment_info["source_document_type"] == "pdf"
            assert fragment_info["is_fragment"] is True
            assert fragment_info["fragment_id"] == f"{i + 1}_of_{expected_fragments}"

            # Verify page ranges
            expected_start = i * pages_per_fragment
            expected_end = min((i + 1) * pages_per_fragment, total_pages)
            assert fragment_info["start_page"] == expected_start
            assert fragment_info["end_page"] == expected_end
            assert fragment_info["pages_in_fragment"] == expected_end - expected_start

            total_pages_verified += fragment_info["pages_in_fragment"]

            # Decode and verify the fragment PDF
            fragment_pdf_base64 = fragment_metadata["content"]
            fragment_pdf_bytes = base64.b64decode(fragment_pdf_base64)

            # Verify it's valid PDF by attempting to load it
            fragment_pdf = pdfium.PdfDocument(fragment_pdf_bytes)
            assert len(fragment_pdf) == fragment_info["pages_in_fragment"]
            fragment_pdf.close()

        # Verify all pages accounted for
        assert total_pages_verified == total_pages

    @pytest.mark.parametrize("pages_per_fragment", [5, 10, 20])
    def test_pdf_metadata_preservation_with_real_pdf(self, test_pdf_path, mock_message, pages_per_fragment):
        """Test that PDF internal metadata is preserved in fragments."""
        # Arrange
        # Create a 30-page PDF for testing
        pdf_bytes = _create_pdf_with_pages(test_pdf_path, 30)

        # Load it to check if it has metadata
        test_pdf = pdfium.PdfDocument(pdf_bytes)
        original_metadata = {}
        for key in ["Title", "Author", "Subject", "Creator", "Producer"]:
            try:
                value = test_pdf.get_metadata_value(key)
                if value:
                    original_metadata[key] = value
            except:
                pass
        test_pdf.close()

        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        metadata = {"content": pdf_base64}
        df = pd.DataFrame({"document_type": ["pdf"], "metadata": [json.dumps(metadata)]})

        mock_message.payload.return_value = df

        # Act
        result = module_under_test.fragment_pdf(mock_message, pages_per_fragment=pages_per_fragment)

        # Assert
        expected_fragments = (30 + pages_per_fragment - 1) // pages_per_fragment
        assert len(result) == expected_fragments

        # Check each fragment preserves original PDF metadata
        for fragment_message in result:
            payload_df = fragment_message.payload.call_args[0][0]
            fragment_metadata = json.loads(payload_df.iloc[0]["metadata"])

            # Original PDF metadata should be in fragment_info
            if original_metadata:  # Only check if source had metadata
                preserved_metadata = fragment_metadata["fragment_info"]["original_pdf_metadata"]
                for key, value in original_metadata.items():
                    assert preserved_metadata.get(key) == value
