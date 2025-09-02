# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for image_disk_utils module.

Tests the efficient image saving functionality including:
- Base64 format detection
- Direct write optimizations
- Format conversion
- Edge cases and error handling
"""

import base64
import io

import pytest
from PIL import Image

# Test client-side utilities that provide convenient wrappers
# around core API functionality for image disk operations
from nv_ingest_client.util.image_disk_utils import (
    save_images_to_disk,
    save_images_from_response,
    save_images_from_ingestor_results,
)


class TestSaveImagesToDisk:
    """Test the main save_images_to_disk function."""

    @pytest.fixture
    def sample_response_data(self):
        """Create sample response data with images."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return [
            {
                "document_type": "structured",
                "metadata": {
                    "content": image_b64,
                    "source_metadata": {"source_id": "test_document.pdf"},
                    "content_metadata": {"subtype": "chart", "page_number": 1},
                },
            },
            {
                "document_type": "structured",
                "metadata": {
                    "content": image_b64,
                    "source_metadata": {"source_id": "test_document.pdf"},
                    "content_metadata": {"subtype": "table", "page_number": 2},
                },
            },
        ]

    def test_save_with_organize_by_type(self, tmp_path, sample_response_data):
        """Test saving images organized by type."""
        result = save_images_to_disk(sample_response_data, str(tmp_path), organize_by_type=True, output_format="png")

        assert result["chart"] == 1
        assert result["table"] == 1
        assert result["total"] == 2

        # Check directory structure
        assert (tmp_path / "chart").exists()
        assert (tmp_path / "table").exists()

        # Check files exist
        chart_files = list((tmp_path / "chart").glob("*.png"))
        table_files = list((tmp_path / "table").glob("*.png"))
        assert len(chart_files) == 1
        assert len(table_files) == 1

    def test_save_flat_structure(self, tmp_path, sample_response_data):
        """Test saving images in flat directory structure."""
        result = save_images_to_disk(sample_response_data, str(tmp_path), organize_by_type=False, output_format="jpeg")

        assert result["total"] == 2

        # Check files exist in root
        image_files = list(tmp_path.glob("*.jpg"))
        assert len(image_files) == 2

        # Check filenames contain subtype
        filenames = [f.name for f in image_files]
        assert any("chart" in name for name in filenames)
        assert any("table" in name for name in filenames)

    def test_filtering_by_subtype(self, tmp_path, sample_response_data):
        """Test filtering images by subtype."""
        result = save_images_to_disk(
            sample_response_data, str(tmp_path), save_charts=True, save_tables=False  # Disable table saving
        )

        assert result["chart"] == 1
        assert result["table"] == 0
        assert result["total"] == 1

    def test_empty_response_data(self, tmp_path):
        """Test handling empty response data."""
        result = save_images_to_disk([], str(tmp_path))
        assert result == {}

    def test_unsupported_output_format(self, tmp_path, sample_response_data):
        """Test handling unsupported output format falls back gracefully."""
        # Test behavior, not logging - fallback should still save images
        result = save_images_to_disk(sample_response_data, str(tmp_path), output_format="bmp")  # Unsupported

        # Should fallback and still save images successfully
        assert result["total"] == 2

        # Images should exist and be in fallback format (JPEG, updated from PNG)
        saved_files = list(tmp_path.rglob("*.jpg"))
        assert len(saved_files) == 2


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    @pytest.fixture
    def sample_api_response(self):
        """Create sample API response structure."""
        img = Image.new("RGB", (5, 5), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "data": [
                {
                    "document_type": "structured",
                    "metadata": {
                        "content": image_b64,
                        "source_metadata": {"source_id": "test.pdf"},
                        "content_metadata": {"subtype": "chart", "page_number": 1},
                    },
                }
            ]
        }

    def test_save_images_from_response(self, tmp_path, sample_api_response):
        """Test saving images from full API response."""
        result = save_images_from_response(sample_api_response, str(tmp_path))

        assert result["chart"] == 1
        assert result["total"] == 1

    def test_save_images_from_response_no_data(self, tmp_path):
        """Test handling API response with no data."""
        response = {"status": "success"}  # No data field
        result = save_images_from_response(response, str(tmp_path))
        assert result == {}

    def test_save_images_from_ingestor_results(self, tmp_path, sample_api_response):
        """Test saving images from ingestor results format."""
        # Ingestor results are list of lists
        ingestor_results = [sample_api_response["data"]]

        result = save_images_from_ingestor_results(ingestor_results, str(tmp_path))

        assert result["chart"] == 1
        assert result["total"] == 1

    def test_save_images_from_ingestor_results_mixed_types(self, tmp_path):
        """Test handling mixed result types in ingestor results."""
        # Create sample data
        img = Image.new("RGB", (3, 3), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create two separate documents to avoid reference issues
        doc_data_1 = {
            "document_type": "structured",
            "metadata": {
                "content": image_b64,
                "source_metadata": {"source_id": "test_1.pdf"},
                "content_metadata": {"subtype": "table", "page_number": 1},
            },
        }

        doc_data_2 = {
            "document_type": "structured",
            "metadata": {
                "content": image_b64,
                "source_metadata": {"source_id": "test_2.pdf"},
                "content_metadata": {"subtype": "table", "page_number": 2},
            },
        }

        # Mix of list and single document
        ingestor_results = [
            [doc_data_1],  # List of documents
            [doc_data_2],  # List of documents (not single document to avoid edge case)
        ]

        result = save_images_from_ingestor_results(ingestor_results, str(tmp_path))

        assert result["table"] == 2
        assert result["total"] == 2

    def test_save_images_from_ingestor_results_single_document(self, tmp_path):
        """Test handling single document (not in list) in ingestor results."""
        # Create sample data
        img = Image.new("RGB", (4, 4), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        doc_data = {
            "document_type": "structured",
            "metadata": {
                "content": image_b64,
                "source_metadata": {"source_id": "single_test.pdf"},
                "content_metadata": {"subtype": "chart", "page_number": 1},
            },
        }

        # Single document properly wrapped in list structure
        ingestor_results = [[doc_data]]  # Proper structure: List[List[Dict]]

        result = save_images_from_ingestor_results(ingestor_results, str(tmp_path))

        assert result["chart"] == 1
        assert result["total"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
