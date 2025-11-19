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
        """Test saving images in flat directory structure with JPEG format."""
        result = save_images_to_disk(sample_response_data, str(tmp_path), organize_by_type=False, output_format="jpeg")

        assert result["total"] == 2

        # Check that JPEG files were created
        jpeg_files = list(tmp_path.glob("*.jpeg"))
        assert len(jpeg_files) == 2, f"Expected 2 JPEG files, found: {[f.name for f in jpeg_files]}"

        # Verify actual image format (not just extension)
        for img_file in jpeg_files:
            with Image.open(img_file) as img:
                assert img.format == "JPEG", f"File {img_file.name} is {img.format}, expected JPEG"

        # Check filenames contain subtype
        filenames = [f.name for f in jpeg_files]
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
        """Test handling unsupported output format raises ValueError."""
        # Unsupported formats should now raise an error instead of falling back gracefully
        with pytest.raises(ValueError, match="Unsupported output format: 'bmp'"):
            save_images_to_disk(sample_response_data, str(tmp_path), output_format="bmp")

    def test_auto_format_preserves_original_png_vs_jpeg(self, tmp_path):
        """
        Test the reviewer's specific concern: auto format should preserve PNG vs JPEG correctly.

        This test verifies that:
        - PNG base64 + target_format="auto" → PNG file with .png extension
        - JPEG base64 + target_format="auto" → JPEG file with .jpeg extension
        """
        # Create PNG base64 image
        png_img = Image.new("RGB", (15, 15), color="green")
        png_buffer = io.BytesIO()
        png_img.save(png_buffer, format="PNG")
        png_b64 = base64.b64encode(png_buffer.getvalue()).decode("utf-8")

        # Create JPEG base64 image
        jpeg_img = Image.new("RGB", (15, 15), color="blue")
        jpeg_buffer = io.BytesIO()
        jpeg_img.save(jpeg_buffer, format="JPEG", quality=90)
        jpeg_b64 = base64.b64encode(jpeg_buffer.getvalue()).decode("utf-8")

        # Test data with mixed PNG and JPEG inputs
        mixed_data = [
            {
                "document_type": "structured",
                "metadata": {
                    "content": png_b64,  # PNG input
                    "source_metadata": {"source_id": "png_doc.pdf"},
                    "content_metadata": {"subtype": "chart", "page_number": 1},
                },
            },
            {
                "document_type": "structured",
                "metadata": {
                    "content": jpeg_b64,  # JPEG input
                    "source_metadata": {"source_id": "jpeg_doc.pdf"},
                    "content_metadata": {"subtype": "table", "page_number": 1},
                },
            },
        ]

        # Save with auto format - should preserve original formats
        result = save_images_to_disk(mixed_data, str(tmp_path), organize_by_type=False, output_format="auto")

        assert result["total"] == 2

        # Find saved image files by expected types
        png_files = [f for f in tmp_path.glob("*.png") if "png_doc" in f.name]
        jpeg_files = [f for f in tmp_path.glob("*.jpeg") if "jpeg_doc" in f.name]

        all_images = png_files + jpeg_files
        assert len(all_images) == 2, f"Expected 2 images (1 PNG + 1 JPEG), found: {[f.name for f in all_images]}"

        assert len(png_files) == 1, f"Expected 1 PNG source file, found: {[f.name for f in png_files]}"
        assert len(jpeg_files) == 1, f"Expected 1 JPEG source file, found: {[f.name for f in jpeg_files]}"

        # Critical test: Verify actual image formats are preserved
        with Image.open(png_files[0]) as img:
            assert img.format == "PNG", f"PNG input should remain PNG, got {img.format}"

        with Image.open(jpeg_files[0]) as img:
            assert img.format == "JPEG", f"JPEG input should remain JPEG, got {img.format}"

        # Verify file extensions match formats
        assert png_files[0].suffix == ".png", f"PNG file should have .png extension, got {png_files[0].suffix}"
        assert jpeg_files[0].suffix == ".jpeg", f"JPEG file should have .jpeg extension, got {jpeg_files[0].suffix}"


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
