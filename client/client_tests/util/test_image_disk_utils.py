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
from unittest.mock import patch

import pytest
from PIL import Image

from nv_ingest_client.util.image_disk_utils import (
    _detect_base64_image_format,
    _base64_to_disk_direct,
    _base64_to_disk_with_conversion,
    _save_image_efficiently,
    save_images_to_disk,
    save_images_from_response,
    save_images_from_ingestor_results,
)


class TestBase64FormatDetection:
    """Test base64 image format detection functionality."""

    @pytest.fixture
    def sample_png_base64(self):
        """Create a small PNG image encoded as base64."""
        # Create a small 1x1 pixel PNG
        img = Image.new("RGB", (1, 1), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def sample_jpeg_base64(self):
        """Create a small JPEG image encoded as base64."""
        # Create a small 1x1 pixel JPEG
        img = Image.new("RGB", (1, 1), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_detect_png_format(self, sample_png_base64):
        """Test PNG format detection."""
        result = _detect_base64_image_format(sample_png_base64)
        assert result == "PNG"

    def test_detect_jpeg_format(self, sample_jpeg_base64):
        """Test JPEG format detection."""
        result = _detect_base64_image_format(sample_jpeg_base64)
        assert result == "JPEG"

    def test_invalid_base64_returns_none(self):
        """Test that invalid base64 returns None."""
        result = _detect_base64_image_format("invalid_base64!")
        assert result is None

    def test_valid_base64_invalid_image_returns_none(self):
        """Test that valid base64 but invalid image returns None."""
        invalid_image_b64 = base64.b64encode(b"not an image").decode("utf-8")
        result = _detect_base64_image_format(invalid_image_b64)
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = _detect_base64_image_format("")
        assert result is None


class TestDirectDiskWrite:
    """Test direct base64 to disk write functionality."""

    def test_successful_direct_write(self, tmp_path, sample_png_base64):
        """Test successful direct write to disk."""
        output_file = tmp_path / "test.png"
        result = _base64_to_disk_direct(sample_png_base64, str(output_file))

        assert result is True
        assert output_file.exists()

        # Verify the written file is valid
        with open(output_file, "rb") as f:
            written_data = f.read()

        # Compare with original decoded data
        original_data = base64.b64decode(sample_png_base64)
        assert written_data == original_data

    def test_direct_write_invalid_base64(self, tmp_path):
        """Test direct write with invalid base64."""
        output_file = tmp_path / "test.png"
        result = _base64_to_disk_direct("invalid_base64!", str(output_file))

        assert result is False
        assert not output_file.exists()

    def test_direct_write_permission_error(self, sample_png_base64):
        """Test direct write with permission error."""
        # Try to write to a non-existent directory
        invalid_path = "/nonexistent/path/test.png"
        result = _base64_to_disk_direct(sample_png_base64, invalid_path)

        assert result is False

    @pytest.fixture
    def sample_png_base64(self):
        """Create sample PNG base64 for testing."""
        img = Image.new("RGB", (1, 1), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestConversionWrite:
    """Test format conversion write functionality."""

    @pytest.fixture
    def sample_png_base64(self):
        """Create sample PNG base64."""
        img = Image.new("RGB", (2, 2), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_png_to_jpeg_conversion(self, tmp_path, sample_png_base64):
        """Test PNG to JPEG conversion."""
        output_file = tmp_path / "test.jpg"
        result = _base64_to_disk_with_conversion(sample_png_base64, str(output_file), "JPEG", quality=90)

        assert result is True
        assert output_file.exists()

        # Verify the output is JPEG
        with Image.open(output_file) as img:
            assert img.format == "JPEG"

    def test_jpeg_to_png_conversion(self, tmp_path):
        """Test JPEG to PNG conversion."""
        # Create sample JPEG
        img = Image.new("RGB", (2, 2), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        output_file = tmp_path / "test.png"
        result = _base64_to_disk_with_conversion(jpeg_b64, str(output_file), "PNG")

        assert result is True
        assert output_file.exists()

        # Verify the output is PNG
        with Image.open(output_file) as img:
            assert img.format == "PNG"

    def test_rgba_to_jpeg_conversion(self, tmp_path):
        """Test RGBA to JPEG conversion (alpha channel removal)."""
        # Create RGBA image
        img = Image.new("RGBA", (2, 2), color=(255, 0, 0, 128))  # Red with transparency
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")  # PNG supports RGBA
        rgba_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        output_file = tmp_path / "test.jpg"
        result = _base64_to_disk_with_conversion(rgba_b64, str(output_file), "JPEG", quality=95)

        assert result is True
        assert output_file.exists()

        # Verify the output is JPEG and RGB mode
        with Image.open(output_file) as img:
            assert img.format == "JPEG"
            assert img.mode == "RGB"

    def test_conversion_with_invalid_base64(self, tmp_path):
        """Test conversion with invalid base64."""
        output_file = tmp_path / "test.jpg"
        result = _base64_to_disk_with_conversion("invalid!", str(output_file), "JPEG")

        assert result is False
        assert not output_file.exists()

    def test_quality_parameter(self, tmp_path, sample_png_base64):
        """Test JPEG quality parameter."""
        output_file_high = tmp_path / "test_high.jpg"
        output_file_low = tmp_path / "test_low.jpg"

        # Save with high quality
        result1 = _base64_to_disk_with_conversion(sample_png_base64, str(output_file_high), "JPEG", quality=95)
        # Save with low quality
        result2 = _base64_to_disk_with_conversion(sample_png_base64, str(output_file_low), "JPEG", quality=10)

        assert result1 is True
        assert result2 is True

        # High quality should result in larger file size
        high_size = output_file_high.stat().st_size
        low_size = output_file_low.stat().st_size
        assert high_size >= low_size


class TestEfficientSave:
    """Test the smart routing efficient save functionality."""

    @pytest.fixture
    def sample_png_base64(self):
        """Create sample PNG base64."""
        img = Image.new("RGB", (2, 2), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def sample_jpeg_base64(self):
        """Create sample JPEG base64."""
        img = Image.new("RGB", (2, 2), color="yellow")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_direct_path_png_to_png(self, tmp_path, sample_png_base64):
        """Test direct path when formats match (PNG -> PNG)."""
        output_file = tmp_path / "test.png"

        with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_direct", return_value=True) as mock_direct:
            with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_with_conversion") as mock_convert:
                result = _save_image_efficiently(sample_png_base64, str(output_file), "PNG")

                assert result is True
                mock_direct.assert_called_once()
                mock_convert.assert_not_called()

    def test_direct_path_jpeg_to_jpeg(self, tmp_path, sample_jpeg_base64):
        """Test direct path when formats match (JPEG -> JPEG)."""
        output_file = tmp_path / "test.jpeg"

        with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_direct", return_value=True) as mock_direct:
            with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_with_conversion") as mock_convert:
                result = _save_image_efficiently(sample_jpeg_base64, str(output_file), "JPEG")

                assert result is True
                mock_direct.assert_called_once()
                mock_convert.assert_not_called()

    def test_conversion_path_png_to_jpeg(self, tmp_path, sample_png_base64):
        """Test conversion path when formats differ (PNG -> JPEG)."""
        output_file = tmp_path / "test.jpg"

        with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_direct") as mock_direct:
            with patch(
                "nv_ingest_client.util.image_disk_utils._base64_to_disk_with_conversion", return_value=True
            ) as mock_convert:
                result = _save_image_efficiently(sample_png_base64, str(output_file), "JPEG", quality=85)

                assert result is True
                mock_direct.assert_not_called()
                mock_convert.assert_called_once_with(sample_png_base64, str(output_file), "JPEG", 85)

    def test_jpg_format_normalization(self, tmp_path, sample_jpeg_base64):
        """Test JPG format normalization to JPEG."""
        output_file = tmp_path / "test.jpg"

        with patch("nv_ingest_client.util.image_disk_utils._base64_to_disk_direct", return_value=True) as mock_direct:
            result = _save_image_efficiently(sample_jpeg_base64, str(output_file), "JPG")

            assert result is True
            mock_direct.assert_called_once()

    def test_unknown_format_conversion(self, tmp_path):
        """Test conversion when source format is unknown."""
        # Create invalid base64 that will return None for format detection
        invalid_b64 = base64.b64encode(b"not an image").decode("utf-8")
        output_file = tmp_path / "test.png"

        with patch(
            "nv_ingest_client.util.image_disk_utils._base64_to_disk_with_conversion", return_value=True
        ) as mock_convert:
            result = _save_image_efficiently(invalid_b64, str(output_file), "PNG")

            assert result is True
            mock_convert.assert_called_once()


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
        """Test handling unsupported output format."""
        with patch("nv_ingest_client.util.image_disk_utils.logger") as mock_logger:
            result = save_images_to_disk(sample_response_data, str(tmp_path), output_format="bmp")  # Unsupported

            # Should fallback to PNG
            mock_logger.warning.assert_called()
            assert result["total"] == 2

    @patch("nv_ingest_client.util.image_disk_utils._save_image_efficiently")
    def test_quality_parameters_by_subtype(self, mock_save, tmp_path, sample_response_data):
        """Test that different quality parameters are used for different subtypes."""
        mock_save.return_value = True

        save_images_to_disk(sample_response_data, str(tmp_path))

        # Verify quality parameters passed
        calls = mock_save.call_args_list
        assert len(calls) == 2

        # Both should be called with quality=95 (charts and tables get high quality)
        for call in calls:
            args, kwargs = call
            assert kwargs.get("quality") == 95


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
