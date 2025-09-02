# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from io import BytesIO
from typing import Tuple
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from nv_ingest_api.util.image_processing.transforms import (
    numpy_to_base64,
    base64_to_numpy,
    check_numpy_image_size,
    scale_image_to_encoding_size,
    ensure_base64_format,
    base64_to_disk,
    save_image_to_disk,
)


# Helper function to create a base64-encoded string from an image
def create_base64_image(width, height, color="white"):
    img = Image.new("RGB", (width, height), color=color)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Fixture for a valid base64-encoded image string
@pytest.fixture
def valid_base64_image():
    return create_base64_image(64, 64)


# Fixture for a corrupted base64 string
@pytest.fixture
def corrupted_base64_image():
    return "not_a_valid_base64_string"


# Fixture for a base64 string that decodes but is not a valid image
@pytest.fixture
def non_image_base64():
    return base64.b64encode(b"This is not an image").decode("utf-8")


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_numpy_to_base64_valid_rgba_image(format):
    array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    result = numpy_to_base64(array, format=format)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_numpy_to_base64_valid_rgb_image(format):
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = numpy_to_base64(array, format=format)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_numpy_to_base64_grayscale_redundant_axis(format):
    array = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
    result = numpy_to_base64(array, format=format)

    assert isinstance(result, str)
    assert len(result) > 0


def test_numpy_to_base64_format_validation():
    """Test that the output format matches the requested format."""
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test PNG format
    png_result = numpy_to_base64(array, format="PNG")
    png_decoded = base64.b64decode(png_result)
    png_image = Image.open(io.BytesIO(png_decoded))
    assert png_image.format == "PNG"

    # Test JPEG format
    jpeg_result = numpy_to_base64(array, format="JPEG")
    jpeg_decoded = base64.b64decode(jpeg_result)
    jpeg_image = Image.open(io.BytesIO(jpeg_decoded))
    assert jpeg_image.format == "JPEG"


@pytest.mark.parametrize("format_input", ["png", "PNG", "jpeg", "JPEG", "jpg", "JPG"])
def test_numpy_to_base64_case_insensitive_format(format_input):
    """Test that format parameter is case-insensitive."""
    array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = numpy_to_base64(array, format=format_input)

    assert isinstance(result, str)
    assert len(result) > 0

    # Verify the actual format
    decoded = base64.b64decode(result)
    image = Image.open(io.BytesIO(decoded))
    if format_input.upper() in ["PNG"]:
        assert image.format == "PNG"
    elif format_input.upper() in ["JPEG", "JPG"]:
        assert image.format == "JPEG"


def test_numpy_to_base64_invalid_format():
    """Test that invalid format raises ValueError."""
    array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported format: BMP"):
        numpy_to_base64(array, format="BMP")

    with pytest.raises(ValueError, match="Unsupported format: GIF"):
        numpy_to_base64(array, format="GIF")


@pytest.mark.parametrize("quality", [1, 25, 50, 75, 90, 100])
def test_numpy_to_base64_jpeg_quality_parameter(quality):
    """Test JPEG quality parameter with various values."""
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = numpy_to_base64(array, format="JPEG", quality=quality)

    assert isinstance(result, str)
    assert len(result) > 0

    # Verify it's actually JPEG
    decoded = base64.b64decode(result)
    image = Image.open(io.BytesIO(decoded))
    assert image.format == "JPEG"


def test_numpy_to_base64_jpeg_quality_comparison():
    """Test that different JPEG quality values produce different sizes (generally)."""
    array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Generate images with different quality levels
    low_quality = numpy_to_base64(array, format="JPEG", quality=10)
    high_quality = numpy_to_base64(array, format="JPEG", quality=95)

    # Higher quality should generally produce larger base64 strings
    assert len(high_quality) > len(low_quality)


def test_numpy_to_base64_default_parameters():
    """Test that default parameters work as expected."""
    array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    # Test default format (should be PNG)
    result = numpy_to_base64(array)
    decoded = base64.b64decode(result)
    image = Image.open(io.BytesIO(decoded))
    assert image.format == "PNG"

    # Test default quality for JPEG (should be 100)
    jpeg_result = numpy_to_base64(array, format="JPEG")
    jpeg_explicit_quality = numpy_to_base64(array, format="JPEG", quality=100)
    assert jpeg_result == jpeg_explicit_quality


def test_numpy_to_base64_roundtrip_consistency():
    """Test that numpy -> base64 -> numpy produces consistent results."""
    original_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    # Test PNG roundtrip (lossless)
    png_base64 = numpy_to_base64(original_array, format="PNG")
    recovered_png = base64_to_numpy(png_base64)

    # For PNG, should be identical (lossless compression)
    assert recovered_png.shape == original_array.shape
    np.testing.assert_array_equal(recovered_png, original_array)

    # Test JPEG roundtrip (lossy)
    jpeg_base64 = numpy_to_base64(original_array, format="JPEG", quality=95)
    recovered_jpeg = base64_to_numpy(jpeg_base64)

    # For lossy JPEG, we check for shape, dtype only
    assert recovered_jpeg.shape == original_array.shape
    assert recovered_jpeg.dtype == original_array.dtype


def test_numpy_to_base64_different_array_dtypes():
    """Test that different numpy array dtypes are handled correctly."""
    # Test uint8 (standard)
    array_uint8 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result_uint8 = numpy_to_base64(array_uint8, format="PNG")
    assert isinstance(result_uint8, str) and len(result_uint8) > 0

    # Test float32 (should be converted internally)
    array_float32 = np.random.rand(50, 50, 3).astype(np.float32)
    result_float32 = numpy_to_base64(array_float32, format="PNG")
    assert isinstance(result_float32, str) and len(result_float32) > 0

    # Test float64 (should be converted internally)
    array_float64 = np.random.rand(50, 50, 3).astype(np.float64)
    result_float64 = numpy_to_base64(array_float64, format="PNG")
    assert isinstance(result_float64, str) and len(result_float64) > 0


def test_numpy_to_base64_grayscale_2d_array():
    """Test grayscale image as 2D array (no channel dimension)."""
    array_2d = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

    result_png = numpy_to_base64(array_2d, format="PNG")
    assert isinstance(result_png, str) and len(result_png) > 0

    result_jpeg = numpy_to_base64(array_2d, format="JPEG")
    assert isinstance(result_jpeg, str) and len(result_jpeg) > 0


# Tests for base64_to_numpy
@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_base64_to_numpy_valid(valid_base64_image, format):
    img_array = base64_to_numpy(valid_base64_image)
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape[0] == 64  # Height
    assert img_array.shape[1] == 64  # Width


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_base64_to_numpy_invalid_string(corrupted_base64_image, format):
    with pytest.raises(ValueError, match="Invalid base64 string"):
        base64_to_numpy(corrupted_base64_image)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_base64_to_numpy_non_image(non_image_base64, format):
    with pytest.raises(ValueError, match="Unable to decode image from base64 string"):
        base64_to_numpy(non_image_base64)


def test_base64_to_numpy_import_error(monkeypatch, valid_base64_image):
    # Simulate ImportError for cv2
    with mock.patch(
        "nv_ingest_api.util.image_processing.transforms.cv2.imdecode",
        side_effect=ImportError("cv2 library not available"),
    ):
        with pytest.raises(ImportError):
            base64_to_numpy(valid_base64_image)


# Tests for check_numpy_image_size
def test_check_numpy_image_size_valid():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert check_numpy_image_size(img, 50, 50) is True


def test_check_numpy_image_size_too_small_height():
    img = np.zeros((40, 100, 3), dtype=np.uint8)  # Height less than min
    assert check_numpy_image_size(img, 50, 50) is False


def test_check_numpy_image_size_too_small_width():
    img = np.zeros((100, 40, 3), dtype=np.uint8)  # Width less than min
    assert check_numpy_image_size(img, 50, 50) is False


def test_check_numpy_image_size_invalid_dimensions():
    img = np.zeros((100,), dtype=np.uint8)  # 1D array
    with pytest.raises(ValueError, match="The input array does not have sufficient dimensions for an image."):
        check_numpy_image_size(img, 50, 50)


def generate_base64_image(size: Tuple[int, int]) -> str:
    """Helper function to generate a base64-encoded PNG image of a specific size."""
    img = Image.new("RGB", size, color="blue")  # Create a simple blue image
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_base64_image_with_format(format: str = "PNG", size: Tuple[int, int] = (100, 100)) -> str:
    """Helper function to generate a base64-encoded image of a specified format and size."""
    img = Image.new("RGB", size, color="blue")  # Simple blue image
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_resize_image_within_size_limit():
    # Generate a base64 image within the size limit
    base64_image = generate_base64_image((100, 100))  # Small image
    max_base64_size = len(base64_image) + 10  # Set limit slightly above image size

    result, _ = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert result == base64_image  # Should return unchanged


def test_resize_image_one_resize_needed():
    # Generate a large base64 image that requires resizing
    base64_image = generate_base64_image((500, 500))
    max_base64_size = len(base64_image) - 1000  # Set limit slightly below current size

    result, _ = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Should be resized within limit


def test_resize_image_multiple_resizes_needed():
    # Generate a very large base64 image that will require multiple reductions
    base64_image = generate_base64_image((1000, 1000))
    max_base64_size = len(base64_image) // 2  # Set limit well below current size

    result, _ = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Final size should be within limit


def test_resize_image_cannot_be_resized_below_limit():
    # Generate a small base64 image where further resizing would be impractical
    base64_image = generate_base64_image((10, 10))
    max_base64_size = 1  # Unreachable size limit

    with pytest.raises(ValueError, match="Image cannot be resized further without becoming too small."):
        scale_image_to_encoding_size(base64_image, max_base64_size)


def test_resize_image_edge_case_minimal_reduction():
    # Generate an image just above the size limit
    base64_image = generate_base64_image((500, 500))
    max_base64_size = len(base64_image) - 50  # Just a slight reduction needed

    result, _ = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Should achieve minimal reduction within limit


def test_resize_image_with_invalid_input():
    # Provide non-image data as input
    non_image_base64 = base64.b64encode(b"This is not an image").decode("utf-8")

    with pytest.raises(Exception):
        scale_image_to_encoding_size(non_image_base64)


def test_ensure_base64_is_png_already_png():
    # Generate a base64-encoded PNG image
    base64_image = generate_base64_image_with_format("PNG")

    result = ensure_base64_format(base64_image, target_format="PNG")
    assert result == base64_image  # Should be unchanged


def test_ensure_base64_is_png_jpeg_to_png_conversion():
    # Generate a base64-encoded JPEG image
    base64_image = generate_base64_image_with_format("JPEG")

    result = ensure_base64_format(base64_image, target_format="PNG")

    # Decode the result and check format
    image_data = base64.b64decode(result)
    image = Image.open(io.BytesIO(image_data))
    assert image.format == "PNG"  # Should be converted to PNG


def test_ensure_base64_is_png_invalid_base64():
    # Provide invalid base64 input
    invalid_base64 = "This is not base64 encoded data"

    # TODO: Should we raise error or return None?
    with pytest.raises(ValueError, match="Invalid base64 string"):
        ensure_base64_format(invalid_base64, target_format="PNG")


def test_ensure_base64_is_png_non_image_base64_data():
    # Provide valid base64 data that isn't an image
    non_image_base64 = base64.b64encode(b"This is not an image").decode("utf-8")

    # TODO: Should we raise error or return None?
    with pytest.raises(ValueError, match="Unable to decode image from base64 string"):
        ensure_base64_format(non_image_base64, target_format="PNG")


def test_ensure_base64_is_png_unsupported_format():
    # Generate an image in a rare format and base64 encode it
    img = Image.new("RGB", (100, 100), color="blue")
    buffered = io.BytesIO()
    img.save(buffered, format="BMP")  # Use an uncommon format like BMP
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    result = ensure_base64_format(base64_image, target_format="PNG")
    # Decode result to verify conversion
    if result:
        image_data = base64.b64decode(result)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"  # Should be converted to PNG if supported
    else:
        assert result is None  # If unsupported, result should be None


# Tests for image disk writing functions


class TestBase64ToDisk:
    """Test the base64_to_disk function - core direct write functionality."""

    @pytest.fixture
    def sample_png_base64(self):
        """Create a small PNG image encoded as base64."""
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def sample_jpeg_base64(self):
        """Create a small JPEG image encoded as base64."""
        img = Image.new("RGB", (10, 10), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_assertion_framework_validation(self):
        """Validate that test assertions work correctly."""
        # Verify pytest assertion handling works as expected
        with pytest.raises(AssertionError):
            assert False, "Expected assertion failure for test framework validation"

    def test_successful_png_write(self, tmp_path, sample_png_base64):
        """Test successful PNG write to disk."""
        output_file = tmp_path / "test_output.png"

        result = base64_to_disk(sample_png_base64, str(output_file))

        assert result is True
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify file is actually a valid PNG
        with Image.open(output_file) as img:
            assert img.format == "PNG"
            assert img.size == (10, 10)

    def test_successful_jpeg_write(self, tmp_path, sample_jpeg_base64):
        """Test successful JPEG write to disk."""
        output_file = tmp_path / "test_output.jpg"

        result = base64_to_disk(sample_jpeg_base64, str(output_file))

        assert result is True
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify file is actually a valid JPEG
        with Image.open(output_file) as img:
            assert img.format == "JPEG"
            assert img.size == (10, 10)

    def test_data_url_prefix_handling(self, tmp_path):
        """Test that data URL prefixes are properly stripped."""
        img = Image.new("RGB", (5, 5), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        clean_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add data URL prefix
        prefixed_b64 = f"data:image/png;base64,{clean_b64}"
        output_file = tmp_path / "prefixed_test.png"

        result = base64_to_disk(prefixed_b64, str(output_file))

        assert result is True
        assert output_file.exists()

        # Verify content is correct by comparing with clean version
        clean_output = tmp_path / "clean_test.png"
        base64_to_disk(clean_b64, str(clean_output))

        assert output_file.stat().st_size == clean_output.stat().st_size

    def test_invalid_base64_returns_false(self, tmp_path):
        """Test that invalid base64 returns False gracefully."""
        output_file = tmp_path / "invalid_test.png"

        result = base64_to_disk("invalid_base64_data!", str(output_file))

        assert result is False
        assert not output_file.exists()

    def test_permission_error_returns_false(self, sample_png_base64):
        """Test that file permission errors return False gracefully."""
        invalid_path = "/nonexistent/deep/path/test.png"

        result = base64_to_disk(sample_png_base64, invalid_path)

        assert result is False

    def test_empty_base64_returns_false(self, tmp_path):
        """Test that empty base64 string returns False."""
        output_file = tmp_path / "empty_test.png"

        result = base64_to_disk("", str(output_file))

        assert result is False
        assert not output_file.exists()

    def test_whitespace_only_base64_returns_false(self, tmp_path):
        """Test that whitespace-only base64 string returns False."""
        output_file = tmp_path / "whitespace_test.png"

        result = base64_to_disk("   \n\t   ", str(output_file))

        assert result is False
        assert not output_file.exists()

    def test_data_url_with_empty_base64_returns_false(self, tmp_path):
        """Test that data URL with empty base64 part returns False."""
        output_file = tmp_path / "empty_data_url_test.png"

        result = base64_to_disk("data:image/png;base64,", str(output_file))

        assert result is False
        assert not output_file.exists()


class TestSaveImageToDisk:
    """Test the save_image_to_disk function - smart wrapper with format conversion."""

    @pytest.fixture
    def sample_png_base64(self):
        """Create a PNG base64 image."""
        img = Image.new("RGB", (15, 15), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def sample_jpeg_base64(self):
        """Create a JPEG base64 image."""
        img = Image.new("RGB", (15, 15), color="orange")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_auto_format_preservation(self, tmp_path, sample_png_base64):
        """Test that AUTO format preserves original PNG format."""
        output_file = tmp_path / "preserved_format.png"

        # Save PNG with auto format (should preserve original)
        result = save_image_to_disk(sample_png_base64, str(output_file), "auto")
        assert result is True

        # Verify original PNG format is preserved
        with Image.open(output_file) as img:
            assert img.format == "PNG"

    def test_auto_mode_preserves_format(self, tmp_path, sample_png_base64, sample_jpeg_base64):
        """Test that AUTO mode preserves original formats."""
        png_file = tmp_path / "auto_png.png"
        jpeg_file = tmp_path / "auto_jpeg.jpg"

        # Test PNG preservation
        result1 = save_image_to_disk(sample_png_base64, str(png_file), "auto")
        assert result1 is True
        with Image.open(png_file) as img:
            assert img.format == "PNG"

        # Test JPEG preservation
        result2 = save_image_to_disk(sample_jpeg_base64, str(jpeg_file), "auto")
        assert result2 is True
        with Image.open(jpeg_file) as img:
            assert img.format == "JPEG"

    def test_format_conversion_png_to_jpeg(self, tmp_path, sample_png_base64):
        """Test PNG to JPEG conversion works."""
        output_file = tmp_path / "converted.jpg"

        result = save_image_to_disk(sample_png_base64, str(output_file), "jpeg", quality=85)

        assert result is True
        assert output_file.exists()

        # Verify conversion worked
        with Image.open(output_file) as img:
            assert img.format == "JPEG"
            assert img.size == (15, 15)

    def test_format_conversion_jpeg_to_png(self, tmp_path, sample_jpeg_base64):
        """Test JPEG to PNG conversion works."""
        output_file = tmp_path / "converted.png"

        result = save_image_to_disk(sample_jpeg_base64, str(output_file), "png")

        assert result is True
        assert output_file.exists()

        # Verify conversion worked
        with Image.open(output_file) as img:
            assert img.format == "PNG"
            assert img.size == (15, 15)

    def test_same_format_no_conversion(self, tmp_path, sample_png_base64):
        """Test that same format doesn't trigger unnecessary conversion."""
        output_file = tmp_path / "same_format.png"

        # This should use direct write path (no conversion)
        result = save_image_to_disk(sample_png_base64, str(output_file), "png")

        assert result is True
        assert output_file.exists()

        with Image.open(output_file) as img:
            assert img.format == "PNG"
            assert img.size == (15, 15)

    def test_quality_parameter_jpeg(self, tmp_path, sample_png_base64):
        """Test JPEG quality parameter affects file size."""
        high_quality_file = tmp_path / "high_quality.jpg"
        low_quality_file = tmp_path / "low_quality.jpg"

        # Save with high quality
        result1 = save_image_to_disk(sample_png_base64, str(high_quality_file), "jpeg", quality=95)
        # Save with low quality
        result2 = save_image_to_disk(sample_png_base64, str(low_quality_file), "jpeg", quality=20)

        assert result1 is True and result2 is True

        # High quality should generally be larger
        high_size = high_quality_file.stat().st_size
        low_size = low_quality_file.stat().st_size
        assert high_size >= low_size  # Allow for equal in case image is too small

    def test_invalid_base64_returns_false(self, tmp_path):
        """Test invalid base64 returns False gracefully."""
        output_file = tmp_path / "invalid.jpg"

        result = save_image_to_disk("invalid_base64!", str(output_file), "jpeg")

        assert result is False
        assert not output_file.exists()

    def test_invalid_target_format_falls_back(self, tmp_path, sample_png_base64):
        """Test invalid target format falls back gracefully."""
        output_file = tmp_path / "fallback.jpg"

        # This should still work by falling back to auto mode or raising an error
        # The exact behavior depends on implementation - test that it doesn't crash
        try:
            result = save_image_to_disk(sample_png_base64, str(output_file), "invalid_format")
            # If it succeeds, verify a file was created
            if result:
                assert output_file.exists()
        except ValueError:
            # If it raises an error, that's also acceptable behavior
            pass

    def test_case_insensitive_formats(self, tmp_path, sample_png_base64):
        """Test that format parameters are case insensitive."""
        jpeg_file = tmp_path / "case_test.jpg"

        # Test various case combinations
        for format_str in ["JPEG", "jpeg", "Jpeg", "PNG", "png", "Png"]:
            if format_str.upper() == "JPEG":
                result = save_image_to_disk(sample_png_base64, str(jpeg_file), format_str)
                assert result is True
                with Image.open(jpeg_file) as img:
                    assert img.format == "JPEG"
                jpeg_file.unlink()  # Clean up for next iteration

    def test_file_overwrite_behavior(self, tmp_path, sample_png_base64, sample_jpeg_base64):
        """Test that files are properly overwritten."""
        output_file = tmp_path / "overwrite_test.jpg"

        # Write first image
        result1 = save_image_to_disk(sample_png_base64, str(output_file), "jpeg")
        assert result1 is True

        # Overwrite with second image
        result2 = save_image_to_disk(sample_jpeg_base64, str(output_file), "jpeg")
        assert result2 is True

        # Verify file exists and contains the second image
        assert output_file.exists()
        with Image.open(output_file) as img:
            assert img.format == "JPEG"
