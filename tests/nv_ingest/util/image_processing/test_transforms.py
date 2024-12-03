# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import numpy as np
import pytest

from io import BytesIO
from PIL import Image
from typing import Tuple
from unittest import mock

from nv_ingest.util.image_processing.transforms import numpy_to_base64, base64_to_numpy, check_numpy_image_size, \
    scale_image_to_encoding_size, ensure_base64_is_png


# Helper function to create a base64-encoded string from an image
def create_base64_image(width, height, color="white"):
    img = Image.new('RGB', (width, height), color=color)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
    return base64.b64encode(b"This is not an image").decode('utf-8')


def test_numpy_to_base64_valid_rgba_image():
    array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0


def test_numpy_to_base64_valid_rgb_image():
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0


def test_numpy_to_base64_grayscale_redundant_axis():
    array = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
    result = numpy_to_base64(array)

    assert isinstance(result, str)
    assert len(result) > 0


# Tests for base64_to_numpy
def test_base64_to_numpy_valid(valid_base64_image):
    img_array = base64_to_numpy(valid_base64_image)
    assert isinstance(img_array, np.ndarray)
    assert img_array.shape[0] == 64  # Height
    assert img_array.shape[1] == 64  # Width


def test_base64_to_numpy_invalid_string(corrupted_base64_image):
    with pytest.raises(ValueError, match="Invalid base64 string"):
        base64_to_numpy(corrupted_base64_image)


def test_base64_to_numpy_non_image(non_image_base64):
    with pytest.raises(ValueError, match="Unable to decode image from base64 string"):
        base64_to_numpy(non_image_base64)


def test_base64_to_numpy_import_error(monkeypatch, valid_base64_image):
    # Simulate ImportError for PIL by patching import_module
    with mock.patch("PIL.Image.open", side_effect=ImportError("PIL library not available")):
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


def generate_base64_image_with_format(format: str = 'PNG', size: Tuple[int, int] = (100, 100)) -> str:
    """Helper function to generate a base64-encoded image of a specified format and size."""
    img = Image.new("RGB", size, color="blue")  # Simple blue image
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_resize_image_within_size_limit():
    # Generate a base64 image within the size limit
    base64_image = generate_base64_image((100, 100))  # Small image
    max_base64_size = len(base64_image) + 10  # Set limit slightly above image size

    result = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert result == base64_image  # Should return unchanged


def test_resize_image_one_resize_needed():
    # Generate a large base64 image that requires resizing
    base64_image = generate_base64_image((500, 500))
    max_base64_size = len(base64_image) - 1000  # Set limit slightly below current size

    result = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Should be resized within limit


def test_resize_image_multiple_resizes_needed():
    # Generate a very large base64 image that will require multiple reductions
    base64_image = generate_base64_image((1000, 1000))
    max_base64_size = len(base64_image) // 2  # Set limit well below current size

    result = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Final size should be within limit


def test_resize_image_cannot_be_resized_below_limit():
    # Generate a small base64 image where further resizing would be impractical
    base64_image = generate_base64_image((10, 10))
    max_base64_size = 1  # Unreachable size limit

    with pytest.raises(ValueError, match="height and width must be > 0"):
        scale_image_to_encoding_size(base64_image, max_base64_size)


def test_resize_image_edge_case_minimal_reduction():
    # Generate an image just above the size limit
    base64_image = generate_base64_image((500, 500))
    max_base64_size = len(base64_image) - 50  # Just a slight reduction needed

    result = scale_image_to_encoding_size(base64_image, max_base64_size)
    assert len(result) <= max_base64_size  # Should achieve minimal reduction within limit


def test_resize_image_with_invalid_input():
    # Provide non-image data as input
    non_image_base64 = base64.b64encode(b"This is not an image").decode("utf-8")

    with pytest.raises(Exception):
        scale_image_to_encoding_size(non_image_base64)


def test_ensure_base64_is_png_already_png():
    # Generate a base64-encoded PNG image
    base64_image = generate_base64_image_with_format("PNG")

    result = ensure_base64_is_png(base64_image)
    assert result == base64_image  # Should be unchanged


def test_ensure_base64_is_png_jpeg_to_png_conversion():
    # Generate a base64-encoded JPEG image
    base64_image = generate_base64_image_with_format("JPEG")

    result = ensure_base64_is_png(base64_image)

    # Decode the result and check format
    image_data = base64.b64decode(result)
    image = Image.open(io.BytesIO(image_data))
    assert image.format == "PNG"  # Should be converted to PNG


def test_ensure_base64_is_png_invalid_base64():
    # Provide invalid base64 input
    invalid_base64 = "This is not base64 encoded data"

    result = ensure_base64_is_png(invalid_base64)
    assert result is None  # Should return None for invalid input


def test_ensure_base64_is_png_non_image_base64_data():
    # Provide valid base64 data that isn’t an image
    non_image_base64 = base64.b64encode(b"This is not an image").decode("utf-8")

    result = ensure_base64_is_png(non_image_base64)
    assert result is None  # Should return None for non-image data


def test_ensure_base64_is_png_unsupported_format():
    # Generate an image in a rare format and base64 encode it
    img = Image.new("RGB", (100, 100), color="blue")
    buffered = io.BytesIO()
    img.save(buffered, format="BMP")  # Use an uncommon format like BMP
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    result = ensure_base64_is_png(base64_image)
    # Decode result to verify conversion
    if result:
        image_data = base64.b64decode(result)
        image = Image.open(io.BytesIO(image_data))
        assert image.format == "PNG"  # Should be converted to PNG if supported
    else:
        assert result is None  # If unsupported, result should be None
