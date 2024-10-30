# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from unittest import mock

from nv_ingest.util.image_processing.transforms import numpy_to_base64, base64_to_numpy, check_numpy_image_size


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
