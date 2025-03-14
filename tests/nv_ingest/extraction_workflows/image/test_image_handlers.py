import io
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from nv_ingest.extraction_workflows.image.image_handlers import convert_svg_to_bitmap
from nv_ingest.extraction_workflows.image.image_handlers import extract_page_element_images
from nv_ingest.extraction_workflows.image.image_handlers import load_and_preprocess_image
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent


def test_load_and_preprocess_image_jpeg():
    """Test loading and preprocessing a JPEG image."""
    # Create a small sample image and save it to a BytesIO stream as JPEG
    image = Image.new("RGB", (10, 10), color="red")
    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")
    image_stream.seek(0)

    # Load and preprocess the image
    result = load_and_preprocess_image(image_stream)

    # Check the output type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)
    assert result.dtype == np.float32
    assert np.all(result[:, :, 0] == 254)  # All red pixels
    assert np.all(result[:, :, 1] == 0)  # No green
    assert np.all(result[:, :, 2] == 0)  # No blue


def test_load_and_preprocess_image_png():
    """Test loading and preprocessing a PNG image."""
    # Create a small sample image and save it to a BytesIO stream as PNG
    image = Image.new("RGB", (5, 5), color="blue")
    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")
    image_stream.seek(0)

    # Load and preprocess the image
    result = load_and_preprocess_image(image_stream)

    # Check the output type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 5, 3)
    assert result.dtype == np.float32
    assert np.all(result[:, :, 0] == 0)  # No red
    assert np.all(result[:, :, 1] == 0)  # No green
    assert np.all(result[:, :, 2] == 255)  # All blue pixels


def test_load_and_preprocess_image_invalid_format():
    """Test that an invalid image format raises an error."""
    # Create a BytesIO stream with non-image content
    invalid_stream = io.BytesIO(b"This is not an image file")

    # Expect an OSError when trying to open a non-image stream
    try:
        load_and_preprocess_image(invalid_stream)
    except OSError as e:
        assert "cannot identify image file" in str(e)


def test_load_and_preprocess_image_corrupt_image():
    """Test that a corrupt image raises an error."""
    # Create a valid JPEG header but corrupt the rest
    corrupt_stream = io.BytesIO(b"\xFF\xD8\xFF\xE0" + b"\x00" * 10)

    # Expect an OSError when trying to open a corrupt image stream
    try:
        load_and_preprocess_image(corrupt_stream)
    except OSError as e:
        assert "cannot identify image file" in str(e)


@pytest.mark.xfail
def test_convert_svg_to_bitmap_basic_svg():
    """Test converting a simple SVG to a bitmap image."""
    # Sample SVG image data (a small red square)
    svg_data = b"""
    <svg width="10" height="10" xmlns="http://www.w3.org/2000/svg">
        <rect width="10" height="10" style="fill:red;"/>
    </svg>
    """
    image_stream = io.BytesIO(svg_data)

    # Convert SVG to bitmap
    result = convert_svg_to_bitmap(image_stream)

    # Check the output type, shape, and color values
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)
    assert result.dtype == np.float32
    assert np.all(result[:, :, 0] == 255)  # Red channel fully on
    assert np.all(result[:, :, 1] == 0)  # Green channel off
    assert np.all(result[:, :, 2] == 0)  # Blue channel off


@pytest.mark.xfail
def test_convert_svg_to_bitmap_large_svg():
    """Test converting a larger SVG to ensure scalability."""
    # Large SVG image data (blue rectangle 100x100)
    svg_data = b"""
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="100" height="100" style="fill:blue;"/>
    </svg>
    """
    image_stream = io.BytesIO(svg_data)

    # Convert SVG to bitmap
    result = convert_svg_to_bitmap(image_stream)

    # Check the output type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.float32
    assert np.all(result[:, :, 0] == 0)  # Red channel off
    assert np.all(result[:, :, 1] == 0)  # Green channel off
    assert np.all(result[:, :, 2] == 255)  # Blue channel fully on


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Mock function to simulate cropping an image."""
    h1, w1, h2, w2 = bbox
    return image[int(h1) : int(h2), int(w1) : int(w2)]


def test_extract_page_element_images_empty_annotations():
    """Test when annotation_dict has no objects to extract."""
    annotation_dict = {"table": [], "chart": []}
    original_image = np.random.rand(640, 640, 3)
    page_elements = []

    extract_page_element_images(annotation_dict, original_image, 0, page_elements)

    # Expect no entries added to page_elements since there are no objects
    assert page_elements == []


def test_extract_page_element_images_single_table():
    """Test extraction with a single table bounding box."""
    annotation_dict = {"table": [[64, 64, 192, 192, 0.8]], "chart": []}
    original_image = np.random.rand(640, 640, 3)
    page_elements = []

    extract_page_element_images(annotation_dict, original_image, 0, page_elements)

    # Expect one entry in page_elements for the table
    assert len(page_elements) == 1
    page_idx, cropped_image_data = page_elements[0]
    assert page_idx == 0
    assert isinstance(cropped_image_data, CroppedImageWithContent)

    # Verify attribute values
    assert cropped_image_data.content == ""
    assert cropped_image_data.type_string == "table"
    assert cropped_image_data.bbox == (64, 64, 192, 192)  # Scaled bounding box from (0.1, 0.1, 0.3, 0.3)
    assert cropped_image_data.max_width == 640
    assert cropped_image_data.max_height == 640
    assert isinstance(cropped_image_data.image, str)  # Assuming the image is base64-encoded


def test_extract_page_element_images_single_chart():
    """Test extraction with a single chart bounding box."""
    annotation_dict = {"table": [], "chart": [[256, 256, 384, 384, 0.9]]}
    original_image = np.random.rand(640, 640, 3)
    page_elements = []

    extract_page_element_images(annotation_dict, original_image, 1, page_elements)

    # Expect one entry in page_elements for the chart
    assert len(page_elements) == 1
    page_idx, cropped_image_data = page_elements[0]
    assert page_idx == 1
    assert isinstance(cropped_image_data, CroppedImageWithContent)
    assert cropped_image_data.type_string == "chart"
    assert cropped_image_data.bbox == (256, 256, 384, 384)  # Scaled bounding box


def test_extract_page_element_images_multiple_objects():
    """Test extraction with multiple table and chart objects."""
    annotation_dict = {
        "table": [[0.1, 0.1, 0.3, 0.3, 0.8], [0.5, 0.5, 0.7, 0.7, 0.85]],
        "chart": [[0.2, 0.2, 0.4, 0.4, 0.9]],
    }
    original_image = np.random.rand(640, 640, 3)
    page_elements = []

    extract_page_element_images(annotation_dict, original_image, 2, page_elements)

    # Expect three entries in page_elements: two tables and one chart
    assert len(page_elements) == 3
    for page_idx, cropped_image_data in page_elements:
        assert page_idx == 2
        assert isinstance(cropped_image_data, CroppedImageWithContent)
        assert cropped_image_data.type_string in ["table", "chart"]
        assert cropped_image_data.bbox is not None  # Bounding box should be defined


def test_extract_page_element_images_invalid_bounding_box():
    """Test with an invalid bounding box to check handling of incorrect coordinates."""
    annotation_dict = {"table": [[704, 704, 960, 960, 0.9]], "chart": []}  # Out of bounds
    original_image = np.random.rand(640, 640, 3)
    page_elements = []

    extract_page_element_images(annotation_dict, original_image, 3, page_elements)

    # Verify that the function processes the bounding box as is
    assert len(page_elements) == 1
    page_idx, cropped_image_data = page_elements[0]
    assert page_idx == 3
    assert isinstance(cropped_image_data, CroppedImageWithContent)
    assert cropped_image_data.type_string == "table"
    assert cropped_image_data.bbox == (704, 704, 960, 960)  # Scaled bounding box with out-of-bounds values
