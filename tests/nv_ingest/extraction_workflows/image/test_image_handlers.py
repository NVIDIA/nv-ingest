import urllib
from pyexpat import ExpatError
from xml.etree.ElementTree import ParseError

from wand.exceptions import WandException

from nv_ingest.extraction_workflows.image.image_handlers import load_and_preprocess_image, convert_svg_to_bitmap, \
    extract_table_and_chart_images
from PIL import Image
import io
import numpy as np
from typing import List, Tuple

from nv_ingest.extraction_workflows.image.image_handlers import process_inference_results
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


def test_process_inference_results_basic_case():
    """Test process_inference_results with a typical valid input."""

    # Simulated model output array for a single image with several detections.
    # Array format is (batch_size, num_detections, 85) - 80 classes + 5 box coordinates
    # For simplicity, use random values for the boxes and class predictions.
    output_array = np.zeros((1, 3, 85), dtype=np.float32)

    # Mock bounding box coordinates
    output_array[0, 0, :4] = [0.5, 0.5, 0.2, 0.2]  # x_center, y_center, width, height
    output_array[0, 1, :4] = [0.6, 0.6, 0.2, 0.2]
    output_array[0, 2, :4] = [0.7, 0.7, 0.2, 0.2]

    # Mock object confidence scores
    output_array[0, :, 4] = [0.8, 0.9, 0.85]

    # Mock class scores (set class 1 with highest confidence for simplicity)
    output_array[0, 0, 5 + 1] = 0.7
    output_array[0, 1, 5 + 1] = 0.75
    output_array[0, 2, 5 + 1] = 0.72

    original_image_shapes = [(640, 640)]  # Original shape of the image before resizing

    # Process inference results with thresholds that should retain all mock detections
    results = process_inference_results(
        output_array,
        original_image_shapes,
        num_classes=80,
        conf_thresh=0.5,
        iou_thresh=0.5,
        min_score=0.1,
        final_thresh=0.3,
    )

    # Check output structure
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)

    # Validate bounding box scaling and structure
    assert "chart" in results[0] or "table" in results[0]
    if "chart" in results[0]:
        assert isinstance(results[0]["chart"], list)
        assert len(results[0]["chart"]) > 0
        # Check bounding box format for each detected "chart" item (5 values per box)
        for bbox in results[0]["chart"]:
            assert len(bbox) == 5  # [x1, y1, x2, y2, score]
            assert bbox[4] >= 0.3  # score meets final threshold

    print("Processed inference results:", results)


def test_process_inference_results_multiple_images():
    """Test with multiple images to verify batch processing."""
    # Simulate model output with 2 images and 3 detections each
    output_array = np.zeros((2, 3, 85), dtype=np.float32)
    # Set bounding boxes and confidence for the mock detections
    output_array[0, 0, :5] = [0.5, 0.5, 0.2, 0.2, 0.8]
    output_array[0, 1, :5] = [0.6, 0.6, 0.2, 0.2, 0.7]
    output_array[1, 0, :5] = [0.4, 0.4, 0.1, 0.1, 0.9]
    # Assign class confidences for classes 0 and 1
    output_array[0, 0, 5 + 1] = 0.75
    output_array[0, 1, 5 + 1] = 0.65
    output_array[1, 0, 5 + 0] = 0.8

    original_image_shapes = [(640, 640), (800, 800)]

    results = process_inference_results(
        output_array,
        original_image_shapes,
        num_classes=80,
        conf_thresh=0.5,
        iou_thresh=0.5,
        min_score=0.1,
        final_thresh=0.3,
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, dict)
        if "chart" in result:
            assert all(len(bbox) == 5 and bbox[4] >= 0.3 for bbox in result["chart"])


def test_process_inference_results_high_confidence_threshold():
    """Test with a high confidence threshold to verify filtering."""
    output_array = np.zeros((1, 5, 85), dtype=np.float32)
    # Set low confidence scores below the threshold
    output_array[0, :, 4] = [0.2, 0.3, 0.4, 0.4, 0.2]
    output_array[0, :, 5] = [0.5] * 5  # Class confidence

    original_image_shapes = [(640, 640)]

    results = process_inference_results(
        output_array,
        original_image_shapes,
        num_classes=80,
        conf_thresh=0.9,  # High confidence threshold
        iou_thresh=0.5,
        min_score=0.1,
        final_thresh=0.3,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] == {}  # No detections should pass the high confidence threshold


def test_process_inference_results_varied_num_classes():
    """Test compatibility with different model class counts."""
    output_array = np.zeros((1, 3, 25), dtype=np.float32)  # 20 classes + 5 box coords
    # Assign box, object confidence, and class scores
    output_array[0, 0, :5] = [0.5, 0.5, 0.2, 0.2, 0.8]
    output_array[0, 1, :5] = [0.6, 0.6, 0.3, 0.3, 0.7]
    output_array[0, 0, 5 + 1] = 0.9  # Assign highest confidence to class 1

    original_image_shapes = [(640, 640)]

    results = process_inference_results(
        output_array,
        original_image_shapes,
        num_classes=20,  # Different class count
        conf_thresh=0.5,
        iou_thresh=0.5,
        min_score=0.1,
        final_thresh=0.3,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "chart" in results[0]
    assert len(results[0]["chart"]) > 0  # Verify detections processed correctly with 20 classes


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Mock function to simulate cropping an image."""
    h1, w1, h2, w2 = bbox
    return image[int(h1):int(h2), int(w1):int(w2)]


def test_extract_table_and_chart_images_empty_annotations():
    """Test when annotation_dict has no objects to extract."""
    annotation_dict = {"table": [], "chart": []}
    original_image = np.random.rand(640, 640, 3)
    tables_and_charts = []

    extract_table_and_chart_images(annotation_dict, original_image, 0, tables_and_charts)

    # Expect no entries added to tables_and_charts since there are no objects
    assert tables_and_charts == []


def test_extract_table_and_chart_images_single_table():
    """Test extraction with a single table bounding box."""
    annotation_dict = {"table": [[0.1, 0.1, 0.3, 0.3, 0.8]], "chart": []}
    original_image = np.random.rand(640, 640, 3)
    tables_and_charts = []

    extract_table_and_chart_images(annotation_dict, original_image, 0, tables_and_charts)

    # Expect one entry in tables_and_charts for the table
    assert len(tables_and_charts) == 1
    page_idx, cropped_image_data = tables_and_charts[0]
    assert page_idx == 0
    assert isinstance(cropped_image_data, CroppedImageWithContent)

    # Verify attribute values
    assert cropped_image_data.content == ""
    assert cropped_image_data.type_string == "table"
    assert cropped_image_data.bbox == (64, 64, 192, 192)  # Scaled bounding box from (0.1, 0.1, 0.3, 0.3)
    assert cropped_image_data.max_width == 640
    assert cropped_image_data.max_height == 640
    assert isinstance(cropped_image_data.image, str)  # Assuming the image is base64-encoded


def test_extract_table_and_chart_images_single_chart():
    """Test extraction with a single chart bounding box."""
    annotation_dict = {"table": [], "chart": [[0.4, 0.4, 0.6, 0.6, 0.9]]}
    original_image = np.random.rand(640, 640, 3)
    tables_and_charts = []

    extract_table_and_chart_images(annotation_dict, original_image, 1, tables_and_charts)

    # Expect one entry in tables_and_charts for the chart
    assert len(tables_and_charts) == 1
    page_idx, cropped_image_data = tables_and_charts[0]
    assert page_idx == 1
    assert isinstance(cropped_image_data, CroppedImageWithContent)
    assert cropped_image_data.type_string == "chart"
    assert cropped_image_data.bbox == (256, 256, 384, 384)  # Scaled bounding box


def test_extract_table_and_chart_images_multiple_objects():
    """Test extraction with multiple table and chart objects."""
    annotation_dict = {
        "table": [[0.1, 0.1, 0.3, 0.3, 0.8], [0.5, 0.5, 0.7, 0.7, 0.85]],
        "chart": [[0.2, 0.2, 0.4, 0.4, 0.9]]
    }
    original_image = np.random.rand(640, 640, 3)
    tables_and_charts = []

    extract_table_and_chart_images(annotation_dict, original_image, 2, tables_and_charts)

    # Expect three entries in tables_and_charts: two tables and one chart
    assert len(tables_and_charts) == 3
    for page_idx, cropped_image_data in tables_and_charts:
        assert page_idx == 2
        assert isinstance(cropped_image_data, CroppedImageWithContent)
        assert cropped_image_data.type_string in ["table", "chart"]
        assert cropped_image_data.bbox is not None  # Bounding box should be defined


def test_extract_table_and_chart_images_invalid_bounding_box():
    """Test with an invalid bounding box to check handling of incorrect coordinates."""
    annotation_dict = {"table": [[1.1, 1.1, 1.5, 1.5, 0.9]], "chart": []}  # Out of bounds
    original_image = np.random.rand(640, 640, 3)
    tables_and_charts = []

    extract_table_and_chart_images(annotation_dict, original_image, 3, tables_and_charts)

    # Verify that the function processes the bounding box as is
    assert len(tables_and_charts) == 1
    page_idx, cropped_image_data = tables_and_charts[0]
    assert page_idx == 3
    assert isinstance(cropped_image_data, CroppedImageWithContent)
    assert cropped_image_data.type_string == "table"
    assert cropped_image_data.bbox == (704, 704, 960, 960)  # Scaled bounding box with out-of-bounds values
