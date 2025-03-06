import base64
import random
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from nv_ingest.util.nim.yolox import YoloxPageElementsModelInterface
from nv_ingest.util.nim.yolox import YOLOX_PAGE_V1_FINAL_SCORE
from nv_ingest.util.nim.yolox import YOLOX_PAGE_V2_FINAL_SCORE


@pytest.fixture
def model_interface():
    return YoloxPageElementsModelInterface(yolox_model_name="nv-yolox-page-elements-v2")


def create_test_image(width=800, height=600, color=(255, 0, 0)):
    """
    Creates a simple RGB image as a NumPy array.

    Parameters:
    ----------
    width : int
        Width of the image.
    height : int
        Height of the image.
    color : tuple
        RGB color of the image.

    Returns:
    -------
    np.ndarray
        The generated image as a NumPy array.
    """
    image = Image.new("RGB", (width, height), color)
    return np.array(image)


def create_base64_image(width=1024, height=1024, color=(255, 0, 0)):
    """
    Creates a base64-encoded PNG image string.

    Parameters:
    ----------
    width : int
        Width of the image.
    height : int
        Height of the image.
    color : tuple
        RGB color of the image.

    Returns:
    -------
    str
        Base64-encoded string of the image.
    """
    with BytesIO() as buffer:
        image = Image.new("RGB", (width, height), color)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_name_returns_yolox(model_interface):
    model_interface = YoloxPageElementsModelInterface()
    assert model_interface.name() == "yolox-page-elements"


def test_prepare_data_for_inference_valid(model_interface):
    images = [create_test_image(), create_test_image(width=640, height=480)]
    input_data = {"images": images}
    result = model_interface.prepare_data_for_inference(input_data)
    assert "original_image_shapes" in result
    assert len(result["original_image_shapes"]) == len(images)
    for original_shape, image in zip(result["original_image_shapes"], images):
        assert original_shape[:2] == image.shape[:2]


def test_prepare_data_for_inference_missing_images(model_interface):
    input_data = {}
    with pytest.raises(KeyError, match="'images'"):
        model_interface.prepare_data_for_inference(input_data)


def test_prepare_data_for_inference_invalid_image_format(model_interface):
    """
    Test prepare_data_for_inference with images that are not NumPy arrays.
    Expects a ValueError to be raised.
    """
    images = ["not_a_numpy_array", create_test_image()]
    input_data = {"images": images}
    with pytest.raises(ValueError):
        model_interface.prepare_data_for_inference(input_data)


def test_format_input_grpc(model_interface):
    """
    Test that for the gRPC protocol:
      - The input images are resized and reordered to have shape (B, 3, 1024, 1024).
      - The returned batch data includes the original images and their original shapes.
    """
    images = [create_test_image(), create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)

    # format_input returns a tuple: (batched_inputs, formatted_batch_data)
    batched_inputs, batch_data = model_interface.format_input(prepared_data, "grpc", max_batch_size=2)

    # Check batched_inputs is a list of NumPy arrays
    assert isinstance(batched_inputs, list)
    # Since max_batch_size=2 and we provided 2 images, there should be one chunk
    assert len(batched_inputs) == 1
    batched_array = batched_inputs[0]

    # Verify dtype and shape: expected shape (number_of_images, 3, 1024, 1024)
    assert batched_array.dtype == np.float32
    assert batched_array.shape[0] == len(images)
    assert batched_array.shape[1:] == (3, 1024, 1024)

    # Verify that the batch_data correctly includes the original images and their shapes.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1  # one chunk since all images were batched together
    bd = batch_data[0]
    assert "images" in bd and "original_image_shapes" in bd
    # The original images should match the ones provided in input_data.
    assert bd["images"] == images
    # The original shapes stored during prepare_data_for_inference should be returned.
    assert bd["original_image_shapes"] == prepared_data["original_image_shapes"]


def test_format_input_http(model_interface):
    """
    Test that for the HTTP protocol:
      - The formatted payload is a JSON-serializable dict with an "input" key containing a list
        of image dictionaries. Each image dictionary must have a "type" key with value "image_url"
        and a "url" starting with "data:image/png;base64,".
      - The accompanying batch data correctly contains the original images and their original shapes.
    """
    images = [create_test_image(), create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)

    # format_input returns a tuple: (payload_batches, formatted_batch_data)
    payload_batches, batch_data = model_interface.format_input(prepared_data, "http", max_batch_size=2)

    # Verify payload structure.
    assert isinstance(payload_batches, list)
    # Since max_batch_size=2 and we have 2 images, we expect one payload chunk.
    assert len(payload_batches) == 1
    payload = payload_batches[0]
    assert "input" in payload
    assert isinstance(payload["input"], list)
    for item in payload["input"]:
        assert "type" in item
        assert item["type"] == "image_url"
        assert "url" in item
        assert item["url"].startswith("data:image/png;base64,")

    # Verify that batch_data is returned correctly.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1  # one batch chunk
    bd = batch_data[0]
    assert "images" in bd and "original_image_shapes" in bd
    assert bd["images"] == images
    assert bd["original_image_shapes"] == prepared_data["original_image_shapes"]


def test_format_input_invalid_protocol(model_interface):
    images = [create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(prepared_data, "invalid_protocol", max_batch_size=1)


def test_parse_output_grpc(model_interface):
    response = np.random.rand(2, 100, 85).astype(np.float32)
    parsed_output = model_interface.parse_output(response, "grpc")
    assert isinstance(parsed_output, np.ndarray)
    assert parsed_output.shape == response.shape
    assert parsed_output.dtype == np.float32


def test_parse_output_http_valid(model_interface):
    response = {
        "data": [
            {
                "index": 0,
                "bounding_boxes": {
                    "table": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2, "confidence": 0.9}],
                    "chart": [{"x_min": 0.3, "y_min": 0.3, "x_max": 0.4, "y_max": 0.4, "confidence": 0.8}],
                    "title": [{"x_min": 0.5, "y_min": 0.5, "x_max": 0.6, "y_max": 0.6, "confidence": 0.95}],
                    "infographic": [{"x_min": 0.7, "y_min": 0.7, "x_max": 0.8, "y_max": 0.8, "confidence": 0.85}],
                },
            },
            {
                "index": 1,
                "bounding_boxes": {
                    "table": [{"x_min": 0.15, "y_min": 0.15, "x_max": 0.25, "y_max": 0.25, "confidence": 0.85}],
                    "chart": [{"x_min": 0.35, "y_min": 0.35, "x_max": 0.45, "y_max": 0.45, "confidence": 0.75}],
                    "title": [{"x_min": 0.55, "y_min": 0.55, "x_max": 0.65, "y_max": 0.65, "confidence": 0.92}],
                    "infographic": [{"x_min": 0.75, "y_min": 0.75, "x_max": 0.85, "y_max": 0.85, "confidence": 0.82}],
                },
            },
        ]
    }
    parsed_output = model_interface.parse_output(response, "http")
    assert parsed_output == [
        {
            "table": [[0.1, 0.1, 0.2, 0.2, 0.9]],
            "chart": [[0.3, 0.3, 0.4, 0.4, 0.8]],
            "title": [[0.5, 0.5, 0.6, 0.6, 0.95]],
            "infographic": [[0.7, 0.7, 0.8, 0.8, 0.85]],
        },
        {
            "table": [[0.15, 0.15, 0.25, 0.25, 0.85]],
            "chart": [[0.35, 0.35, 0.45, 0.45, 0.75]],
            "title": [[0.55, 0.55, 0.65, 0.65, 0.92]],
            "infographic": [[0.75, 0.75, 0.85, 0.85, 0.82]],
        },
    ]


def test_parse_output_invalid_protocol(model_interface):
    response = "Some response"
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.parse_output(response, "invalid_protocol")


def test_process_inference_results_grpc(model_interface):
    output_array = np.random.rand(2, 100, 85).astype(np.float32)
    original_image_shapes = [(800, 600, 3), (640, 480, 3)]
    inference_results = model_interface.process_inference_results(
        output_array,
        "grpc",
        original_image_shapes=original_image_shapes,
    )
    assert isinstance(inference_results, list)
    assert len(inference_results) == 2
    for result in inference_results:
        assert isinstance(result, dict)
        if "table" in result:
            for bbox in result["table"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["table"]
        if "chart" in result:
            for bbox in result["chart"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["chart"]
        if "infographic" in result:
            for bbox in result["infographic"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["infographic"]
        if "title" in result:
            assert isinstance(result["title"], list)


def test_process_inference_results_http(model_interface):
    output = [
        {
            "table": [[random.random() for _ in range(5)] for _ in range(10)],
            "chart": [[random.random() for _ in range(5)] for _ in range(10)],
            "title": [[random.random() for _ in range(5)] for _ in range(10)],
        }
        for _ in range(10)
    ]
    original_image_shapes = [[100, 100] for _ in range(10)]
    inference_results = model_interface.process_inference_results(
        output,
        "http",
        original_image_shapes=original_image_shapes,
    )
    assert isinstance(inference_results, list)
    assert len(inference_results) == 10
    for result in inference_results:
        assert isinstance(result, dict)
        if "table" in result:
            for bbox in result["table"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["table"]
        if "chart" in result:
            for bbox in result["chart"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["chart"]
        if "infographic" in result:
            for bbox in result["infographic"]:
                assert bbox[4] >= YOLOX_PAGE_V2_FINAL_SCORE["infographic"]
        if "title" in result:
            assert isinstance(result["title"], list)
