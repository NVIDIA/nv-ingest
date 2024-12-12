import pytest
import numpy as np
from io import BytesIO
import base64
from PIL import Image

from nv_ingest.util.nim.yolox import YoloxModelInterface


@pytest.fixture
def model_interface():
    return YoloxModelInterface()


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
    assert model_interface.name() == "yolox"


def test_prepare_data_for_inference_valid(model_interface):
    images = [create_test_image(), create_test_image(width=640, height=480)]
    input_data = {"images": images}
    result = model_interface.prepare_data_for_inference(input_data)
    assert "resized_images" in result
    assert "original_image_shapes" in result
    assert len(result["resized_images"]) == len(images)
    assert len(result["original_image_shapes"]) == len(images)
    for original_shape, resized_image, image in zip(result["original_image_shapes"], result["resized_images"], images):
        assert resized_image.shape == (1024, 1024, 3)
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
    images = [create_test_image(), create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)
    formatted_input = model_interface.format_input(prepared_data, "grpc")
    assert isinstance(formatted_input, np.ndarray)
    assert formatted_input.dtype == np.float32
    assert formatted_input.shape[0] == len(images)
    assert formatted_input.shape[1:] == (3, 1024, 1024)


def test_format_input_http(model_interface):
    images = [create_test_image(), create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)
    formatted_input = model_interface.format_input(prepared_data, "http")
    assert "messages" in formatted_input
    assert isinstance(formatted_input["messages"], list)
    for message in formatted_input["messages"]:
        assert "content" in message
        for content in message["content"]:
            assert "type" in content
            assert content["type"] == "image_url"
            assert "image_url" in content
            assert "url" in content["image_url"]
            assert content["image_url"]["url"].startswith("data:image/png;base64,")


def test_format_input_invalid_protocol(model_interface):
    images = [create_test_image()]
    input_data = {"images": images}
    prepared_data = model_interface.prepare_data_for_inference(input_data)
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.format_input(prepared_data, "invalid_protocol")


def test_parse_output_grpc(model_interface):
    response = np.random.rand(2, 100, 85).astype(np.float32)
    parsed_output = model_interface.parse_output(response, "grpc")
    assert isinstance(parsed_output, np.ndarray)
    assert parsed_output.shape == response.shape
    assert parsed_output.dtype == np.float32


def test_parse_output_http_valid(model_interface):
    response = {
        "data": [
            [
                {
                    "type": "table",
                    "bboxes": [{"xmin": 0.1, "ymin": 0.1, "xmax": 0.2, "ymax": 0.2, "confidence": 0.9}],
                },
                {
                    "type": "chart",
                    "bboxes": [{"xmin": 0.3, "ymin": 0.3, "xmax": 0.4, "ymax": 0.4, "confidence": 0.8}],
                },
                {"type": "title", "bboxes": [{"xmin": 0.5, "ymin": 0.5, "xmax": 0.6, "ymax": 0.6, "confidence": 0.95}]},
            ],
            [
                {
                    "type": "table",
                    "bboxes": [{"xmin": 0.15, "ymin": 0.15, "xmax": 0.25, "ymax": 0.25, "confidence": 0.85}],
                },
                {
                    "type": "chart",
                    "bboxes": [{"xmin": 0.35, "ymin": 0.35, "xmax": 0.45, "ymax": 0.45, "confidence": 0.75}],
                },
                {
                    "type": "title",
                    "bboxes": [{"xmin": 0.55, "ymin": 0.55, "xmax": 0.65, "ymax": 0.65, "confidence": 0.92}],
                },
            ],
        ]
    }
    scaling_factors = [(1.0, 1.0), (1.0, 1.0)]
    data = {"scaling_factors": scaling_factors}
    parsed_output = model_interface.parse_output(response, "http", data)
    assert isinstance(parsed_output, np.ndarray)
    assert parsed_output.shape == (2, 3, 85)
    assert parsed_output.dtype == np.float32


def test_parse_output_invalid_protocol(model_interface):
    response = "Some response"
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        model_interface.parse_output(response, "invalid_protocol")


def test_process_inference_results(model_interface):
    output_array = np.random.rand(2, 100, 85).astype(np.float32)
    original_image_shapes = [(800, 600, 3), (640, 480, 3)]
    inference_results = model_interface.process_inference_results(
        output_array,
        original_image_shapes=original_image_shapes,
        num_classes=3,
        conf_thresh=0.5,
        iou_thresh=0.4,
        min_score=0.3,
        final_thresh=0.6,
    )
    assert isinstance(inference_results, list)
    assert len(inference_results) == 2
    for result in inference_results:
        assert isinstance(result, dict)
        if "table" in result:
            for bbox in result["table"]:
                assert bbox[4] >= 0.6
        if "chart" in result:
            for bbox in result["chart"]:
                assert bbox[4] >= 0.6
        if "title" in result:
            assert isinstance(result["title"], list)
