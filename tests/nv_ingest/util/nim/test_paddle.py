import json
from unittest.mock import patch

import pytest
import base64
import io
import numpy as np
from PIL import Image

from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

_MODULE_UNDER_TEST = "nv_ingest.util.nim.paddle"


def create_valid_base64_image(width=32, height=32, color=(127, 127, 127)):
    """
    Create a simple (width x height) solid-color image in-memory
    and return its Base64-encoded PNG string.
    """
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded_img


def create_valid_grpc_response_batched(text="mock_text"):
    """
    Create a gRPC response in shape (3, n).
      - row 0 => bounding boxes
      - row 1 => text predictions
      - row 2 => extra data / metadata

    For a single item, we get (3,1).
    """
    # Example bounding boxes: one list with a single bounding box of 4 corners
    bounding_boxes = [[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]
    # Example text predictions
    text_predictions = [[text]]
    # Some arbitrary extra data
    extra_data = "mock_extra_data"

    # Encode each row as JSON bytes
    bb_json = json.dumps(bounding_boxes).encode("utf-8")
    txt_json = json.dumps(text_predictions).encode("utf-8")
    extra_json = json.dumps(extra_data).encode("utf-8")

    # Return shape => (3,1)
    #   row 0 -> bounding_boxes
    #   row 1 -> text_predictions
    #   row 2 -> extra_data
    return np.array([[bb_json], [txt_json], [extra_json]], dtype=object)


@pytest.fixture
def paddle_ocr_model():
    return PaddleOCRModelInterface()


@pytest.fixture
def mock_paddle_http_response():
    return {
        "data": [
            {
                "text_detections": [
                    {
                        "text_prediction": {"text": "mock_text", "confidence": 0.99},
                        "bounding_box": {
                            "points": [
                                {"x": 0.1, "y": 0.2},
                                {"x": 0.2, "y": 0.2},
                                {"x": 0.2, "y": 0.3},
                                {"x": 0.1, "y": 0.3},
                            ],
                            "confidence": None,
                        },
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_paddle_grpc_response():
    bboxes = b"[[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]"
    texts = b'[["mock_text"]]'
    scores = b"[[0.99]]"

    return np.array([bboxes, texts, scores])


def test_prepare_data_for_inference(paddle_ocr_model):
    """
    Previously, we expected 'image_array' in result and stored _width, _height.
    Now, we expect 'image_arrays' with exactly one element if there's a single base64_image.
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

        # Now we store a list of arrays under 'image_arrays'
        assert "image_arrays" in result
        assert len(result["image_arrays"]) == 1
        assert result["image_arrays"][0].shape == (100, 100, 3)
        assert data["image_dims"][0] == (100, 100)


def test_format_input_grpc(paddle_ocr_model):
    """
    For gRPC, the images are processed with preprocess_image_for_paddle and batched.
    The test verifies that:
      - The returned batched input has the correct shape: (batch_size, H, W, C).
      - The accompanying batch data correctly includes the original image_arrays and image_dims.
    """
    with patch(f"{_MODULE_UNDER_TEST}.preprocess_image_for_paddle") as mock_preprocess:
        # Force the preprocessing to return an array with shape (32, 32, 3)
        mock_preprocess.return_value = np.zeros((32, 32, 3))
        # Supply both "image_arrays" and "image_dims" as required by the implementation.
        img = np.zeros((32, 32, 3))
        data = {"image_arrays": [img], "image_dims": [(32, 32)]}
        batches, batch_data = paddle_ocr_model.format_input(data, protocol="grpc", max_batch_size=1)
        # Verify batched input shape; preprocessing adds a batch dimension via np.expand_dims.
        result = batches[0]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 32, 32, 3)
        # Verify that the batch_data correctly reflects the original image and dimensions.
        assert isinstance(batch_data, list)
        assert len(batch_data) == 1
        bd = batch_data[0]
        assert "image_arrays" in bd and "image_dims" in bd
        assert bd["image_arrays"] == [img]
        assert bd["image_dims"] == [(32, 32)]


def test_format_input_http(paddle_ocr_model):
    """
    For HTTP in non-legacy mode, after prepare_data_for_inference (which populates both
    "image_arrays" and "image_dims"), the formatted payload should use the new structure:
       {"input": [ {"type": "image_url", "url": "data:image/png;base64,..."}, ... ]}
    In addition, the accompanying batch data should contain the original images and dimensions.
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    # prepare_data_for_inference adds "image_arrays" and "image_dims"
    data = paddle_ocr_model.prepare_data_for_inference(data)
    batches, batch_data = paddle_ocr_model.format_input(data, protocol="http", max_batch_size=1)
    result = batches[0]
    # Check that the new (non-legacy) branch is used.
    assert "input" in result
    assert isinstance(result["input"], list)
    assert len(result["input"]) == 1
    first_item = result["input"][0]
    assert first_item["type"] == "image_url"
    assert first_item["url"].startswith("data:image/png;base64,")
    assert len(first_item["url"]) > len("data:image/png;base64,")
    # Also verify the returned batch data.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 1
    bd = batch_data[0]
    assert "image_arrays" in bd and "image_dims" in bd


def test_parse_output_http_pseudo_markdown(paddle_ocr_model, mock_paddle_http_response):
    """
    parse_output should now return a list of (content, table_content_format) tuples.
    e.g. [("| mock_text |\n", "pseudo_markdown")]
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((3, 100, 100))

        data = {"base64_image": "mock_base64_string"}
        _ = paddle_ocr_model.prepare_data_for_inference(data)

    result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http")
    # It's a list with one tuple => (content, format).
    assert len(result) == 1
    assert result[0][0] == "| mock_text |"
    assert result[0][1] == "pseudo_markdown"


def test_parse_output_http_simple(paddle_ocr_model, mock_paddle_http_response):
    """
    The new parse_output also returns a list of (content, format).
    If we pass table_content_format="simple", the code uses " ".join(...) logic.
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        _ = paddle_ocr_model.prepare_data_for_inference(data)

    result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http")
    # Should be (bounding_boxes, text_predictions)
    assert len(result) == 1
    assert result[0][0] == [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]
    assert result[0][1] == ["mock_text"]


def test_parse_output_grpc_simple(paddle_ocr_model):
    """
    parse_output with gRPC & table_content_format="simple" => [("mock_text", "simple")]
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    paddle_ocr_model.prepare_data_for_inference(data)

    data["_dimensions"] = [
        {
            "new_width": 1,
            "new_height": 1,
            "pad_width": 0,
            "pad_height": 0,
            "scale_factor": 1.0,
        }
    ]
    grpc_response = create_valid_grpc_response_batched("mock_text")
    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc", data=data)

    assert len(result) == 1
    assert result[0][0] == [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]
    assert result[0][1] == ["mock_text"]
