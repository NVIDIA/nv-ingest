import json
from unittest.mock import MagicMock
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
    Now we place the images under 'image_arrays' and return a batched input for gRPC.
    """
    with patch(f"{_MODULE_UNDER_TEST}.preprocess_image_for_paddle") as mock_preprocess:
        mock_preprocess.return_value = (np.zeros((32, 32, 3)), {})

        # For gRPC, we rely on 'image_arrays'
        data = {"image_arrays": [np.zeros((32, 32, 3))]}
        result = paddle_ocr_model.format_input(data, protocol="grpc")

        # shape => (batch_size, 32, 32, 3). We have batch_size=1.
        assert result.shape == (1, 32, 32, 3)


def test_format_input_http(paddle_ocr_model):
    """
    For HTTP, we expect a payload with 'input': [...] containing a valid base64 PNG.
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}

    # MUST call prepare_data_for_inference first, so data has "image_arrays"
    data = paddle_ocr_model.prepare_data_for_inference(data)

    result = paddle_ocr_model.format_input(data, protocol="http")

    # {"input": [ {"type":"image_url","url": "..."} ]}
    assert "input" in result
    assert len(result["input"]) == 1
    first_item = result["input"][0]
    assert first_item["type"] == "image_url"
    assert first_item["url"].startswith("data:image/png;base64,")
    # Optionally, check that it's non-empty after the prefix
    assert len(first_item["url"]) > len("data:image/png;base64,")


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
