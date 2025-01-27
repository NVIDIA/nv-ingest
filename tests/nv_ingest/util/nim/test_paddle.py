import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import base64
import io
import numpy as np
from PIL import Image

from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle
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
    Create a gRPC response in shape (B,2), column0 => bounding boxes, column1 => text predictions
    bounding_boxes is a list-of-lists-of-lists, text is a list-of-lists
    """
    # Example bounding boxes: 1 set with 4 corners
    bounding_boxes = [[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]
    text_predictions = [[text]]

    # Encode each column as JSON bytes
    bb_json = json.dumps(bounding_boxes).encode("utf-8")
    txt_json = json.dumps(text_predictions).encode("utf-8")

    # Return shape => (1,2)
    return np.array([[bb_json, txt_json]], dtype=object)


@pytest.fixture
def paddle_ocr_model():
    return PaddleOCRModelInterface(paddle_version="0.2.1")


@pytest.fixture
def legacy_paddle_ocr_model():
    return PaddleOCRModelInterface(paddle_version="0.2.0")


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

        # We also store dimensions in self._dims
        assert paddle_ocr_model._dims == [(100, 100)]


def test_format_input_grpc(paddle_ocr_model):
    """
    Now we place the images under 'image_arrays' and return a batched input for gRPC.
    """
    with patch(f"{_MODULE_UNDER_TEST}.preprocess_image_for_paddle") as mock_preprocess:
        mock_preprocess.return_value = np.zeros((32, 32, 3))

        # For gRPC, we rely on 'image_arrays'
        data = {"image_arrays": [np.zeros((32, 32, 3))]}
        result = paddle_ocr_model.format_input(data, protocol="grpc")

        # shape => (batch_size, 32, 32, 3). We have batch_size=1.
        assert result.shape == (1, 32, 32, 3)


def test_format_input_http(paddle_ocr_model):
    """
    For HTTP, if not legacy, we expect a payload with 'input': [...]
    containing a valid base64 PNG.
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}

    # MUST call prepare_data_for_inference first, so data has "image_arrays"
    data = paddle_ocr_model.prepare_data_for_inference(data)

    result = paddle_ocr_model.format_input(data, protocol="http")

    # For non-legacy => {"input": [ {"type":"image_url","url": "..."} ]}
    assert "input" in result
    assert len(result["input"]) == 1
    first_item = result["input"][0]
    assert first_item["type"] == "image_url"
    assert first_item["url"].startswith("data:image/png;base64,")
    # Optionally, check that it's non-empty after the prefix
    assert len(first_item["url"]) > len("data:image/png;base64,")


def test_format_input_http_legacy(legacy_paddle_ocr_model):
    """
    For legacy version (<0.2.1-rc2), the code should produce the 'messages' structure.
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}

    data = legacy_paddle_ocr_model.prepare_data_for_inference(data)
    result = legacy_paddle_ocr_model.format_input(data, protocol="http")

    # Now we expect => {"messages":[{"content":[ { "type":"image_url",
    # "image_url":{"url":"data:image/png;base64,..."}}, ... ]}]}
    assert "messages" in result
    assert len(result["messages"]) == 1
    content_list = result["messages"][0]["content"]
    assert len(content_list) == 1
    item = content_list[0]
    assert item["type"] == "image_url"
    assert item["image_url"]["url"].startswith("data:image/png;base64,")


def test_parse_output_http_pseudo_markdown(paddle_ocr_model, mock_paddle_http_response):
    """
    parse_output should now return a list of (content, table_content_format) tuples.
    e.g. [("| mock_text |\n", "pseudo_markdown")]
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        _ = paddle_ocr_model.prepare_data_for_inference(data)

    result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http")
    # It's a list with one tuple => (content, format).
    assert len(result) == 1
    assert result[0][0] == "| mock_text |\n"
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

    result = paddle_ocr_model.parse_output(mock_paddle_http_response, protocol="http", table_content_format="simple")
    # Should be [("mock_text", "simple")]
    assert len(result) == 1
    assert result[0][0] == "mock_text"
    assert result[0][1] == "simple"


def test_parse_output_http_simple_legacy(legacy_paddle_ocr_model):
    """
    For the legacy version, parse_output also returns a list of (content, format),
    but it forces 'simple' format if the user requested something else.
    """
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        _ = legacy_paddle_ocr_model.prepare_data_for_inference(data)

    mock_legacy_paddle_http_response = {"data": [{"content": "mock_text"}]}

    result = legacy_paddle_ocr_model.parse_output(
        mock_legacy_paddle_http_response, protocol="http", table_content_format="foo"
    )
    # Expect => [("mock_text", "simple")]
    assert len(result) == 1
    assert result[0][0] == "mock_text"
    assert result[0][1] == "simple"


def test_parse_output_grpc_pseudo_markdown(paddle_ocr_model):
    """
    Provide a valid (1,2) shape for bounding boxes & text.
    The interface should parse them into [("| mock_text |\n", "pseudo_markdown")].
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    paddle_ocr_model.prepare_data_for_inference(data)

    # Create a valid shape => (1,2), with bounding-box JSON, text JSON
    grpc_response = create_valid_grpc_response_batched("mock_text")

    # parse_output with default => pseudo_markdown for non-legacy
    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc")

    assert len(result) == 1
    content, fmt = result[0]
    # content might contain a markdown table row => "| mock_text |"
    assert "mock_text" in content
    assert fmt == "pseudo_markdown"


def test_parse_output_grpc_simple(paddle_ocr_model):
    """
    parse_output with gRPC & table_content_format="simple" => [("mock_text", "simple")]
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    paddle_ocr_model.prepare_data_for_inference(data)

    grpc_response = create_valid_grpc_response_batched("mock_text")
    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc", table_content_format="simple")

    assert len(result) == 1
    content, fmt = result[0]
    assert content == "mock_text"
    assert fmt == "simple"


def test_parse_output_grpc_legacy(legacy_paddle_ocr_model):
    """
    For legacy gRPC, we also return a list of (content, format).
    We force 'simple' if table_content_format was not 'simple'.
    """
    valid_b64 = create_valid_base64_image()
    data = {"base64_image": valid_b64}
    legacy_paddle_ocr_model.prepare_data_for_inference(data)

    grpc_response = create_valid_grpc_response_batched("mock_text")

    # Pass a non-"simple" format => should be forced to 'simple' in legacy
    result = legacy_paddle_ocr_model.parse_output(grpc_response, protocol="grpc", table_content_format="foo")
    assert len(result) == 1
    assert result[0][0] == "mock_text"
    assert result[0][1] == "simple"
