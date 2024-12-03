from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

_MODULE_UNDER_TEST = "nv_ingest.util.nim.paddle"


@pytest.fixture
def paddle_ocr_model():
    return PaddleOCRModelInterface(paddle_version="0.2.1")


@pytest.fixture
def legacy_paddle_ocr_model():
    return PaddleOCRModelInterface(paddle_version="0.2.0")


def test_prepare_data_for_inference(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

        assert "image_array" in result
        assert result["image_array"].shape == (100, 100, 3)
        assert paddle_ocr_model._width == 100
        assert paddle_ocr_model._height == 100


def test_format_input_grpc(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.preprocess_image_for_paddle") as mock_preprocess:
        mock_preprocess.return_value = np.zeros((32, 32, 3))

        data = {"image_array": np.zeros((32, 32, 3))}
        result = paddle_ocr_model.format_input(data, protocol="grpc")

        assert result.shape == (1, 32, 32, 3)


def test_format_input_http(paddle_ocr_model):
    data = {"base64_image": "mock_base64_string"}
    result = paddle_ocr_model.format_input(data, protocol="http")

    assert "input" in result
    assert result["input"][0]["type"] == "image_url"
    assert result["input"][0]["url"] == "data:image/png;base64,mock_base64_string"


def test_format_input_http_legacy(legacy_paddle_ocr_model):
    data = {"base64_image": "mock_base64_string"}
    result = legacy_paddle_ocr_model.format_input(data, protocol="http")

    assert "messages" in result
    content = result["messages"][0]["content"][0]
    assert content["type"] == "image_url"
    assert content["image_url"]["url"] == "data:image/png;base64,mock_base64_string"


def test_parse_output_http_pseudo_markdown(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

    http_response = {
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

    result = paddle_ocr_model.parse_output(http_response, protocol="http")
    assert result == "| mock_text |\n"


def test_parse_output_http_simple(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

    http_response = {
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

    result = paddle_ocr_model.parse_output(http_response, protocol="http", table_content_format="simple")
    assert result == "mock_text"


def test_parse_output_grpc_pseudo_markdown(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

    bboxes = b"[[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]"
    texts = b'[["mock_text"]]'
    scores = b"[[0.99]]"

    grpc_response = np.array([bboxes, texts, scores])

    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc")
    assert result == "| mock_text |\n"


def test_parse_output_grpc_simple(paddle_ocr_model):
    with patch(f"{_MODULE_UNDER_TEST}.base64_to_numpy") as mock_base64_to_numpy:
        mock_base64_to_numpy.return_value = np.zeros((100, 100, 3))

        data = {"base64_image": "mock_base64_string"}
        result = paddle_ocr_model.prepare_data_for_inference(data)

    bboxes = b"[[[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]]]"
    texts = b'[["mock_text"]]'
    scores = b"[[0.99]]"

    grpc_response = np.array([bboxes, texts, scores])

    result = paddle_ocr_model.parse_output(grpc_response, protocol="grpc", table_content_format="simple")
    assert result == "mock_text"
