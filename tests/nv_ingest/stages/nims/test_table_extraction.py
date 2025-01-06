import base64
from io import BytesIO
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import requests
from PIL import Image

from nv_ingest.stages.nim.table_extraction import _extract_table_data
from nv_ingest.stages.nim.table_extraction import _update_metadata
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

# Constants for minimum image size
PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32

MODULE_UNDER_TEST = "nv_ingest.stages.nim.table_extraction"


# Mocked PaddleOCRModelInterface
class MockPaddleOCRModelInterface:
    def __init__(self, paddle_version=None):
        self.paddle_version = paddle_version

    def prepare_data_for_inference(self, data):
        return data

    def format_input(self, data, protocol, **kwargs):
        return data

    def parse_output(self, response, protocol, **kwargs):
        table_content = (
            "Chart 1 This chart shows some gadgets, and some very fictitious costs "
            "Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 "
            "$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium "
            "desk fan Cost"
        )
        table_content_format = "simple"

        return table_content, table_content_format

    def process_inference_results(self, output, **kwargs):
        return output


@pytest.fixture
def mock_paddle_client_and_requests():
    # Create a mocked PaddleOCRModelInterface
    model_interface = MockPaddleOCRModelInterface()
    # Create a mocked NimClient with the mocked model_interface
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))

    # Mock response for requests.post
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"index": 0, "content": "Mocked content from PaddleOCR", "object": "string"}],
        "model": "paddleocr",
        "usage": None,
    }

    # Patching create_inference_client and requests.post
    with patch(f"{MODULE_UNDER_TEST}.create_inference_client", return_value=paddle_client) as mock_create_client, patch(
        "requests.post", return_value=mock_response
    ) as mock_requests_post:
        yield paddle_client, mock_create_client, mock_requests_post


# Fixture for common mock setup (inference failure)
# Fixture for common mock setup (inference failure)
@pytest.fixture
def mock_paddle_client_and_requests_failure():
    # Create a mocked PaddleOCRModelInterface
    model_interface = MockPaddleOCRModelInterface()
    # Create a mocked NimClient with the mocked model_interface
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))

    # Mock the infer method to raise an exception to simulate an inference failure
    paddle_client.infer = Mock(side_effect=Exception("Inference error"))

    # Patching create_inference_client
    with patch(f"{MODULE_UNDER_TEST}.create_inference_client", return_value=paddle_client) as mock_create_client:
        yield paddle_client, mock_create_client


# Fixture to create a sample image and encode it in base64
@pytest.fixture
def base64_encoded_image():
    # Create a simple image using PIL
    img = Image.new("RGB", (64, 64), color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    # Encode the image to base64
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str


# Fixture for a small image (below minimum size)
# Fixture for small base64-encoded image
@pytest.fixture
def base64_encoded_small_image():
    # Generate a small image (e.g., 10x10 pixels) and encode it in base64
    small_image = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", small_image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image


# Test function for _extract_table_data with an image that is too small
def test_extract_table_data_image_too_small(base64_encoded_small_image):
    data = {
        "metadata": [
            {
                "content": base64_encoded_small_image,
                "content_metadata": {"type": "image", "subtype": "table"},
                "table_metadata": {"table_content": ""},
            }
        ]
    }
    df = pd.DataFrame(data)

    # Mock 'validated_config' and its attributes
    validated_config = Mock()
    stage_config = Mock()
    validated_config.stage_config = stage_config
    stage_config.paddle_endpoints = ("mock_endpoint_grpc", "mock_endpoint_http")
    stage_config.auth_token = "mock_token"
    stage_config.paddle_infer_protocol = "http"

    trace_info = {}

    # Mock the NimClient to return a specific result
    mock_nim_client = Mock(spec=NimClient)

    # Simulate that inference is skipped due to small image
    def mock_infer(data, model_name, **kwargs):
        # Simulate behavior when image is too small: return empty result or raise an exception
        raise Exception("Image too small for inference")

    mock_nim_client.infer.side_effect = mock_infer

    # Patch 'create_inference_client' to return the mocked NimClient
    with patch(f"{MODULE_UNDER_TEST}.create_inference_client", return_value=mock_nim_client), patch(
        f"{MODULE_UNDER_TEST}.get_version", return_value="0.1.0"
    ):
        # Since the image is too small, we expect the table_content to remain unchanged
        updated_df, _ = _extract_table_data(df, {}, validated_config, trace_info)

    # Verify that 'table_content' remains empty
    assert updated_df.loc[0, "metadata"]["table_metadata"]["table_content"] == ""


# Fixture for a sample DataFrame
@pytest.fixture
def sample_dataframe(base64_encoded_image):
    data = {
        "metadata": [
            {
                "content": base64_encoded_image,
                "content_metadata": {"type": "structured", "subtype": "table"},
                "table_metadata": {"table_content": ""},
            }
        ]
    }
    df = pd.DataFrame(data)
    return df


# Fixture for DataFrame with missing metadata
@pytest.fixture
def dataframe_missing_metadata():
    data = {"other_data": ["no metadata here"]}
    df = pd.DataFrame(data)
    return df


# Fixture for DataFrame where content_metadata doesn't meet conditions
@pytest.fixture
def dataframe_non_table(base64_encoded_image):
    data = {
        "metadata": [
            {
                "content": base64_encoded_image,
                "content_metadata": {"type": "text", "subtype": "paragraph"},  # Not "structured"  # Not "table"
                "table_metadata": {"table_content": ""},
            }
        ]
    }
    df = pd.DataFrame(data)
    return df


# Tests for _update_metadata
def test_update_metadata_missing_metadata():
    row = pd.Series({"other_data": "not metadata"})
    model_interface = PaddleOCRModelInterface()
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))
    trace_info = {}
    with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
        _update_metadata(row, paddle_client, trace_info)


def test_update_metadata_non_table_content(dataframe_non_table):
    row = dataframe_non_table.iloc[0]
    model_interface = PaddleOCRModelInterface()
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))
    trace_info = {}
    result = _update_metadata(row, paddle_client, trace_info)
    # The metadata should remain unchanged
    assert result == row["metadata"]


def test_update_metadata_image_too_small_1(base64_encoded_small_image):
    row = pd.Series(
        {
            "metadata": {
                "content": base64_encoded_small_image,
                "content_metadata": {"type": "structured", "subtype": "table"},
                "table_metadata": {"table_content": ""},
            }
        }
    )
    model_interface = PaddleOCRModelInterface()
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))
    trace_info = {}
    result = _update_metadata(row, paddle_client, trace_info)
    # Since the image is too small, table_content should remain unchanged
    assert result["table_metadata"]["table_content"] == ""


def test_update_metadata_successful_update(sample_dataframe, mock_paddle_client_and_requests):
    model_interface = MockPaddleOCRModelInterface()
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))

    row = sample_dataframe.iloc[0]
    trace_info = {}
    result = _update_metadata(row, paddle_client, trace_info)

    # Expected content from the mocked response
    expected_content = (
        "Chart 1 This chart shows some gadgets, and some very fictitious costs "
        "Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 "
        "$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium "
        "desk fan Cost"
    )

    # The table_content should be updated with expected_content
    assert result["table_metadata"]["table_content"] == expected_content


def test_update_metadata_inference_failure(sample_dataframe, mock_paddle_client_and_requests_failure):
    model_interface = MockPaddleOCRModelInterface()
    paddle_client = NimClient(model_interface, "http", ("mock_endpoint_grpc", "mock_endpoint_http"))

    # Mock response to simulate requests.post behavior
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = RuntimeError("HTTP request failed: Inference error")
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "index": 0,
                "content": (
                    "Chart 1 This chart shows some gadgets, and some very fictitious costs "
                    "Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 "
                    "$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium "
                    "desk fan Cost"
                ),
                "object": "string",
            }
        ],
        "model": "paddleocr",
        "usage": None,
    }

    row = sample_dataframe.iloc[0]
    trace_info = {}
    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="HTTP request failed: Inference error"):
            _update_metadata(row, paddle_client, trace_info)


# Tests for _extract_table_data
def test_extract_table_data_successful(sample_dataframe, mock_paddle_client_and_requests):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    trace_info = {}

    with patch(f"{MODULE_UNDER_TEST}.get_version", return_value="0.3.3"):
        updated_df, trace_info_out = _extract_table_data(sample_dataframe, {}, validated_config, trace_info)

    # Expected content from the mocked response
    expected_content = (
        "Chart 1 This chart shows some gadgets, and some very fictitious costs "
        "Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 "
        "$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium "
        "desk fan Cost"
    )
    assert updated_df.loc[0, "metadata"]["table_metadata"]["table_content"] == expected_content
    assert trace_info_out == {"trace_info": trace_info}

    # Verify that the mocked methods were called
    mock_create_client.assert_called_once()
    mock_requests_post.assert_called_once()


def test_extract_table_data_missing_metadata(dataframe_missing_metadata, mock_paddle_client_and_requests):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    trace_info = {}

    with patch(f"{MODULE_UNDER_TEST}.get_version", return_value="0.2.1"):
        with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
            _extract_table_data(dataframe_missing_metadata, {}, validated_config, trace_info)

    # Verify that the mocked methods were called
    mock_create_client.assert_called_once()
    # Since metadata is missing, requests.post should not be called
    mock_requests_post.assert_not_called()


def test_extract_table_data_inference_failure(sample_dataframe, mock_paddle_client_and_requests_failure):
    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    trace_info = {}

    with patch(f"{MODULE_UNDER_TEST}.get_version", return_value="0.1.0"):
        with pytest.raises(Exception, match="Inference error"):
            _extract_table_data(sample_dataframe, {}, validated_config, trace_info)


def test_extract_table_data_image_too_small_2(base64_encoded_small_image):
    data = {
        "metadata": [
            {
                "content": base64_encoded_small_image,
                "content_metadata": {"type": "structured", "subtype": "table"},
                "table_metadata": {"table_content": ""},
            }
        ]
    }
    df = pd.DataFrame(data)

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    model_interface = PaddleOCRModelInterface()
    trace_info = {}

    def mock_create_inference_client(endpoints, model_interface, auth_token, infer_protocol):
        paddle_client = NimClient(model_interface, "http", ("mock_httpendpoint", "mock_grpc_endpoint"))

        return paddle_client

    # Mock response to simulate requests.post behavior
    mock_response = Mock()
    mock_response.raise_for_status = Mock()  # Does nothing
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "index": 0,
                "content": (
                    "Chart 1 This chart shows some gadgets, and some very fictitious costs "
                    "Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 "
                    "$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium "
                    "desk fan Cost"
                ),
                "object": "string",
            }
        ],
        "model": "paddleocr",
        "usage": None,
    }

    with patch(f"{MODULE_UNDER_TEST}.create_inference_client", side_effect=mock_create_inference_client), patch(
        f"{MODULE_UNDER_TEST}.get_version", return_value="0.1.0"
    ), patch("requests.post", return_value=mock_response):
        updated_df, _ = _extract_table_data(df, {}, validated_config, trace_info)

    # The table_content should remain unchanged because the image is too small
    assert updated_df.loc[0, "metadata"]["table_metadata"]["table_content"] == ""
