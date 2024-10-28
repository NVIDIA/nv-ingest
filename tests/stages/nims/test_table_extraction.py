import pytest
import pandas as pd
import base64
import requests
from unittest.mock import Mock, patch
from io import BytesIO
from PIL import Image
from nv_ingest.stages.nim.table_extraction import _update_metadata, _extract_table_data

# Constants for minimum image size
PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32

MODULE_UNDER_TEST = "nv_ingest.stages.nim.table_extraction"


# Fixture for common mock setup
@pytest.fixture
def mock_paddle_client_and_requests():
    # Dummy client as a dictionary with 'endpoint_url' and 'headers'
    paddle_client = {
        'endpoint_url': 'http://mock_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }

    # Mock response for requests.post
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        'object': 'list',
        'data': [{
            'index': 0,
            'content': ('Chart 1 This chart shows some gadgets, and some very fictitious costs '
                        'Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 '
                        '$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium '
                        'desk fan Cost'),
            'object': 'string'
        }],
        'model': 'paddleocr',
        'usage': None
    }

    # Patching create_inference_client and requests.post
    with patch(f'{MODULE_UNDER_TEST}.create_inference_client', return_value=paddle_client) as mock_create_client, \
            patch('requests.post', return_value=mock_response) as mock_requests_post:
        yield paddle_client, mock_create_client, mock_requests_post


# Fixture for common mock setup (inference failure)
@pytest.fixture
def mock_paddle_client_and_requests_failure():
    # Dummy client as a dictionary with 'endpoint_url' and 'headers'
    paddle_client = {
        'endpoint_url': 'http://mock_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }

    # Mock response for requests.post to raise an HTTPError
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Inference error")
    mock_response.json.return_value = {}

    # Patching create_inference_client and requests.post
    with patch(f'{MODULE_UNDER_TEST}.create_inference_client', return_value=paddle_client) as mock_create_client, \
            patch('requests.post', return_value=mock_response) as mock_requests_post:
        yield paddle_client, mock_create_client, mock_requests_post


# Fixture to create a sample image and encode it in base64
@pytest.fixture
def base64_encoded_image():
    # Create a simple image using PIL
    img = Image.new('RGB', (64, 64), color='white')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    # Encode the image to base64
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


# Fixture for a small image (below minimum size)
@pytest.fixture
def base64_encoded_small_image():
    img = Image.new('RGB', (16, 16), color='white')  # Smaller than minimum size
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


# Fixture for a sample DataFrame
@pytest.fixture
def sample_dataframe(base64_encoded_image):
    data = {
        "metadata": [{
            "content": base64_encoded_image,
            "content_metadata": {
                "type": "structured",
                "subtype": "table"
            },
            "table_metadata": {
                "table_content": ""
            }
        }]
    }
    df = pd.DataFrame(data)
    return df


# Fixture for DataFrame with missing metadata
@pytest.fixture
def dataframe_missing_metadata():
    data = {
        "other_data": ["no metadata here"]
    }
    df = pd.DataFrame(data)
    return df


# Fixture for DataFrame where content_metadata doesn't meet conditions
@pytest.fixture
def dataframe_non_table(base64_encoded_image):
    data = {
        "metadata": [{
            "content": base64_encoded_image,
            "content_metadata": {
                "type": "text",  # Not "structured"
                "subtype": "paragraph"  # Not "table"
            },
            "table_metadata": {
                "table_content": ""
            }
        }]
    }
    df = pd.DataFrame(data)
    return df


# Dummy paddle client that simulates the external service
class DummyPaddleClient:
    def infer(self, *args, **kwargs):
        return "{'object': 'list', 'data': [{'index': 0, 'content': 'Chart 1 This chart shows some gadgets, and some very fictitious costs Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 $40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium desk fan Cost', 'object': 'string'}], 'model': 'paddleocr', 'usage': None}"

    def close(self):
        pass


# Tests for _update_metadata
def test_update_metadata_missing_metadata():
    row = pd.Series({
        "other_data": "not metadata"
    })
    paddle_client = DummyPaddleClient()
    trace_info = {}
    with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
        _update_metadata(row, paddle_client, "0.1.0", trace_info)


def test_update_metadata_non_table_content(dataframe_non_table):
    row = dataframe_non_table.iloc[0]
    paddle_client = DummyPaddleClient()
    trace_info = {}
    result = _update_metadata(row, paddle_client, "0.1.0", trace_info)
    # The metadata should remain unchanged
    assert result == row["metadata"]


def test_update_metadata_image_too_small(base64_encoded_small_image):
    row = pd.Series({
        "metadata": {
            "content": base64_encoded_small_image,
            "content_metadata": {
                "type": "structured",
                "subtype": "table"
            },
            "table_metadata": {
                "table_content": ""
            }
        }
    })
    paddle_client = DummyPaddleClient()
    trace_info = {}
    result = _update_metadata(row, paddle_client, "0.1.1", trace_info)
    # Since the image is too small, table_content should remain unchanged
    assert result["table_metadata"]["table_content"] == ""


def test_update_metadata_successful_update(sample_dataframe, mock_paddle_client_and_requests):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests

    row = sample_dataframe.iloc[0]
    trace_info = {}
    result = _update_metadata(row, paddle_client, "0.2.0", trace_info)

    # Expected content from the mocked response
    expected_content = ('Chart 1 This chart shows some gadgets, and some very fictitious costs '
                        'Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 '
                        '$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium '
                        'desk fan Cost')

    # The table_content should be updated with expected_content
    assert result["table_metadata"]["table_content"] == expected_content

    # Verify that requests.post was called
    mock_requests_post.assert_called_once()


def test_update_metadata_inference_failure(sample_dataframe, mock_paddle_client_and_requests_failure):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests_failure

    row = sample_dataframe.iloc[0]
    trace_info = {}

    with pytest.raises(RuntimeError, match="HTTP request failed: Inference error"):
        _update_metadata(row, paddle_client, "0.2.0", trace_info)

    # Verify that requests.post was called and raised an exception
    mock_requests_post.assert_called_once()


# Tests for _extract_table_data
def test_extract_table_data_successful(sample_dataframe, mock_paddle_client_and_requests):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    trace_info = {}

    with patch(f'{MODULE_UNDER_TEST}.get_version', return_value="0.3.3"):
        updated_df, trace_info_out = _extract_table_data(sample_dataframe, {}, validated_config, trace_info)

    # Expected content from the mocked response
    expected_content = ('Chart 1 This chart shows some gadgets, and some very fictitious costs '
                        'Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 '
                        '$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium '
                        'desk fan Cost')
    assert updated_df.loc[0, 'metadata']['table_metadata']['table_content'] == expected_content
    assert trace_info_out == trace_info

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

    with patch(f'{MODULE_UNDER_TEST}.get_version', return_value="0.2.1"):
        with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
            _extract_table_data(dataframe_missing_metadata, {}, validated_config, trace_info)

    # Verify that the mocked methods were called
    mock_create_client.assert_called_once()
    # Since metadata is missing, requests.post should not be called
    mock_requests_post.assert_not_called()


def test_extract_table_data_inference_failure(sample_dataframe, mock_paddle_client_and_requests_failure):
    paddle_client, mock_create_client, mock_requests_post = mock_paddle_client_and_requests_failure

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    trace_info = {}

    with patch(f'{MODULE_UNDER_TEST}.get_version', return_value="0.1.0"):
        with pytest.raises(RuntimeError, match="HTTP request failed: Inference error"):
            _extract_table_data(sample_dataframe, {}, validated_config, trace_info)

    # Verify that create_inference_client was called
    mock_create_client.assert_called_once()
    # Verify that requests.post was called and raised an exception
    mock_requests_post.assert_called_once()


def test_extract_table_data_image_too_small(base64_encoded_small_image):
    data = {
        "metadata": [{
            "content": base64_encoded_small_image,
            "content_metadata": {
                "type": "structured",
                "subtype": "table"
            },
            "table_metadata": {
                "table_content": ""
            }
        }]
    }
    df = pd.DataFrame(data)

    validated_config = Mock()
    validated_config.stage_config.paddle_endpoints = "mock_endpoint"
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.paddle_infer_protocol = "mock_protocol"

    # Dummy client as a dictionary with 'endpoint_url' and 'headers'
    paddle_client = {
        'endpoint_url': 'http://mock_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }
    trace_info = {}

    def mock_create_inference_client(endpoints, auth_token, protocol):
        return paddle_client

    # Mock response to simulate requests.post behavior
    mock_response = Mock()
    mock_response.raise_for_status = Mock()  # Does nothing
    mock_response.json.return_value = {
        'object': 'list',
        'data': [{
            'index': 0,
            'content': ('Chart 1 This chart shows some gadgets, and some very fictitious costs '
                        'Gadgets and their cost $160.00 $140.00 $120.00 $100.00 $80.00 $60.00 '
                        '$40.00 $20.00 $- Hammer Powerdrill Bluetooth speaker Minifridge Premium '
                        'desk fan Cost'),
            'object': 'string'
        }],
        'model': 'paddleocr',
        'usage': None
    }

    with patch(f'{MODULE_UNDER_TEST}.create_inference_client', side_effect=mock_create_inference_client), \
            patch(f'{MODULE_UNDER_TEST}.get_version', return_value="0.1.0"), \
            patch('requests.post', return_value=mock_response):
        updated_df, _ = _extract_table_data(df, {}, validated_config, trace_info)

    # The table_content should remain unchanged because the image is too small
    assert updated_df.loc[0, 'metadata']['table_metadata']['table_content'] == ""
