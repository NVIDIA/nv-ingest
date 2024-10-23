import pytest
import pandas as pd
from unittest.mock import Mock, patch
from nv_ingest.stages.nim.chart_extraction import _update_metadata, \
    _extract_chart_data  # Adjust the import as per your module
import requests

MODULE_UNDER_TEST = "nv_ingest.stages.nim.chart_extraction"  # Replace with your actual module name


# Sample data for testing
@pytest.fixture
def base64_encoded_image():
    # Create a simple image and encode it to base64
    from PIL import Image
    from io import BytesIO
    import base64

    img = Image.new('RGB', (64, 64), color='white')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


@pytest.fixture
def sample_dataframe(base64_encoded_image):
    data = {
        "metadata": [{
            "content": base64_encoded_image,
            "content_metadata": {
                "type": "structured",
                "subtype": "chart"
            },
            "table_metadata": {
                "table_content": "original_content"
            }
        }]
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def dataframe_missing_metadata():
    data = {
        "other_data": ["no metadata here"]
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def dataframe_non_chart(base64_encoded_image):
    data = {
        "metadata": [{
            "content": base64_encoded_image,
            "content_metadata": {
                "type": "text",  # Not "structured"
                "subtype": "paragraph"  # Not "chart"
            },
            "table_metadata": {
                "table_content": "original_content"
            }
        }]
    }
    df = pd.DataFrame(data)
    return df


# Common mock fixtures
@pytest.fixture
def mock_clients_and_requests():
    # Dummy clients as dictionaries with 'endpoint_url' and 'headers'
    deplot_client = {
        'endpoint_url': 'http://deplot_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }
    cached_client = {
        'endpoint_url': 'http://cached_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }

    # Mock response for requests.post (successful inference)
    mock_response_deplot = Mock()
    mock_response_deplot.raise_for_status = Mock()  # Does nothing
    mock_response_deplot.json.return_value = {
        'object': 'list',
        'data': [{
            'index': 0,
            'content': 'deplot_result_content',
            'object': 'string'
        }],
        'model': 'deplot',
        'usage': None
    }

    mock_response_cached = Mock()
    mock_response_cached.raise_for_status = Mock()  # Does nothing
    mock_response_cached.json.return_value = {
        'object': 'list',
        'data': [{
            'index': 0,
            'content': 'cached_result_content',
            'object': 'string'
        }],
        'model': 'cached',
        'usage': None
    }

    # Patching create_inference_client and requests.post
    with patch(f'{MODULE_UNDER_TEST}.create_inference_client') as mock_create_client, \
            patch('requests.post') as mock_requests_post:
        # Mock create_inference_client to return dummy clients
        def side_effect_create_inference_client(endpoints, auth_token, protocol):
            if 'deplot' in endpoints[0]:
                return deplot_client
            elif 'cached' in endpoints[0]:
                return cached_client
            else:
                return None

        mock_create_client.side_effect = side_effect_create_inference_client

        # Mock requests.post to return different responses based on URL
        def side_effect_requests_post(url, *args, **kwargs):
            if 'deplot' in url:
                return mock_response_deplot
            elif 'cached' in url:
                return mock_response_cached
            else:
                return Mock()

        mock_requests_post.side_effect = side_effect_requests_post

        yield deplot_client, cached_client, mock_create_client, mock_requests_post


@pytest.fixture
def mock_clients_and_requests_failure():
    # Dummy clients as dictionaries with 'endpoint_url' and 'headers'
    deplot_client = {
        'endpoint_url': 'http://deplot_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }
    cached_client = {
        'endpoint_url': 'http://cached_endpoint_url',
        'headers': {'Authorization': 'Bearer mock_token'}
    }

    # Mock response for requests.post to raise an HTTPError
    mock_response_failure = Mock()
    mock_response_failure.raise_for_status.side_effect = requests.exceptions.HTTPError("Inference error")
    mock_response_failure.json.return_value = {}

    # Patching create_inference_client and requests.post
    with patch(f'{MODULE_UNDER_TEST}.create_inference_client') as mock_create_client, \
            patch('requests.post', return_value=mock_response_failure) as mock_requests_post:
        # Mock create_inference_client to return dummy clients
        def side_effect_create_inference_client(endpoints, auth_token, protocol):
            if 'deplot' in endpoints[0]:
                return deplot_client
            elif 'cached' in endpoints[0]:
                return cached_client
            else:
                return None

        mock_create_client.side_effect = side_effect_create_inference_client

        yield deplot_client, cached_client, mock_create_client, mock_requests_post


# Tests for _update_metadata
def test_update_metadata_missing_metadata(dataframe_missing_metadata, mock_clients_and_requests):
    deplot_client, cached_client, _, _ = mock_clients_and_requests

    row = dataframe_missing_metadata.iloc[0]
    trace_info = {}
    with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
        _update_metadata(row, cached_client, deplot_client, trace_info)


def test_update_metadata_non_chart_content(dataframe_non_chart, mock_clients_and_requests):
    deplot_client, cached_client, _, _ = mock_clients_and_requests

    row = dataframe_non_chart.iloc[0]
    trace_info = {}
    result = _update_metadata(row, cached_client, deplot_client, trace_info)
    # The metadata should remain unchanged
    assert result == row["metadata"]


@pytest.mark.xfail
def test_update_metadata_successful_update(sample_dataframe, mock_clients_and_requests):
    deplot_client, cached_client, _, _ = mock_clients_and_requests

    row = sample_dataframe.iloc[0]
    trace_info = {}
    result = _update_metadata(row, cached_client, deplot_client, trace_info)
    # The table_content should be updated with combined result
    expected_content = 'Combined content: cached_result_content + deplot_result_content'
    assert result["table_metadata"]["table_content"] == expected_content


@pytest.mark.xfail
def test_update_metadata_inference_failure(sample_dataframe, mock_clients_and_requests_failure):
    deplot_client, cached_client, _, mock_requests_post = mock_clients_and_requests_failure

    row = sample_dataframe.iloc[0]
    trace_info = {}

    with pytest.raises(RuntimeError, match="An error occurred during inference: Inference error"):
        _update_metadata(row, cached_client, deplot_client, trace_info)

    # Verify that requests.post was called and raised an exception
    assert mock_requests_post.call_count >= 1  # At least one call failed


@pytest.mark.xfail
def test_extract_chart_data_successful(sample_dataframe, mock_clients_and_requests):
    deplot_client, cached_client, mock_create_client, mock_requests_post = mock_clients_and_requests

    validated_config = Mock()
    validated_config.stage_config.deplot_endpoints = ("http://deplot_endpoint", None)
    validated_config.stage_config.cached_endpoints = ("http://cached_endpoint", None)
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.deplot_infer_protocol = "mock_protocol"
    validated_config.stage_config.cached_infer_protocol = "mock_protocol"

    trace_info = {}

    updated_df, trace_info_out = _extract_chart_data(sample_dataframe, {}, validated_config, trace_info)

    # Expected content from the combined results
    expected_content = 'Combined content: cached_result_content + deplot_result_content'
    assert updated_df.loc[0, 'metadata']['table_metadata']['table_content'] == expected_content
    assert trace_info_out == trace_info

    # Verify that the mocked methods were called
    assert mock_create_client.call_count == 2  # deplot and cached clients created
    assert mock_requests_post.call_count == 2  # deplot and cached inference called


def test_extract_chart_data_missing_metadata(dataframe_missing_metadata, mock_clients_and_requests):
    deplot_client, cached_client, _, _ = mock_clients_and_requests

    validated_config = Mock()
    validated_config.stage_config.deplot_endpoints = ("http://deplot_endpoint", None)
    validated_config.stage_config.cached_endpoints = ("http://cached_endpoint", None)
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.deplot_infer_protocol = "mock_protocol"
    validated_config.stage_config.cached_infer_protocol = "mock_protocol"

    trace_info = {}

    with pytest.raises(ValueError, match="Row does not contain 'metadata'."):
        _extract_chart_data(dataframe_missing_metadata, {}, validated_config, trace_info)


@pytest.mark.xfail
def test_extract_chart_data_inference_failure(sample_dataframe, mock_clients_and_requests_failure):
    deplot_client, cached_client, mock_create_client, mock_requests_post = mock_clients_and_requests_failure

    validated_config = Mock()
    validated_config.stage_config.deplot_endpoints = ("http://deplot_endpoint", None)
    validated_config.stage_config.cached_endpoints = ("http://cached_endpoint", None)
    validated_config.stage_config.auth_token = "mock_token"
    validated_config.stage_config.deplot_infer_protocol = "mock_protocol"
    validated_config.stage_config.cached_infer_protocol = "mock_protocol"

    trace_info = {}

    with pytest.raises(RuntimeError, match="An error occurred during inference: Inference error"):
        _extract_chart_data(sample_dataframe, {}, validated_config, trace_info)

    # Verify that the mocked methods were called
    assert mock_create_client.call_count == 2
    assert mock_requests_post.call_count >= 1  # At least one call failed
