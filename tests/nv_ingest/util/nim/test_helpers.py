import pytest
from unittest.mock import Mock, patch
from typing import Tuple, Optional
import numpy as np

from nv_ingest.util.nim.helpers import NimClient

MODULE_UNDER_TEST = 'nv_ingest.util.nim.helpers'


class MockModelInterface:
    def prepare_data_for_inference(self, data):
        # Simulate data preparation
        return data

    def format_input(self, data, protocol: str, **kwargs):
        # Return different data based on the protocol
        if protocol == 'grpc':
            return np.array([1, 2, 3], dtype=np.float32)
        elif protocol == 'http':
            return {'input': 'formatted_data'}
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str):
        # Simulate parsing the output
        return f"parsed_output_{protocol}"

    def process_inference_results(self, output, **kwargs):
        # Simulate processing the results
        return f"processed_{output}"


# Fixtures for endpoints
@pytest.fixture
def grpc_endpoints():
    return ('grpc_endpoint', None)


@pytest.fixture
def http_endpoints():
    return (None, 'http_endpoint')


@pytest.fixture
def both_endpoints():
    return ('grpc_endpoint', 'http_endpoint')


@pytest.fixture
def mock_model_interface():
    return MockModelInterface()


# Black-box tests for NimClient

# Test initialization with valid gRPC parameters
def test_nimclient_init_grpc_valid(mock_model_interface, grpc_endpoints):
    client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)
    assert client.protocol == 'grpc'


# Test initialization with valid HTTP parameters
def test_nimclient_init_http_valid(mock_model_interface, http_endpoints):
    client = NimClient(mock_model_interface, 'http', http_endpoints, auth_token='test_token')
    assert client.protocol == 'http'
    assert 'Authorization' in client.headers
    assert client.headers['Authorization'] == 'Bearer test_token'


# Test initialization with invalid protocol
def test_nimclient_init_invalid_protocol(mock_model_interface, both_endpoints):
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        NimClient(mock_model_interface, 'invalid_protocol', both_endpoints)


# Test initialization missing gRPC endpoint
def test_nimclient_init_missing_grpc_endpoint(mock_model_interface):
    with pytest.raises(ValueError, match="gRPC endpoint must be provided for gRPC protocol"):
        NimClient(mock_model_interface, 'grpc', (None, 'http_endpoint'))


# Test initialization missing HTTP endpoint
def test_nimclient_init_missing_http_endpoint(mock_model_interface):
    with pytest.raises(ValueError, match="HTTP endpoint must be provided for HTTP protocol"):
        NimClient(mock_model_interface, 'http', ('grpc_endpoint', None))


# Test infer with gRPC protocol
def test_nimclient_infer_grpc(mock_model_interface, grpc_endpoints):
    data = {'input_data': 'test'}

    # Mock the gRPC client
    with patch(f'{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient') as mock_grpc_client:
        # Instantiate the NimClient after the patch is in place
        client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)

        # Mock the infer response
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        result = client.infer(data, model_name='test_model')

    assert result == 'processed_parsed_output_grpc'


# Test infer with HTTP protocol
def test_nimclient_infer_http(mock_model_interface, http_endpoints):
    data = {'input_data': 'test'}
    client = NimClient(mock_model_interface, 'http', http_endpoints)

    # Mock the HTTP request
    with patch(f'{MODULE_UNDER_TEST}.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {'output': 'response_data'}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = client.infer(data, model_name='test_model')

    assert result == 'processed_parsed_output_http'


# Test infer raises exception on HTTP error
def test_nimclient_infer_http_error(mock_model_interface, http_endpoints):
    data = {'input_data': 'test'}

    with patch(f'{MODULE_UNDER_TEST}.requests.post') as mock_post:
        client = NimClient(mock_model_interface, 'http', http_endpoints)
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Inference error")
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="HTTP Inference error"):
            client.infer(data, model_name='test_model')


# Test infer raises exception on gRPC error
def test_nimclient_infer_grpc_error(mock_model_interface, grpc_endpoints):
    data = {'input_data': 'test'}

    with patch(f'{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient') as mock_grpc_client:
        client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)
        mock_grpc_client.return_value.infer.side_effect = Exception("gRPC Inference error")

        with pytest.raises(Exception, match="gRPC Inference error"):
            client.infer(data, model_name='test_model')


# Test infer raises exception on invalid protocol
def test_nimclient_infer_invalid_protocol(mock_model_interface, both_endpoints):
    client = NimClient(mock_model_interface, 'grpc', both_endpoints)
    client.protocol = 'invalid_protocol'

    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        client.infer({}, model_name='test_model')


# Test close method for gRPC protocol
def test_nimclient_close_grpc(mock_model_interface, grpc_endpoints):
    with patch(f'{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient') as mock_grpc_client:
        client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)
        mock_grpc_instance = mock_grpc_client.return_value
        client.close()


# Test close method for HTTP protocol
def test_nimclient_close_http(mock_model_interface, http_endpoints):
    client = NimClient(mock_model_interface, 'http', http_endpoints)
    # Calling close should not raise an exception
    client.close()


# Test that NimClient handles exceptions from model_interface methods
def test_nimclient_infer_model_interface_exception(mock_model_interface, grpc_endpoints):
    data = {'input_data': 'test'}
    client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)

    # Simulate exception in prepare_data_for_inference
    mock_model_interface.prepare_data_for_inference = Mock(side_effect=Exception("Preparation error"))

    with pytest.raises(Exception, match="Preparation error"):
        client.infer(data, model_name='test_model')


# Test that NimClient handles exceptions from parse_output
def test_nimclient_infer_parse_output_exception(mock_model_interface, grpc_endpoints):
    data = {'input_data': 'test'}

    # Mock the gRPC client
    with patch(f'{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient') as mock_grpc_client:
        client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        # Simulate exception in parse_output
        mock_model_interface.parse_output = Mock(side_effect=Exception("Parsing error"))

        with pytest.raises(Exception, match="Parsing error"):
            client.infer(data, model_name='test_model')


# Test that NimClient handles exceptions from process_inference_results
def test_nimclient_infer_process_results_exception(mock_model_interface, grpc_endpoints):
    data = {'input_data': 'test'}

    # Mock the gRPC client
    with patch(f'{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient') as mock_grpc_client:
        client = NimClient(mock_model_interface, 'grpc', grpc_endpoints)
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        # Simulate exception in process_inference_results
        mock_model_interface.process_inference_results = Mock(side_effect=Exception("Processing error"))

        with pytest.raises(Exception, match="Processing error"):
            client.infer(data, model_name='test_model')
