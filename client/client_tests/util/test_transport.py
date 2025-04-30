import pytest
import numpy as np
from unittest.mock import Mock, patch
from tritonclient.grpc import InferenceServerClient
from nv_ingest_client.util.transport import (
    infer_microservice,
    infer_batch_http,
    infer_with_http,
    infer_batch,
    infer_with_grpc,
)


# Test data fixtures
@pytest.fixture
def sample_data():
    return [
        {"metadata": {"content": "test content 1"}},
        {"metadata": {"content": "test content 2"}},
    ]


@pytest.fixture
def mock_http_response():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    return mock_response


@pytest.fixture
def mock_grpc_client():
    mock_client = Mock(spec=InferenceServerClient)
    # Set up nested config properly
    config_mock = Mock()
    config_mock.max_batch_size = 10
    model_config_mock = Mock()
    model_config_mock.config = config_mock
    mock_client.get_model_config.return_value = model_config_mock
    return mock_client


# HTTP inference tests
@patch("requests.post")
def test_infer_batch_http_success(mock_post, mock_http_response):
    mock_post.return_value = mock_http_response

    result = infer_batch_http(
        text_batch=["test1", "test2"],
        model_name="test_model",
        embedding_endpoint="http://test-endpoint",
    )

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


@patch("requests.post")
def test_infer_batch_http_with_api_key(mock_post, mock_http_response):
    mock_post.return_value = mock_http_response

    infer_batch_http(
        text_batch=["test"],
        model_name="test_model",
        embedding_endpoint="http://test-endpoint",
        nvidia_api_key="test-key",
    )

    # Verify API key was included in headers
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"


@patch("requests.post")
def test_infer_batch_http_error(mock_post):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    with pytest.raises(ValueError, match="Failed retrieving embedding results"):
        infer_batch_http(text_batch=["test"], model_name="test_model", embedding_endpoint="http://test-endpoint")


@patch("nv_ingest_client.util.transport.infer_batch_http")
def test_infer_with_http(mock_infer_batch, sample_data):
    mock_infer_batch.return_value = [[0.1, 0.2, 0.3]]

    result = infer_with_http(
        payload=sample_data, model_name="test_model", embedding_endpoint="http://test-endpoint", batch_size=1
    )

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.1, 0.2, 0.3]


# gRPC inference tests
def test_infer_batch(mock_grpc_client):
    mock_result = Mock()
    mock_result.as_numpy.side_effect = [
        np.array([10, 20]),  # token_count
        np.array([[0.1, 0.2], [0.3, 0.4]]),  # embeddings
    ]
    mock_grpc_client.infer.return_value = mock_result

    token_count, embeddings = infer_batch(
        text_batch=["test1", "test2"],
        client=mock_grpc_client,
        model_name="test_model",
        parameters={"input_type": "passage", "truncate": "END"},
    )

    assert isinstance(token_count, np.ndarray)
    assert isinstance(embeddings, np.ndarray)
    assert token_count.tolist() == [10, 20]
    assert embeddings.tolist() == [[0.1, 0.2], [0.3, 0.4]]


@patch("tritonclient.grpc.InferenceServerClient")
def test_infer_with_grpc(MockInferenceServerClient, sample_data):
    # Create a mock client instance
    mock_client = Mock(spec=InferenceServerClient)
    MockInferenceServerClient.return_value = mock_client

    # Mock the get_model_config method
    mock_config = Mock()
    mock_config.max_batch_size = 10
    mock_model_config = Mock()
    mock_model_config.config = mock_config
    mock_client.get_model_config.return_value = mock_model_config  # Set the return value!

    # Mock the infer method
    mock_result = Mock()
    mock_result.as_numpy.side_effect = [
        np.array([10, 20]),  # token_count
        np.array([[0.1, 0.2], [0.3, 0.4]]),  # embeddings
    ]
    mock_client.infer.return_value = mock_result

    result = infer_with_grpc(
        text_ls=sample_data,
        model_name="test_model",
        grpc_client=mock_client,
    )

    # Verify the mock was called correctly
    mock_client.get_model_config.assert_called_once_with(model_name="test_model")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

    # Verify infer was called with correct parameters
    mock_client.infer.assert_called()

    # Verify the result is correct
    assert np.array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))


# Integration tests
def test_infer_microservice_http(sample_data):
    with patch("nv_ingest_client.util.transport.infer_with_http") as mock_http:
        mock_http.return_value = [[0.1, 0.2], [0.3, 0.4]]

        result = infer_microservice(
            data=sample_data, model_name="test_model", embedding_endpoint="http://test-endpoint", grpc=False
        )

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]


def test_infer_microservice_grpc(sample_data):
    with patch("nv_ingest_client.util.transport.infer_with_grpc") as mock_grpc:
        mock_grpc.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = infer_microservice(
            data=sample_data, model_name="test_model", embedding_endpoint="localhost:8001", grpc=True
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
