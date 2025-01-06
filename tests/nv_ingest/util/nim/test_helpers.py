# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch
import numpy as np
import packaging.version
import pytest

import requests

from nv_ingest.util.nim.helpers import (
    NimClient,
    create_inference_client,
    preprocess_image_for_paddle,
    generate_url,
    remove_url_endpoints,
    is_ready,
    get_version,
)

MODULE_UNDER_TEST = "nv_ingest.util.nim.helpers"


class MockModelInterface:
    def prepare_data_for_inference(self, data):
        # Simulate data preparation
        return data

    def format_input(self, data, protocol: str, **kwargs):
        # Return different data based on the protocol
        if protocol == "grpc":
            return np.array([1, 2, 3], dtype=np.float32)
        elif protocol == "http":
            return {"input": "formatted_data"}
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str, data):
        # Simulate parsing the output
        return f"parsed_output_{protocol}"

    def process_inference_results(self, output, **kwargs):
        # Simulate processing the results
        return f"processed_{output}"


@pytest.fixture
def mock_backoff(mocker):
    """
    Mock backoff functionality to avoid actual delays during testing.
    """
    return mocker.patch(f"{MODULE_UNDER_TEST}.backoff")


@pytest.fixture
def mock_requests_get():
    with patch(f"{MODULE_UNDER_TEST}.requests.get") as mock_get:
        yield mock_get


# Fixtures for endpoints
@pytest.fixture
def sample_image():
    """
    Returns a sample image array of shape (height, width, channels) with random pixel values.
    """
    height, width = 800, 600  # Example dimensions
    image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


# Fixtures for endpoints
@pytest.fixture
def grpc_endpoint():
    return "grpc_endpoint"


@pytest.fixture
def http_endpoint():
    return "http_endpoint"


@pytest.fixture
def empty_endpoint():
    return ""


@pytest.fixture
def grpc_endpoints():
    return ("grpc_endpoint", None)


@pytest.fixture
def http_endpoints():
    return (None, "http_endpoint")


@pytest.fixture
def both_endpoints():
    return ("grpc_endpoint", "http_endpoint")


@pytest.fixture
def mock_model_interface():
    return MockModelInterface()


# Black-box tests for NimClient


# Test initialization with valid gRPC parameters
def test_nimclient_init_grpc_valid(mock_model_interface, grpc_endpoints):
    client = NimClient(mock_model_interface, "grpc", grpc_endpoints)
    assert client.protocol == "grpc"


# Test initialization with valid HTTP parameters
def test_nimclient_init_http_valid(mock_model_interface, http_endpoints):
    client = NimClient(mock_model_interface, "http", http_endpoints, auth_token="test_token")
    assert client.protocol == "http"
    assert "Authorization" in client.headers
    assert client.headers["Authorization"] == "Bearer test_token"


# Test initialization with invalid protocol
def test_nimclient_init_invalid_protocol(mock_model_interface, both_endpoints):
    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        NimClient(mock_model_interface, "invalid_protocol", both_endpoints)


# Test initialization missing gRPC endpoint
def test_nimclient_init_missing_grpc_endpoint(mock_model_interface):
    with pytest.raises(ValueError, match="gRPC endpoint must be provided for gRPC protocol"):
        NimClient(mock_model_interface, "grpc", (None, "http_endpoint"))


# Test initialization missing HTTP endpoint
def test_nimclient_init_missing_http_endpoint(mock_model_interface):
    with pytest.raises(ValueError, match="HTTP endpoint must be provided for HTTP protocol"):
        NimClient(mock_model_interface, "http", ("grpc_endpoint", None))


# Test infer with gRPC protocol
def test_nimclient_infer_grpc(mock_model_interface, grpc_endpoints):
    data = {"input_data": "test"}

    # Mock the gRPC client
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        # Instantiate the NimClient after the patch is in place
        client = NimClient(mock_model_interface, "grpc", grpc_endpoints)

        # Mock the infer response
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        result = client.infer(data, model_name="test_model")

    assert result == "processed_parsed_output_grpc"


# Test infer with HTTP protocol
def test_nimclient_infer_http(mock_model_interface, http_endpoints):
    data = {"input_data": "test"}
    client = NimClient(mock_model_interface, "http", http_endpoints)

    # Mock the HTTP request
    with patch(f"{MODULE_UNDER_TEST}.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {"output": "response_data"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = client.infer(data, model_name="test_model")

    assert result == "processed_parsed_output_http"


# Test infer raises exception on HTTP error
def test_nimclient_infer_http_error(mock_model_interface, http_endpoints):
    data = {"input_data": "test"}

    with patch(f"{MODULE_UNDER_TEST}.requests.post") as mock_post:
        client = NimClient(mock_model_interface, "http", http_endpoints)
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Inference error")
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="HTTP Inference error"):
            client.infer(data, model_name="test_model")


# Test infer raises exception on gRPC error
def test_nimclient_infer_grpc_error(mock_model_interface, grpc_endpoints):
    data = {"input_data": "test"}

    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        client = NimClient(mock_model_interface, "grpc", grpc_endpoints)
        mock_grpc_client.return_value.infer.side_effect = Exception("gRPC Inference error")

        with pytest.raises(Exception, match="gRPC Inference error"):
            client.infer(data, model_name="test_model")


# Test infer raises exception on invalid protocol
def test_nimclient_infer_invalid_protocol(mock_model_interface, both_endpoints):
    client = NimClient(mock_model_interface, "grpc", both_endpoints)
    client.protocol = "invalid_protocol"

    with pytest.raises(ValueError, match="Invalid protocol specified. Must be 'grpc' or 'http'."):
        client.infer({}, model_name="test_model")


# Test close method for gRPC protocol
def test_nimclient_close_grpc(mock_model_interface, grpc_endpoints):
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        client = NimClient(mock_model_interface, "grpc", grpc_endpoints)
        mock_grpc_instance = mock_grpc_client.return_value
        client.close()


# Test close method for HTTP protocol
def test_nimclient_close_http(mock_model_interface, http_endpoints):
    client = NimClient(mock_model_interface, "http", http_endpoints)
    # Calling close should not raise an exception
    client.close()


# Test that NimClient handles exceptions from model_interface methods
def test_nimclient_infer_model_interface_exception(mock_model_interface, grpc_endpoints):
    data = {"input_data": "test"}
    client = NimClient(mock_model_interface, "grpc", grpc_endpoints)

    # Simulate exception in prepare_data_for_inference
    mock_model_interface.prepare_data_for_inference = Mock(side_effect=Exception("Preparation error"))

    with pytest.raises(Exception, match="Preparation error"):
        client.infer(data, model_name="test_model")


# Test that NimClient handles exceptions from parse_output
def test_nimclient_infer_parse_output_exception(mock_model_interface, grpc_endpoints):
    data = {"input_data": "test"}

    # Mock the gRPC client
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        client = NimClient(mock_model_interface, "grpc", grpc_endpoints)
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        # Simulate exception in parse_output
        mock_model_interface.parse_output = Mock(side_effect=Exception("Parsing error"))

        with pytest.raises(Exception, match="Parsing error"):
            client.infer(data, model_name="test_model")


# Test that NimClient handles exceptions from process_inference_results
def test_nimclient_infer_process_results_exception(mock_model_interface, grpc_endpoints):
    data = {"input_data": "test"}

    # Mock the gRPC client
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        client = NimClient(mock_model_interface, "grpc", grpc_endpoints)
        mock_response = Mock()
        mock_response.as_numpy.return_value = np.array([1, 2, 3])
        mock_grpc_client.return_value.infer.return_value = mock_response

        # Simulate exception in process_inference_results
        mock_model_interface.process_inference_results = Mock(side_effect=Exception("Processing error"))

        with pytest.raises(Exception, match="Processing error"):
            client.infer(data, model_name="test_model")


# create_inference_client


# Test Case 1: infer_protocol is None, both endpoints provided
def test_create_inference_client_both_endpoints(mock_model_interface, grpc_endpoint, http_endpoint):
    client = create_inference_client(
        endpoints=(grpc_endpoint, http_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol=None,
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "grpc"  # Should default to 'grpc' if both endpoints are provided


# Test Case 2: infer_protocol is None, only grpc_endpoint provided
def test_create_inference_client_grpc_only(mock_model_interface, grpc_endpoint, empty_endpoint):
    client = create_inference_client(
        endpoints=(grpc_endpoint, empty_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol=None,
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "grpc"


# Test Case 3: infer_protocol is None, only http_endpoint provided
def test_create_inference_client_http_only(mock_model_interface, empty_endpoint, http_endpoint):
    client = create_inference_client(
        endpoints=(empty_endpoint, http_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol=None,
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "http"


# Test Case 4: infer_protocol is 'grpc', grpc_endpoint provided
def test_create_inference_client_infer_protocol_grpc(mock_model_interface, grpc_endpoint, empty_endpoint):
    client = create_inference_client(
        endpoints=(grpc_endpoint, empty_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol="grpc",
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "grpc"


# Test Case 5: infer_protocol is 'http', http_endpoint provided
def test_create_inference_client_infer_protocol_http(mock_model_interface, empty_endpoint, http_endpoint):
    client = create_inference_client(
        endpoints=(empty_endpoint, http_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol="http",
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "http"


# Test Case 6: infer_protocol is 'grpc', but grpc_endpoint is empty
def test_create_inference_client_infer_protocol_grpc_no_endpoint(mock_model_interface, empty_endpoint, http_endpoint):
    with pytest.raises(ValueError, match="gRPC endpoint must be provided for gRPC protocol"):
        create_inference_client(
            endpoints=(empty_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="grpc",
        )


# Test Case 7: infer_protocol is 'http', but http_endpoint is empty
def test_create_inference_client_infer_protocol_http_no_endpoint(mock_model_interface, grpc_endpoint, empty_endpoint):
    with pytest.raises(ValueError, match="HTTP endpoint must be provided for HTTP protocol"):
        create_inference_client(
            endpoints=(grpc_endpoint, empty_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="http",
        )


# Test Case 8: infer_protocol is invalid
def test_create_inference_client_invalid_infer_protocol(mock_model_interface, grpc_endpoint, http_endpoint):
    with pytest.raises(ValueError, match="Invalid infer_protocol specified. Must be 'grpc' or 'http'."):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="invalid_protocol",
        )


# Test Case 9: infer_protocol is None, endpoints are empty
def test_create_inference_client_no_endpoints(mock_model_interface, empty_endpoint):
    with pytest.raises(ValueError, match="Invalid infer_protocol specified. Must be 'grpc' or 'http'."):
        create_inference_client(
            endpoints=(empty_endpoint, empty_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol=None,
        )


# Test Case 10: infer_protocol is None, grpc_endpoint is whitespace
def test_create_inference_client_grpc_endpoint_whitespace(mock_model_interface, http_endpoint):
    grpc_endpoint = "   "
    client = create_inference_client(
        endpoints=(grpc_endpoint, http_endpoint),
        model_interface=mock_model_interface,
        auth_token="test_token",
        infer_protocol=None,
    )
    assert isinstance(client, NimClient)
    assert client.protocol == "http"  # Should default to 'http' since grpc_endpoint is empty/whitespace


# Test Case 11: Check that NimClient is instantiated with correct parameters
def test_create_inference_client_nimclient_parameters(mock_model_interface, grpc_endpoint, http_endpoint):
    infer_protocol = "grpc"
    auth_token = "test_token"

    # Mock NimClient to capture the initialization parameters
    with patch(f"{MODULE_UNDER_TEST}.NimClient") as mock_nim_client_class:
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token=auth_token,
            infer_protocol=infer_protocol,
        )
        mock_nim_client_class.assert_called_once_with(
            mock_model_interface, infer_protocol, (grpc_endpoint, http_endpoint), auth_token
        )


# Test Case 12: infer_protocol is 'grpc', grpc_endpoint is None
def test_create_inference_client_grpc_endpoint_none(mock_model_interface, http_endpoint):
    grpc_endpoint = None
    with pytest.raises(ValueError, match="gRPC endpoint must be provided for gRPC protocol"):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="grpc",
        )


# Test Case 13: infer_protocol is 'http', http_endpoint is None
def test_create_inference_client_http_endpoint_none(mock_model_interface, grpc_endpoint):
    http_endpoint = None
    with pytest.raises(ValueError, match="HTTP endpoint must be provided for HTTP protocol"):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="http",
        )


# Test Case 14: infer_protocol is None, both endpoints are None
def test_create_inference_client_endpoints_none(mock_model_interface):
    grpc_endpoint = None
    http_endpoint = None
    with pytest.raises(ValueError, match="Invalid infer_protocol specified. Must be 'grpc' or 'http'."):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol=None,
        )


# Test Case 15: infer_protocol is 'grpc', but grpc_endpoint is whitespace
def test_create_inference_client_grpc_endpoint_whitespace_with_infer_protocol(mock_model_interface, http_endpoint):
    grpc_endpoint = None
    with pytest.raises(ValueError, match="gRPC endpoint must be provided for gRPC protocol"):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="grpc",
        )


# Test Case 16: infer_protocol is 'http', but http_endpoint is whitespace
def test_create_inference_client_http_endpoint_whitespace_with_infer_protocol(mock_model_interface, grpc_endpoint):
    http_endpoint = None
    with pytest.raises(ValueError, match="HTTP endpoint must be provided for HTTP protocol"):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol="http",
        )


# Test Case 17: infer_protocol is None, grpc_endpoint is empty, http_endpoint is whitespace
def test_create_inference_client_http_endpoint_whitespace_no_infer_protocol(mock_model_interface, empty_endpoint):
    grpc_endpoint = ""
    http_endpoint = None
    with pytest.raises(ValueError, match="Invalid infer_protocol specified. Must be 'grpc' or 'http'."):
        create_inference_client(
            endpoints=(grpc_endpoint, http_endpoint),
            model_interface=mock_model_interface,
            auth_token="test_token",
            infer_protocol=None,
        )


# Preprocess image for paddle
def test_preprocess_image_paddle_version_none(sample_image):
    """
    Test that when paddle_version is None, the function returns the input image unchanged.
    """
    result = preprocess_image_for_paddle(sample_image, paddle_version=None)
    assert np.array_equal(
        result, sample_image
    ), "The output should be the same as the input when paddle_version is None."


def test_preprocess_image_paddle_version_old(sample_image):
    """
    Test that when paddle_version is less than '0.2.0-rc1', the function returns the input image unchanged.
    """
    result = preprocess_image_for_paddle(sample_image, paddle_version="0.1.0")
    assert np.array_equal(
        result, sample_image
    ), "The output should be the same as the input when paddle_version is less than '0.2.0-rc1'."


def test_preprocess_image_paddle_version_new(sample_image):
    """
    Test that when paddle_version is '0.2.0-rc1' or higher, the function processes the image.
    """
    result = preprocess_image_for_paddle(sample_image, paddle_version="0.2.0-rc1")
    assert not np.array_equal(
        result, sample_image
    ), "The output should be different from the input when paddle_version is '0.2.0-rc1' or higher."
    assert result.shape[0] == sample_image.shape[2], "The output should have shape (channels, height, width)."


def test_preprocess_image_transpose(sample_image):
    """
    Test that the output image is transposed correctly.
    """
    result = preprocess_image_for_paddle(sample_image, paddle_version="0.2.0")
    # The output should have shape (channels, height, width)
    assert result.shape[0] == sample_image.shape[2], "The output should have channels in the first dimension."
    assert result.shape[1] > 0 and result.shape[2] > 0, "The output height and width should be greater than zero."


def test_preprocess_image_dtype(sample_image):
    """
    Test that the output image has dtype float32.
    """
    result = preprocess_image_for_paddle(sample_image, paddle_version="0.2.0")
    assert result.dtype == np.float32, "The output image should have dtype float32."


def test_preprocess_image_large_image():
    """
    Test processing of a large image.
    """
    image = np.random.randint(0, 256, size=(3000, 2000, 3), dtype=np.uint8)
    result = preprocess_image_for_paddle(image, paddle_version="0.2.0")
    height, width = image.shape[:2]
    scale_factor = 960 / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    expected_height = ((new_height + 31) // 32) * 32
    expected_width = ((new_width + 31) // 32) * 32
    assert (
        result.shape[1] == expected_height and result.shape[2] == expected_width
    ), "The output shape is incorrect for a large image."


def test_preprocess_image_small_image():
    """
    Test processing of a small image.
    """
    image = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    result = preprocess_image_for_paddle(image, paddle_version="0.2.0")
    height, width = image.shape[:2]
    scale_factor = 960 / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    expected_height = ((new_height + 31) // 32) * 32
    expected_width = ((new_width + 31) // 32) * 32
    assert (
        result.shape[1] == expected_height and result.shape[2] == expected_width
    ), "The output shape is incorrect for a small image."


def test_preprocess_image_non_multiple_of_32():
    """
    Test that images with dimensions not multiples of 32 are padded correctly.
    """
    image = np.random.randint(0, 256, size=(527, 319, 3), dtype=np.uint8)
    result = preprocess_image_for_paddle(image, paddle_version="0.2.0")
    height, width = image.shape[:2]
    scale_factor = 960 / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    expected_height = ((new_height + 31) // 32) * 32
    expected_width = ((new_width + 31) // 32) * 32
    assert (
        result.shape[1] == expected_height and result.shape[2] == expected_width
    ), "The image should be padded to the next multiple of 32."


def test_preprocess_image_dtype_uint8():
    """
    Test that the function works with images of dtype uint8.
    """
    image = np.random.randint(0, 256, size=(700, 500, 3), dtype=np.uint8)
    result = preprocess_image_for_paddle(image, paddle_version="0.2.0")
    assert result.dtype == np.float32, "The output image should be converted to dtype float32."


def test_preprocess_image_max_dimension_less_than_960():
    """
    Test that images with max dimension less than 960 are scaled up.
    """
    image = np.random.randint(0, 256, size=(800, 600, 3), dtype=np.uint8)
    result = preprocess_image_for_paddle(image, paddle_version="0.2.0")
    height, width = image.shape[:2]
    scale_factor = 960 / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    expected_height = ((new_height + 31) // 32) * 32
    expected_width = ((new_width + 31) // 32) * 32
    assert (
        result.shape[1] == expected_height and result.shape[2] == expected_width
    ), "The image should be scaled up to have max dimension 960."


def test_preprocess_image_zero_dimension():
    """
    Test that the function handles images with zero dimensions.
    """
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        preprocess_image_for_paddle(image, paddle_version="0.2.0")


def test_preprocess_image_invalid_input():
    """
    Test that the function handles invalid input types.
    """
    image = "not an image"
    with pytest.raises(Exception):
        preprocess_image_for_paddle(image, paddle_version="0.2.0")


def test_preprocess_image_different_paddle_versions(sample_image):
    """
    Test the function with different paddle_version inputs.
    """
    versions = ["0.1.0", "0.2.0-rc0", "0.2.0-rc1", "0.2.1"]
    for version in versions:
        result = preprocess_image_for_paddle(sample_image, paddle_version=version)
        if packaging.version.parse(version) < packaging.version.parse("0.2.0-rc1"):
            assert np.array_equal(
                result, sample_image
            ), f"The output should be the same as the input when paddle_version is {version}."
        else:
            assert not np.array_equal(
                result, sample_image
            ), f"The output should be different from the input when paddle_version is {version}."


# Tests for `remove_url_endpoints`
@pytest.mark.parametrize(
    "input_url, expected_output",
    [
        ("http://deplot:8000/v1/chat/completions", "http://deplot:8000"),
        ("http://example.com/v1/api/resource", "http://example.com"),
        ("https://example.com/v1", "https://example.com"),
        ("https://example.com/v1/", "https://example.com"),
        ("http://localhost:8080/v1/something", "http://localhost:8080"),
        ("http://localhost:8080", "http://localhost:8080"),  # No "/v1" in URL
        ("http://example.com/path/without/v1", "http://example.com/path/without"),
        ("http://example.com/v1path/extra", "http://example.com"),  # "/v1" as part of path
    ],
)
def test_remove_url_endpoints(input_url, expected_output):
    """
    Test the `remove_url_endpoints` function for various cases of input URLs.
    """
    result = remove_url_endpoints(input_url)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# Tests for `generate_url`
@pytest.mark.parametrize(
    "input_url, expected_output",
    [
        ("http://example.com", "http://example.com"),  # Already has `http://`
        ("https://example.com", "https://example.com"),  # Already has `https://`
        ("example.com", "http://example.com"),  # Missing `http://`
        ("localhost:8080", "http://localhost:8080"),  # Missing `http://`
        ("http://localhost:8080", "http://localhost:8080"),  # Already has `http://`
        ("https://localhost:8080", "https://localhost:8080"),  # Already has `https://`
        ("127.0.0.1:5000", "http://127.0.0.1:5000"),  # Missing `http://`
        ("http://127.0.0.1:5000", "http://127.0.0.1:5000"),  # Already has `http://`
        ("https://127.0.0.1:5000", "https://127.0.0.1:5000"),  # Already has `https://`
        ("", "http://"),  # Empty string input
    ],
)
def test_generate_url(input_url, expected_output):
    """
    Test the `generate_url` function for various cases of input URLs.
    """
    result = generate_url(input_url)
    assert result == expected_output, f"Expected {expected_output}, got {result}"


# Edge cases and error handling
def test_remove_url_endpoints_empty_string():
    """
    Test `remove_url_endpoints` with an empty string.
    """
    result = remove_url_endpoints("")
    assert result == "", "Expected an empty string when input is empty."


def test_generate_url_no_http_pattern():
    """
    Test `generate_url` with a completely invalid URL without HTTP pattern.
    """
    result = generate_url("invalid_url_without_http")
    assert result == "http://invalid_url_without_http", "Expected 'http://' to be prepended to invalid URL."


def test_generate_url_already_http():
    """
    Test `generate_url` when the input already starts with `http://`.
    """
    url = "http://already_valid_url"
    result = generate_url(url)
    assert result == url, "Expected the URL to remain unchanged when it already starts with 'http://'."


def test_is_ready_service_not_configured():
    """
    Test that the service is marked as ready when the endpoint is None or empty.
    """
    assert is_ready(None, "/health/ready") is True
    assert is_ready("", "/health/ready") is True


def test_is_ready_nvidia_service():
    """
    Test that URLs for ai.api.nvidia.com are automatically marked as ready.
    """
    assert is_ready("https://ai.api.nvidia.com", "/health/ready") is True
    assert is_ready("http://ai.api.nvidia.com", "/health/ready") is True


def test_is_ready_success(mock_requests_get):
    """
    Test that the function returns True when the HTTP endpoint returns a 200 status.
    """
    mock_requests_get.return_value = Mock(status_code=200)
    result = is_ready("http://example.com", "/health/ready")
    assert result is True
    mock_requests_get.assert_called_once_with("http://example.com/health/ready", timeout=5)


def test_is_ready_not_ready(mock_requests_get):
    """
    Test that the function returns False when the HTTP endpoint returns a 503 status.
    """
    mock_requests_get.return_value = Mock(status_code=503)
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    mock_requests_get.assert_called_once_with("http://example.com/health/ready", timeout=5)


def test_is_ready_confusing_status(mock_requests_get, caplog):
    """
    Test that the function logs a warning for non-200/503 status codes and returns False.
    """
    mock_requests_get.return_value = Mock(status_code=400, json=lambda: {"error": "Bad Request"})
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    mock_requests_get.assert_called_once_with("http://example.com/health/ready", timeout=5)
    assert "HTTP Status: 400" in caplog.text
    assert "Response Payload: {'error': 'Bad Request'}" in caplog.text


def test_is_ready_http_error(mock_requests_get, caplog):
    """
    Test that the function returns False and logs a warning when an HTTP error occurs.
    """
    mock_requests_get.side_effect = requests.HTTPError("HTTP error")
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    assert "produced a HTTP error: HTTP error" in caplog.text


def test_is_ready_timeout(mock_requests_get, caplog):
    """
    Test that the function returns False and logs a warning when a timeout occurs.
    """
    mock_requests_get.side_effect = requests.Timeout
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    assert "request timed out" in caplog.text


def test_is_ready_connection_error(mock_requests_get, caplog):
    """
    Test that the function returns False and logs a warning when a connection error occurs.
    """
    mock_requests_get.side_effect = ConnectionError("Connection failed")
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    assert "A connection error for 'http://example.com/health/ready' occurred" in caplog.text


def test_is_ready_generic_request_exception(mock_requests_get, caplog):
    """
    Test that the function returns False and logs a warning for generic RequestException errors.
    """
    mock_requests_get.side_effect = requests.RequestException("Generic error")
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    assert "An error occurred: Generic error" in caplog.text


def test_is_ready_unexpected_exception(mock_requests_get, caplog):
    """
    Test that the function returns False and logs a warning for unexpected exceptions.
    """
    mock_requests_get.side_effect = Exception("Unexpected error")
    result = is_ready("http://example.com", "/health/ready")
    assert result is False
    assert "Exception: Unexpected error" in caplog.text


def test_is_ready_ready_endpoint_format(mock_requests_get):
    """
    Test that the function appends the ready endpoint correctly.
    """
    mock_requests_get.return_value = Mock(status_code=200)
    result = is_ready("http://example.com/", "health/ready")
    assert result is True
    mock_requests_get.assert_called_once_with("http://example.com/health/ready", timeout=5)


def test_is_ready_generate_url_integration(mock_requests_get):
    """
    Test that the function correctly generates the URL when `http://` is missing.
    """
    mock_requests_get.return_value = Mock(status_code=200)
    result = is_ready("example.com", "/health/ready")
    assert result is True
    mock_requests_get.assert_called_once_with("http://example.com/health/ready", timeout=5)


def test_get_version_cache(mock_requests_get):
    """
    Test that the function uses the cache for subsequent calls with the same arguments.
    """
    mock_requests_get.return_value = Mock(status_code=200, json=lambda: {"version": "1.2.3"})
    result1 = get_version("http://example.com", "/v1/metadata", "version")
    result2 = get_version("http://example.com", "/v1/metadata", "version")

    assert result1 == "1.2.3"
    assert result2 == "1.2.3"
