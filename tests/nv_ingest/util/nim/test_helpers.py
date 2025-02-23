# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch
import numpy as np
import pytest

from nv_ingest_api.primitives.nim import NimClient

MODULE_UNDER_TEST = "nv_ingest.util.nim.helpers"


# ---------------------------------------------------------------------
# Dummy model interface for testing
# ---------------------------------------------------------------------
class DummyModelInterface:
    def name(self):
        return "DummyModel"

    def prepare_data_for_inference(self, data):
        # Simulate some preparation by adding a flag.
        data["prepared"] = True
        # Also, simulate storing original shapes (for later use).
        data["original_image_shapes"] = [(100, 100)]
        return data

    def format_input(self, data, protocol: str, max_batch_size: int, **kwargs):
        # For testing, return a tuple of (formatted_batches, batch_data)
        if protocol == "grpc":
            # Return one numpy array (batch) and accompanying batch_data.
            return ([np.ones((1, 10), dtype=np.float32)], [{"dummy": "batch_data"}])
        elif protocol == "http":
            # Return one payload dictionary and accompanying batch_data.
            return ([{"input": "http_payload"}], [{"dummy": "batch_data"}])
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str, data, **kwargs):
        # For testing, simply return a fixed string depending on the protocol.
        if protocol == "grpc":
            return "parsed_grpc"
        elif protocol == "http":
            return "parsed_http"
        else:
            raise ValueError("Invalid protocol")

    def process_inference_results(self, parsed_output, **kwargs):
        # For testing, prepend "processed_" to the parsed output.
        return f"processed_{parsed_output}"


# ---------------------------------------------------------------------
# Fixtures for endpoints
# ---------------------------------------------------------------------
@pytest.fixture
def grpc_endpoints():
    # For grpc, the first element is the gRPC endpoint; the HTTP endpoint is unused.
    return ("grpc_endpoint", None)


@pytest.fixture
def http_endpoints():
    # For HTTP, the second element is the HTTP endpoint; the gRPC endpoint is unused.
    return (None, "http_endpoint")


# ---------------------------------------------------------------------
# Black‑box tests for NimClient
# ---------------------------------------------------------------------


def test_init_invalid_protocol():
    dummy_interface = DummyModelInterface()
    with pytest.raises(ValueError, match="Invalid protocol specified"):
        NimClient(dummy_interface, "invalid", ("grpc_endpoint", "http_endpoint"))


def test_init_missing_grpc_endpoint():
    dummy_interface = DummyModelInterface()
    with pytest.raises(ValueError, match="gRPC endpoint must be provided"):
        # For grpc, the first element of endpoints must be non‑empty.
        NimClient(dummy_interface, "grpc", ("", "http_endpoint"))


def test_init_missing_http_endpoint():
    dummy_interface = DummyModelInterface()
    with pytest.raises(ValueError, match="HTTP endpoint must be provided"):
        # For http, the second element must be non‑empty.
        NimClient(dummy_interface, "http", ("grpc_endpoint", ""))


def test_init_http_auth_token():
    dummy_interface = DummyModelInterface()
    # Patch generate_url to return a dummy URL.
    with patch(f"{MODULE_UNDER_TEST}.generate_url", return_value="http://example.com") as mock_gen:
        client = NimClient(dummy_interface, "http", (None, "http_endpoint"), auth_token="secret")
        assert client.endpoint_url == "http://example.com"
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer secret"


def test_infer_grpc_success(grpc_endpoints):
    dummy_interface = DummyModelInterface()
    # Patch the gRPC client so that its infer() and get_model_config() behave as expected.
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value

        # Simulate get_model_config returning a config with max_batch_size = 2.
        fake_config = Mock()
        fake_config.config = Mock(max_batch_size=2)
        fake_client.get_model_config.return_value = fake_config

        # Simulate a successful inference response.
        fake_response = Mock()
        # The _grpc_infer method calls as_numpy("output"); we can return any dummy numpy array.
        fake_response.as_numpy.return_value = np.array([0])
        fake_client.infer.return_value = fake_response

        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        data = {"input_data": "test"}

        result = client.infer(data, model_name="dummy_model")
        # Expected flow:
        # 1. DummyModelInterface.prepare_data_for_inference adds "prepared": True and an "original_image_shapes" key.
        # 2. format_input returns a dummy numpy array and batch_data.
        # 3. _grpc_infer returns fake_response.as_numpy("output").
        # 4. parse_output returns "parsed_grpc" and then process_inference_results returns "processed_parsed_grpc".
        assert result == ["processed_parsed_grpc"]


def test_infer_http_success(http_endpoints):
    dummy_interface = DummyModelInterface()
    # Patch requests.post and generate_url so that HTTP inference works.
    with patch(f"{MODULE_UNDER_TEST}.requests.post") as mock_post, patch(
        f"{MODULE_UNDER_TEST}.generate_url", return_value="http://example.com"
    ):
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"dummy": "response"}
        fake_response.raise_for_status = lambda: None
        mock_post.return_value = fake_response

        client = NimClient(dummy_interface, "http", http_endpoints)
        data = {"input_data": "test"}

        result = client.infer(data, model_name="dummy_model")
        # Expected: parse_output returns "parsed_http" and process_inference_results returns "processed_parsed_http".
        assert result == ["processed_parsed_http"]


def test_infer_http_retry_failure(http_endpoints):
    dummy_interface = DummyModelInterface()
    # Patch requests.post so that it always returns an HTTP error.
    with patch(f"{MODULE_UNDER_TEST}.requests.post") as mock_post, patch(
        f"{MODULE_UNDER_TEST}.generate_url", return_value="http://example.com"
    ):
        fake_response = Mock()
        fake_response.status_code = 500
        fake_response.raise_for_status.side_effect = Exception("HTTP Inference error")
        mock_post.return_value = fake_response

        client = NimClient(dummy_interface, "http", http_endpoints, max_retries=2, timeout=0.1)
        data = {"input_data": "test"}
        with pytest.raises(Exception, match="HTTP Inference error"):
            client.infer(data, model_name="dummy_model")


def test_infer_grpc_infer_exception(grpc_endpoints):
    dummy_interface = DummyModelInterface()
    # Patch the gRPC client so that its infer() call fails.
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value
        fake_client.infer.side_effect = Exception("gRPC Inference error")
        fake_config = Mock()
        fake_config.config = Mock(max_batch_size=1)
        fake_client.get_model_config.return_value = fake_config

        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        data = {"input_data": "test"}
        with pytest.raises(RuntimeError, match="gRPC Inference error"):
            client.infer(data, model_name="dummy_model")


def test_infer_parse_output_exception(grpc_endpoints):
    # In this test the dummy model interface will raise an exception during parse_output.
    class FaultyModelInterface(DummyModelInterface):
        def parse_output(self, response, protocol: str, data, **kwargs):
            raise Exception("Parsing error")

    dummy_interface = FaultyModelInterface()
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value
        fake_response = Mock()
        fake_response.as_numpy.return_value = np.array([0])
        fake_client.infer.return_value = fake_response
        fake_config = Mock()
        fake_config.config = Mock(max_batch_size=1)
        fake_client.get_model_config.return_value = fake_config

        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        data = {"input_data": "test"}
        with pytest.raises(RuntimeError, match="Parsing error"):
            client.infer(data, model_name="dummy_model")


def test_infer_process_results_exception(grpc_endpoints):
    # In this test the dummy model interface will raise an exception during process_inference_results.
    class FaultyModelInterface(DummyModelInterface):
        def process_inference_results(self, parsed_output, **kwargs):
            raise Exception("Processing error")

    dummy_interface = FaultyModelInterface()
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value
        fake_response = Mock()
        fake_response.as_numpy.return_value = np.array([0])
        fake_client.infer.return_value = fake_response
        fake_config = Mock()
        fake_config.config = Mock(max_batch_size=1)
        fake_client.get_model_config.return_value = fake_config

        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        data = {"input_data": "test"}
        with pytest.raises(RuntimeError, match="Processing error"):
            client.infer(data, model_name="dummy_model")


def test_close_grpc(grpc_endpoints):
    dummy_interface = DummyModelInterface()
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value
        fake_client.close = Mock()
        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        client.close()
        fake_client.close.assert_called_once()


def test_try_set_max_batch_size(grpc_endpoints):
    dummy_interface = DummyModelInterface()
    with patch(f"{MODULE_UNDER_TEST}.grpcclient.InferenceServerClient") as mock_grpc_client:
        fake_client = mock_grpc_client.return_value
        fake_config = Mock()
        fake_config.config = Mock(max_batch_size=4)
        fake_client.get_model_config.return_value = fake_config

        client = NimClient(dummy_interface, "grpc", grpc_endpoints)
        client.try_set_max_batch_size("dummy_model")
        assert client._max_batch_sizes.get("dummy_model") == 4
