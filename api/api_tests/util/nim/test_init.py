# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import MagicMock, patch
import nv_ingest_api.util.nim as module_under_test


def test_create_inference_client_defaults_to_grpc_when_grpc_endpoint_present():
    mock_interface = MagicMock()
    with patch(f"{module_under_test.__name__}.NimClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        result = module_under_test.create_inference_client(
            ("grpc://localhost:50051", "http://localhost:8000"),
            model_interface=mock_interface,
        )

        mock_client.assert_called_once_with(
            mock_interface, "grpc", ("grpc://localhost:50051", "http://localhost:8000"), None, 120.0, 5
        )
        assert result == client_instance


def test_create_inference_client_defaults_to_http_when_no_grpc_endpoint():
    mock_interface = MagicMock()
    with patch(f"{module_under_test.__name__}.NimClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        result = module_under_test.create_inference_client(
            ("", "http://localhost:8000"),
            model_interface=mock_interface,
        )

        mock_client.assert_called_once_with(mock_interface, "http", ("", "http://localhost:8000"), None, 120.0, 5)
        assert result == client_instance


def test_create_inference_client_honors_explicit_protocol_grpc():
    mock_interface = MagicMock()
    with patch(f"{module_under_test.__name__}.NimClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        result = module_under_test.create_inference_client(
            ("grpc://localhost:50051", "http://localhost:8000"),
            model_interface=mock_interface,
            infer_protocol="grpc",
        )

        mock_client.assert_called_once_with(
            mock_interface, "grpc", ("grpc://localhost:50051", "http://localhost:8000"), None, 120.0, 5
        )
        assert result == client_instance


def test_create_inference_client_honors_explicit_protocol_http():
    mock_interface = MagicMock()
    with patch(f"{module_under_test.__name__}.NimClient") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        result = module_under_test.create_inference_client(
            ("grpc://localhost:50051", "http://localhost:8000"),
            model_interface=mock_interface,
            infer_protocol="http",
            auth_token="token123",
            timeout=60,
            max_retries=3,
        )

        mock_client.assert_called_once_with(
            mock_interface, "http", ("grpc://localhost:50051", "http://localhost:8000"), "token123", 60, 3
        )
        assert result == client_instance


def test_create_inference_client_raises_for_invalid_protocol():
    mock_interface = MagicMock()
    with pytest.raises(ValueError, match="Invalid infer_protocol specified. Must be 'grpc' or 'http'."):
        module_under_test.create_inference_client(
            ("grpc://localhost:50051", "http://localhost:8000"),
            model_interface=mock_interface,
            infer_protocol="invalid",
        )
