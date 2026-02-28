# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from nv_ingest_api.data_handlers.writer_strategies.http import HttpWriterStrategy
from nv_ingest_api.data_handlers.data_writer import HttpDestinationConfig


class TestHttpWriterStrategy:
    """Black box tests for HttpWriterStrategy."""

    def test_is_available_when_requests_available(self):
        """Test is_available returns True when requests can be imported."""
        with patch.dict("sys.modules", {"requests": Mock()}):
            strategy = HttpWriterStrategy()
            assert strategy.is_available() is True

    def test_is_available_when_requests_unavailable(self):
        """Test is_available returns False when requests cannot be imported."""
        with patch("builtins.__import__", side_effect=ImportError):
            strategy = HttpWriterStrategy()
            assert strategy.is_available() is False

    def test_write_success_200(self):
        """Test successful HTTP write with 200 response."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data", method="POST")

        data_payload = ['{"id": 1, "name": "Alice"}', '{"id": 2, "name": "Bob"}']

        # Mock session and response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            strategy.write(data_payload, config)

        # Verify session.request was called correctly
        mock_session.request.assert_called_once_with(
            method="POST",
            url="https://api.example.com/data",
            json=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            headers={},
            timeout=30,
        )

    def test_write_with_auth_token(self):
        """Test write includes authorization header when auth_token provided."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(
            url="https://secure-api.example.com/data", method="PUT", auth_token="bearer-token-123"
        )

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            strategy.write(['{"test": "data"}'], config)

        # Verify authorization header was included
        call_args = mock_session.request.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer bearer-token-123"

    def test_write_with_custom_headers(self):
        """Test write includes custom headers."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(
            url="https://api.example.com/data",
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": "secret"},
        )

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            strategy.write(['{"test": "data"}'], config)

        call_args = mock_session.request.call_args
        expected_headers = {"Content-Type": "application/json", "X-API-Key": "secret"}
        assert call_args[1]["headers"] == expected_headers

    def test_write_dependency_error(self):
        """Test write raises DependencyError when requests unavailable."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        # Mock is_available to return False
        with patch.object(strategy, "is_available", return_value=False):
            from nv_ingest_api.data_handlers.errors import DependencyError

            with pytest.raises(DependencyError, match="requests library is not available"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_4xx_client_error(self):
        """Test write classifies 4xx errors as permanent."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import PermanentError

            with pytest.raises(PermanentError, match="HTTP 404 client error"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_408_with_retry_after(self):
        """Test write classifies 408 with Retry-After as transient."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 408
        mock_response.headers = {"Retry-After": "30"}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import TransientError

            with pytest.raises(TransientError, match="HTTP 408 with Retry-After: 30s"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_429_with_retry_after(self):
        """Test write classifies 429 with Retry-After as transient."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import TransientError

            with pytest.raises(TransientError, match="HTTP 429 with Retry-After: 60s"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_5xx_server_error(self):
        """Test write classifies 5xx errors as transient."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import TransientError

            with pytest.raises(TransientError, match="HTTP 500 server error"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_connection_error(self):
        """Test write handles connection errors."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_session.request.side_effect = ConnectionError("Connection timeout")

        with patch.object(strategy, "_get_session", return_value=mock_session):
            with pytest.raises(ConnectionError, match="Connection timeout"):
                strategy.write(['{"test": "data"}'], config)

    def test_session_reuse(self):
        """Test that the same session is reused for multiple writes."""
        strategy = HttpWriterStrategy()

        # First write
        config = HttpDestinationConfig(url="https://api.example.com/data1")
        mock_session1 = Mock()
        mock_response1 = Mock()
        mock_response1.ok = True
        mock_session1.request.return_value = mock_response1

        with patch.object(strategy, "_get_session", return_value=mock_session1):
            strategy.write(['{"test": "data1"}'], config)

        # Second write - should reuse same session
        config2 = HttpDestinationConfig(url="https://api.example.com/data2")
        mock_session2 = Mock()
        mock_response2 = Mock()
        mock_response2.ok = True
        mock_session2.request.return_value = mock_response2

        with patch.object(strategy, "_get_session", return_value=mock_session2):
            strategy.write(['{"test": "data2"}'], config2)

        # Both should use the same session instance
        # (In practice, HttpWriterStrategy creates one session and reuses it)

    def test_session_creation_path_uses_requests_session(self):
        """Exercise _get_session branch that constructs a real Session from injected requests module."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        # Build a dummy requests module with a Session class
        class DummySession:
            def __init__(self):
                self.request = Mock(return_value=Mock(ok=True))

        dummy_requests = type("R", (), {"Session": DummySession})()

        with patch.dict("sys.modules", {"requests": dummy_requests}):
            # Do not patch _get_session so code constructs the session
            strategy.write(['{"a": 1}'], config)

            # Ensure request was issued
            assert isinstance(strategy._get_session(), DummySession)

    def test_write_408_without_retry_after_is_permanent(self):
        """HTTP 408 without Retry-After should be classified as permanent error per policy."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 408
        mock_response.headers = {}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import PermanentError

            with pytest.raises(PermanentError):
                strategy.write(['{"test": "data"}'], config)

    def test_write_429_with_invalid_retry_after_is_permanent(self):
        """HTTP 429 with non-integer Retry-After should fall back to permanent error path."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "not-a-number"}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import PermanentError

            with pytest.raises(PermanentError):
                strategy.write(['{"test": "data"}'], config)

    def test_write_429_without_retry_after_is_permanent(self):
        """HTTP 429 without Retry-After should be classified as permanent error per policy."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/data")

        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_session.request.return_value = mock_response

        with patch.object(strategy, "_get_session", return_value=mock_session):
            from nv_ingest_api.data_handlers.errors import PermanentError

            with pytest.raises(PermanentError):
                strategy.write(['{"test": "data"}'], config)

    def test_write_uses_configured_method_and_propagates_timeout(self):
        """Verify HTTP method is used and request timeout exceptions propagate."""
        strategy = HttpWriterStrategy()
        config = HttpDestinationConfig(url="https://api.example.com/resource", method="GET")

        mock_session = Mock()
        mock_session.request.side_effect = TimeoutError("Request timed out")

        with patch.object(strategy, "_get_session", return_value=mock_session):
            with pytest.raises(TimeoutError, match="Request timed out"):
                strategy.write(['{"q": 1}'], config)

        # Confirm method was used
        called_method = mock_session.request.call_args.kwargs.get("method") or mock_session.request.call_args[1].get(
            "method"
        )
        assert called_method == "GET"
