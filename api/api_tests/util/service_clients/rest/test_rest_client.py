# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch, MagicMock

import requests

from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema
from nv_ingest_api.util.service_clients.rest.rest_client import RestClient


@pytest.fixture
def dummy_rest_client():
    return RestClient("localhost", 8080)


@pytest.fixture
def mock_response_with_context():
    def _make_mock_response(status_code=200, text=None, iter_content=None, headers=None):
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text or ""
        mock_response.headers = headers or {}
        mock_response.iter_content.return_value = iter_content or []

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_response
        mock_context.__exit__.return_value = None
        return mock_context

    return _make_mock_response


def test_ping_success(dummy_rest_client, mock_response_with_context):
    mock_response = mock_response_with_context(status_code=200)
    with patch.object(dummy_rest_client.get_client(), "get", return_value=mock_response) as mock_get:
        result = dummy_rest_client.ping()
        assert isinstance(result, ResponseSchema)
        assert result.response_code == 0
        assert result.response_reason == "Ping OK"
        mock_get.assert_called_once_with(dummy_rest_client._base_url, timeout=(5.0, 10.0))


def test_ping_failure(dummy_rest_client):
    with patch.object(dummy_rest_client.get_client(), "get", side_effect=requests.exceptions.ConnectionError()):
        result = dummy_rest_client.ping()
        assert result.response_code == 1
        assert "Ping failed" in result.response_reason


def test_fetch_message_success(dummy_rest_client, mock_response_with_context):
    mock_response = mock_response_with_context(status_code=200, iter_content=[b'{"result":"ok"}'])
    with patch.object(dummy_rest_client.get_client(), "get", return_value=mock_response) as mock_get:
        result = dummy_rest_client.fetch_message("job123")
        assert result.response_code == 0
        assert '{"result":"ok"}' in result.response
        mock_get.assert_called_once()


def test_fetch_message_terminal_error(dummy_rest_client, mock_response_with_context):
    mock_response = mock_response_with_context(status_code=404, text="Not Found")
    with patch.object(dummy_rest_client.get_client(), "get", return_value=mock_response):
        result = dummy_rest_client.fetch_message("job123")
        assert result.response_code == 1
        assert "Terminal response code" in result.response_reason


def test_fetch_message_202_not_ready(dummy_rest_client, mock_response_with_context):
    mock_response = mock_response_with_context(status_code=202)
    with patch.object(dummy_rest_client.get_client(), "get", return_value=mock_response):
        result = dummy_rest_client.fetch_message("job123")
        assert result.response_code == 2
        assert "not ready" in result.response_reason.lower()


def test_submit_message_success(dummy_rest_client):
    mock_response = MagicMock(status_code=200, text='"job-456"', headers={"x-trace-id": "trace-abc"})
    with patch.object(dummy_rest_client.get_client(), "post", return_value=mock_response):
        result = dummy_rest_client.submit_message("ignored_channel", '{"job": "do something"}')
        assert result.response_code == 0
        assert "job-456" in result.response
        assert result.trace_id == "trace-abc"
        assert result.transaction_id == "job-456"


def test_submit_message_terminal_error(dummy_rest_client):
    mock_response = MagicMock(status_code=422, text="Unprocessable Entity", headers={})  # Explicit empty headers
    with patch.object(dummy_rest_client.get_client(), "post", return_value=mock_response):
        result = dummy_rest_client.submit_message("ignored_channel", '{"job": "do something"}')
        assert result.response_code == 1
        assert "Terminal response code" in result.response_reason


def test_perform_retry_backoff_limit_exceeded(dummy_rest_client):
    dummy_rest_client._max_retries = 3
    with pytest.raises(RuntimeError):
        dummy_rest_client.perform_retry_backoff(3)


def test_perform_retry_backoff_sleeps(monkeypatch, dummy_rest_client):
    dummy_rest_client._max_retries = 5
    dummy_rest_client._max_backoff = 8
    recorded_delays = []

    def fake_sleep(delay):
        recorded_delays.append(delay)

    monkeypatch.setattr("time.sleep", fake_sleep)
    retry_count = dummy_rest_client.perform_retry_backoff(2)
    assert retry_count == 3
    assert recorded_delays[0] == 4  # 2^2 = 4


def test_generate_url_from_full_url():
    url = RestClient._generate_url("https://myapi.com:1234/some/path", 8080)
    assert url == "https://myapi.com:1234/some/path"


def test_generate_url_from_host_port():
    url = RestClient._generate_url("myapi.com", 8080)
    assert url == "http://myapi.com:8080"
