# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import redis
from unittest.mock import MagicMock, patch
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient
from nv_ingest_api.util.service_clients.client_base import FetchMode


@pytest.fixture
def dummy_redis_client():
    return RedisClient("localhost", 6379)


def test_ping_success(dummy_redis_client):
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    dummy_redis_client._client = mock_client

    assert dummy_redis_client.ping() is True
    mock_client.ping.assert_called_once()


def test_ping_failure(dummy_redis_client):
    mock_client = MagicMock()
    mock_client.ping.side_effect = redis.RedisError("test error")
    dummy_redis_client._client = mock_client

    assert dummy_redis_client.ping() is False
    assert dummy_redis_client._client is None


def test_get_client_reconnects(dummy_redis_client):
    # Simulate client is None and ping on reconnect works
    dummy_redis_client._client = None
    mock_redis = MagicMock()
    dummy_redis_client._redis_allocator = MagicMock(return_value=mock_redis)
    mock_redis.ping.return_value = True

    client = dummy_redis_client.get_client()
    assert client == mock_redis
    dummy_redis_client._redis_allocator.assert_called_once()


def test_get_client_reconnect_fails(dummy_redis_client):
    dummy_redis_client._client = None
    dummy_redis_client._redis_allocator = MagicMock(side_effect=Exception("fail"))

    with pytest.raises(RuntimeError):
        dummy_redis_client.get_client()


def test_submit_message_success(dummy_redis_client):
    mock_client = MagicMock()
    dummy_redis_client._client = mock_client

    dummy_redis_client.submit_message("channel", '{"msg": "test"}', ttl_seconds=123)
    mock_client.pipeline.return_value.rpush.assert_called_with("channel", '{"msg": "test"}')
    mock_client.pipeline.return_value.expire.assert_called_with("channel", 123)
    mock_client.pipeline.return_value.execute.assert_called_once()


def test_submit_message_retries_and_fails(dummy_redis_client):
    dummy_redis_client._max_retries = 2

    # Create a new pipeline mock for each retry to simulate 'client becoming None' and being reconnected
    mock_pipeline = MagicMock()
    mock_pipeline.execute.side_effect = redis.RedisError("fail")

    # The get_client always returns a client whose pipeline always errors
    mock_client = MagicMock()
    mock_client.pipeline.return_value = mock_pipeline

    with patch.object(dummy_redis_client, "get_client", return_value=mock_client):
        with pytest.raises(ValueError, match="Failed to submit to Redis after 3 attempts: fail"):
            dummy_redis_client.submit_message("channel", '{"msg": "test"}', ttl_seconds=123)


def test_fetch_message_destructive_success(dummy_redis_client):
    dummy_redis_client._fetch_mode = FetchMode.DESTRUCTIVE

    with patch.object(
        dummy_redis_client, "_fetch_first_or_all_fragments_destructive", return_value={"result": "ok"}
    ) as mock_fetch:
        result = dummy_redis_client.fetch_message("channel", timeout=5)
        assert result == {"result": "ok"}
        mock_fetch.assert_called_once()


def test_fetch_message_non_destructive_success(dummy_redis_client):
    dummy_redis_client._fetch_mode = FetchMode.NON_DESTRUCTIVE

    with patch.object(
        dummy_redis_client,
        "_fetch_fragments_non_destructive",
        return_value=[{"fragment": 0, "data": [1]}, {"fragment": 1, "data": [2]}],
    ):
        with patch.object(dummy_redis_client, "_combine_fragments", return_value={"data": [1, 2]}) as mock_combine:
            result = dummy_redis_client.fetch_message("channel", timeout=5)
            assert result == {"data": [1, 2]}
            mock_combine.assert_called_once()


def test_fetch_message_unexpected_error(dummy_redis_client):
    dummy_redis_client._fetch_mode = FetchMode.DESTRUCTIVE

    with patch.object(dummy_redis_client, "_fetch_first_or_all_fragments_destructive", side_effect=Exception("boom")):
        with pytest.raises(ValueError, match="Unexpected error during fetch: boom"):
            dummy_redis_client.fetch_message("channel", timeout=5)


def test_combine_fragments_success():
    fragments = [
        {"fragment": 0, "data": [1, 2]},
        {"fragment": 1, "data": [3, 4]},
    ]
    combined = RedisClient._combine_fragments(fragments)
    assert combined["data"] == [1, 2, 3, 4]


def test_combine_fragments_empty_raises():
    with pytest.raises(ValueError, match="Cannot combine empty list of fragments"):
        RedisClient._combine_fragments([])


def test_max_retries_setter_validation(dummy_redis_client):
    with pytest.raises(ValueError, match="max_retries must be a non-negative integer."):
        dummy_redis_client.max_retries = -1
