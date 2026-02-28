# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from nv_ingest_api.data_handlers.writer_strategies.redis import RedisWriterStrategy
from nv_ingest_api.data_handlers.data_writer import RedisDestinationConfig


class TestRedisWriterStrategy:
    """Black box tests for RedisWriterStrategy."""

    def test_is_available_when_redis_available(self):
        """Test is_available returns True when Redis client can be imported."""
        with patch.dict("sys.modules", {"nv_ingest_api.util.service_clients.redis.redis_client": Mock()}):
            strategy = RedisWriterStrategy()
            assert strategy.is_available() is True

    def test_is_available_when_redis_unavailable(self):
        """Test is_available returns False when Redis client cannot be imported."""
        with patch.dict("sys.modules", {"nv_ingest_api.util.service_clients.redis.redis_client": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                strategy = RedisWriterStrategy()
                assert strategy.is_available() is False

    def test_write_success(self):
        """Test successful write to Redis."""
        # Create mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.submit_message = Mock()

        # Create strategy and config
        strategy = RedisWriterStrategy()
        config = RedisDestinationConfig(host="localhost", port=6379, db=0, channel="test_channel")

        # Mock the RedisClient import and instantiation at the path used in strategy.write()
        with patch.object(strategy, "is_available", return_value=True):
            with patch(
                "nv_ingest_api.util.service_clients.redis.redis_client.RedisClient", return_value=mock_redis_client
            ) as MockRedisClient:
                # Execute write
                data_payload = ['{"key": "value1"}', '{"key": "value2"}']
                strategy.write(data_payload, config)

                # Verify Redis client was created with correct parameters
                MockRedisClient.assert_called_once_with(host="localhost", port=6379, db=0, password=None)

        # Verify messages were submitted
        assert mock_redis_client.submit_message.call_count == 2
        mock_redis_client.submit_message.assert_any_call("test_channel", '{"key": "value1"}')
        mock_redis_client.submit_message.assert_any_call("test_channel", '{"key": "value2"}')

    def test_write_with_password(self):
        """Test write with authentication password."""
        mock_redis_client = Mock()
        mock_redis_client.submit_message = Mock()

        strategy = RedisWriterStrategy()
        config = RedisDestinationConfig(
            host="secure.redis.com", port=6380, db=1, password="secret123", channel="secure_channel"
        )

        with patch.object(strategy, "is_available", return_value=True):
            with patch(
                "nv_ingest_api.util.service_clients.redis.redis_client.RedisClient", return_value=mock_redis_client
            ) as MockRedisClient:
                data_payload = ['{"data": "test"}']
                strategy.write(data_payload, config)

                MockRedisClient.assert_called_once_with(host="secure.redis.com", port=6380, db=1, password="secret123")

    def test_write_dependency_error(self):
        """Test write raises DependencyError when Redis unavailable."""
        strategy = RedisWriterStrategy()
        config = RedisDestinationConfig(channel="test")

        # Mock is_available to return False
        with patch.object(strategy, "is_available", return_value=False):
            from nv_ingest_api.data_handlers.errors import DependencyError

            with pytest.raises(DependencyError, match="Redis client library is not available"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_redis_connection_error(self):
        """Test write handles Redis connection errors."""
        mock_redis_client = Mock()
        mock_redis_client.submit_message.side_effect = ConnectionError("Connection failed")

        strategy = RedisWriterStrategy()
        config = RedisDestinationConfig(channel="test")

        with patch.object(strategy, "is_available", return_value=True):
            with patch(
                "nv_ingest_api.util.service_clients.redis.redis_client.RedisClient", return_value=mock_redis_client
            ):
                with pytest.raises(ConnectionError, match="Connection failed"):
                    strategy.write(['{"test": "data"}'], config)
