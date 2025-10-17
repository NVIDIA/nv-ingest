# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest_api.data_handlers.writer_strategies import (
    get_writer_strategy,
    RedisWriterStrategy,
    FilesystemWriterStrategy,
    HttpWriterStrategy,
    KafkaWriterStrategy,
)


class TestGetWriterStrategy:
    """Black box tests for get_writer_strategy factory function."""

    def test_get_redis_strategy(self):
        """Test get_writer_strategy returns RedisWriterStrategy for 'redis'."""
        strategy = get_writer_strategy("redis")
        assert isinstance(strategy, RedisWriterStrategy)

    def test_get_filesystem_strategy(self):
        """Test get_writer_strategy returns FilesystemWriterStrategy for 'filesystem'."""
        strategy = get_writer_strategy("filesystem")
        assert isinstance(strategy, FilesystemWriterStrategy)

    def test_get_http_strategy(self):
        """Test get_writer_strategy returns HttpWriterStrategy for 'http'."""
        strategy = get_writer_strategy("http")
        assert isinstance(strategy, HttpWriterStrategy)

    def test_get_kafka_strategy(self):
        """Test get_writer_strategy returns KafkaWriterStrategy for 'kafka'."""
        strategy = get_writer_strategy("kafka")
        assert isinstance(strategy, KafkaWriterStrategy)

    def test_get_unknown_strategy_raises_value_error(self):
        """Test get_writer_strategy raises ValueError for unknown strategy type."""
        with pytest.raises(ValueError, match="Unsupported destination type: unknown"):
            get_writer_strategy("unknown")

    def test_get_case_sensitive_strategy(self):
        """Test get_writer_strategy is case sensitive."""
        with pytest.raises(ValueError, match="Unsupported destination type: REDIS"):
            get_writer_strategy("REDIS")

    def test_get_strategy_error_message_lists_supported_types(self):
        """Test error message lists all supported destination types."""
        with pytest.raises(ValueError) as exc_info:
            get_writer_strategy("invalid")

        error_message = str(exc_info.value)
        assert "redis" in error_message
        assert "filesystem" in error_message
        assert "http" in error_message
        assert "kafka" in error_message
        assert "Supported:" in error_message

    def test_get_strategy_returns_same_instance(self):
        """Test get_writer_strategy returns the same instance for repeated calls."""
        strategy1 = get_writer_strategy("redis")
        strategy2 = get_writer_strategy("redis")
        assert strategy1 is strategy2  # Same instance (singleton pattern)

    def test_all_supported_strategies_are_available(self):
        """Test all supported strategies can be retrieved without error."""
        supported_types = ["redis", "filesystem", "http", "kafka"]

        for strategy_type in supported_types:
            strategy = get_writer_strategy(strategy_type)
            assert strategy is not None
            assert hasattr(strategy, "write")
            assert hasattr(strategy, "is_available")
