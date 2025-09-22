# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch, MagicMock

from nv_ingest_api.data_handlers.writer_strategies.kafka import KafkaWriterStrategy
from nv_ingest_api.data_handlers.data_writer import KafkaDestinationConfig


class TestKafkaWriterStrategy:
    """Black box tests for KafkaWriterStrategy."""

    def test_is_available_when_kafka_available(self):
        """Test is_available returns True when kafka-python can be imported."""
        with patch.dict("sys.modules", {"kafka": Mock()}):
            strategy = KafkaWriterStrategy()
            assert strategy.is_available() is True

    def test_is_available_when_kafka_unavailable(self):
        """Test is_available returns False when kafka-python cannot be imported."""
        with patch("builtins.__import__", side_effect=ImportError):
            strategy = KafkaWriterStrategy()
            assert strategy.is_available() is False

    def test_write_success_basic(self):
        """Test successful write to Kafka with basic configuration."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="test-topic")

        data_payload = ['{"id": 1, "message": "test1"}', '{"id": 2, "message": "test2"}']

        # Mock KafkaProducer and futures
        mock_producer = Mock()
        mock_future1 = Mock()
        mock_future1.get.return_value = None
        mock_future2 = Mock()
        mock_future2.get.return_value = None

        mock_producer.send.side_effect = [mock_future1, mock_future2]

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                strategy.write(data_payload, config)

                # Verify producer was created with correct config
                args, kwargs = dummy_kafka_module.KafkaProducer.call_args
                assert kwargs["bootstrap_servers"] == ["localhost:9092"]
                assert kwargs["security_protocol"] == "PLAINTEXT"
                # verify serializer callable exists
                assert callable(kwargs["value_serializer"])

        # Verify messages were sent
        assert mock_producer.send.call_count == 2
        mock_producer.send.assert_any_call("test-topic", value={"id": 1, "message": "test1"}, key=None)
        mock_producer.send.assert_any_call("test-topic", value={"id": 2, "message": "test2"}, key=None)

        # Verify flush was called
        mock_producer.flush.assert_called_once()

        # Verify producer was closed
        mock_producer.close.assert_called_once()

    def test_value_serializer_string_mode(self):
        """When value_serializer='string', values should be sent as bytes of str(payload)."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["localhost:9092"],
            topic="string-topic",
            value_serializer="string",
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka = MagicMock()
            dummy_kafka.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka}):
                strategy.write(['{"foo": "bar"}'], config)

                # Grab the value_serializer used
                _, kwargs = dummy_kafka.KafkaProducer.call_args
                serializer = kwargs["value_serializer"]
                # Validate serializer behavior
                assert serializer("abc") == b"abc"

    def test_sasl_without_ssl(self):
        """SASL over PLAINTEXT should pass SASL fields without SSL settings."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["localhost:9092"],
            topic="sasl-topic",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="pass",
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka = MagicMock()
            dummy_kafka.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka}):
                strategy.write(['{"a": 1}'], config)

                _, kwargs = dummy_kafka.KafkaProducer.call_args
                assert kwargs["security_protocol"] == "SASL_PLAINTEXT"
                assert kwargs["sasl_mechanism"] == "PLAIN"
                assert kwargs["sasl_plain_username"] == "user"
                assert kwargs["sasl_plain_password"] == "pass"
                # SSL keys should not be present
                assert "ssl_cafile" not in kwargs
                assert "ssl_certfile" not in kwargs
                assert "ssl_keyfile" not in kwargs

    def test_partial_ssl_config(self):
        """Only cafile provided should be passed through without cert/key."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["localhost:9092"], topic="ssl-topic", ssl_cafile="/path/to/ca.pem"
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka = MagicMock()
            dummy_kafka.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka}):
                strategy.write(['{"x": 1}'], config)

                _, kwargs = dummy_kafka.KafkaProducer.call_args
                assert kwargs["ssl_cafile"] == "/path/to/ca.pem"
                assert "ssl_certfile" not in kwargs
                assert "ssl_keyfile" not in kwargs

    def test_key_serializer_string_without_id(self):
        """When key_serializer='string' but payload has no id, key should be None."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["localhost:9092"], topic="key-topic", key_serializer="string"
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka = MagicMock()
            dummy_kafka.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka}):
                strategy.write(['{"no_id": 1}'], config)

        # Ensure send was called with key=None
        mock_producer.send.assert_called_once()
        args, kwargs = mock_producer.send.call_args
        assert kwargs.get("key") is None

    def test_write_with_ssl_authentication(self):
        """Test write with SSL and SASL authentication."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["secure-kafka:9093"],
            topic="secure-topic",
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="pass",
            ssl_cafile="/path/to/ca.pem",
            ssl_certfile="/path/to/client.pem",
            ssl_keyfile="/path/to/client.key",
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                strategy.write(['{"test": "data"}'], config)

                # Verify producer was created with SSL and SASL config
                _, kwargs = dummy_kafka_module.KafkaProducer.call_args
                expected_config = {
                    "bootstrap_servers": ["secure-kafka:9093"],
                    "security_protocol": "SASL_SSL",
                    "sasl_mechanism": "PLAIN",
                    "sasl_plain_username": "user",
                    "sasl_plain_password": "pass",
                    "ssl_cafile": "/path/to/ca.pem",
                    "ssl_certfile": "/path/to/client.pem",
                    "ssl_keyfile": "/path/to/client.key",
                }

                for key, value in expected_config.items():
                    assert kwargs[key] == value

    def test_write_with_key_serializer(self):
        """Test write with key serializer configured."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(
            bootstrap_servers=["localhost:9092"], topic="test-topic", key_serializer="string"
        )

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                strategy.write(['{"id": 123, "data": "test"}'], config)

        # Verify key was serialized
        mock_producer.send.assert_called_once()
        call_args = mock_producer.send.call_args
        assert call_args[1]["key"] == b"123"  # key_serializer should encode to bytes

    def test_write_dependency_error(self):
        """Test write raises DependencyError when kafka-python unavailable."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="test-topic")

        # Mock is_available to return False
        with patch.object(strategy, "is_available", return_value=False):
            from nv_ingest_api.data_handlers.errors import DependencyError

            with pytest.raises(DependencyError, match="kafka-python library is not available"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_kafka_connection_error(self):
        """Test write handles Kafka connection errors."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="test-topic")

        mock_producer = Mock()
        mock_producer.send.side_effect = Exception("Kafka connection failed")

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                with pytest.raises(Exception, match="Kafka connection failed"):
                    strategy.write(['{"test": "data"}'], config)

        # Verify producer was still closed even on error
        mock_producer.close.assert_called_once()

    def test_write_kafka_send_timeout(self):
        """Test write handles Kafka send timeouts."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="test-topic")

        mock_producer = Mock()
        mock_future = Mock()
        mock_future.get.side_effect = TimeoutError("Send timeout")
        mock_producer.send.return_value = mock_future

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                with pytest.raises(TimeoutError, match="Send timeout"):
                    strategy.write(['{"test": "data"}'], config)

    def test_write_empty_payload(self):
        """Test write with empty payload."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="test-topic")

        mock_producer = Mock()

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                strategy.write([], config)

        # Should not call send for empty payload
        mock_producer.send.assert_not_called()
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()

    def test_write_multiple_messages_batch(self):
        """Test write batches multiple messages correctly."""
        strategy = KafkaWriterStrategy()
        config = KafkaDestinationConfig(bootstrap_servers=["localhost:9092"], topic="batch-topic")

        data_payload = ['{"msg": "1"}', '{"msg": "2"}', '{"msg": "3"}']

        mock_producer = Mock()
        mock_futures = [Mock() for _ in range(3)]
        for future in mock_futures:
            future.get.return_value = None
        mock_producer.send.side_effect = mock_futures

        with patch.object(strategy, "is_available", return_value=True):
            dummy_kafka_module = MagicMock()
            dummy_kafka_module.KafkaProducer = MagicMock(return_value=mock_producer)
            with patch.dict("sys.modules", {"kafka": dummy_kafka_module}):
                strategy.write(data_payload, config)

        # Verify all messages were sent
        assert mock_producer.send.call_count == 3
        # Verify flush was called after all sends
        mock_producer.flush.assert_called_once()
        # Verify producer was closed
        mock_producer.close.assert_called_once()
