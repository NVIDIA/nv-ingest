# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
import time
from typing import Literal
from unittest.mock import patch, MagicMock

import pytest
import ray
from pydantic import ValidationError

from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    RedisClientConfig,
    BrokerParamsRedis,
    SimpleClientConfig,
    MessageBrokerTaskSourceConfig,
    MessageBrokerTaskSourceStage,
)


# Initialize Ray once at the module level
@pytest.fixture(scope="module", autouse=True)
def ray_fixture():
    """Initialize Ray for the entire test module."""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestMessageBrokerConfiguration:
    """Black box tests for message broker configuration classes."""

    def test_redis_config_valid(self):
        """
        Test that a valid Redis configuration can be created.

        This test verifies that a RedisClientConfig can be instantiated with
        valid parameters and that default values are applied correctly.
        """
        # Create a minimal valid configuration
        config = RedisClientConfig(client_type="redis", host="localhost", port=6379)

        # Check required fields are set correctly
        assert config.client_type == "redis"
        assert config.host == "localhost"
        assert config.port == 6379

        # Check default values are applied
        assert config.max_retries == 5
        assert config.max_backoff == 5.0
        assert config.connection_timeout == 30.0

        # Check nested broker_params defaults
        assert config.broker_params.db == 0
        assert config.broker_params.use_ssl is False

    def test_redis_config_custom_values(self):
        """
        Test that a Redis configuration accepts custom values.

        This test verifies that a RedisClientConfig properly stores custom values
        for all configurable parameters, including nested broker_params.
        """
        # Create configuration with custom values
        config = RedisClientConfig(
            client_type="redis",
            host="redis.example.com",
            port=7000,
            max_retries=10,
            max_backoff=2.5,
            connection_timeout=15.0,
            broker_params=BrokerParamsRedis(db=3, use_ssl=True),
        )

        # Check all values are set correctly
        assert config.client_type == "redis"
        assert config.host == "redis.example.com"
        assert config.port == 7000
        assert config.max_retries == 10
        assert config.max_backoff == 2.5
        assert config.connection_timeout == 15.0
        assert config.broker_params.db == 3
        assert config.broker_params.use_ssl is True

    def test_simple_config_valid(self):
        """
        Test that a valid Simple configuration can be created.

        This test verifies that a SimpleClientConfig can be instantiated with
        valid parameters and that default values are applied correctly.
        """
        # Create a minimal valid configuration
        config = SimpleClientConfig(client_type="simple", host="localhost", port=7671)

        # Check required fields are set correctly
        assert config.client_type == "simple"
        assert config.host == "localhost"
        assert config.port == 7671

        # Check default values are applied
        assert config.max_retries == 5
        assert config.max_backoff == 5.0
        assert config.connection_timeout == 30.0
        assert config.broker_params == {}

    def test_broker_task_source_config_with_redis(self):
        """
        Test MessageBrokerTaskSourceConfig with Redis client configuration.

        This test verifies that a MessageBrokerTaskSourceConfig can be instantiated
        with a Redis client configuration and that all values are set correctly.
        """
        # Create a task source config with Redis client
        config = MessageBrokerTaskSourceConfig(
            broker_client=RedisClientConfig(client_type="redis", host="redis.example.com", port=6379),
            task_queue="test_queue",
            poll_interval=0.5,
        )

        # Check values are set correctly
        assert config.broker_client.client_type == "redis"
        assert config.broker_client.host == "redis.example.com"
        assert config.broker_client.port == 6379
        assert config.task_queue == "test_queue"
        assert config.poll_interval == 0.5

    def test_broker_task_source_config_with_simple(self):
        """
        Test MessageBrokerTaskSourceConfig with Simple client configuration.

        This test verifies that a MessageBrokerTaskSourceConfig can be instantiated
        with a Simple client configuration and that all values are set correctly.
        """
        # Create a task source config with Simple client
        config = MessageBrokerTaskSourceConfig(
            broker_client=SimpleClientConfig(client_type="simple", host="localhost", port=7671),
            task_queue="simple_queue",
        )

        # Check values are set correctly
        assert config.broker_client.client_type == "simple"
        assert config.broker_client.host == "localhost"
        assert config.broker_client.port == 7671
        assert config.task_queue == "simple_queue"

    def test_invalid_redis_config(self):
        """
        Test that invalid Redis configurations are rejected.

        This test verifies that a RedisClientConfig raises appropriate validation
        errors when required fields are missing or invalid values are provided.
        """
        # Missing required host
        with pytest.raises(ValidationError):
            RedisClientConfig(client_type="redis", port=6379)

        # Missing required port
        with pytest.raises(ValidationError):
            RedisClientConfig(client_type="redis", host="localhost")

        # Invalid max_retries (negative)
        with pytest.raises(ValidationError):
            RedisClientConfig(client_type="redis", host="localhost", port=6379, max_retries=-1)

        # Invalid max_backoff (zero)
        with pytest.raises(ValidationError):
            RedisClientConfig(client_type="redis", host="localhost", port=6379, max_backoff=0)

        # Invalid connection_timeout (zero)
        with pytest.raises(ValidationError):
            RedisClientConfig(client_type="redis", host="localhost", port=6379, connection_timeout=0)

    def test_invalid_broker_task_source_config(self):
        """
        Test that invalid MessageBrokerTaskSourceConfig are rejected.

        This test verifies that a MessageBrokerTaskSourceConfig raises appropriate
        validation errors when required fields are missing or invalid values are provided.
        """
        # Missing required broker_client
        with pytest.raises(ValidationError):
            MessageBrokerTaskSourceConfig(task_queue="test_queue")

        # Missing required task_queue
        with pytest.raises(ValidationError):
            MessageBrokerTaskSourceConfig(
                broker_client=RedisClientConfig(client_type="redis", host="localhost", port=6379)
            )

        # Invalid poll_interval (zero)
        with pytest.raises(ValidationError):
            MessageBrokerTaskSourceConfig(
                broker_client=RedisClientConfig(client_type="redis", host="localhost", port=6379),
                task_queue="test_queue",
                poll_interval=0,
            )

        # Invalid poll_interval (negative)
        with pytest.raises(ValidationError):
            MessageBrokerTaskSourceConfig(
                broker_client=RedisClientConfig(client_type="redis", host="localhost", port=6379),
                task_queue="test_queue",
                poll_interval=-0.1,
            )


class TestMessageBrokerTaskSourceStage:
    """Black box tests for MessageBrokerTaskSourceStage."""

    def test_stage_initialization(self):
        """
        Test that the MessageBrokerTaskSourceStage can be initialized.

        This test verifies that a MessageBrokerTaskSourceStage can be created
        as a Ray actor with a valid configuration.
        """
        # Create a valid configuration
        config = MessageBrokerTaskSourceConfig(
            broker_client=SimpleClientConfig(client_type="simple", host="localhost", port=7671),
            task_queue="test_queue",
            poll_interval=0.01,  # Fast polling for testing
        )

        # Create the stage as a Ray actor
        stage_ref = MessageBrokerTaskSourceStage.remote(config)

        # Verify the stage was created by checking if it has the expected methods
        assert hasattr(stage_ref, "start")
        assert hasattr(stage_ref, "stop")
        assert hasattr(stage_ref, "get_stats")
        assert hasattr(stage_ref, "set_output_queue")
        assert hasattr(stage_ref, "pause")
        assert hasattr(stage_ref, "resume")

        # Get initial stats to verify the stage is accessible
        stats = ray.get(stage_ref.get_stats.remote())

        # Check that stats has the expected structure
        assert isinstance(stats, dict)
        assert "processed" in stats
        assert stats["processed"] == 0  # No messages processed yet

        # Cleanup
        ray.kill(stage_ref)

    def test_stage_lifecycle_methods(self):
        """
        Test the lifecycle methods of MessageBrokerTaskSourceStage.

        This test verifies that the start, stop, pause, and resume methods
        of the stage work as expected from a black box perspective.
        """
        # Create a valid configuration
        config = MessageBrokerTaskSourceConfig(
            broker_client=SimpleClientConfig(client_type="simple", host="localhost", port=7671),
            task_queue="test_queue",
            poll_interval=0.01,
        )

        # Create the stage
        stage_ref = MessageBrokerTaskSourceStage.remote(config)

        # Test start method
        start_result = ray.get(stage_ref.start.remote())
        assert start_result is True, "Start method should return True on success"

        # Test that calling start again returns False (already started)
        start_again_result = ray.get(stage_ref.start.remote())
        assert start_again_result is False, "Start method should return False when already started"

        # Test pause method
        pause_result = ray.get(stage_ref.pause.remote())
        assert pause_result is True, "Pause method should return True on success"

        # Test resume method
        resume_result = ray.get(stage_ref.resume.remote())
        assert resume_result is True, "Resume method should return True on success"

        # Test stop method
        stop_result = ray.get(stage_ref.stop.remote())
        assert stop_result is True, "Stop method should return True on success"

        # Cleanup
        ray.kill(stage_ref)

    def test_set_output_queue(self):
        """
        Test the set_output_queue method of MessageBrokerTaskSourceStage.

        This test verifies that the set_output_queue method correctly accepts
        a queue handle and returns True on success.
        """
        # Create a valid configuration
        config = MessageBrokerTaskSourceConfig(
            broker_client=RedisClientConfig(client_type="redis", host="localhost", port=6379), task_queue="test_queue"
        )

        # Create the stage
        stage_ref = MessageBrokerTaskSourceStage.remote(config)

        # Create a mock queue (just use an object for black box testing)
        mock_queue = object()

        # Test set_output_queue method
        set_queue_result = ray.get(stage_ref.set_output_queue.remote(mock_queue))
        assert set_queue_result is True, "set_output_queue should return True on success"

        # Cleanup
        ray.kill(stage_ref)

    def test_swap_queues(self):
        """
        Test the swap_queues method of MessageBrokerTaskSourceStage.

        This test verifies that the swap_queues method correctly accepts
        a new queue handle and returns True on success.
        """
        # Create a valid configuration
        config = MessageBrokerTaskSourceConfig(
            broker_client=RedisClientConfig(client_type="redis", host="localhost", port=6379), task_queue="test_queue"
        )

        # Create the stage
        stage_ref = MessageBrokerTaskSourceStage.remote(config)

        # Create a mock queue (just use an object for black box testing)
        new_queue = object()

        # Test swap_queues method
        swap_result = ray.get(stage_ref.swap_queues.remote(new_queue))
        assert swap_result is True, "swap_queues should return True on success"

        # Cleanup
        ray.kill(stage_ref)

    def test_get_stats_structure(self):
        """
        Test the structure of stats returned by get_stats method.

        This test verifies that the get_stats method returns a dictionary
        with the expected keys and value types.
        """
        # Create a valid configuration
        config = MessageBrokerTaskSourceConfig(
            broker_client=SimpleClientConfig(client_type="simple", host="localhost", port=7671), task_queue="test_queue"
        )

        # Create the stage
        stage_ref = MessageBrokerTaskSourceStage.remote(config)

        # Start the stage to initialize stats
        ray.get(stage_ref.start.remote())

        # Get stats
        stats = ray.get(stage_ref.get_stats.remote())

        # Check that stats has all expected keys
        expected_keys = [
            "active_processing",
            "delta_processed",
            "elapsed",
            "errors",
            "failed",
            "processed",
            "processing_rate_cps",
            "successful_queue_reads",
            "successful_queue_writes",
            "queue_full",
        ]

        for key in expected_keys:
            assert key in stats, f"Stats should contain key '{key}'"

        # Check value types
        assert isinstance(stats["active_processing"], int)
        assert isinstance(stats["delta_processed"], int)
        assert isinstance(stats["elapsed"], (int, float))
        assert isinstance(stats["errors"], int)
        assert isinstance(stats["failed"], int)
        assert isinstance(stats["processed"], int)
        assert isinstance(stats["processing_rate_cps"], (int, float))
        assert isinstance(stats["successful_queue_reads"], int)
        assert isinstance(stats["successful_queue_writes"], int)
        assert isinstance(stats["queue_full"], int)

        # Stop the stage
        ray.get(stage_ref.stop.remote())

        # Cleanup
        ray.kill(stage_ref)

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        self.output_queue = queue_handle
        self._logger.info("Output queue set: %s", queue_handle)
        return True

    @ray.method(num_returns=1)
    def pause(self) -> bool:
        """
        Pause the stage. This clears the pause event, causing the processing loop
        to block before writing to the output queue.

        Returns
        -------
        bool
            True after the stage is paused.
        """
        self._pause_event.clear()
        self._logger.info("Stage paused.")

        return True

    @ray.method(num_returns=1)
    def resume(self) -> bool:
        """
        Resume the stage. This sets the pause event, allowing the processing loop
        to proceed with writing to the output queue.

        Returns
        -------
        bool
            True after the stage is resumed.
        """
        self._pause_event.set()
        self._logger.info("Stage resumed.")
        return True

    @ray.method(num_returns=1)
    def swap_queues(self, new_queue: any) -> bool:
        """
        Swap in a new output queue for this stage.
        This method pauses the stage, waits for any current processing to finish,
        replaces the output queue, and then resumes the stage.
        """
        self._logger.info("Swapping output queue: pausing stage first.")
        self.pause()
        self.set_output_queue(new_queue)
        self._logger.info("Output queue swapped. Resuming stage.")
        self.resume()
        return True
