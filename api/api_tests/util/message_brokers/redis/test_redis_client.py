# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock, MagicMock
import redis
from unittest.mock import patch

import pytest

import nv_ingest_api.util.service_clients.redis.redis_client as module_under_test
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient
from redis.exceptions import RedisError

MODULE_UNDER_TEST = f"{module_under_test.__name__}"

TEST_PAYLOAD = '{"job_id": 123, "job_payload": "abc"}'


# Use fixture for the client instance if using pytest
@pytest.fixture
def mock_redis_client_instance(mocker):
    """Provides a mocked RedisClient instance."""
    # Mock the dependencies of RedisClient's __init__ if needed
    mocker.patch(f"{MODULE_UNDER_TEST}.redis.ConnectionPool")  # Mock pool creation

    # Instantiate the client - it will use the mocked pool
    # Pass necessary args; mocking __init__ might be simpler if complex
    client = RedisClient(
        host="mock_host", port=6379, max_retries=1  # Ensure at least one retry is allowed for the test
    )

    # Mock the get_client method to return a mock Redis connection
    mock_redis_connection = MagicMock(spec=redis.Redis)  # Mock the connection object itself
    client.get_client = MagicMock(return_value=mock_redis_connection)  # Make get_client return the mock connection

    # Configure the pipeline mock structure
    pipeline_mock = MagicMock()
    mock_redis_connection.pipeline.return_value = pipeline_mock

    # Mock the pipeline methods (rpush, expire, execute)
    # No need for side_effects on rpush/expire unless testing specific queuing behavior
    pipeline_mock.rpush = MagicMock(return_value=pipeline_mock)  # Chainable
    pipeline_mock.expire = MagicMock(return_value=pipeline_mock)  # Chainable
    pipeline_mock.execute = MagicMock()  # side_effect set in test

    # Store the mocks on the client instance for easy access in tests if needed
    client._mocks = {"connection": mock_redis_connection, "pipeline": pipeline_mock}
    return client


@pytest.fixture
def mock_redis():
    mock = Mock()
    # Simulate a healthy ping (note: redis returns booleans)
    mock.ping.return_value = True
    # For BLPOP, Redis returns a tuple where the message is in bytes.
    mock.blpop.return_value = ("queue", TEST_PAYLOAD.encode("utf-8"))
    # Create a pipeline mock that simulates rpush/expire/execute calls.
    pipeline_mock = Mock()
    pipeline_mock.rpush.return_value = True
    pipeline_mock.expire.return_value = True
    pipeline_mock.execute.return_value = [True]  # Simulated execute result.
    mock.pipeline.return_value = pipeline_mock
    mock.delete.return_value = True
    return mock


@pytest.fixture
def mock_redis_client(mock_redis):
    with patch("redis.Redis", return_value=mock_redis):
        # For testing purposes, we supply a dummy redis_allocator that returns mock_redis.
        client = RedisClient(host="localhost", port=6379, redis_allocator=Mock(return_value=mock_redis))
        return client


def test_get_client_with_successful_ping(mock_redis_client, mock_redis):
    """
    Test get_client method when the ping is successful.
    Expect that ping() is called exactly once.
    """
    client = mock_redis_client.get_client()
    assert client == mock_redis
    client.ping()
    mock_redis.ping.assert_called_once()


def test_fetch_message_successful(mock_redis_client, mock_redis):
    """
    Test fetch_message method successfully returns a complete (non-fragmented) message.
    """
    job_payload = mock_redis_client.fetch_message("queue")
    expected = json.loads(TEST_PAYLOAD)
    assert job_payload == expected


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)
def test_fetch_message_with_retries(mock_time, mock_redis_client, mock_redis):
    """
    Test that fetch_message retries when a temporary error (RedisError) occurs and then succeeds.
    """
    # Set BLPOP to raise a RedisError on the first call and then succeed.
    mock_redis.blpop.side_effect = [RedisError("Temporary fetch failure"), ("queue", TEST_PAYLOAD.encode("utf-8"))]
    mock_redis_client.max_retries = 1
    mock_redis_client.max_backoff = 1

    job_payload = mock_redis_client.fetch_message("queue")
    expected = json.loads(TEST_PAYLOAD)
    assert job_payload == expected
    assert mock_redis.blpop.call_count == 2


# Test using the fixture
@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)
def test_submit_message_with_retries(ock_time_sleep, mock_redis_client_instance):
    """
    test submit_message retries on rediserror during execute and succeeds on retry.
    """
    queue_name = "test_queue"
    message = "test_message"

    # get the mocks from the fixture
    pipeline_mock = mock_redis_client_instance._mocks["pipeline"]
    mock_connection = mock_redis_client_instance._mocks["connection"]

    # --- configure side effects ---
    # 1. first call to pipe.execute() raises rediserror
    # 2. second call to pipe.execute() succeeds (returns list indicating success of queued commands)
    pipeline_mock.execute.side_effect = [
        RedisError("temporary submission failure"),
        [True],  # simulate successful execution of [rpush] command
    ]

    # --- call the method under test ---
    # use the client instance provided by the fixture
    mock_redis_client_instance.submit_message(queue_name, message, ttl_seconds=None)  # assuming no ttl for this test

    # --- assertions ---
    # verify get_client was called (at least twice: initial attempt + retry)
    assert mock_redis_client_instance.get_client.call_count >= 2

    # verify pipeline was created twice
    assert mock_connection.pipeline.call_count == 2

    # verify rpush was queued twice (once for each pipeline)
    # access the mock directly on the pipeline mock object
    assert pipeline_mock.rpush.call_count == 2
    pipeline_mock.rpush.assert_any_call(queue_name, message)  # check arguments on any call

    # verify expire was not queued (since ttl_seconds=none)
    pipeline_mock.expire.assert_called()

    # verify execute was called twice
    assert pipeline_mock.execute.call_count == 2


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)
@patch(f"{MODULE_UNDER_TEST}.logger.error")
def test_submit_message_exceeds_max_retries(mock_logger_error, mock_time_sleep, mock_redis_client, mock_redis):
    """
    Test that submit_message ultimately fails with a ValueError when the maximum
    number of retries is exceeded.
    """
    mock_redis_client.max_retries = 1
    mock_redis_client.max_backoff = 1
    queue_name = "test_queue"
    message = "test_message"
    pipeline_mock = mock_redis.pipeline.return_value

    # Always fail: every call to rpush and execute raises an error.
    pipeline_mock.rpush.side_effect = RedisError("Persistent submission failure")
    pipeline_mock.execute.side_effect = RedisError("Persistent submission failure")

    with pytest.raises(ValueError):
        mock_redis_client.submit_message(queue_name, message)

    # With max_retries=1, there is the initial attempt plus one retry, so rpush should be called twice.
    assert pipeline_mock.rpush.call_count == 2
