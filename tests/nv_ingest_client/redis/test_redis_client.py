# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from nv_ingest_client.message_clients.redis.redis_client import RedisClient
from redis import RedisError

MODULE_UNDER_TEST = "nv_ingest.util.redis.redis_client"

TEST_PAYLOAD = '{"job_id": 123, "job_payload": "abc"}'


# Note: Some of the tests in this file are inconsistent with CONTRIBUTING.md's outlined behavior for blackbox style
# testing. In this particular instance we do care about the implementation details of the RedisClient class, so we are
# checking to see that the class is correctly handling errors and retries. To do that, we need to check that it is
# attempting to call the expected Redis methods.


@pytest.fixture
def mock_redis():
    mock = Mock()
    # Simulate ping response for a healthy Redis connection
    mock.ping.return_value = True
    # Simulate blpop response
    mock.blpop.return_value = ("queue", TEST_PAYLOAD)
    # By default, simulate a successful rpush operation
    mock.rpush.return_value = True
    mock.delete.return_value = True

    return mock


@pytest.fixture
def mock_redis_client(mock_redis):
    client = RedisClient(host="localhost", port=6379, redis_allocator=Mock(return_value=mock_redis))
    return client


def test_get_client_with_successful_ping(mock_redis_client, mock_redis):
    """
    Test get_client method when the ping is successful.
    """

    assert mock_redis_client.get_client() == mock_redis

    mock_redis.ping.assert_called_once()


def test_get_client_reconnects_on_failed_ping(mock_redis_client, mock_redis):
    """
    Test get_client method reconnects when ping fails initially.
    """
    mock_redis.ping.side_effect = [
        RedisError("Ping failed"),
        True,
    ]  # Fail first, succeed next
    assert mock_redis_client.get_client() == mock_redis
    # Assert ping was called twice: fail then success
    assert mock_redis.ping.call_count == 2


def test_fetch_message_successful(mock_redis_client, mock_redis):
    """
    Test fetch_message method successfully fetches a message.
    """
    job_payload = mock_redis_client.fetch_message("queue")
    assert job_payload == TEST_PAYLOAD
    mock_redis.blpop.assert_called_once_with(["queue"], 10)


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)  # Mock time.sleep to prevent actual sleeping
def test_fetch_message_with_retries(mock_time, mock_redis_client, mock_redis):
    """
    Test fetch_message method retries on RedisError and eventually succeeds.
    """
    mock_redis.blpop.side_effect = [
        RedisError("Temporary fetch failure"),
        ("queue", TEST_PAYLOAD),
    ]
    mock_redis_client.max_retries = 1  # Allow one retry
    mock_redis_client.max_backoff = 1

    job_payload = mock_redis_client.fetch_message("queue")
    assert job_payload == TEST_PAYLOAD
    # Assert blpop was called twice due to the retry logic
    assert mock_redis.blpop.call_count == 2


def test_fetch_message_exceeds_max_retries(mock_redis_client, mock_redis):
    """
    Test fetch_message method exceeds max retries and raises RedisError.
    """
    mock_redis.blpop.side_effect = RedisError("Persistent fetch failure")
    mock_redis_client.max_retries = 1  # Allow one retry

    with pytest.raises(ValueError):
        mock_redis_client.fetch_message("queue")
    # Assert blpop was called twice: initial attempt + 1 retry
    assert mock_redis.blpop.call_count == 2


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)  # Mock time.sleep to skip actual sleep
def test_submit_message_success(mock_time_sleep, mock_redis_client, mock_redis):
    """
    Test successful message submission to Redis.
    """
    queue_name = "test_queue"
    message = "test_message"

    mock_redis_client.submit_message(queue_name, message)

    mock_redis.rpush.assert_called_once_with(queue_name, message)


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)
def test_submit_message_with_retries(mock_time_sleep, mock_redis_client, mock_redis):
    """
    Test message submission retries on RedisError and eventually succeeds.
    """
    mock_redis.rpush.side_effect = [
        RedisError("Temporary submission failure"),
        True,
    ]  # Fail first, succeed next

    queue_name = "test_queue"
    message = "test_message"

    mock_redis_client.submit_message(queue_name, message)

    # Assert rpush was called twice due to the retry logic
    assert mock_redis.rpush.call_count == 2


@patch(f"{MODULE_UNDER_TEST}.time.sleep", return_value=None)
def test_submit_message_exceeds_max_retries(mock_time_sleep, mock_redis_client, mock_redis):
    """
    Test failure to submit message after exceeding maximum retries.
    """
    mock_redis_client.max_retries = 1
    mock_redis_client.max_backoff = 1
    mock_redis.rpush.side_effect = RedisError("Persistent submission failure")

    queue_name = "test_queue"
    message = "test_message"

    with pytest.raises(RedisError):
        mock_redis_client.submit_message(queue_name, message)

    # Assert that rpush was called 2 times: initial attempt + 1 retry (max_retries=1 in the fixture)
    assert mock_redis.rpush.call_count == 1


# Test case for successful message submission and fetching
@patch("json.dumps", side_effect=json.dumps)
@patch("json.loads", side_effect=json.loads)
def test_submit_and_fetch_message_success(mock_json_loads, mock_json_dumps, mock_redis_client, mock_redis):
    job_payload = {"task": "data"}
    serialized_payload = json.dumps(job_payload)
    mock_redis.blpop.return_value = ("response_channel", json.dumps({"data": "result"}))

    # Simulate submitting the job payload to the task queue
    mock_redis_client.submit_message("task_queue", serialized_payload)
    mock_redis.rpush.assert_called_once_with("task_queue", serialized_payload)

    # Simulate fetching the response
    response = mock_redis_client.fetch_message("response_channel", timeout=90)

    assert response == json.dumps({"data": "result"})


# Test case for handling timeouts during message fetch
def test_fetch_message_timeout(mock_redis_client, mock_redis):
    mock_redis.blpop.return_value = None  # Simulate timeout

    rval = mock_redis_client.fetch_message("response_channel", timeout=10)

    assert rval is None


# Test case for handling errors during message submission
def test_submit_message_error(mock_redis_client, mock_redis):
    mock_redis_client.max_retries = 1
    mock_redis_client.max_backoff = 1
    mock_redis.rpush.side_effect = RedisError("Submission failed")

    with pytest.raises(RedisError):
        mock_redis_client.submit_message("task_queue", "some data")

    # Ensure clean up if job submission fails
    mock_redis.delete.assert_not_called()
