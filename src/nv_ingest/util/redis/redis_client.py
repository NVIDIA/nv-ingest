# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import time
import traceback

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(
        self,
        host,
        port,
        db=0,
        max_retries=0,
        max_backoff=32,
        connection_timeout=300,
        max_pool_size=128,
        use_ssl=False,
        redis_allocator=redis.Redis,  # New parameter for custom Redis allocator
    ):
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.connection_timeout = connection_timeout
        self.use_ssl = use_ssl
        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            socket_connect_timeout=self.connection_timeout,
            max_connections=max_pool_size,
        )
        self.redis_allocator = redis_allocator
        # Use the custom allocator if provided
        self.client = self.redis_allocator(connection_pool=self.pool)
        self.retries = 0

    def _connect(self):
        if not self.ping():
            logger.debug("Reconnecting to Redis")
            self.client = self.redis_allocator(connection_pool=self.pool)

    def get_client(self):
        if self.client is None or not self.ping():
            self._connect()
        return self.client

    def ping(self):
        try:
            self.client.ping()
            return True
        except (RedisError, AttributeError):
            return False

    def fetch_message(self, task_queue):
        retries = 0
        while True:
            try:
                _, job_payload = self.get_client().blpop([task_queue])
                return job_payload
            except RedisError as err:
                retries += 1
                logger.error(f"Redis error during fetch: {err}")
                backoff_delay = min(2**retries, self.max_backoff)

                if self.max_retries == 0 or retries <= self.max_retries:
                    logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to fetch message from {task_queue} after {retries} attempts.")
                    raise

                self.client = None  # Invalidate client to force reconnection

    def submit_message(self, queue_name, message):
        """
        Updated method to submit a message to a specified Redis queue with retries on failure.

        :param queue_name: The name of the queue to submit the message to.
        :param message: The message to submit.
        """
        retries = 0
        while True:
            try:
                self.get_client().rpush(queue_name, message)
                logger.debug(f"Message submitted to {queue_name}")
                return
            except RedisError as e:
                logger.error(f"Failed to submit message, retrying... Error: {e}")
                self.client = None  # Invalidate client to force reconnection
                retries += 1
                backoff_delay = min(2**retries, self.max_backoff)

                if self.max_retries == 0 or retries < self.max_retries:
                    logger.error(f"Submit attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to submit message to {queue_name} after {retries} attempts.")
                    raise

    def submit_job(
        self,
        task_queue,
        job_payload,
        response_channel,
        response_channel_expiration,
        timeout=90,
    ):
        """
        Submits a job to a specified task queue and waits for a response on a specified response
        channel.

        :param task_queue: The Redis queue to submit the job to.
        :param job_payload: The job payload, expected to be a dictionary.
        :param response_channel: The Redis channel to listen for responses on.
        :param response_channel_expiration: Expiration time in seconds for the response channel.
        :param timeout: Timeout in seconds to wait for a response.
        :return: The response data as a dictionary if a response is received.
        :raises RuntimeError: If no response is received within the timeout period.
        """

        logger.debug(f"Submitting job to queue '{task_queue}' with response channel '{response_channel}'")
        try:
            # Serialize the job payload to a JSON string if it's not already a string
            serialized_job_payload = json.dumps(job_payload) if not isinstance(job_payload, str) else job_payload

            # Submit the job payload to the specified queue
            self.submit_message(task_queue, serialized_job_payload)

            # Set an expiration for the response channel to ensure cleanup
            self.get_client().expire(response_channel, response_channel_expiration)

            logger.debug(f"Waiting for response on channel '{response_channel}' for up to {timeout} " "seconds...")

            response = self.get_client().blpop(response_channel, timeout=timeout)

            if response:
                _, response_data = response
                # Delete the response channel after retrieving the response to ensure cleanup
                return json.loads(response_data)
            else:
                # Ensure the response channel is cleaned up if no response is received
                logger.error("No response received within timeout period")
                raise RuntimeError("No response received within timeout period")
        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Ensure the response channel is cleaned up if an exception occurs
            self.get_client().delete(response_channel)
