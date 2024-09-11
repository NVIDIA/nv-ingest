# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: skip-file

import logging
import time
from typing import Any
from typing import Optional

import redis
from redis.exceptions import RedisError

from nv_ingest_client.message_clients import MessageClientBase

logger = logging.getLogger(__name__)


class RedisClient(MessageClientBase):
    """
    A client for interfacing with Redis, providing mechanisms for sending and receiving messages
    with retry logic and connection management.

    Parameters
    ----------
    host : str
        The hostname of the Redis server.
    port : int
        The port number of the Redis server.
    db : int, optional
        The database number to connect to. Default is 0.
    max_retries : int, optional
        The maximum number of retry attempts for operations. Default is 0 (no retries).
    max_backoff : int, optional
        The maximum backoff delay between retries in seconds. Default is 32 seconds.
    connection_timeout : int, optional
        The timeout in seconds for connecting to the Redis server. Default is 300 seconds.
    max_pool_size : int, optional
        The maximum number of connections in the Redis connection pool. Default is 128.
    use_ssl : bool, optional
        Specifies if SSL should be used for the connection. Default is False.
    redis_allocator : Any, optional
        The Redis client allocator, allowing for custom Redis client instances. Default is redis.Redis.

    Attributes
    ----------
    client : Any
        The Redis client instance used for operations.
    """

    def __init__(
            self,
            host: str,
            port: int,
            db: int = 0,
            max_retries: int = 0,
            max_backoff: int = 32,
            connection_timeout: int = 300,
            max_pool_size: int = 128,
            use_ssl: bool = False,
            redis_allocator: Any = redis.Redis,  # Type hint as 'Any' due to dynamic nature
    ):
        self._host = host
        self._port = port
        self._db = db
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout
        self._use_ssl = use_ssl
        self._pool = redis.ConnectionPool(
            host=self._host,
            port=self._port,
            db=self._db,
            socket_connect_timeout=self._connection_timeout,
            max_connections=max_pool_size,
        )
        self._redis_allocator = redis_allocator
        self._client = self._redis_allocator(connection_pool=self._pool)
        self._retries = 0

    def _connect(self) -> None:
        """
        Attempts to reconnect to the Redis server if the current connection is not responsive.
        """
        if not self.ping():
            logger.debug("Reconnecting to Redis")
            self._client = self._redis_allocator(connection_pool=self._pool)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self._max_retries = value

    def get_client(self) -> Any:
        """
        Returns a Redis client instance, reconnecting if necessary.

        Returns
        -------
        Any
            The Redis client instance.
        """
        if self._client is None or not self.ping():
            self._connect()
        return self._client

    def ping(self) -> bool:
        """
        Checks if the Redis server is responsive.

        Returns
        -------
        bool
            True if the server responds to a ping, False otherwise.
        """
        try:
            self._client.ping()
            return True
        except (RedisError, AttributeError):
            return False

    def fetch_message_morpheus(self, task_queue):
        retries = 0
        while True:
            try:
                _, job_payload = self.get_client().blpop([task_queue])
                return job_payload
            except RedisError as err:
                retries += 1
                logger.error(f"Redis error during fetch: {err}")
                backoff_delay = min(2 ** retries, self.max_backoff)

                if self.max_retries == 0 or retries <= self.max_retries:
                    logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to fetch message from {task_queue} after {retries} attempts.")
                    raise

                self.client = None  # Invalidate client to force reconnection

    def fetch_message(self, fetch_channel: str, timeout: float = 300) -> Optional[str]:
        """
        Fetches a message from the specified queue with retries on failure.

        Parameters
        ----------
        fetch_channel: str
            Channel to fetch the message from.

        timeout : float
            The timeout in seconds for blocking until a message is available.

        Returns
        -------
        Optional[str]
            The fetched message, or None if no message could be fetched.

        Raises
        ------
        ValueError
            If fetching the message fails after the specified number of retries or due to other critical errors.
        """
        retries = 0
        while True:
            try:
                response = self.get_client().blpop([fetch_channel], timeout)
                if response and response[1]:
                    return response[1]
                return None
            except RedisError as err:
                retries += 1
                logger.error(f"Redis error during fetch: {err}")
                backoff_delay = min(2 ** retries, self._max_backoff)

                if self.max_retries > 0 and retries <= self.max_retries:
                    logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to fetch message from {fetch_channel} after {retries} attempts.")
                    raise ValueError(f"Failed to fetch message from Redis queue after {retries} attempts: {err}")

                # Invalidate client to force reconnection on the next try
                self._client = None
            except Exception as e:
                # Handle non-Redis specific exceptions
                logger.error(f"Unexpected error during fetch from {fetch_channel}: {e}")
                raise ValueError(f"Unexpected error during fetch: {e}")

    def submit_message(self, channel_name: str, message: str) -> None:
        """
        Submits a message to a specified Redis queue with retries on failure.

        Parameters
        ----------
        channel_name : str
            The name of the queue to submit the message to.
        message : str
            The message to submit.

        Raises
        ------
        RedisError
            If submitting the message fails after the specified number of retries.
        """
        retries = 0
        while True:
            try:
                self.get_client().rpush(channel_name, message)
                logger.debug(f"Message submitted to {channel_name}")
                break
            except RedisError as e:
                logger.error(f"Failed to submit message, retrying... Error: {e}")
                self._client = None  # Invalidate client to force reconnection
                retries += 1
                backoff_delay = min(2 ** retries, self._max_backoff)

                if self.max_retries == 0 or retries < self.max_retries:
                    logger.error(f"Submit attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to submit message to {channel_name} after {retries} attempts.")
                    raise
