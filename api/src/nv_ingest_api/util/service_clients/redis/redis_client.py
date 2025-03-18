# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import redis
from redis.exceptions import RedisError

from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase

# pylint: skip-file

logger = logging.getLogger(__name__)


class RedisClient(MessageBrokerClientBase):
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

    def _check_response(
        self, channel_name: str, timeout: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]:
        """
        Checks for a response from the Redis queue and processes it into a message, fragment, and fragment count.

        Parameters
        ----------
        channel_name : str
            The name of the Redis channel from which to receive the response.
        timeout : float
            The time in seconds to wait for a response from the Redis queue before timing out.

        Returns
        -------
        Tuple[Optional[Dict[str, Any]], Optional[int], Optional[int]]
            A tuple containing:
                - message: A dictionary containing the decoded message if successful,
                    or None if no message was retrieved.
                - fragment: An integer representing the fragment number of the message,
                    or None if no fragment was found.
                - fragment_count: An integer representing the total number of message fragments,
                    or None if no fragment count was found.

        Raises
        ------
        ValueError
            If the message retrieved from Redis cannot be decoded from JSON.
        """

        response = self.get_client().blpop([channel_name], timeout)
        if response is None:
            raise TimeoutError("No response was received in the specified timeout period")

        if len(response) > 1 and response[1]:
            try:
                message = json.loads(response[1])
                fragment = message.get("fragment", 0)
                fragment_count = message.get("fragment_count", 1)

                return message, fragment, fragment_count
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
                raise ValueError(f"Failed to decode message from Redis: {e}")

        return None, None, None

    def fetch_message(self, channel_name: str, timeout: float = 10) -> Optional[Union[str, Dict]]:
        """
        Fetches a message from the specified queue with retries on failure. If the message is fragmented, it will
        continue fetching fragments until all parts have been collected.

        Parameters
        ----------
        channel_name: str
            Channel to fetch the message from.
        timeout : float
            The timeout in seconds for blocking until a message is available. If we receive a multi-part message,
            this value will be temporarily extended in order to collect all fragments.

        Returns
        -------
        Optional[str or Dict]
            The full fetched message, or None if no message could be fetched after retries.

        Raises
        ------
        ValueError
            If fetching the message fails after the specified number of retries or due to other critical errors.
        """
        accumulated_time = 0
        collected_fragments = []
        fragment_count = None
        retries = 0

        logger.debug(f"Starting fetch_message on channel '{channel_name}' with timeout {timeout}s.")

        while True:
            try:
                # Attempt to fetch a message from the Redis queue
                message, fragment, fragment_count = self._check_response(channel_name, timeout)
                logger.debug(f"Fetched fragment: {fragment} (fragment_count: {fragment_count}).")

                if message is not None:
                    if fragment_count == 1:
                        return message

                    collected_fragments.append(message)
                    logger.debug(f"Collected {len(collected_fragments)} of {fragment_count} fragments so far.")

                    # If we have collected all fragments, combine and return
                    if len(collected_fragments) == fragment_count:
                        logger.debug("All fragments received. Sorting and combining fragments.")
                        # Sort fragments by the 'fragment' field to ensure correct order
                        collected_fragments.sort(key=lambda x: x["fragment"])
                        reconstructed_message = self._combine_fragments(collected_fragments)
                        logger.debug("Message reconstructed successfully. Returning combined message.")
                        return reconstructed_message
                else:
                    logger.debug("Received empty response; returning None.")
                    return message

            except TimeoutError:
                # When fragments are expected but not all received before timeout
                if fragment_count and fragment_count > 1:
                    accumulated_time += timeout
                    logger.debug(
                        f"Timeout occurred waiting for fragments. "
                        f"Accumulated timeout: {accumulated_time}s (Threshold: {timeout * fragment_count}s)."
                    )
                    if accumulated_time >= (timeout * fragment_count):
                        err_msg = f"Failed to reconstruct message from {channel_name} after {accumulated_time} sec."
                        logger.error(err_msg)
                        raise ValueError(err_msg)
                else:
                    raise  # This is expected in many cases, so re-raise it

            except RedisError as err:
                retries += 1
                logger.error(f"Redis error during fetch: {err}")
                backoff_delay = min(2**retries, self._max_backoff)

                if self.max_retries > 0 and retries <= self.max_retries:
                    logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to fetch message from {channel_name} after {retries} attempts.")
                    raise ValueError(f"Failed to fetch message from Redis queue after {retries} attempts: {err}")

                # Invalidate client to force reconnection on the next try
                self._client = None

            except Exception as e:
                # Handle non-Redis specific exceptions
                logger.error(f"Unexpected error during fetch from {channel_name}: {e}")
                raise ValueError(f"Unexpected error during fetch: {e}")

    @staticmethod
    def _combine_fragments(fragments: List[Dict[str, Any]]) -> Dict:
        """
        Combines multiple message fragments into a single message by extending the 'data' elements,
        retaining the 'status' and 'description' of the first fragment, and removing 'fragment' and 'fragment_counts'.

        Parameters
        ----------
        fragments : List[Dict[str, Any]]
            A list of fragments to be combined.

        Returns
        -------
        str
            The combined message as a JSON string, containing 'status', 'description', and combined 'data'.
        """
        if not fragments:
            raise ValueError("Fragments list is empty")

        # Use 'status' and 'description' from the first fragment
        combined_message = {
            "status": fragments[0]["status"],
            "description": fragments[0]["description"],
            "data": [],
            "trace": fragments[0].get("trace", {}),
        }

        # Combine the 'data' elements from all fragments
        for fragment in fragments:
            combined_message["data"].extend(fragment["data"])

        return combined_message

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
                backoff_delay = min(2**retries, self._max_backoff)

                if self.max_retries == 0 or retries < self.max_retries:
                    logger.error(f"Submit attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to submit message to {channel_name} after {retries} attempts.")
                    raise
