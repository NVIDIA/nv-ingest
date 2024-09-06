# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: skip-file

import json
import logging
import time
from typing import Any
from typing import Optional

import httpx
import requests
from nv_ingest_client.message_clients import MessageClientBase
from nv_ingest_client.primitives.jobs.job_state import JobState

logger = logging.getLogger(__name__)


class RestClient(MessageClientBase):
    """
    A client for interfacing with the nv-ingest HTTP endpoint, providing mechanisms for sending and receiving messages
    with retry logic and connection management.

    Parameters
    ----------
    host : str
        The hostname of the HTTP server.
    port : int
        The port number of the HTTP server.
    max_retries : int, optional
        The maximum number of retry attempts for operations. Default is 0 (no retries).
    max_backoff : int, optional
        The maximum backoff delay between retries in seconds. Default is 32 seconds.
    connection_timeout : int, optional
        The timeout in seconds for connecting to the Redis server. Default is 300 seconds.
    http_allocator : Any, optional
        The HTTP client allocator.

    Attributes
    ----------
    client : Any
        The HTTP client instance used for operations.
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_retries: int = 0,
        max_backoff: int = 32,
        connection_timeout: int = 300,
        http_allocator: Any = httpx.AsyncClient,
    ):
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout
        self._http_allocator = http_allocator
        self._client = self._http_allocator()
        self._retries = 0

        self._submit_endpoint = "/v1/submit_job"
        self._fetch_endpoint = "/v1/fetch_job"

    def _connect(self) -> None:
        """
        Attempts to reconnect to the HTTP server if the current connection is not responsive.
        """
        if not self.ping():
            logger.debug("Reconnecting to HTTP server")
            self._client = self._http_allocator()

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self._max_retries = value

    def get_client(self) -> Any:
        """
        Returns a HTTP client instance, reconnecting if necessary.

        Returns
        -------
        Any
            The HTTP client instance.
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
        except (httpx.HTTPError, AttributeError):
            return False

    def fetch_message(self, job_state: JobState, timeout: float = 10) -> Optional[str]:
        """
        Fetches a message from the specified queue with retries on failure.

        Parameters
        ----------
        job_state : JobState
            The JobState of the message to be fetched.
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
                # Fetch via HTTP
                url = f"http://{self._host}:{self._port}{self._fetch_endpoint}/{job_state.job_id}"
                print(f"Fetch message at URL: {url}")
                result = requests.get(url)
                logger.debug(f"Fetch Message submitted to http endpoint {self._submit_endpoint}")
                print(f"RestClient fetch Result: {result}")
                break
            except httpx.HTTPError as err:
                retries += 1
                logger.error(f"REST error during fetch: {err}")
                backoff_delay = min(2**retries, self._max_backoff)

                if self.max_retries > 0 and retries <= self.max_retries:
                    logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"Failed to fetch message from {job_state.channel_name} after {retries} attempts.")
                    raise ValueError(f"Failed to fetch message from HTTP endpoint after {retries} attempts: {err}")

                # Invalidate client to force reconnection on the next try
                self._client = None
            except Exception as e:
                # Handle non-http specific exceptions
                logger.error(f"Unexpected error during fetch from {job_state.channel_name}: {e}")
                raise ValueError(f"Unexpected error during fetch: {e}")

    def submit_message(self, _: str, message: str) -> str:
        """
        Submits a message to a specified HTTP endpoint with retries on failure.

        Parameters
        ----------
        channel_name : str
            Not used as part of RestClient
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
                # Submit via HTTP
                url = f"http://{self._host}:{self._port}{self._submit_endpoint}"
                result = requests.post(url, json=json.loads(message), headers={"Content-Type": "application/json"})
                logger.debug(f"Message submitted to http endpoint {self._submit_endpoint}")
                print(f"RestClient submission Result: {result}")
                print(f"Response.text: {result.text}")
                print(f"Response.json: {result.json()}")
                break
            except httpx.HTTPError as e:
                logger.error(f"Failed to submit job, retrying... Error: {e}")
                self._client = None  # Invalidate client to force reconnection
                retries += 1
                backoff_delay = min(2**retries, self._max_backoff)

                if self.max_retries == 0 or retries < self.max_retries:
                    logger.error(f"Submit attempt failed, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(
                        f"Failed to submit message to http endpoint {self._submit_endpoint} after {retries} attempts."
                    )
                    raise
