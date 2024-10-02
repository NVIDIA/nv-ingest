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

import logging
import time
from typing import Any
from typing import Optional

import httpx
import requests
import re

from nv_ingest_client.message_clients import MessageClientBase

logger = logging.getLogger(__name__)

# HTTP Response Statuses that result in marking submission as failed
# 4XX - Any 4XX status is considered a client derived error and will result in failure
# 5XX - Not all 500's are terminal but most are. Those which are listed below
_TERMINAL_RESPONSE_STATUSES = [
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
    414, 415, 416, 417, 418, 421, 422, 423, 424, 425, 426, 428, 429, 431, 451,
    500, 501, 503, 505, 506, 507, 508, 510, 511
]


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

    def generate_url(self, user_provided_url, user_provided_port) -> str:
        """Examines the user defined URL for http*://. If that
        pattern is detected the URL is used as provided by the user.
        If that pattern does not exist then the assumption is made that
        the endpoint is simply `http://` and that is prepended
        to the user supplied endpoint.

        Args:
            user_provided_url str: Endpoint where the Rest service is running

        Returns:
            str: Fully validated URL
        """
        if not re.match(r'^https?://', user_provided_url):
            # Add the default `http://` if its not already present in the URL
            user_provided_url = f"http://{user_provided_url}:{user_provided_port}"
        else:
            user_provided_url = f"{user_provided_url}:{user_provided_port}"
        return user_provided_url

    def fetch_message(self, job_id: str, timeout: float = 10) -> Optional[str]:
        """
        Fetches a message from the specified queue with retries on failure.

        Parameters
        ----------
        job_id: str
            The server-side job identifier.
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
                url = f"{self.generate_url(self._host, self._port)}{self._fetch_endpoint}/{job_id}"
                logger.debug(f"Invoking fetch_message http endpoint @ '{url}'")
                result = requests.get(url)

                response_code = result.status_code
                if response_code in _TERMINAL_RESPONSE_STATUSES:
                    # Any terminal response code results in a RuntimeError
                    raise RuntimeError(f"A terminal response code: {response_code} was received \
                                       when fetching JobSpec: {job_id} with server response \
                                        '{result.text}'")
                else:
                    # If the result contains a 200 then return the raw JSON string response
                    if response_code == 200:
                        return result.text
                    elif response_code == 202:
                        raise TimeoutError("Job is not ready yet. Retry later.")
                    else:
                        # We could just let this exception bubble, but we capture for clarity
                        # we may also choose to use more specific exceptions in the future
                        try:
                            retries = self.perform_retry_backoff(retries)
                        except RuntimeError as rte:
                            raise rte

            except httpx.HTTPError as err:
                logger.error(f"Error during fetching, retrying... Error: {err}")
                self._client = None  # Invalidate client to force reconnection
                try:
                    retries = self.perform_retry_backoff(retries)
                except RuntimeError:
                    # This RuntimeError is captured from reaching max number of retries
                    # however, we are in an except for httpx error, so we should raise
                    # that exception to ensure the most visibility to the root cause
                    raise
            except TimeoutError:
                raise
            except Exception as e:
                # Handle non-http specific exceptions
                logger.error(f"Unexpected error during fetch from {url}: {e}")
                raise ValueError(f"Unexpected error during fetch: {e}")

    def submit_message(self, _: str, message: str) -> str:
        """
        Submits a JobSpec to a specified HTTP endpoint with retries on failure.

        Parameters
        ----------
        channel_name : str
            Not used as part of RestClient but defined in MessageClientBase
        message: str
            The message to submit.

        Raises
        ------
        httpx.HTTPError
            Any HTTP related errors that occur while attempting to submit the JobSpec

        RuntimeError
            Raised if the maximum number of retry attempts has been reached for a submission
        """
        retries = 0
        while True:
            try:
                # Submit via HTTP
                url = f"{self.generate_url(self._host, self._port)}{self._submit_endpoint}"
                result = requests.post(url, json={"payload": message}, headers={"Content-Type": "application/json"})

                response_code = result.status_code
                if response_code in _TERMINAL_RESPONSE_STATUSES:
                    # Any terminal response code results in a RuntimeError
                    raise RuntimeError(f"A terminal response code: {response_code} was received \
                                       when submitting JobSpec: {'TODO'} with server response \
                                        '{result.text}'")
                else:
                    # If 200 we are good, otherwise let's try again
                    if response_code == 200:
                        logger.debug(f"JobSpec successfully submitted to http \
                                     endpoint {self._submit_endpoint}, Resulting JobId: {result.json()}")
                        # The REST interface returns a JobId, so we capture that here

                        return result.json()
                    else:
                        # We could just let this exception bubble, but we capture for clarity
                        # we may also choose to use more specific exceptions in the future
                        try:
                            retries = self.perform_retry_backoff(retries)
                        except RuntimeError as rte:
                            raise rte

            except httpx.HTTPError as e:
                logger.error(f"Failed to submit job, retrying... Error: {e}")
                self._client = None  # Invalidate client to force reconnection
                try:
                    retries = self.perform_retry_backoff(retries)
                except RuntimeError:
                    # This RuntimeError is captured from reaching max number of retries
                    # however, we are in an except for httpx error, so we should raise
                    # that exception to ensure the most visibility to the root cause
                    raise e
            except Exception as e:
                # Handle non-http specific exceptions
                logger.error(f"Unexpected error during submission of JobSpec to {url}: {e}")
                raise ValueError(f"Unexpected error during JobSpec submission: {e}")

    def perform_retry_backoff(self, existing_retries) -> int:
        """
        Attempts to perform a backoff retry delay. This function accepts the
        current number of retries that have been attempted and compares
        that with the maximum number of retries allowed. If the current
        number of retries excedes the max then a RuntimeError is raised.

        Parameters
        ----------
        existing_retries : int
            The number of retries that have been attempting for this submission thus far
        job_id: str
            The server-side job identifier

        Returns
        -------
        int
            The updated number of retry attempts that have been made for this submission

        Raises
        ------
        RuntimeError
            Raised if the maximum number of retry attempts has been reached.
        """
        backoff_delay = min(2 ** existing_retries, self._max_backoff)
        logger.debug(f"Retry #: {existing_retries} of max_retries: {self.max_retries} \
                        | current backoff_delay: {backoff_delay} of max_backoff: {self._max_backoff}")

        if self.max_retries > 0 and existing_retries <= self.max_retries:
            logger.error(f"Fetch attempt failed, retrying in {backoff_delay}s...")
            time.sleep(backoff_delay)
        else:
            raise RuntimeError(f"Max retry attempts of {self.max_retries} reached")

        return existing_retries + 1
