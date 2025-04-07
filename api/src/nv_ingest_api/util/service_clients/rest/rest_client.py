# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

import logging
import re
import time
from typing import Any, Union, Tuple

import httpx
import requests

from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema
from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase

logger = logging.getLogger(__name__)

# HTTP Response Statuses that result in marking submission as failed
# 4XX - Any 4XX status is considered a client derived error and will result in failure
# 5XX - Not all 500's are terminal but most are. Those which are listed below
_TERMINAL_RESPONSE_STATUSES = [
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    421,
    422,
    423,
    424,
    425,
    426,
    428,
    429,
    431,
    451,
    500,
    501,
    503,
    505,
    506,
    507,
    508,
    510,
    511,
]


class RestClient(MessageBrokerClientBase):
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
        The timeout in seconds for connecting to the HTTP server. Default is 300 seconds.
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

        self._base_url = f"{self.generate_url(self._host, self._port)}"

    def _connect(self) -> None:
        """
        Attempts to reconnect to the HTTP server if the current connection is not responsive.
        """
        ping_result = self.ping()

        if ping_result.response_code != 0:
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
        if self._client is None:
            self._connect()
        return self._client

    def ping(self) -> ResponseSchema:
        """
        Checks if the HTTP server is responsive.

        Returns
        -------
        bool
            True if the server responds to a ping, False otherwise.
        """
        try:
            # Implement a simple GET request to a health endpoint or root
            self._client.ping()
            return ResponseSchema(response_code=0)
        except (httpx.HTTPError, AttributeError):
            return ResponseSchema(response_code=1, response_reason="Failed to ping HTTP server")

    @staticmethod
    def generate_url(user_provided_url, user_provided_port) -> str:
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
        if not re.match(r"^https?://", user_provided_url):
            # Add the default `http://` if it's not already present in the URL
            user_provided_url = f"http://{user_provided_url}:{user_provided_port}"
        else:
            user_provided_url = f"{user_provided_url}:{user_provided_port}"
        return user_provided_url

    def fetch_message(self, job_id: str, timeout: Union[float, Tuple[float, float]] = (10, 600)) -> ResponseSchema:
        """
        Fetches a message (job result) from the specified endpoint with retries on *connection* failures.
        Terminal HTTP status codes (like 400) result in an immediate failure ResponseSchema without retry attempts
        for that specific call.

        Parameters
        ----------
        job_id : str
            The server-side job identifier.
        timeout : Union[float, Tuple[float, float]]
            Timeout for the request. Can be a single float (connect and read) or a tuple (connect, read).
            Defaults to (10 seconds connect, 600 seconds read).

        Returns
        -------
        ResponseSchema
            The fetched message wrapped in a ResponseSchema object. Code 0 on success (200),
            Code 1 on terminal errors (e.g., 4xx, 5xx defined in _TERMINAL_RESPONSE_STATUSES),
            Code 2 for job not ready (202), or Code 1 after exhausting retries for connection issues.
        """
        retries = 0
        url = f"{self._base_url}{self._fetch_endpoint}/{job_id}"
        logger.debug(f"Attempting to fetch message for job ID {job_id} from '{url}'")

        # Ensure timeout is a tuple
        if isinstance(timeout, (int, float)):
            req_timeout = (min(float(timeout), 30.0), float(timeout))
        elif isinstance(timeout, tuple) and len(timeout) == 2:
            req_timeout = timeout
        else:
            logger.warning(f"Invalid timeout format: {timeout}. Using default (10, 600).")
            req_timeout = (10, 600)

        while True:
            try:
                with requests.get(url, timeout=req_timeout, stream=True) as result:  # Direct use of requests
                    response_code = result.status_code
                    trace_id = result.headers.get("x-trace-id")

                    # --- Terminal Error Handling (Includes 400) ---
                    if response_code in _TERMINAL_RESPONSE_STATUSES:
                        # Log the specific terminal error
                        error_reason = (
                            f"Terminal response code {response_code} received when fetching Job Result for ID:"
                            f" {job_id}. "
                            f"Server Response: {result.text[:500]}"  # Include part of the response for context
                        )
                        logger.error(error_reason)
                        # Return failure immediately (Code 1), do not retry this specific call
                        return ResponseSchema(
                            response_code=1,  # Use 1 for terminal/unrecoverable errors
                            response_reason=error_reason,
                            response=result.text,  # Return full text if available
                            trace_id=trace_id,
                        )
                    # --- End Terminal Error Handling ---

                    if response_code == 200:
                        # Success path... (handle streaming response)
                        try:
                            response_chunks = []
                            for chunk in result.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    response_chunks.append(chunk)
                            full_response = b"".join(response_chunks).decode("utf-8")
                            logger.debug(f"Successfully fetched and decoded result for job ID {job_id}")
                            return ResponseSchema(
                                response_code=0,  # Use 0 for success
                                response_reason="OK",
                                response=full_response,
                                trace_id=trace_id,
                            )
                        except Exception as e:
                            logger.error(f"Error processing streamed response for job ID {job_id}: {e}")
                            return ResponseSchema(
                                response_code=1,  # Treat processing error as failure
                                response_reason=f"Failed to process response stream: {e}",
                                trace_id=trace_id,
                            )

                    elif response_code == 202:
                        # Job is not ready yet - return distinct code/reason
                        logger.debug(f"Job ID {job_id} not ready yet (202 Accepted).")
                        return ResponseSchema(
                            response_code=2,  # Use 2 for "not ready" / retryable timeout condition
                            response_reason="Job is not ready yet. Retry later.",
                            trace_id=trace_id,
                        )

                    else:
                        # Handle other unexpected non-terminal, non-success codes
                        logger.warning(
                            f"Received unexpected status code {response_code} for job ID {job_id}."
                            "Attempting retry if configured."
                        )
                        # Fall through to retry logic based on perform_retry_backoff capability for non-terminal errors

            # --- Exception Handling (Connection Errors, Timeouts, etc.) ---
            except (ConnectionError, requests.HTTPError, requests.exceptions.ConnectionError) as err:
                try:
                    retries = self.perform_retry_backoff(retries)
                    # Optional: Recreate client or session if needed based on error type
                    # if "Connection refused" in str(err): time.sleep(10)
                    continue  # Go to next iteration of while loop
                except RuntimeError as rte:
                    # Max retries reached for connection errors
                    logger.error(f"Max retries reached for job ID {job_id} due to connection errors. Error: {rte}")
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(err))
                except TimeoutError:  # If perform_retry_backoff raises this
                    # Decide how to handle timeout during backoff - likely treat as failure
                    return ResponseSchema(response_code=1, response_reason="Timeout during retry backoff")

            # --- General Exception Handling ---
            except Exception as e:
                logger.exception(f"Unexpected error during fetch from {url} for job ID {job_id}: {e}")
                return ResponseSchema(
                    response_code=1, response_reason=f"Unexpected error during fetch: {e}", response=None
                )

            # If code reaches here, it implies a non-terminal HTTP error occurred
            #   (e.g., 503 without hitting max retries yet)
            # Attempt retry for these cases based on perform_retry_backoff
            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                # Max retries reached for non-terminal HTTP errors
                logger.error(
                    f"Max retries reached for job ID {job_id} after non-terminal HTTP error {response_code}. "
                    f"Error: {rte}"
                )
                last_response_text = result.text if "result" in locals() and hasattr(result, "text") else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries reached after HTTP {response_code}: {rte}",
                    response=last_response_text,
                    trace_id=trace_id if "trace_id" in locals() else None,
                )

    def submit_message(self, channel_name: str, message: str, for_nv_ingest: bool = False) -> ResponseSchema:
        """
        Submits a JobSpec to a specified HTTP endpoint with retries on failure.

        Parameters
        ----------
        channel_name : str
            Not used as part of RestClient but defined in MessageClientBase.
        message : str
            The message to submit.
        for_nv_ingest : bool
            Not used as part of RestClient but defined in MessageClientBase.

        Returns
        -------
        ResponseSchema
            The response from the server wrapped in a ResponseSchema object.
        """
        retries = 0
        while True:
            try:
                # Submit via HTTP
                url = f"{self._base_url}{self._submit_endpoint}"
                result = requests.post(url, json={"payload": message}, headers={"Content-Type": "application/json"})

                response_code = result.status_code
                if response_code in _TERMINAL_RESPONSE_STATUSES:
                    # Terminal response code; return error ResponseSchema
                    return ResponseSchema(
                        response_code=1,
                        response_reason=f"Terminal response code {response_code} received when submitting JobSpec",
                        trace_id=result.headers.get("x-trace-id"),
                    )
                else:
                    # If 200 we are good, otherwise let's try again
                    if response_code == 200:
                        logger.debug(f"JobSpec successfully submitted to http endpoint {self._submit_endpoint}")
                        # The REST interface returns a JobId, so we capture that here
                        x_trace_id = result.headers.get("x-trace-id")
                        return ResponseSchema(
                            response_code=0,
                            response_reason="OK",
                            response=result.text,
                            transaction_id=result.text,
                            trace_id=x_trace_id,
                        )
                    else:
                        # Retry the operation
                        retries = self.perform_retry_backoff(retries)
            except requests.RequestException as e:
                logger.error(f"Failed to submit job, retrying... Error: {e}")
                self._client = None  # Invalidate client to force reconnection
                if "Connection refused" in str(e):
                    logger.debug(
                        "Connection refused encountered during submission; sleeping for 10 seconds before retrying."
                    )
                    time.sleep(10)
                try:
                    retries = self.perform_retry_backoff(retries)
                except RuntimeError as rte:
                    # Max retries reached
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(e))
            except Exception as e:
                # Handle non-http specific exceptions
                logger.error(f"Unexpected error during submission of JobSpec to {url}: {e}")
                return ResponseSchema(
                    response_code=1, response_reason=f"Unexpected error during JobSpec submission: {e}", response=None
                )

    def perform_retry_backoff(self, existing_retries) -> int:
        """
        Calculates backoff delay and sleeps if retries are allowed.

        Parameters
        ----------
        existing_retries : int
            The number of retries already attempted.

        Returns
        -------
        int
            The updated number of retries (existing_retries + 1).

        Raises
        ------
        RuntimeError
            If the maximum number of retry attempts has been reached.
        """
        if (existing_retries < self.max_retries) or (self.max_retries == 0):
            backoff_delay = min(2**existing_retries, self._max_backoff)
            logger.info(
                f"Operation failed. Retrying attempt {existing_retries + 1}/{self.max_retries} in {backoff_delay}s..."
            )
            time.sleep(backoff_delay)
            return existing_retries + 1
        else:
            raise RuntimeError(f"Max retry attempts ({self.max_retries}) reached")
