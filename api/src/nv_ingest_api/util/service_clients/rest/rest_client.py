# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

import logging
import re
import time
from typing import Any, Union, Tuple, Optional
from urllib.parse import urlparse

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
        connection_timeout: int = 300,  # Keep original parameter name
        http_allocator: Any = httpx.AsyncClient,  # Restore original default and usage
    ):
        """
        Initialize the RestClient (Restored original signature).
        """
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout  # Store original parameter
        self._http_allocator = http_allocator  # Store allocator
        self._client = self._http_allocator()  # Use the allocator
        self._retries = 0  # Original attribute

        self._submit_endpoint = "/v1/submit_job"
        self._fetch_endpoint = "/v1/fetch_job"
        # Regenerate base_url using original helper - assuming it exists or is similar
        self._base_url = self.generate_url(self._host, self._port)

    # Restore original generate_url helper IF it was different
    @staticmethod
    def generate_url(user_provided_url, user_provided_port) -> str:
        """Examines the user defined URL for http*://. If that
        pattern is detected the URL is used as provided by the user.
        If that pattern does not exist then the assumption is made that
        the endpoint is simply `http://` and that is prepended
        to the user supplied endpoint.

        Args:
            user_provided_url str: Endpoint where the Rest service is running
            user_provided_port int: Port for the service.

        Returns:
            str: Fully validated URL
        """
        # Ensure URL is treated as string
        url_str = str(user_provided_url)
        if not re.match(r"^https?://", url_str):
            # Add the default `http://` if it's not already present in the URL
            base_url = f"http://{url_str}:{user_provided_port}"
        else:
            parsed_url = urlparse(url_str)
            if parsed_url.port:
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"  # Use existing netloc
            else:
                base_url = f"{parsed_url.scheme}://{parsed_url.hostname}:{user_provided_port}"
            if parsed_url.path and parsed_url.path != "/":
                base_url += parsed_url.path  # Preserve path if present

        return base_url.rstrip("/")  # Ensure no trailing slash

    def _get_request_timeout(
        self, specific_timeout: Optional[Union[float, Tuple[float, float]]]
    ) -> Tuple[float, float]:
        """
        Determines the (connect, read) timeout tuple.
        Uses self._connection_timeout as a default reference, primarily for connect.
        """
        default_connect = float(self._connection_timeout)  # Use stored value as connect default
        default_read = 600.0  # Default read timeout (adjust if needed)

        if specific_timeout is None:
            return (default_connect, default_read)
        elif isinstance(specific_timeout, (int, float)):
            # Single float interpreted as read timeout
            return (default_connect, float(specific_timeout))
        elif isinstance(specific_timeout, tuple) and len(specific_timeout) == 2:
            try:
                return (float(specific_timeout[0]), float(specific_timeout[1]))
            except (ValueError, TypeError):
                logger.warning(f"Invalid tuple values in specific_timeout: {specific_timeout}. Using defaults.")
                return (default_connect, default_read)
        else:
            logger.warning(f"Invalid timeout format provided: {specific_timeout}. Using defaults.")
            return (default_connect, default_read)

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

    def fetch_message(self, job_id: str, timeout: Optional[Union[float, Tuple[float, float]]] = None) -> ResponseSchema:
        retries = 0
        url = f"{self._base_url}{self._fetch_endpoint}/{job_id}"
        req_timeout = self._get_request_timeout(timeout)
        logger.debug(f"Attempting fetch for job ID {job_id} from '{url}' with timeout {req_timeout}")

        while True:
            try:
                with requests.get(url, timeout=req_timeout, stream=True) as result:
                    response_code = result.status_code
                    trace_id = result.headers.get("x-trace-id")

                    if response_code in _TERMINAL_RESPONSE_STATUSES:
                        error_reason = f"Terminal response code {response_code} fetching {job_id}."
                        logger.error(f"{error_reason} Response: {result.text[:200]}")
                        return ResponseSchema(
                            response_code=1, response_reason=error_reason, response=result.text, trace_id=trace_id
                        )
                    elif response_code == 200:
                        try:
                            full_response = b"".join(c for c in result.iter_content(1024 * 1024) if c).decode("utf-8")
                            logger.debug(f"Success fetching {job_id}")
                            return ResponseSchema(
                                response_code=0, response_reason="OK", response=full_response, trace_id=trace_id
                            )
                        except Exception as e:
                            logger.error(f"Stream processing error for {job_id}: {e}")
                            return ResponseSchema(
                                response_code=1, response_reason=f"Stream processing error: {e}", trace_id=trace_id
                            )
                    elif response_code == 202:
                        logger.debug(f"Job {job_id} not ready (202)")
                        return ResponseSchema(
                            response_code=2, response_reason="Job not ready yet. Retry later.", trace_id=trace_id
                        )
                    else:  # Non-200/202/Terminal
                        logger.warning(f"Unexpected status {response_code} for {job_id}. Retrying if possible.")
                        # Fall through

            # --- Exception Handling ---
            except requests.exceptions.RequestException as err:
                logger.warning(f"RequestException fetching {job_id}: {err}. Attempting retry...")
                try:
                    retries = self.perform_retry_backoff(retries)
                    continue
                except RuntimeError as rte:
                    logger.error(f"Max retries hit fetching {job_id} after RequestException: {rte}")
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(err))
                except TimeoutError:
                    logger.error(f"Timeout during backoff for {job_id}")
                    return ResponseSchema(response_code=1, response_reason="Timeout during retry backoff")
            except Exception as e:
                logger.exception(f"Unexpected error fetching {job_id}: {e}")
                # *** Use kwargs ***
                return ResponseSchema(response_code=1, response_reason=f"Unexpected fetch error: {e}")

            # --- Retry Logic for fall-through codes ---
            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit fetching {job_id} after HTTP {response_code}: {rte}")
                resp_text = result.text[:500] if result and hasattr(result, "text") else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text,
                    trace_id=trace_id,
                )

    def submit_message(
        self,
        channel_name: str,
        message: str,
        for_nv_ingest: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> ResponseSchema:
        retries = 0
        url = f"{self._base_url}{self._submit_endpoint}"
        headers = {"Content-Type": "application/json"}
        req_timeout = self._get_request_timeout(timeout)
        logger.debug(f"Attempting submit to '{url}' with timeout {req_timeout}")

        while True:
            try:
                result = requests.post(url, json={"payload": message}, headers=headers, timeout=req_timeout)
                response_code = result.status_code
                trace_id = result.headers.get("x-trace-id")

                if response_code in _TERMINAL_RESPONSE_STATUSES:
                    error_reason = f"Terminal response code {response_code} submitting job."
                    logger.error(f"{error_reason} Response: {result.text[:200]}")
                    return ResponseSchema(
                        response_code=1, response_reason=error_reason, response=result.text, trace_id=trace_id
                    )
                elif response_code == 200:
                    server_job_id_raw = result.text
                    cleaned_job_id = server_job_id_raw.strip('"')
                    logger.debug(f"Submit successful. Server Job ID: {cleaned_job_id}, Trace: {trace_id}")
                    return ResponseSchema(
                        response_code=0,
                        response_reason="OK",
                        response=server_job_id_raw,
                        transaction_id=cleaned_job_id,
                        trace_id=trace_id,
                    )
                else:  # Non-200/Terminal
                    logger.warning(f"Unexpected status {response_code} on submit. Retrying if possible.")
                    # Fall through

            # --- Exception Handling ---
            except requests.exceptions.RequestException as err:
                logger.warning(f"RequestException submitting job: {err}. Attempting retry...")
                try:
                    retries = self.perform_retry_backoff(retries)
                    continue
                except RuntimeError as rte:
                    logger.error(f"Max retries hit submitting job after RequestException: {rte}")
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(err))
                except TimeoutError:
                    logger.error("Timeout during backoff for submit")
                    return ResponseSchema(response_code=1, response_reason="Timeout during retry backoff")
            except Exception as e:
                logger.exception(f"Unexpected error submitting job: {e}")
                return ResponseSchema(response_code=1, response_reason=f"Unexpected submit error: {e}")

            # --- Retry Logic for fall-through codes ---
            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit submitting job after HTTP {response_code}: {rte}")
                resp_text = result.text[:500] if result and hasattr(result, "text") else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text,
                    trace_id=trace_id,
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
            logger.debug(
                f"Operation failed. Retrying attempt {existing_retries + 1}/{self.max_retries} in {backoff_delay}s..."
            )
            time.sleep(backoff_delay)
            return existing_retries + 1
        else:
            raise RuntimeError(f"Max retry attempts ({self.max_retries}) reached")
