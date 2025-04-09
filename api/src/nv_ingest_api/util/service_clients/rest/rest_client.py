# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import time
from typing import Any, Union, Tuple, Optional, Dict, Callable
from urllib.parse import urlparse

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
    A client for interfacing with an HTTP endpoint (e.g., nv-ingest), providing
    mechanisms for sending and receiving messages with retry logic using the
    `requests` library by default, but allowing a custom HTTP client allocator.

    Extends MessageBrokerClientBase for interface compatibility.
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_retries: int = 0,
        max_backoff: int = 32,
        default_connect_timeout: float = 300.0,
        default_read_timeout: float = None,
        http_allocator: Optional[Callable[[], Any]] = None,  # Default to None
    ):
        """
        Initializes the RestClient.

        By default, uses `requests.Session`. If `http_allocator` is provided,
        it will be called to instantiate the client. Note that if a custom
        allocator is used, the internal methods (`fetch_message`, `submit_message`)
        might need adjustments if the allocated client's API differs significantly
        from `requests.Session`.

        Parameters
        ----------
        host : str
            The hostname or IP address of the HTTP server.
        port : int
            The port number of the HTTP server.
        max_retries : int, optional
            Maximum number of retry attempts for connection errors or specific
            retryable HTTP statuses. Default is 0.
        max_backoff : int, optional
            Maximum backoff delay between retries, in seconds. Default is 32.
        default_connect_timeout : float, optional
            Default timeout in seconds for establishing a connection. Default is 10.0.
        default_read_timeout : float, optional
            Default timeout in seconds for waiting for data after connection. Default is 600.0.
        http_allocator : Optional[Callable[[], Any]], optional
            A callable (e.g., a class constructor) that returns an HTTP client
            instance. If None, `requests.Session()` is used. Default is None.
        """
        self._host: str = host
        self._port: int = port
        self._max_retries: int = max_retries
        self._max_backoff: int = max_backoff
        self._default_connect_timeout: float = default_connect_timeout
        self._default_read_timeout: float = default_read_timeout
        self._http_allocator: Optional[Callable[[], Any]] = http_allocator

        self._timeout = (self._default_connect_timeout, default_read_timeout)

        # Instantiate the client
        if self._http_allocator is None:
            self._client: Any = requests.Session()
            logger.debug("RestClient initialized using default requests.Session.")
        else:
            try:
                # Use the provided allocator
                self._client = self._http_allocator()
                logger.debug(f"RestClient initialized using provided http_allocator: {self._http_allocator.__name__}")
                if not isinstance(self._client, requests.Session):
                    logger.warning(
                        "Provided http_allocator does not create a requests.Session. "
                        "Internal HTTP calls may fail if the client API is incompatible."
                    )
            except Exception as e:
                logger.exception(
                    f"Failed to instantiate client using provided http_allocator: {e}. "
                    f"Falling back to requests.Session."
                )
                self._client = requests.Session()

        self._submit_endpoint: str = "/v1/submit_job"
        self._fetch_endpoint: str = "/v1/fetch_job"
        self._base_url: str = self._generate_url(self._host, self._port)
        logger.debug(f"RestClient base URL set to: {self._base_url}")

    @staticmethod
    def _generate_url(host: str, port: int) -> str:
        """
        Constructs a base URL, ensuring the scheme and port are present.

        Examines the user defined URL for http*://. If that pattern is detected,
        it uses the provided scheme and hostname. If not, it defaults to `http://`.
        It ensures the port is included in the final base URL.

        Parameters
        ----------
        host : str
            Hostname or full URL where the REST service is running.
        port : int
            Port number for the service.

        Returns
        -------
        str
            A fully constructed base URL (e.g., "http://example.com:8000").
        """
        url_str = str(host)
        if not re.match(r"^https?://", url_str):
            base_url = f"http://{url_str}:{port}"
        else:
            parsed_url = urlparse(url_str)
            scheme = parsed_url.scheme
            hostname = parsed_url.hostname
            if hostname is None:  # Handle case where input might just be scheme://
                raise ValueError(f"Invalid URL provided: {url_str}")
            base_url = f"{scheme}://{hostname}:{port}"
            if parsed_url.path and parsed_url.path.strip("/"):
                base_url += parsed_url.path

        return base_url.rstrip("/")

    @property
    def max_retries(self) -> int:
        """
        int: Maximum number of retry attempts configured for operations.
        """
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        """
        Sets the maximum number of retry attempts.

        Parameters
        ----------
        value : int
            The new maximum number of retries. Must be a non-negative integer.

        Raises
        ------
        ValueError
            If `value` is not a non-negative integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        self._max_retries = value

    def get_client(self) -> Any:
        """
        Returns the underlying HTTP client instance.

        The type of the returned client depends on the `http_allocator` provided
        during initialization (defaults to `requests.Session`).

        Returns
        -------
        Any
            The active HTTP client instance.
        """
        return self._client

    def ping(self) -> ResponseSchema:
        """
        Checks if the HTTP server is responsive by sending a GET request.

        Attempts a GET request to the base URL (`self._base_url`). Assumes a
        successful response (e.g., 2xx) indicates the server is reachable.
        Uses the configured client instance (`self._client`).

        Returns
        -------
        ResponseSchema
            - response_code=0: If the GET request succeeds (status code < 400).
            - response_code=1: If the GET request fails due to connection issues,
                               timeout, or an HTTP error status (>= 400).
        """
        ping_timeout = (min(self._default_connect_timeout, 5.0), 10.0)
        logger.debug(f"Pinging server at {self._base_url} with timeout {ping_timeout}")
        try:
            # --- Assumes self._client has a 'get' method compatible with requests ---
            if isinstance(self._client, requests.Session):
                response = self._client.get(self._base_url, timeout=ping_timeout)
                response.raise_for_status()  # Check for 4xx/5xx
                logger.debug(f"Ping successful (Status: {response.status_code})")
                return ResponseSchema(response_code=0, response_reason="Ping OK")
            else:
                logger.warning(f"Cannot perform standard ping with client type {type(self._client)}. Assuming OK.")
                # Return success optimistically, or implement type-specific ping
                return ResponseSchema(response_code=0, response_reason="Ping skipped for non-default client")

        except requests.exceptions.RequestException as e:
            error_reason = f"Failed to ping HTTP server at {self._base_url}: {e}"
            logger.warning(error_reason)
            return ResponseSchema(response_code=1, response_reason=error_reason)
        except Exception as e:
            error_reason = f"Unexpected error during ping to {self._base_url}: {e}"
            logger.exception(error_reason)
            return ResponseSchema(response_code=1, response_reason=error_reason)

    def fetch_message(self, job_id: str, timeout: Optional[Union[float, Tuple[float, float]]] = None) -> ResponseSchema:
        """
        Fetches a job result message from the server's fetch endpoint.

        Handles retries for connection errors and non-terminal HTTP errors based
        on `max_retries` configuration. Treats specific HTTP statuses
        (_TERMINAL_RESPONSE_STATUSES) as immediate failures. Treats HTTP 202
        as a signal that the job is not ready. Uses the configured client instance.

        Parameters
        ----------
        job_id : str
            The server-assigned identifier of the job result to fetch.
        timeout : Optional[Union[float, Tuple[float, float]]], optional
            Specific timeout override for this request. Interpreted by
            `_get_request_timeout`. Uses client defaults if None. Default is None.

        Returns
        -------
        ResponseSchema
            - response_code=0: Success (HTTP 200), `response` contains message body.
            - response_code=1: Terminal failure (e.g., 404, 400, 5xx, max retries).
            - response_code=2: Job not ready yet (HTTP 202).

        Raises
        ------
        TypeError
            If the configured `self._client` does not support the required HTTP GET method
            or arguments (e.g., if a non-requests-compatible allocator was used).
        """
        retries: int = 0
        url: str = f"{self._base_url}{self._fetch_endpoint}/{job_id}"

        req_timeout = self._timeout

        while True:
            result: Optional[Any] = None  # Type depends on client
            trace_id: Optional[str] = None
            response_code: int = -1

            try:
                # --- Client-specific GET call ---
                if isinstance(self._client, requests.Session):
                    with self._client.get(url, timeout=req_timeout, stream=True) as result:
                        response_code = result.status_code
                        trace_id = result.headers.get("x-trace-id")
                        response_text = result.text  # Read text for potential error reporting

                        if response_code in _TERMINAL_RESPONSE_STATUSES:
                            error_reason = f"Terminal response code {response_code} fetching {job_id}."
                            logger.error(f"{error_reason} Response: {response_text[:200]}")
                            return ResponseSchema(
                                response_code=1, response_reason=error_reason, response=response_text, trace_id=trace_id
                            )
                        elif response_code == 200:
                            try:
                                full_response = b"".join(c for c in result.iter_content(1024 * 1024) if c).decode(
                                    "utf-8"
                                )
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
                        else:  # Non-200/202/Terminal code received
                            logger.warning(f"Unexpected status {response_code} for {job_id}. Retrying if possible.")
                            # Fall through to retry logic below
                else:
                    raise TypeError(
                        f"Unsupported client type for fetch_message: {type(self._client)}. "
                        "Requires requests.Session compatible API or specific implementation."
                    )

            # --- Exception Handling ---
            # Catch requests exception specifically if using requests client
            except requests.exceptions.RequestException as err:
                logger.warning(
                    f"RequestException fetching {job_id}: {err}. Attempting retry ({retries + 1}/{self.max_retries})..."
                )
                try:
                    retries = self.perform_retry_backoff(retries)
                    continue
                except RuntimeError as rte:
                    logger.error(f"Max retries hit fetching {job_id} after RequestException: {rte}")
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(err))
            except Exception as e:
                logger.exception(f"Unexpected error fetching {job_id}: {e}")
                return ResponseSchema(response_code=1, response_reason=f"Unexpected fetch error: {e}")

            # --- Retry Logic for Non-Terminal HTTP Errors (if fall-through occurred) ---
            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit fetching {job_id} after HTTP {response_code}: {rte}")
                # Use response_text captured earlier if available
                resp_text_snippet = response_text[:500] if "response_text" in locals() else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text_snippet,  # Return snippet or full if needed and safe
                    trace_id=trace_id,
                )

    def submit_message(
        self,
        channel_name: str,
        message: str,
        for_nv_ingest: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> ResponseSchema:
        """
        Submits a job message payload to the server's submit endpoint.

        Handles retries for connection errors and non-terminal HTTP errors based
        on `max_retries` configuration. Treats specific HTTP statuses
        (_TERMINAL_RESPONSE_STATUSES) as immediate failures. Uses the configured client instance.

        Parameters
        ----------
        channel_name : str
            Not used by RestClient, part of the base class interface.
        message : str
            The JSON string representing the JobSpec payload (to be wrapped).
        for_nv_ingest : bool, optional
            Not used by RestClient. Default is False.
        timeout : Optional[Union[float, Tuple[float, float]]], optional
            Specific timeout override for this request. Interpreted by
            `_get_request_timeout`. Uses client defaults if None. Default is None.

        Returns
        -------
        ResponseSchema
            - response_code=0: Success (HTTP 200), `response` contains raw server response (job ID),
                               `transaction_id` contains cleaned job ID.
            - response_code=1: Terminal failure (e.g., 422, 400, 5xx, max retries).

        Raises
        ------
        TypeError
            If the configured `self._client` does not support the required HTTP POST method
            or arguments (e.g., if a non-requests-compatible allocator was used).
        """
        retries: int = 0
        url: str = f"{self._base_url}{self._submit_endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        # Prepare the payload according to the server's expected structure { "payload": "..." }
        request_payload: Dict[str, str] = {"payload": message}

        req_timeout = self._timeout

        while True:
            result: Optional[Any] = None  # Type depends on client
            trace_id: Optional[str] = None
            response_code: int = -1

            try:
                # --- Client-specific POST call ---
                if isinstance(self._client, requests.Session):
                    result = self._client.post(
                        url,
                        json=request_payload,  # Use json parameter for requests
                        headers=headers,
                        timeout=req_timeout,
                    )
                    response_code = result.status_code
                    trace_id = result.headers.get("x-trace-id")
                    response_text = result.text  # Read text for potential error/success reporting

                    if response_code in _TERMINAL_RESPONSE_STATUSES:
                        error_reason = f"Terminal response code {response_code} submitting job."
                        logger.error(f"{error_reason} Response: {response_text[:200]}")
                        return ResponseSchema(
                            response_code=1, response_reason=error_reason, response=response_text, trace_id=trace_id
                        )
                    elif response_code == 200:
                        server_job_id_raw = response_text
                        cleaned_job_id = server_job_id_raw.strip('"')
                        logger.debug(f"Submit successful. Server Job ID: {cleaned_job_id}, Trace: {trace_id}")
                        return ResponseSchema(
                            response_code=0,
                            response_reason="OK",
                            response=server_job_id_raw,
                            transaction_id=cleaned_job_id,
                            trace_id=trace_id,
                        )
                    else:  # Non-200/Terminal code received
                        logger.warning(f"Unexpected status {response_code} on submit. Retrying if possible.")
                        # Fall through to retry logic below
                else:
                    raise TypeError(
                        f"Unsupported client type for submit_message: {type(self._client)}. "
                        "Requires requests.Session compatible API or specific implementation."
                    )

            # --- Exception Handling ---
            except requests.exceptions.RequestException as err:
                logger.warning(
                    f"RequestException submitting job: {err}. Attempting retry ({retries + 1}/{self.max_retries})..."
                )
                try:
                    retries = self.perform_retry_backoff(retries)
                    continue
                except RuntimeError as rte:
                    logger.error(f"Max retries hit submitting job after RequestException: {rte}")
                    return ResponseSchema(response_code=1, response_reason=str(rte), response=str(err))
            except Exception as e:
                logger.exception(f"Unexpected error submitting job: {e}")
                return ResponseSchema(response_code=1, response_reason=f"Unexpected submit error: {e}")

            # --- Retry Logic for Non-Terminal HTTP Errors (if fall-through occurred) ---
            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit submitting job after HTTP {response_code}: {rte}")
                # Use response_text captured earlier if available
                resp_text_snippet = response_text[:500] if "response_text" in locals() else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text_snippet,
                    trace_id=trace_id,
                )

    def perform_retry_backoff(self, existing_retries: int) -> int:
        """
        Performs exponential backoff sleep if retries are permitted.

        Calculates the delay using exponential backoff (2^retries) capped by
        `self._max_backoff`. Sleeps for the calculated delay if the number of
        `existing_retries` is less than `self.max_retries`.

        Parameters
        ----------
        existing_retries : int
            The number of retries already attempted for the current operation.

        Returns
        -------
        int
            The incremented retry count (`existing_retries + 1`).

        Raises
        ------
        RuntimeError
            If `existing_retries` is greater than or equal to `self.max_retries`
            (and `self.max_retries` is > 0), indicating maximum retries reached.
        """
        # Allow infinite retries if max_retries is 0
        if self.max_retries > 0 and existing_retries >= self.max_retries:
            raise RuntimeError(f"Max retry attempts ({self.max_retries}) reached")

        # Calculate backoff delay: 2^retries, capped by max_backoff
        backoff_delay = min(2**existing_retries, self._max_backoff)

        # Log retry attempt details
        retry_attempt_num = existing_retries + 1
        max_retry_display = self.max_retries if self.max_retries > 0 else "infinite"
        logger.debug(
            f"Operation failed. Retrying attempt {retry_attempt_num}/{max_retry_display} " f"in {backoff_delay:.2f}s..."
        )

        # Sleep for the calculated delay
        time.sleep(backoff_delay)

        # Return the updated retry count
        return retry_attempt_num
