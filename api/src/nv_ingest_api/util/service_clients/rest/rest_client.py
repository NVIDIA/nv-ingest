# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import time
from typing import Any, Union, Tuple, Optional, Dict, Callable
from urllib.parse import urlparse

import requests

from nv_ingest_api.internal.schemas.message_brokers.response_schema import (
    ResponseSchema,
)
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
    A client for interfacing with an HTTP endpoint (e.g., nv-ingest), providing mechanisms for sending
    and receiving messages with retry logic using the `requests` library by default, but allowing a custom
    HTTP client allocator.

    Extends MessageBrokerClientBase for interface compatibility.
    """

    def __init__(
        self,
        host: str,
        port: int,
        max_retries: int = 0,
        max_backoff: int = 32,
        default_connect_timeout: float = 300.0,
        default_read_timeout: Optional[float] = None,
        http_allocator: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the RestClient.

        By default, uses `requests.Session`. If `http_allocator` is provided, it will be called to instantiate
        the client. If a custom allocator is used, the internal methods (`fetch_message`, `submit_message`)
        might need adjustments if the allocated client's API differs significantly from `requests.Session`.

        Parameters
        ----------
        host : str
            The hostname or IP address of the HTTP server.
        port : int
            The port number of the HTTP server.
        max_retries : int, optional
            Maximum number of retry attempts for connection errors or specific retryable HTTP statuses. Default is 0.
        max_backoff : int, optional
            Maximum backoff delay between retries, in seconds. Default is 32.
        default_connect_timeout : float, optional
            Default timeout in seconds for establishing a connection. Default is 300.0.
        default_read_timeout : float, optional
            Default timeout in seconds for waiting for data after connection. Default is None.
        http_allocator : Optional[Callable[[], Any]], optional
            A callable that returns an HTTP client instance. If None, `requests.Session()` is used.
        **kwargs : dict
            Additional keyword arguments. Supported keys:
            - api_version : str, optional
                API version to use ('v1' or 'v2'). Defaults to 'v1' if not specified.
                Invalid versions will log a warning and fall back to 'v1'.
            - base_url : str, optional
                Override the generated base URL.
            - headers : dict, optional
                Additional headers to include in requests.
            - auth : optional
                Authentication configuration for requests.

        Returns
        -------
        None
        """
        self._host: str = host
        self._port: int = port
        self._max_retries: int = max_retries
        self._max_backoff: int = max_backoff
        self._default_connect_timeout: float = default_connect_timeout
        self._default_read_timeout: Optional[float] = default_read_timeout
        self._http_allocator: Optional[Callable[[], Any]] = http_allocator

        self._timeout: Tuple[float, Optional[float]] = (
            self._default_connect_timeout,
            default_read_timeout,
        )

        if self._http_allocator is None:
            self._client: Any = requests.Session()
            logger.debug("RestClient initialized using default requests.Session.")
        else:
            try:
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

        # Validate and normalize API version to prevent misconfiguration
        # Default to v1 for backwards compatibility if not explicitly provided
        VALID_API_VERSIONS = {"v1", "v2"}
        raw_api_version = kwargs.get("api_version", "v1")
        api_version = str(raw_api_version).strip().lower()

        if api_version not in VALID_API_VERSIONS:
            logger.warning(
                f"Invalid API version '{raw_api_version}' specified. "
                f"Valid versions are: {VALID_API_VERSIONS}. Falling back to 'v1'."
            )
            api_version = "v1"

        self._api_version = api_version
        self._submit_endpoint: str = f"/{api_version}/submit_job"
        self._fetch_endpoint: str = f"/{api_version}/fetch_job"
        self._base_url: str = kwargs.get("base_url") or self._generate_url(self._host, self._port)
        self._headers = kwargs.get("headers", {})
        self._auth = kwargs.get("auth", None)

        logger.debug(f"RestClient base URL set to: {self._base_url}")
        logger.info(
            f"RestClient using API version: {api_version} (endpoints: {self._submit_endpoint}, {self._fetch_endpoint})"
        )

    @staticmethod
    def _generate_url(host: str, port: int) -> str:
        """
        Constructs a base URL from host and port, intelligently handling schemes and existing ports.

        Parameters
        ----------
        host : str
            Hostname, IP address, or full URL (e.g., "localhost", "192.168.1.100",
            "http://example.com", "https://api.example.com:8443/v1").
        port : int
            The default port number to use if the host string does not explicitly specify one.

        Returns
        -------
        str
            A fully constructed base URL string, including scheme, hostname, port,
            and any original path, without a trailing slash.

        Raises
        ------
        ValueError
            If the host string appears to be a URL but lacks a valid hostname.
        """
        url_str: str = str(host).strip()
        scheme: str = "http"
        parsed_path: Optional[str] = None
        effective_port: int = port
        hostname: Optional[str] = None

        if re.match(r"^https?://", url_str, re.IGNORECASE):
            parsed_url = urlparse(url_str)
            hostname = parsed_url.hostname
            if hostname is None:
                raise ValueError(f"Invalid URL provided in host string: '{url_str}'. Could not parse a valid hostname.")
            scheme = parsed_url.scheme
            if parsed_url.port is not None:
                effective_port = parsed_url.port
            else:
                effective_port = port
            if parsed_url.path and parsed_url.path.strip("/"):
                parsed_path = parsed_url.path
        else:
            hostname = url_str
            effective_port = port

        if not hostname:
            raise ValueError(f"Could not determine a valid hostname from input: '{host}'")

        base_url: str = f"{scheme}://{hostname}:{effective_port}"
        if parsed_path:
            if not parsed_path.startswith("/"):
                parsed_path = "/" + parsed_path
            base_url += parsed_path

        final_url: str = base_url.rstrip("/")
        logger.debug(f"Generated base URL: {final_url}")
        return final_url

    @property
    def max_retries(self) -> int:
        """
        Maximum number of retry attempts configured for operations.

        Returns
        -------
        int
            The maximum number of retries.
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
            If value is not a non-negative integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        self._max_retries = value

    def get_client(self) -> Any:
        """
        Returns the underlying HTTP client instance.

        Returns
        -------
        Any
            The active HTTP client instance.
        """
        return self._client

    def ping(self) -> "ResponseSchema":
        """
        Checks if the HTTP server endpoint is responsive using an HTTP GET request.

        Returns
        -------
        ResponseSchema
            An object encapsulating the outcome:
            - response_code = 0 indicates success (HTTP status code < 400).
            - response_code = 1 indicates failure, with details in response_reason.
        """
        ping_timeout: Tuple[float, float] = (
            min(self._default_connect_timeout, 5.0),
            10.0,
        )
        logger.debug(f"Attempting to ping server at {self._base_url} with timeout {ping_timeout}")
        try:
            if isinstance(self._client, requests.Session):
                response: requests.Response = self._client.get(self._base_url, timeout=ping_timeout)
                response.raise_for_status()
                logger.debug(f"Ping successful to {self._base_url} (Status: {response.status_code})")
                return ResponseSchema(response_code=0, response_reason="Ping OK")
        except requests.exceptions.RequestException as e:
            error_reason: str = f"Ping failed due to RequestException for {self._base_url}: {e}"
            logger.warning(error_reason)
            return ResponseSchema(response_code=1, response_reason=error_reason)
        except Exception as e:
            error_reason: str = f"Unexpected error during ping to {self._base_url}: {e}"
            logger.exception(error_reason)
            return ResponseSchema(response_code=1, response_reason=error_reason)

    def fetch_message(
        self, job_id: str, timeout: Optional[Union[float, Tuple[float, float]]] = None
    ) -> "ResponseSchema":
        """
        Fetches a job result message from the server's fetch endpoint.

        Handles retries for connection errors and non-terminal HTTP errors based on the max_retries configuration.
        Specific HTTP statuses are treated as immediate failures (terminal) or as job not ready (HTTP 202).

        Parameters
        ----------
        job_id : str
            The server-assigned identifier of the job to fetch.
        timeout : float or tuple of float, optional
            Specific timeout override for this request.

        Returns
        -------
        ResponseSchema
            - response_code = 0: Success (HTTP 200) with the job result.
            - response_code = 1: Terminal failure (e.g., 404, 400, 5xx, or max retries exceeded).
            - response_code = 2: Job not ready (HTTP 202).

        Raises
        ------
        TypeError
            If the configured client does not support the required HTTP GET method.
        """
        # Ensure headers are included
        headers = {"Content-Type": "application/json"}
        headers.update(self._headers)

        retries: int = 0
        url: str = f"{self._base_url}{self._fetch_endpoint}/{job_id}"
        # Derive per-call timeout if provided; otherwise use default
        if timeout is None:
            req_timeout: Tuple[float, Optional[float]] = self._timeout
        else:
            if isinstance(timeout, tuple):
                # Expect (connect, read)
                connect_t = float(timeout[0])
                read_t = None if (len(timeout) < 2 or timeout[1] is None) else float(timeout[1])
                req_timeout = (connect_t, read_t)
            else:
                # Single float means override read timeout, keep a small connect timeout
                req_timeout = (min(self._default_connect_timeout, 5.0), float(timeout))

        while True:
            result: Optional[Any] = None
            trace_id: Optional[str] = job_id
            response_code: int = -1

            try:
                if isinstance(self._client, requests.Session):
                    with self._client.get(
                        url,
                        timeout=req_timeout,
                        headers=headers,
                        stream=True,
                        auth=self._auth,
                    ) as result:
                        response_code = result.status_code
                        response_text = result.text

                        if response_code in _TERMINAL_RESPONSE_STATUSES:
                            error_reason: str = f"Terminal response code {response_code} fetching {job_id}."
                            logger.error(f"{error_reason} Response: {response_text[:200]}")
                            return ResponseSchema(
                                response_code=1,
                                response_reason=error_reason,
                                response=response_text,
                                trace_id=trace_id,
                            )
                        elif response_code == 200:
                            try:
                                full_response: str = b"".join(c for c in result.iter_content(1024 * 1024) if c).decode(
                                    "utf-8"
                                )
                                return ResponseSchema(
                                    response_code=0,
                                    response_reason="OK",
                                    response=full_response,
                                    trace_id=trace_id,
                                )
                            except Exception as e:
                                logger.error(f"Stream processing error for {job_id}: {e}")
                                return ResponseSchema(
                                    response_code=1,
                                    response_reason=f"Stream processing error: {e}",
                                    trace_id=trace_id,
                                )
                        elif response_code == 202:
                            logger.debug(f"Job {job_id} not ready (202)")
                            return ResponseSchema(
                                response_code=2,
                                response_reason="Job not ready yet. Retry later.",
                                trace_id=trace_id,
                            )
                        else:
                            logger.warning(f"Unexpected status {response_code} for {job_id}. Retrying if possible.")
                else:
                    raise TypeError(
                        f"Unsupported client type for fetch_message: {type(self._client)}. "
                        f"Requires a requests.Session compatible API."
                    )
            except requests.exceptions.RequestException as err:
                logger.debug(
                    f"RequestException fetching {job_id}: {err}. "
                    f"Attempting retry ({retries + 1}/{self._max_retries})..."
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

            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit fetching {job_id} after HTTP {response_code}: {rte}")
                resp_text_snippet: Optional[str] = response_text[:500] if "response_text" in locals() else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text_snippet,
                    trace_id=trace_id,
                )

    def submit_message(
        self,
        channel_name: str,
        message: str,
        for_nv_ingest: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> "ResponseSchema":
        """
        Submits a job message payload to the server's submit endpoint.

        Handles retries for connection errors and non-terminal HTTP errors based on the max_retries configuration.
        Specific HTTP statuses are treated as immediate failures.

        Parameters
        ----------
        channel_name : str
            Not used by RestClient; included for interface compatibility.
        message : str
            The JSON string representing the job specification payload.
        for_nv_ingest : bool, optional
            Not used by RestClient. Default is False.
        timeout : float or tuple of float, optional
            Specific timeout override for this request.

        Returns
        -------
        ResponseSchema
            - response_code = 0: Success (HTTP 200) with a successful job submission.
            - response_code = 1: Terminal failure (e.g., 422, 400, 5xx, or max retries exceeded).

        Raises
        ------
        TypeError
            If the configured client does not support the required HTTP POST method.
        """
        retries: int = 0
        url: str = f"{self._base_url}{self._submit_endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        request_payload: Dict[str, str] = {"payload": message}
        req_timeout: Tuple[float, Optional[float]] = self._timeout

        # Ensure content-type is present
        headers = {"Content-Type": "application/json"}
        headers.update(self._headers)

        while True:
            result: Optional[Any] = None
            trace_id: Optional[str] = None
            response_code: int = -1

            try:
                if isinstance(self._client, requests.Session):
                    result = self._client.post(
                        url,
                        json=request_payload,
                        headers=headers,
                        auth=self._auth,
                        timeout=req_timeout,
                    )
                    response_code = result.status_code
                    trace_id = result.headers.get("x-trace-id")
                    response_text: str = result.text

                    if response_code in _TERMINAL_RESPONSE_STATUSES:
                        error_reason: str = f"Terminal response code {response_code} submitting job."
                        logger.error(f"{error_reason} Response: {response_text[:200]}")
                        return ResponseSchema(
                            response_code=1,
                            response_reason=error_reason,
                            response=response_text,
                            trace_id=trace_id,
                        )
                    elif response_code == 200:
                        server_job_id_raw: str = response_text
                        cleaned_job_id: str = server_job_id_raw.strip('"')
                        logger.debug(f"Submit successful. Server Job ID: {cleaned_job_id}, Trace: {trace_id}")
                        return ResponseSchema(
                            response_code=0,
                            response_reason="OK",
                            response=server_job_id_raw,
                            transaction_id=cleaned_job_id,
                            trace_id=trace_id,
                        )
                    else:
                        logger.warning(f"Unexpected status {response_code} on submit. Retrying if possible.")
                else:
                    raise TypeError(
                        f"Unsupported client type for submit_message: {type(self._client)}. "
                        f"Requires a requests.Session compatible API."
                    )
            except requests.exceptions.RequestException as err:
                logger.debug(
                    f"RequestException submitting job: {err}. Attempting retry ({retries + 1}/{self._max_retries})..."
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

            try:
                retries = self.perform_retry_backoff(retries)
                continue
            except RuntimeError as rte:
                logger.error(f"Max retries hit submitting job after HTTP {response_code}: {rte}")
                resp_text_snippet: Optional[str] = response_text[:500] if "response_text" in locals() else None
                return ResponseSchema(
                    response_code=1,
                    response_reason=f"Max retries after HTTP {response_code}: {rte}",
                    response=resp_text_snippet,
                    trace_id=trace_id,
                )

    def perform_retry_backoff(self, existing_retries: int) -> int:
        """
        Performs exponential backoff sleep if retries are permitted.

        Calculates the delay using exponential backoff (2^existing_retries) capped by self._max_backoff.
        Sleeps for the calculated delay if the number of existing_retries is less than max_retries.

        Parameters
        ----------
        existing_retries : int
            The number of retries already attempted for the current operation.

        Returns
        -------
        int
            The incremented retry count (existing_retries + 1).

        Raises
        ------
        RuntimeError
            If existing_retries is greater than or equal to max_retries (when max_retries > 0).
        """
        if self._max_retries > 0 and existing_retries >= self._max_retries:
            raise RuntimeError(f"Max retry attempts ({self._max_retries}) reached")
        backoff_delay: int = min(2**existing_retries, self._max_backoff)
        retry_attempt_num: int = existing_retries + 1
        logger.debug(
            f"Operation failed. Retrying attempt "
            f"{retry_attempt_num}/{self._max_retries if self._max_retries > 0 else 'infinite'} "
            f"in {backoff_delay:.2f}s..."
        )
        time.sleep(backoff_delay)
        return retry_attempt_num
