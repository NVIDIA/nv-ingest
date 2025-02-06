# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import threading
import time
from typing import Any
from typing import Optional
from typing import Tuple

import backoff
import cv2
import numpy as np
import requests
import tritonclient.grpc as grpcclient
from packaging import version as pkgversion

from nv_ingest.util.image_processing.transforms import normalize_image
from nv_ingest.util.image_processing.transforms import pad_image
from nv_ingest.util.nim.decorators import multiprocessing_cache
from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)

DEPLOT_MAX_TOKENS = 128
DEPLOT_TEMPERATURE = 1.0
DEPLOT_TOP_P = 1.0


class ModelInterface:
    """
    Base class for defining a model interface that supports preparing input data, formatting it for
    inference, parsing output, and processing inference results.
    """

    def format_input(self, data: dict, protocol: str, max_batch_size: int):
        """
        Format the input data for the specified protocol.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to format the data for.
        """

        raise NotImplementedError("Subclasses should implement this method")

    def parse_output(self, response, protocol: str, data: Optional[dict] = None, **kwargs):
        """
        Parse the output data from the model's inference response.

        Parameters
        ----------
        response : Any
            The response from the model inference.
        protocol : str
            The protocol used ("grpc" or "http").
        data : dict, optional
            Additional input data passed to the function.
        """

        raise NotImplementedError("Subclasses should implement this method")

    def prepare_data_for_inference(self, data: dict):
        """
        Prepare input data for inference by processing or transforming it as required.

        Parameters
        ----------
        data : dict
            The input data to prepare.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def process_inference_results(self, output_array, protocol: str, **kwargs):
        """
        Process the inference results from the model.

        Parameters
        ----------
        output_array : Any
            The raw output from the model.
        kwargs : dict
            Additional parameters for processing.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        raise NotImplementedError("Subclasses should implement this method")


class NimClient:
    """
    A client for interfacing with a model inference server using gRPC or HTTP protocols.
    """

    def __init__(
        self,
        model_interface,
        protocol: str,
        endpoints: Tuple[str, str],
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 5,
    ):
        """
        Initialize the NimClient with the specified model interface, protocol, and server endpoints.

        Parameters
        ----------
        model_interface : ModelInterface
            The model interface implementation to use.
        protocol : str
            The protocol to use ("grpc" or "http").
        endpoints : tuple
            A tuple containing the gRPC and HTTP endpoints.
        auth_token : str, optional
            Authorization token for HTTP requests (default: None).
        timeout : float, optional
            Timeout for HTTP requests in seconds (default: 30.0).

        Raises
        ------
        ValueError
            If an invalid protocol is specified or if required endpoints are missing.
        """

        self.client = None
        self.model_interface = model_interface
        self.protocol = protocol.lower()
        self.auth_token = auth_token
        self.timeout = timeout  # Timeout for HTTP requests
        self.max_retries = max_retries
        self._grpc_endpoint, self._http_endpoint = endpoints
        self._max_batch_sizes = {}
        self._lock = threading.Lock()

        if self.protocol == "grpc":
            if not self._grpc_endpoint:
                raise ValueError("gRPC endpoint must be provided for gRPC protocol")
            logger.debug(f"Creating gRPC client with {self._grpc_endpoint}")
            self.client = grpcclient.InferenceServerClient(url=self._grpc_endpoint)
        elif self.protocol == "http":
            if not self._http_endpoint:
                raise ValueError("HTTP endpoint must be provided for HTTP protocol")
            logger.debug(f"Creating HTTP client with {self._http_endpoint}")
            self.endpoint_url = generate_url(self._http_endpoint)
            self.headers = {"accept": "application/json", "content-type": "application/json"}
            if self.auth_token:
                self.headers["Authorization"] = f"Bearer {self.auth_token}"
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def _fetch_max_batch_size(self, model_name, model_version: str = "") -> int:
        """Fetch the maximum batch size from the Triton model configuration in a thread-safe manner."""
        if model_name in self._max_batch_sizes:
            return self._max_batch_sizes[model_name]

        with self._lock:
            # Double check, just in case another thread set the value while we were waiting
            if model_name in self._max_batch_sizes:
                return self._max_batch_sizes[model_name]

            if not self._grpc_endpoint:
                self._max_batch_sizes[model_name] = 1
                return 1

            try:
                client = self.client if self.client else grpcclient.InferenceServerClient(url=self._grpc_endpoint)
                model_config = client.get_model_config(model_name=model_name, model_version=model_version)
                self._max_batch_sizes[model_name] = model_config.config.max_batch_size
                logger.info(f"Max batch size for model '{model_name}': {self._max_batch_sizes[model_name]}")
            except Exception as e:
                self._max_batch_sizes[model_name] = 1
                logger.warning(f"Failed to retrieve max batch size: {e}, defaulting to 1")

            return self._max_batch_sizes[model_name]

    def try_set_max_batch_size(self, model_name, model_version: str = ""):
        """Attempt to set the max batch size for the model if it is not already set, ensuring thread safety."""
        self._fetch_max_batch_size(model_name, model_version)

    @traceable_func(trace_name="{stage_name}::{model_name}")
    def infer(self, data: dict, model_name: str, **kwargs) -> Any:
        """
        Perform inference using the specified model and input data.

        Parameters
        ----------
        data : dict
            The input data for inference.
        model_name : str
            The name of the model to use for inference.
        kwargs : dict
            Additional parameters for inference.

        Returns
        -------
        Any
            The processed inference results.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """

        try:
            # 1. Retrieve or default to the model's maximum batch size
            batch_size = self._fetch_max_batch_size(model_name)
            max_requested_batch_size = kwargs.get("max_batch_size", batch_size)
            force_requested_batch_size = kwargs.get("force_max_batch_size", False)

            # 1a. In some cases we can't use the absolute max batch size (or don't want to) so we allow override
            # 1b. In some cases we can't reliably retrieve the max batch size so we default to 1 and allow forced
            #   override
            if not force_requested_batch_size:
                max_batch_size = min(batch_size, max_requested_batch_size)
            else:
                max_batch_size = max_requested_batch_size

            # 2. Prepare data for inference
            prepared_data = self.model_interface.prepare_data_for_inference(data)

            # 3. Format the input based on protocol
            #    NOTE: This now returns a list of batches.
            formatted_batches = self.model_interface.format_input(
                prepared_data, protocol=self.protocol, max_batch_size=max_batch_size
            )

            # Container for all parsed outputs
            all_parsed_outputs = []

            # 4. Loop over each batch
            for batch_input in formatted_batches:
                if self.protocol == "grpc":
                    logger.debug("Performing gRPC inference for a batch...")
                    response = self._grpc_infer(batch_input, model_name)
                    logger.debug("gRPC inference received response for a batch")
                elif self.protocol == "http":
                    logger.debug("Performing HTTP inference for a batch...")
                    response = self._http_infer(batch_input)
                    logger.debug("HTTP inference received response for a batch")
                else:
                    raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

                # Parse the output of this batch
                parsed_output = self.model_interface.parse_output(
                    response, protocol=self.protocol, data=prepared_data, **kwargs
                )
                # Accumulate parsed outputs
                all_parsed_outputs.append(parsed_output)

            # 5. Process the parsed outputs for each batch
            all_results = []
            for parsed_output in all_parsed_outputs:
                batch_results = self.model_interface.process_inference_results(
                    parsed_output,
                    original_image_shapes=data.get("original_image_shapes"),
                    protocol=self.protocol,
                    **kwargs,
                )
                # Extend or append based on how `batch_results` is structured
                # (assuming it's a list of result items):
                if isinstance(batch_results, list):
                    all_results.extend(batch_results)
                else:
                    all_results.append(batch_results)

        except Exception as err:
            error_str = f"Error during NimClient inference [{self.model_interface.name()}, {self.protocol}]: {err}"
            logger.error(error_str)
            raise RuntimeError(error_str)

        # 6. Return final accumulated results
        return all_results

    def _grpc_infer(self, formatted_input: np.ndarray, model_name: str) -> np.ndarray:
        """
        Perform inference using the gRPC protocol.

        Parameters
        ----------
        formatted_input : np.ndarray
            The input data formatted as a numpy array.
        model_name : str
            The name of the model to use for inference.

        Returns
        -------
        np.ndarray
            The output of the model as a numpy array.
        """

        input_tensors = [grpcclient.InferInput("input", formatted_input.shape, datatype="FP32")]
        input_tensors[0].set_data_from_numpy(formatted_input)

        outputs = [grpcclient.InferRequestedOutput("output")]
        response = self.client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)
        logger.debug(f"gRPC inference response: {response}")

        return response.as_numpy("output")

    def _http_infer(self, formatted_input: dict) -> dict:
        """
        Perform inference using the HTTP protocol, retrying for timeouts or 5xx errors up to 5 times.

        Parameters
        ----------
        formatted_input : dict
            The input data formatted as a dictionary.

        Returns
        -------
        dict
            The output of the model as a dictionary.

        Raises
        ------
        TimeoutError
            If the HTTP request times out repeatedly, up to the max retries.
        requests.RequestException
            For other HTTP-related errors that persist after max retries.
        """

        base_delay = 2.0
        attempt = 0

        while attempt < self.max_retries:
            try:
                response = requests.post(
                    self.endpoint_url, json=formatted_input, headers=self.headers, timeout=self.timeout
                )
                status_code = response.status_code

                # Check for server-side or rate-limit type errors
                # e.g. 5xx => server error, 429 => too many requests
                if status_code == 429 or status_code == 503 or (500 <= status_code < 600):
                    logger.warning(
                        f"Received HTTP {status_code} ({response.reason}) from "
                        f"{self.model_interface.name()}. Attempt {attempt + 1} of {self.max_retries}."
                    )
                    if attempt == self.max_retries - 1:
                        # No more retries left
                        logger.error(f"Max retries exceeded after receiving HTTP {status_code}.")
                        response.raise_for_status()  # raise the appropriate HTTPError
                    else:
                        # Exponential backoff
                        backoff_time = base_delay * (2**attempt)
                        time.sleep(backoff_time)
                        attempt += 1
                        continue
                else:
                    # Not in our "retry" category => just raise_for_status or return
                    response.raise_for_status()
                    logger.debug(f"HTTP inference response: {response.json()}")
                    return response.json()

            except requests.Timeout:
                # Treat timeouts similarly to 5xx => attempt a retry
                logger.warning(
                    f"HTTP request timed out after {self.timeout} seconds during {self.model_interface.name()} "
                    f"inference. Attempt {attempt + 1} of {self.max_retries}."
                )
                if attempt == self.max_retries - 1:
                    logger.error("Max retries exceeded after repeated timeouts.")
                    raise TimeoutError(
                        f"Repeated timeouts for {self.model_interface.name()} after {attempt + 1} attempts."
                    )
                # Exponential backoff
                backoff_time = base_delay * (2**attempt)
                time.sleep(backoff_time)
                attempt += 1

            except requests.HTTPError as http_err:
                # If we ended up here, it's a non-retryable 4xx or final 5xx after final attempt
                logger.error(f"HTTP request failed with status code {response.status_code}: {http_err}")
                raise

            except requests.RequestException as e:
                # ConnectionError or other non-HTTPError
                logger.error(f"HTTP request encountered a network issue: {e}")
                if attempt == self.max_retries - 1:
                    raise
                # Else retry on next loop iteration
                backoff_time = base_delay * (2**attempt)
                time.sleep(backoff_time)
                attempt += 1

        # If we exit the loop without returning, we've exhausted all attempts
        logger.error(f"Failed to get a successful response after {self.max_retries} retries.")
        raise Exception(f"Failed to get a successful response after {self.max_retries} retries.")

    def close(self):
        if self.protocol == "grpc" and hasattr(self.client, "close"):
            self.client.close()


def create_inference_client(
    endpoints: Tuple[str, str],
    model_interface: ModelInterface,
    auth_token: Optional[str] = None,
    infer_protocol: Optional[str] = None,
) -> NimClient:
    """
    Create a NimClient for interfacing with a model inference server.

    Parameters
    ----------
    endpoints : tuple
        A tuple containing the gRPC and HTTP endpoints.
    model_interface : ModelInterface
        The model interface implementation to use.
    auth_token : str, optional
        Authorization token for HTTP requests (default: None).
    infer_protocol : str, optional
        The protocol to use ("grpc" or "http"). If not specified, it is inferred from the endpoints.

    Returns
    -------
    NimClient
        The initialized NimClient.

    Raises
    ------
    ValueError
        If an invalid infer_protocol is specified.
    """

    grpc_endpoint, http_endpoint = endpoints

    if (infer_protocol is None) and (grpc_endpoint and grpc_endpoint.strip()):
        infer_protocol = "grpc"
    elif infer_protocol is None and http_endpoint:
        infer_protocol = "http"

    if infer_protocol not in ["grpc", "http"]:
        raise ValueError("Invalid infer_protocol specified. Must be 'grpc' or 'http'.")

    return NimClient(model_interface, infer_protocol, endpoints, auth_token)


def preprocess_image_for_paddle(array: np.ndarray, paddle_version: Optional[str] = None) -> np.ndarray:
    """
    Preprocesses an input image to be suitable for use with PaddleOCR by resizing, normalizing, padding,
    and transposing it into the required format.

    This function is intended for preprocessing images to be passed as input to PaddleOCR using GRPC.
    It is not necessary when using the HTTP endpoint.

    Steps:
    -----
    1. Resizes the image while maintaining aspect ratio such that its largest dimension is scaled to 960 pixels.
    2. Normalizes the image using the `normalize_image` function.
    3. Pads the image to ensure both its height and width are multiples of 32, as required by PaddleOCR.
    4. Transposes the image from (height, width, channel) to (channel, height, width), the format expected by PaddleOCR.

    Parameters:
    ----------
    array : np.ndarray
        The input image array of shape (height, width, channels). It should have pixel values in the range [0, 255].

    Returns:
    -------
    np.ndarray
        A preprocessed image with the shape (channels, height, width) and normalized pixel values.
        The image will be padded to have dimensions that are multiples of 32, with the padding color set to 0.

    Notes:
    -----
    - The image is resized so that its largest dimension becomes 960 pixels, maintaining the aspect ratio.
    - After normalization, the image is padded to the nearest multiple of 32 in both dimensions, which is
      a requirement for PaddleOCR.
    - The normalized pixel values are scaled between 0 and 1 before padding and transposing the image.
    """
    if (not paddle_version) or (pkgversion.parse(paddle_version) < pkgversion.parse("0.2.0-rc1")):
        return array

    height, width = array.shape[:2]
    scale_factor = 960 / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized = cv2.resize(array, (new_width, new_height))

    normalized = normalize_image(resized)

    # PaddleOCR NIM (GRPC) requires input shapes to be multiples of 32.
    new_height = (normalized.shape[0] + 31) // 32 * 32
    new_width = (normalized.shape[1] + 31) // 32 * 32
    padded, _ = pad_image(
        normalized, target_height=new_height, target_width=new_width, background_color=0, dtype=np.float32
    )

    # PaddleOCR NIM (GRPC) requires input to be (channel, height, width).
    transposed = padded.transpose((2, 0, 1))

    return transposed


def remove_url_endpoints(url) -> str:
    """Some configurations provide the full endpoint in the URL.
    Ex: http://deplot:8000/v1/chat/completions. For hitting the
    health endpoint we need to get just the hostname:port combo
    that we can append the health/ready endpoint to so we attempt
    to parse that information here.

    Args:
        url str: Incoming URL

    Returns:
        str: URL with just the hostname:port portion remaining
    """
    if "/v1" in url:
        url = url.split("/v1")[0]

    return url


def generate_url(url) -> str:
    """Examines the user defined URL for http*://. If that
    pattern is detected the URL is used as provided by the user.
    If that pattern does not exist then the assumption is made that
    the endpoint is simply `http://` and that is prepended
    to the user supplied endpoint.

    Args:
        url str: Endpoint where the Rest service is running

    Returns:
        str: Fully validated URL
    """
    if not re.match(r"^https?://", url):
        # Add the default `http://` if its not already present in the URL
        url = f"http://{url}"

    return url


def is_ready(http_endpoint: str, ready_endpoint: str) -> bool:
    """
    Check if the server at the given endpoint is ready.

    Parameters
    ----------
    http_endpoint : str
        The HTTP endpoint of the server.
    ready_endpoint : str
        The specific ready-check endpoint.

    Returns
    -------
    bool
        True if the server is ready, False otherwise.
    """

    # IF the url is empty or None that means the service was not configured
    # and is therefore automatically marked as "ready"
    if http_endpoint is None or http_endpoint == "":
        return True

    # If the url is for build.nvidia.com, it is automatically assumed "ready"
    if "ai.api.nvidia.com" in http_endpoint:
        return True

    url = generate_url(http_endpoint)
    url = remove_url_endpoints(url)

    if not ready_endpoint.startswith("/") and not url.endswith("/"):
        ready_endpoint = "/" + ready_endpoint

    url = url + ready_endpoint

    # Call the ready endpoint of the NIM
    try:
        # Use a short timeout to prevent long hanging calls. 5 seconds seems resonable
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            # The NIM is saying it is ready to serve
            return True
        elif resp.status_code == 503:
            # NIM is explicitly saying it is not ready.
            return False
        else:
            # Any other code is confusing. We should log it with a warning
            # as it could be something that might hold up ready state
            logger.warning(f"'{url}' HTTP Status: {resp.status_code} - Response Payload: {resp.json()}")
            return False
    except requests.HTTPError as http_err:
        logger.warning(f"'{url}' produced a HTTP error: {http_err}")
        return False
    except requests.Timeout:
        logger.warning(f"'{url}' request timed out")
        return False
    except ConnectionError:
        logger.warning(f"A connection error for '{url}' occurred")
        return False
    except requests.RequestException as err:
        logger.warning(f"An error occurred: {err} for '{url}'")
        return False
    except Exception as ex:
        # Don't let anything squeeze by
        logger.warning(f"Exception: {ex}")
        return False


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_version(http_endpoint: str, metadata_endpoint: str = "/v1/metadata", version_field: str = "version") -> str:
    """
    Get the version of the server from its metadata endpoint.

    Parameters
    ----------
    http_endpoint : str
        The HTTP endpoint of the server.
    metadata_endpoint : str, optional
        The metadata endpoint to query (default: "/v1/metadata").
    version_field : str, optional
        The field containing the version in the response (default: "version").

    Returns
    -------
    str
        The version of the server, or an empty string if unavailable.
    """

    if (http_endpoint is None) or (http_endpoint == ""):
        return ""

    # TODO: Need a way to match NIM versions to API versions.
    if "ai.api.nvidia.com" in http_endpoint or "api.nvcf.nvidia.com" in http_endpoint:
        return "1.0.0"

    url = generate_url(http_endpoint)
    url = remove_url_endpoints(url)

    if not metadata_endpoint.startswith("/") and not url.endswith("/"):
        metadata_endpoint = "/" + metadata_endpoint

    url = url + metadata_endpoint

    # Call the metadata endpoint of the NIM
    try:
        # Use a short timeout to prevent long hanging calls. 5 seconds seems reasonable
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            version = resp.json().get(version_field, "")
            if version:
                return version
            else:
                # If version field is empty, retry
                logger.warning(f"No version field in response from '{url}'. Retrying.")
                return ""
        else:
            # Any other code is confusing. We should log it with a warning
            logger.warning(f"'{url}' HTTP Status: {resp.status_code} - Response Payload: {resp.text}")
            return ""
    except requests.HTTPError as http_err:
        logger.warning(f"'{url}' produced a HTTP error: {http_err}")
        return ""
    except requests.Timeout:
        logger.warning(f"'{url}' request timed out")
        return ""
    except ConnectionError:
        logger.warning(f"A connection error for '{url}' occurred")
        return ""
    except requests.RequestException as err:
        logger.warning(f"An error occurred: {err} for '{url}'")
        return ""
    except Exception as ex:
        # Don't let anything squeeze by
        logger.warning(f"Exception: {ex}")
        return ""
