# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Optional
from typing import Tuple

import backoff
import cv2
import numpy as np
import packaging
import requests
import tritonclient.grpc as grpcclient

from nv_ingest.util.image_processing.transforms import normalize_image
from nv_ingest.util.image_processing.transforms import pad_image
from nv_ingest.util.nim.decorators import multiprocessing_cache
from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)

DEPLOT_MAX_TOKENS = 128
DEPLOT_TEMPERATURE = 1.0
DEPLOT_TOP_P = 1.0


class ModelInterface:
    def format_input(self, data, protocol: str):
        raise NotImplementedError("Subclasses should implement this method")

    def parse_output(self, response, protocol: str):
        raise NotImplementedError("Subclasses should implement this method")

    def prepare_data_for_inference(self, data):
        raise NotImplementedError("Subclasses should implement this method")

    def process_inference_results(self, output_array, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")


class NimClient:
    def __init__(self, model_interface: ModelInterface, protocol: str, endpoints: Tuple[str, str],
                 auth_token: Optional[str] = None):
        self.model_interface = model_interface
        self.protocol = protocol.lower()
        self.auth_token = auth_token

        grpc_endpoint, http_endpoint = endpoints

        if self.protocol == 'grpc':
            if not grpc_endpoint:
                raise ValueError("gRPC endpoint must be provided for gRPC protocol")
            logger.debug(f"Creating gRPC client with {grpc_endpoint}")
            self.client = grpcclient.InferenceServerClient(url=grpc_endpoint)
        elif self.protocol == 'http':
            if not http_endpoint:
                raise ValueError("HTTP endpoint must be provided for HTTP protocol")
            logger.debug(f"Creating HTTP client with {http_endpoint}")
            self.endpoint_url = generate_url(http_endpoint)
            self.headers = {"accept": "application/json", "content-type": "application/json"}
            if self.auth_token:
                self.headers["Authorization"] = f"Bearer {self.auth_token}"
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def infer(self, data, model_name: str, **kwargs):
        # Prepare data for inference
        prepared_data = self.model_interface.prepare_data_for_inference(data)

        # Format input based on protocol
        formatted_input = self.model_interface.format_input(prepared_data, protocol=self.protocol)

        # Perform inference
        if self.protocol == 'grpc':
            logger.debug("Performing gRPC inference...")
            response = self._grpc_infer(formatted_input, model_name)
        elif self.protocol == 'http':
            logger.debug("Performing HTTP inference...")
            response = self._http_infer(formatted_input)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

        # Parse and process output
        parsed_output = self.model_interface.parse_output(response, protocol=self.protocol)
        results = self.model_interface.process_inference_results(parsed_output,
                                                                 original_image_shapes=data.get(
                                                                     'original_image_shapes'),
                                                                 **kwargs)
        return results

    def _grpc_infer(self, formatted_input, model_name: str):
        input_tensors = [grpcclient.InferInput("input", formatted_input.shape, datatype="FP32")]
        input_tensors[0].set_data_from_numpy(formatted_input)

        outputs = [grpcclient.InferRequestedOutput("output")]
        response = self.client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)
        logger.debug(f"gRPC inference response: {response}")

        return response.as_numpy("output")

    def _http_infer(self, formatted_input):
        response = requests.post(self.endpoint_url, json=formatted_input, headers=self.headers)
        response.raise_for_status()
        logger.debug(f"HTTP inference response: {response.json()}")
        return response.json()

    def close(self):
        if self.protocol == 'grpc' and hasattr(self.client, 'close'):
            self.client.close()


def create_inference_client(
        endpoints: Tuple[str, str],
        model_interface: ModelInterface,
        auth_token: Optional[str] = None,
        infer_protocol: Optional[str] = None,
):
    grpc_endpoint, http_endpoint = endpoints

    if (infer_protocol is None) and (grpc_endpoint and grpc_endpoint.strip()):
        infer_protocol = "grpc"
    elif infer_protocol is None and http_endpoint:
        infer_protocol = "http"

    if infer_protocol not in ['grpc', 'http']:
        raise ValueError("Invalid infer_protocol specified. Must be 'grpc' or 'http'.")

    return NimClient(model_interface, infer_protocol, endpoints, auth_token)


# Perform inference and return predictions
@traceable_func(trace_name="pdf_content_extractor::{model_name}")
def perform_model_inference(client, model_name: str, input_array: np.ndarray):
    """
    Perform inference using the provided model and input data.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        The gRPC client to use for inference.
    model_name : str
        The name of the model to use for inference.
    input_array : np.ndarray
        The input data to feed into the model, formatted as a numpy array.

    Returns
    -------
    np.ndarray
        The output of the model as a numpy array.

    Examples
    --------
    >>> client = create_inference_client("http://localhost:8000")
    >>> input_array = np.random.rand(2, 3, 1024, 1024).astype(np.float32)
    >>> output = perform_model_inference(client, "my_model", input_array)
    >>> output.shape
    (2, 1000)
    """
    input_tensors = [grpcclient.InferInput("input", input_array.shape, datatype="FP32")]
    input_tensors[0].set_data_from_numpy(input_array)

    outputs = [grpcclient.InferRequestedOutput("output")]
    query_response = client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)
    logger.debug(query_response)

    return query_response.as_numpy("output")


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
    if (not paddle_version) or (packaging.version.parse(paddle_version) < packaging.version.parse("0.2.0-rc1")):
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


def is_ready(http_endpoint, ready_endpoint) -> bool:
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


@backoff.on_predicate(backoff.expo, max_time=30)
@multiprocessing_cache(max_calls=100)
def get_version(http_endpoint, metadata_endpoint="/v1/metadata", version_field="version") -> str:
    if http_endpoint is None or http_endpoint == "":
        return ""

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
