# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional
from typing import Tuple

import numpy as np
import re
import requests
import tritonclient.grpc as grpcclient

from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)


def create_inference_client(endpoints: Tuple[str, str], auth_token: Optional[str]):
    """
    Creates an inference client based on the provided endpoints.

    If the gRPC endpoint is provided, a gRPC client is created. Otherwise, an HTTP client is created.

    Parameters
    ----------
    endpoints : Tuple[str, str]
        A tuple containing the gRPC and HTTP endpoints. The first element is the gRPC endpoint, and the second element
        is the HTTP endpoint.
    auth_token : Optional[str]
        The authentication token to be used for the HTTP client, if provided.

    Returns
    -------
    grpcclient.InferenceServerClient or dict
        A gRPC client if the gRPC endpoint is provided, otherwise a dictionary containing the HTTP client details.
    """
    if endpoints[0] and endpoints[0].strip():
        logger.debug(f"Creating gRPC client with {endpoints}")
        return grpcclient.InferenceServerClient(url=endpoints[0])
    else:
        logger.debug(f"Creating HTTP client with {endpoints}")
        headers = {"accept": "application/json", "content-type": "application/json"}

        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        return {"endpoint_url": endpoints[1], "headers": headers}


@traceable_func(trace_name="pdf_content_extractor::{model_name}")
def call_image_inference_model(client, model_name: str, image_data):
    """
    Calls an image inference model using the provided client.

    If the client is a gRPC client, the inference is performed using gRPC. Otherwise, it is performed using HTTP.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient or dict
        The inference client, which can be either a gRPC client or an HTTP client.
    model_name : str
        The name of the model to be used for inference.
    image_data : np.ndarray
        The image data to be used for inference. Should be a NumPy array.

    Returns
    -------
    str or None
        The result of the inference as a string if successful, otherwise `None`.

    Raises
    ------
    RuntimeError
        If the HTTP request fails or if the response format is not as expected.
    """
    if isinstance(client, grpcclient.InferenceServerClient):
        if image_data.ndim == 3:
            image_data = np.expand_dims(image_data, axis=0)
        inputs = [grpcclient.InferInput("input", image_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_data.astype(np.float32))

        outputs = [grpcclient.InferRequestedOutput("output")]

        try:
            result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            return " ".join([output[0].decode("utf-8") for output in result.as_numpy("output")])
        except Exception as e:
            err_msg = f"Inference failed for model {model_name}: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    else:
        base64_img = numpy_to_base64(image_data)

        try:
            url = client["endpoint_url"]
            headers = client["headers"]

            messages = [
                {
                    "role": "user",
                    "content": f"Generate the underlying data table of the figure below: "
                    f'<img src="data:image/png;base64,{base64_img}" />',
                }
            ]
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 128,
                "stream": False,
                "temperature": 1.0,
                "top_p": 1.0,
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the JSON response
            json_response = response.json()

            # Validate the response structure
            if "choices" not in json_response or not json_response["choices"]:
                raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")

            return json_response["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {e}")
        except KeyError as e:
            raise RuntimeError(f"Missing expected key in response: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during inference: {e}")


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
    if '/v1' in url:
        url = url.split('/v1')[0]

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
    if not re.match(r'^https?://', url):
        # Add the default `http://` if its not already present in the URL
        url = f"http://{url}"

    url = remove_url_endpoints(url)

    return url


def is_ready(http_endpoint, ready_endpoint) -> bool:

    # IF the url is empty or None that means the service was not configured
    # and is therefore automatically marked as "ready"
    if http_endpoint is None or http_endpoint == '':
        return True

    url = generate_url(http_endpoint)

    if not ready_endpoint.startswith('/') and not url.endswith('/'):
        ready_endpoint = '/' + ready_endpoint

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
