# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import logging
from typing import Optional
from typing import Tuple

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from PIL import Image

from nv_ingest.util.converters import bytetools

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
            logger.error(f"Inference failed for model {model_name}: {str(e)}")
            return None
    else:
        image = Image.fromarray(image_data)
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            base64_img = bytetools.base64frombytes(buffer.getvalue())

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
