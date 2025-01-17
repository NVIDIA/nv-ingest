# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, Optional

import numpy as np

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class CachedModelInterface(ModelInterface):
    """
    An interface for handling inference with a Cached model, supporting both gRPC and HTTP protocols.
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface ("Cached").
        """
        return "Cached"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for inference by decoding the base64 image into a numpy array.

        Parameters
        ----------
        data : dict
            The input data containing a base64-encoded image.

        Returns
        -------
        dict
            The updated data dictionary with the decoded image array.
        """
        # Expecting base64_image in data
        base64_image = data["base64_image"]
        data["image_array"] = base64_to_numpy(base64_image)
        return data

    def format_input(self, data: Dict[str, Any], protocol: str) -> Any:
        """
        Format input data for the specified protocol.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to use ("grpc" or "http").

        Returns
        -------
        Any
            The formatted input data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        if protocol == "grpc":
            logger.debug("Formatting input for gRPC Cached model")
            # Convert image array to expected format
            image_data = data["image_array"]
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)
            image_data = image_data.astype(np.float32)
            return image_data
        elif protocol == "http":
            logger.debug("Formatting input for HTTP Cached model")
            # Prepare payload for HTTP request
            base64_img = data["base64_image"]
            payload = self._prepare_nim_payload(base64_img)
            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the output from the model's inference response.

        Parameters
        ----------
        response : Any
            The response from the model inference.
        protocol : str
            The protocol used ("grpc" or "http").
        data : dict, optional
            Additional input data passed to the function.

        Returns
        -------
        Any
            The parsed output data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC Cached model")
            # Convert bytes output to string
            return " ".join([output[0].decode("utf-8") for output in response])
        elif protocol == "http":
            logger.debug("Parsing output from HTTP Cached model")
            return self._extract_content_from_nim_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the Cached model.

        Parameters
        ----------
        output : Any
            The raw output from the model.

        Returns
        -------
        Any
            The processed inference results.
        """
        # For Cached model, the output is the chart content as a string
        return output

    def _prepare_nim_payload(self, base64_img: str) -> Dict[str, Any]:
        """
        Prepare a payload for the NIM (HTTP) API using a base64-encoded image.

        Parameters
        ----------
        base64_img : str
            The base64-encoded image string.

        Returns
        -------
        dict
            The formatted payload for the NIM API.
        """
        image_url = f"data:image/png;base64,{base64_img}"
        image = {"type": "image_url", "image_url": {"url": image_url}}

        message = {"content": [image]}
        payload = {"messages": [message]}

        return payload

    def _extract_content_from_nim_response(self, json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a NIM (HTTP) API request.

        Parameters
        ----------
        json_response : dict
            The JSON response from the NIM API.

        Returns
        -------
        Any
            The extracted content from the response.

        Raises
        ------
        RuntimeError
            If the response does not contain the expected "data" key or if it is empty.
        """
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        return json_response["data"][0]["content"]
