# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import io
import logging
import PIL.Image as Image
from typing import Any, Dict, Optional

import numpy as np

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class CachedModelInterface(ModelInterface):
    """
    An interface for handling inference with a Cached model, supporting both gRPC and HTTP protocols,
    including batched input.
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
        Prepare input data for inference by decoding base64 images into numpy arrays.

        Parameters
        ----------
        data : dict
            The input data containing either a single "base64_image" or multiple "base64_images".

        Returns
        -------
        dict
            The updated data dictionary with decoded image arrays stored in "image_arrays".
        """
        # Handle single vs. multiple images.
        # For batch processing, prefer "base64_images".
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")
            data["image_arrays"] = [base64_to_numpy(img) for img in base64_list]
        elif "base64_image" in data:
            # Fallback to single image case; wrap it in a list to keep everything consistent
            data["image_arrays"] = [base64_to_numpy(data["base64_image"])]
        else:
            raise KeyError(
                "Input data must include either 'base64_image' or 'base64_images' with base64-encoded images."
            )

        return data

    def format_input(self, data: Dict[str, Any], protocol: str) -> Any:
        """
        Format input data for the specified protocol (gRPC or HTTP).

        Parameters
        ----------
        data : dict
            The input data to format, containing "image_arrays" (list of np.ndarray).
        protocol : str
            The protocol to use ("grpc" or "http").

        Returns
        -------
        Any
            The formatted input data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified or if the images do not share the same shape.
        """
        if "image_arrays" not in data:
            raise KeyError("Expected 'image_arrays' in data. Make sure prepare_data_for_inference was called.")

        # The arrays we got from prepare_data_for_inference
        image_arrays = data["image_arrays"]

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC Cached model (batched).")

            batched_images = []
            for arr in image_arrays:
                # If shape is (H, W, C), expand to (1, H, W, C)
                # If already has a leading batch dimension, keep it
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)  # -> (1, H, W, C)

                batched_images.append(arr.astype(np.float32))

            if not batched_images:
                raise ValueError("No valid images found for gRPC formatting.")

            # Check that all images match in shape beyond the batch dimension
            # e.g. every array should be (1, H, W, C) with the same (H, W, C)
            shapes = [img.shape[1:] for img in batched_images]  # list of (H, W, C) shapes
            if any(s != shapes[0] for s in shapes[1:]):
                raise ValueError(f"All images must have the same dimensions for gRPC batching. Found: {shapes}")

            # Concatenate along the batch dimension => shape (B, H, W, C)
            batched_input = np.concatenate(batched_images, axis=0)
            return batched_input

        elif protocol == "http":
            logger.debug("Formatting input for HTTP Cached model (batched).")
            # Convert each image back to base64, building a single payload with multiple images
            # to mimic YOLOX or NIM's typical batch approach

            # If your Nim endpoint expects: {"messages":[{"content": [ ... ]}]}
            # we can build that structure.
            content_list = []
            # If data also included the original base64 strings, we could just reuse them,
            # but here let's do the full approach of re-encoding from "image_arrays".
            for arr in image_arrays:
                # Convert from np.uint8 or float -> PIL -> base64
                if arr.dtype != np.uint8:
                    # If your pipeline expects [0,1] floats, you may need to scale 255
                    arr = (arr * 255).astype(np.uint8)
                image_pil = Image.fromarray(arr)
                buffered = io.BytesIO()
                image_pil.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Build item for Nim
                image_item = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                content_list.append(image_item)

            # Nim payload example (similar to your single-image approach, but batched)
            # One message containing multiple images in the "content" array:
            message = {"content": content_list}
            payload = {"messages": [message]}

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
            Additional input data passed to the function (could contain 'image_arrays').

        Returns
        -------
        Any
            The parsed output data (likely a list of strings or a single string).

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC Cached model (batched).")
            parsed = []
            for single_output in response:
                # single_output might be [b'something']
                joined_str = " ".join(o.decode("utf-8") for o in single_output)
                parsed.append(joined_str)
            return parsed

        elif protocol == "http":
            logger.debug("Parsing output from HTTP Cached model (batched).")
            if not isinstance(response, dict):
                raise RuntimeError("Expected JSON/dict response for HTTP, got something else.")

            if "data" not in response or not response["data"]:
                raise RuntimeError("Unexpected response format: 'data' key missing or empty.")

            contents = []
            for item in response["data"]:
                # Each "item" might have a "content" key
                content = item.get("content", "")
                contents.append(content)

            return contents
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
