# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Optional, List

import numpy as np
import logging

from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy

logger = logging.getLogger(__name__)


class DeplotModelInterface(ModelInterface):
    """
    An interface for handling inference with a Deplot model, supporting both gRPC and HTTP protocols,
    now updated to handle multiple base64 images ('base64_images').
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface ("Deplot").
        """
        return "Deplot"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data by decoding one or more base64-encoded images into NumPy arrays.

        Parameters
        ----------
        data : dict
            The input data containing either 'base64_image' (single image)
            or 'base64_images' (multiple images).

        Returns
        -------
        dict
            The updated data dictionary with 'image_arrays': a list of decoded NumPy arrays.
        """

        # Handle a single base64_image or multiple base64_images
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")
            image_arrays = [base64_to_numpy(b64) for b64 in base64_list]

        elif "base64_image" in data:
            # Fallback for single image
            image_arrays = [base64_to_numpy(data["base64_image"])]
        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        data["image_arrays"] = image_arrays

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol (gRPC or HTTP) for Deplot.
        For HTTP, we now construct multiple messages—one per image batch—along with
        corresponding batch data carrying the original image arrays and their dimensions.

        Parameters
        ----------
        data : dict of str -> Any
            The input data dictionary, expected to contain "image_arrays" (a list of np.ndarray).
        protocol : str
            The protocol to use, "grpc" or "http".
        max_batch_size : int
            The maximum number of images per batch.
        kwargs : dict
            Additional parameters to pass to the payload preparation (for HTTP).

        Returns
        -------
        tuple
            (formatted_batches, formatted_batch_data) where:
              - For gRPC: formatted_batches is a list of NumPy arrays, each of shape (B, H, W, C)
                with B <= max_batch_size.
              - For HTTP: formatted_batches is a list of JSON-serializable payload dicts.
              - In both cases, formatted_batch_data is a list of dicts containing:
                    "image_arrays": the list of original np.ndarray images for that batch, and
                    "image_dims":  a list of (height, width) tuples for each image in the batch.

        Raises
        ------
        KeyError
            If "image_arrays" is missing in the data dictionary.
        ValueError
            If the protocol is invalid, or if no valid images are found.
        """
        if "image_arrays" not in data:
            raise KeyError("Expected 'image_arrays' in data. Call prepare_data_for_inference first.")

        image_arrays = data["image_arrays"]
        # Compute image dimensions from each image array.
        image_dims = [(img.shape[0], img.shape[1]) for img in image_arrays]

        # Helper function: chunk a list into sublists of length <= chunk_size.
        def chunk_list(lst: list, chunk_size: int) -> List[list]:
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC Deplot model (potentially batched).")
            processed = []
            for arr in image_arrays:
                # Ensure each image has shape (1, H, W, C)
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)
                arr = arr.astype(np.float32)
                arr /= 255.0  # Normalize to [0,1]
                processed.append(arr)

            if not processed:
                raise ValueError("No valid images found for gRPC formatting.")

            formatted_batches = []
            formatted_batch_data = []
            proc_chunks = chunk_list(processed, max_batch_size)
            orig_chunks = chunk_list(image_arrays, max_batch_size)
            dims_chunks = chunk_list(image_dims, max_batch_size)

            for proc_chunk, orig_chunk, dims_chunk in zip(proc_chunks, orig_chunks, dims_chunks):
                # Concatenate along the batch dimension to form a single input.
                batched_input = np.concatenate(proc_chunk, axis=0)
                formatted_batches.append(batched_input)
                formatted_batch_data.append({"image_arrays": orig_chunk, "image_dims": dims_chunk})
            return formatted_batches, formatted_batch_data

        elif protocol == "http":
            logger.debug("Formatting input for HTTP Deplot model (multiple messages).")
            if "base64_images" in data:
                base64_list = data["base64_images"]
            else:
                base64_list = [data["base64_image"]]

            formatted_batches = []
            formatted_batch_data = []
            b64_chunks = chunk_list(base64_list, max_batch_size)
            orig_chunks = chunk_list(image_arrays, max_batch_size)
            dims_chunks = chunk_list(image_dims, max_batch_size)

            for b64_chunk, orig_chunk, dims_chunk in zip(b64_chunks, orig_chunks, dims_chunks):
                payload = self._prepare_deplot_payload(
                    base64_list=b64_chunk,
                    max_tokens=kwargs.get("max_tokens", 500),
                    temperature=kwargs.get("temperature", 0.5),
                    top_p=kwargs.get("top_p", 0.9),
                )
                formatted_batches.append(payload)
                formatted_batch_data.append({"image_arrays": orig_chunk, "image_dims": dims_chunk})
            return formatted_batches, formatted_batch_data

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the model's inference response.
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC Deplot model (batched).")
            # Each batch element might be returned as a list of bytes. Combine or keep separate as needed.
            results = []
            for item in response:
                # If item is [b'...'], decode and join
                if isinstance(item, list):
                    joined_str = " ".join(o.decode("utf-8") for o in item)
                    results.append(joined_str)
                else:
                    # single bytes or str
                    val = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    results.append(val)
            return results  # Return a list of strings, one per image.

        elif protocol == "http":
            logger.debug("Parsing output from HTTP Deplot model.")
            return self._extract_content_from_deplot_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the Deplot model.

        Parameters
        ----------
        output : Any
            The raw output from the model.
        protocol : str
            The protocol used for inference (gRPC or HTTP).

        Returns
        -------
        Any
            The processed inference results.
        """

        # For Deplot, the output is the chart content as a string
        return output

    @staticmethod
    def _prepare_deplot_payload(
        base64_list: list,
        max_tokens: int = 500,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Prepare an HTTP payload for Deplot that includes one message per image,
        matching the original single-image style:

            messages = [
              {
                "role": "user",
                "content": "Generate ... <img src=\"data:image/png;base64,...\" />"
              },
              {
                "role": "user",
                "content": "Generate ... <img src=\"data:image/png;base64,...\" />"
              },
              ...
            ]

        If your backend expects multiple messages in a single request, this keeps
        the same structure as the single-image code repeated N times.
        """
        messages = []
        # Note: deplot NIM currently only supports a single message per request
        for b64_img in base64_list:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Generate the underlying data table of the figure below: "
                        f'<img src="data:image/png;base64,{b64_img}" />'
                    ),
                }
            )

        payload = {
            "model": "google/deplot",
            "messages": messages,  # multiple user messages now
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
        }
        return payload

    @staticmethod
    def _extract_content_from_deplot_response(json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a Deplot HTTP API request.
        The original code expected a single choice with a single textual content.
        """
        if "choices" not in json_response or not json_response["choices"]:
            raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")

        # If the service only returns one textual result, we return that one.
        return json_response["choices"][0]["message"]["content"]
