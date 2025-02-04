# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import io
import logging
import PIL.Image as Image
from typing import Any, Dict, Optional, List

import numpy as np

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class CachedModelInterface(ModelInterface):
    """
    An interface for handling inference with a Cached model, supporting both gRPC and HTTP
    protocols, including batched input.
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
        Decode base64-encoded images into NumPy arrays, storing them in `data["image_arrays"]`.

        Parameters
        ----------
        data : dict of str -> Any
            The input data containing either:
             - "base64_image": a single base64-encoded image, or
             - "base64_images": a list of base64-encoded images.

        Returns
        -------
        dict of str -> Any
            The updated data dictionary with decoded image arrays stored in
            "image_arrays", where each array has shape (H, W, C).

        Raises
        ------
        KeyError
            If neither 'base64_image' nor 'base64_images' is provided.
        ValueError
            If 'base64_images' is provided but is not a list.
        """
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")
            data["image_arrays"] = [base64_to_numpy(img) for img in base64_list]

        elif "base64_image" in data:
            # Fallback to single image case; wrap it in a list to keep the interface consistent
            data["image_arrays"] = [base64_to_numpy(data["base64_image"])]

        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images' with base64-encoded images.")

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol ("grpc" or "http"), handling batched images.

        Parameters
        ----------
        data : dict of str -> Any
            The input data dictionary, expected to contain "image_arrays" (a list of np.ndarray).
        protocol : str
            The protocol to use, "grpc" or "http".
        max_batch_size : int
            The maximum number of images per batch.

        Returns
        -------
        Any
            A list of formatted input batches. For gRPC, each batch is a NumPy array of shape (B, H, W, C)
            where B <= max_batch_size. For HTTP, each batch is a JSON-serializable dict containing base64‑encoded
            images.

        Raises
        ------
        KeyError
            If "image_arrays" is missing in the data dictionary.
        ValueError
            If the protocol is invalid, or if images have differing shapes for gRPC.
        """
        if "image_arrays" not in data:
            raise KeyError("Expected 'image_arrays' in data. Make sure prepare_data_for_inference was called.")

        image_arrays = data["image_arrays"]

        # Helper to chunk a list into sublists of size up to chunk_size.
        def chunk_list(lst: list, chunk_size: int) -> List[list]:
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC Cached model (batched).")

            batched_images: List[np.ndarray] = []
            for arr in image_arrays:
                # Expand from (H, W, C) to (1, H, W, C) if needed
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)
                # Convert to float32
                batched_images.append(arr.astype(np.float32))

            if not batched_images:
                raise ValueError("No valid images found for gRPC formatting.")

            # Ensure all images have the same shape (excluding batch dimension)
            shapes = [img.shape[1:] for img in batched_images]  # each is (H, W, C)
            if any(s != shapes[0] for s in shapes[1:]):
                raise ValueError(f"All images must have the same dimensions for gRPC batching. Found: {shapes}")

            # Chunk the images into groups of size up to max_batch_size
            batched_image_chunks = chunk_list(batched_images, max_batch_size)
            batched_inputs = []
            for chunk in batched_image_chunks:
                # Concatenate along the batch dimension => shape (B, H, W, C) where B <= max_batch_size
                batched_input = np.concatenate(chunk, axis=0)
                batched_inputs.append(batched_input)
            return batched_inputs

        elif protocol == "http":
            logger.debug("Formatting input for HTTP Cached model (batched).")

            content_list: List[Dict[str, Any]] = []
            for arr in image_arrays:
                # Convert to uint8 if needed, then to PIL Image, then to base64 string
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                image_pil = Image.fromarray(arr)
                buffered = io.BytesIO()
                image_pil.save(buffered, format="PNG")
                base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Build item for Nim-like structure
                image_item = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                content_list.append(image_item)

            # Chunk the list of images into groups of size up to max_batch_size
            content_chunks = chunk_list(content_list, max_batch_size)
            payload_batches = []
            for chunk in content_chunks:
                message = {"content": chunk}
                payload = {"messages": [message]}
                payload_batches.append(payload)
            return payload_batches

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Parse the output from the Cached model's inference response.

        Parameters
        ----------
        response : Any
            The raw response from the model inference.
        protocol : str
            The protocol used ("grpc" or "http").
        data : dict of str -> Any, optional
            Additional input data (unused here, but available for consistency).
        **kwargs : Any
            Additional keyword arguments for future compatibility.

        Returns
        -------
        Any
            The parsed output data (e.g., list of strings), depending on the protocol.

        Raises
        ------
        ValueError
            If the protocol is invalid.
        RuntimeError
            If the HTTP response is not as expected (missing 'data' key).
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC Cached model (batched).")
            parsed: List[str] = []
            # Assume `response` is iterable, each element a list/array of byte strings
            for single_output in response:
                joined_str = " ".join(o.decode("utf-8") for o in single_output)
                parsed.append(joined_str)
            return parsed

        elif protocol == "http":
            logger.debug("Parsing output from HTTP Cached model (batched).")
            if not isinstance(response, dict):
                raise RuntimeError("Expected JSON/dict response for HTTP, got something else.")
            if "data" not in response or not response["data"]:
                raise RuntimeError("Unexpected response format: 'data' key missing or empty.")

            contents: List[str] = []
            for item in response["data"]:
                # Each "item" might have a "content" key
                content = item.get("content", "")
                contents.append(content)

            return contents

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, protocol: str, **kwargs: Any) -> Any:
        """
        Process inference results for the Cached model.

        Parameters
        ----------
        output : Any
            The raw output from the model.
        protocol : str
            The inference protocol used ("grpc" or "http").
        **kwargs : Any
            Additional parameters for post-processing (not used here).

        Returns
        -------
        Any
            The processed inference results, which here is simply returned as-is.
        """
        # For Cached model, we simply return what we parsed (e.g., a list of strings or a single string)
        return output

    def _extract_content_from_nim_response(self, json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a NIM (HTTP) API request.

        Parameters
        ----------
        json_response : dict of str -> Any
            The JSON response from the NIM API.

        Returns
        -------
        Any
            The extracted content from the response.

        Raises
        ------
        RuntimeError
            If the response format is unexpected (missing 'data' or empty).
        """
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        return json_response["data"][0]["content"]
