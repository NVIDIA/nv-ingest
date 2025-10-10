# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import backoff
import numpy as np
import tritonclient.grpc as grpcclient

from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.decorators import multiprocessing_cache
from nv_ingest_api.internal.primitives.nim.model_interface.helpers import preprocess_image_for_paddle
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy

DEFAULT_OCR_MODEL_NAME = "paddle"
NEMORETRIEVER_OCR_MODEL_NAME = "scene_text_wrapper"
NEMORETRIEVER_OCR_ENSEMBLE_MODEL_NAME = "scene_text_ensemble"
NEMORETRIEVER_OCR_BLS_MODEL_NAME = "scene_text_python"


logger = logging.getLogger(__name__)


class OCRModelInterfaceBase(ModelInterface):

    NUM_CHANNELS = 3
    BYTES_PER_ELEMENT = 4  # For float32

    def parse_output(
        self,
        response: Any,
        protocol: str,
        data: Optional[Dict[str, Any]] = None,
        model_name: str = DEFAULT_OCR_MODEL_NAME,
        **kwargs: Any,
    ) -> Any:
        """
        Parse the model's inference response for the given protocol. The parsing
        may handle batched outputs for multiple images.

        Parameters
        ----------
        response : Any
            The raw response from the OCR model.
        protocol : str
            The protocol used for inference, "grpc" or "http".
        data : dict of str -> Any, optional
            Additional data dictionary that may include "image_dims" for bounding box scaling.
        **kwargs : Any
            Additional keyword arguments, such as custom `table_content_format`.

        Returns
        -------
        Any
            The parsed output, typically a list of (content, table_content_format) tuples.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        # Retrieve image dimensions if available
        dims: Optional[List[Tuple[int, int]]] = data.get("image_dims") if data else None

        if protocol == "grpc":
            logger.debug("Parsing output from gRPC OCR model (batched).")
            return self._extract_content_from_ocr_grpc_response(response, dims, model_name=model_name)

        elif protocol == "http":
            logger.debug("Parsing output from HTTP OCR model (batched).")
            return self._extract_content_from_ocr_http_response(response, dims)

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs: Any) -> Any:
        """
        Process inference results for the OCR model.

        Parameters
        ----------
        output : Any
            The raw output parsed from the OCR model.
        **kwargs : Any
            Additional keyword arguments for customization.

        Returns
        -------
        Any
            The post-processed inference results. By default, this simply returns the output
            as the table content (or content list).
        """
        return output

    def does_item_fit_in_batch(self, current_batch, next_request, memory_budget_bytes: int) -> bool:
        """
        Estimates the memory of a potential batch of padded images and checks it
        against the configured budget.
        """
        all_requests = current_batch + [next_request]
        all_dims = [req.dims for req in all_requests]

        potential_max_h = max(d[0] for d in all_dims)
        potential_max_w = max(d[1] for d in all_dims)

        potential_batch_size = len(all_requests)

        potential_memory_bytes = (
            potential_batch_size * potential_max_h * potential_max_w * self.NUM_CHANNELS * self.BYTES_PER_ELEMENT
        )

        return potential_memory_bytes <= memory_budget_bytes

    def _prepare_ocr_payload(self, base64_img: str) -> Dict[str, Any]:
        """
        DEPRECATED by batch logic in format_input. Kept here if you need single-image direct calls.

        Parameters
        ----------
        base64_img : str
            A single base64-encoded image string.

        Returns
        -------
        dict of str -> Any
            The payload in either legacy or new format for OCR's HTTP endpoint.
        """
        image_url = f"data:image/png;base64,{base64_img}"

        image = {"type": "image_url", "url": image_url}
        payload = {"input": [image]}

        return payload

    def _extract_content_from_ocr_http_response(
        self,
        json_response: Dict[str, Any],
        dimensions: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        """
        Extract content from the JSON response of a OCR HTTP API request.

        Parameters
        ----------
        json_response : dict of str -> Any
            The JSON response returned by the OCR endpoint.
        table_content_format : str or None
            The specified format for table content (e.g., 'simple' or 'pseudo_markdown').
        dimensions : list of dict, optional
            A list of dict for each corresponding image, used for bounding box scaling.

        Returns
        -------
        list of (str, str)
            A list of (content, table_content_format) tuples, one for each image result.

        Raises
        ------
        RuntimeError
            If the response format is missing or invalid.
        ValueError
            If the `table_content_format` is unrecognized.
        """
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        results: List[str] = []
        for item_idx, item in enumerate(json_response["data"]):
            text_detections = item.get("text_detections", [])
            text_predictions = []
            bounding_boxes = []
            conf_scores = []
            for td in text_detections:
                text_predictions.append(td["text_prediction"]["text"])
                bounding_boxes.append([[pt["x"], pt["y"]] for pt in td["bounding_box"]["points"]])
                conf_scores.append(td["text_prediction"]["confidence"])

            bounding_boxes, text_predictions, conf_scores = self._postprocess_ocr_response(
                bounding_boxes,
                text_predictions,
                conf_scores,
                dimensions,
                img_index=item_idx,
            )

            results.append([bounding_boxes, text_predictions, conf_scores])

        return results

    def _extract_content_from_ocr_grpc_response(
        self,
        response: np.ndarray,
        dimensions: List[Dict[str, Any]],
        model_name: str = DEFAULT_OCR_MODEL_NAME,
    ) -> List[Tuple[str, str]]:
        """
        Parse a gRPC response for one or more images. The response can have two possible shapes:
          - (3,) for batch_size=1
          - (3, n) for batch_size=n

        In either case:
          response[0, i]: byte string containing bounding box data
          response[1, i]: byte string containing text prediction data
          response[2, i]: (Optional) additional data/metadata (ignored here)

        Parameters
        ----------
        response : np.ndarray
            The raw NumPy array from gRPC. Expected shape: (3,) or (3, n).
        table_content_format : str
            The format of the output text content, e.g. 'simple' or 'pseudo_markdown'.
        dims : list of dict, optional
            A list of dict for each corresponding image, used for bounding box scaling.

        Returns
        -------
        list of (str, str)
            A list of (content, table_content_format) for each image.

        Raises
        ------
        ValueError
            If the response is not a NumPy array or has an unexpected shape,
            or if the `table_content_format` is unrecognized.
        """
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        if model_name in [
            NEMORETRIEVER_OCR_MODEL_NAME,
            NEMORETRIEVER_OCR_ENSEMBLE_MODEL_NAME,
            NEMORETRIEVER_OCR_BLS_MODEL_NAME,
        ]:
            response = response.transpose((1, 0))

        # If we have shape (3,), convert to (3, 1)
        if response.ndim == 1 and response.shape == (3,):
            response = response.reshape(3, 1)
        elif response.ndim != 2 or response.shape[0] != 3:
            raise ValueError(f"Unexpected response shape: {response.shape}. Expecting (3,) or (3, n).")
        batch_size = response.shape[1]
        results: List[Tuple[str, str]] = []

        for i in range(batch_size):
            # 1) Parse bounding boxes
            bboxes_bytestr: bytes = response[0, i]
            bounding_boxes = json.loads(bboxes_bytestr.decode("utf8"))

            # 2) Parse text predictions
            texts_bytestr: bytes = response[1, i]
            text_predictions = json.loads(texts_bytestr.decode("utf8"))

            # 3) Parse confidence scores
            confs_bytestr: bytes = response[2, i]
            conf_scores = json.loads(confs_bytestr.decode("utf8"))

            # Some gRPC responses nest single-item lists; flatten them if needed
            if (
                (isinstance(bounding_boxes, list) and len(bounding_boxes) == 1 and isinstance(bounding_boxes[0], list))
                and (
                    isinstance(text_predictions, list)
                    and len(text_predictions) == 1
                    and isinstance(text_predictions[0], list)
                )
                and (isinstance(conf_scores, list) and len(conf_scores) == 1 and isinstance(conf_scores[0], list))
            ):
                bounding_boxes = bounding_boxes[0]
                text_predictions = text_predictions[0]
                conf_scores = conf_scores[0]

            # 4) Postprocess
            bounding_boxes, text_predictions, conf_scores = self._postprocess_ocr_response(
                bounding_boxes,
                text_predictions,
                conf_scores,
                dimensions,
                img_index=i,
                scale_coordinates=True,
            )

            results.append([bounding_boxes, text_predictions, conf_scores])

        return results

    @staticmethod
    def _postprocess_ocr_response(
        bounding_boxes: List[Any],
        text_predictions: List[str],
        conf_scores: List[float],
        dims: Optional[List[Dict[str, Any]]] = None,
        img_index: int = 0,
        scale_coordinates: bool = True,
        shift_coordinates: bool = True,
    ) -> Tuple[List[Any], List[str]]:
        """
        Convert bounding boxes with normalized coordinates to pixel cooridnates by using
        the dimensions. Also shift the coorindates if the inputs were padded. For multiple images,
        the correct image dimensions (height, width) are retrieved from `dims[img_index]`.

        Parameters
        ----------
        bounding_boxes : list of Any
            A list (per line of text) of bounding boxes, each a list of (x, y) points.
        text_predictions : list of str
            A list of text predictions, one for each bounding box.
        img_index : int, optional
            The index of the image for which bounding boxes are being converted. Default is 0.
        dims : list of dict, optional
            A list of dictionaries, where each dictionary contains image-specific dimensions
            and scaling information:
                - "new_width" (int): The width of the image after processing.
                - "new_height" (int): The height of the image after processing.
                - "pad_width" (int, optional): The width of padding added to the image.
                - "pad_height" (int, optional): The height of padding added to the image.
                - "scale_factor" (float, optional): The scaling factor applied to the image.

        Returns
        -------
        Tuple[List[Any], List[str]]
            Bounding boxes scaled backed to the original dimensions and detected text lines.

        Notes
        -----
        - If `dims` is None or `img_index` is out of range, bounding boxes will not be scaled properly.
        """
        # Default to no scaling if dims are missing or out of range
        if not dims:
            raise ValueError("No image_dims provided.")
        else:
            if img_index >= len(dims):
                logger.warning("Image index out of range for stored dimensions. Using first image dims by default.")
                img_index = 0

        max_width = dims[img_index]["new_width"] if scale_coordinates else 1.0
        max_height = dims[img_index]["new_height"] if scale_coordinates else 1.0
        pad_width = dims[img_index].get("pad_width", 0) if shift_coordinates else 0.0
        pad_height = dims[img_index].get("pad_height", 0) if shift_coordinates else 0.0
        scale_factor = dims[img_index].get("scale_factor", 1.0) if scale_coordinates else 1.0

        bboxes: List[List[float]] = []
        texts: List[str] = []
        confs: List[float] = []

        # Convert normalized coords back to actual pixel coords
        for box, txt, conf in zip(bounding_boxes, text_predictions, conf_scores):
            if box == "nan":
                continue
            points: List[List[float]] = []
            for point in box:
                # Convert normalized coords back to actual pixel coords,
                # and shift them back to their original positions if padded.
                x_pixels = float(point[0]) * max_width - pad_width
                y_pixels = float(point[1]) * max_height - pad_height
                x_original = x_pixels / scale_factor
                y_original = y_pixels / scale_factor
                points.append([x_original, y_original])
            bboxes.append(points)
            texts.append(txt)
            confs.append(conf)

        return bboxes, texts, confs


class PaddleOCRModelInterface(OCRModelInterfaceBase):
    """
    An interface for handling inference with a legacy OCR model, supporting both gRPC and HTTP protocols.
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        return "PaddleOCR"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode one or more base64-encoded images into NumPy arrays, storing them
        alongside their dimensions in `data`.

        Parameters
        ----------
        data : dict of str -> Any
            The input data containing either:
             - 'base64_image': a single base64-encoded image, or
             - 'base64_images': a list of base64-encoded images.

        Returns
        -------
        dict of str -> Any
            The updated data dictionary with the following keys added:
            - "images": List of decoded NumPy arrays of shape (H, W, C).
            - "image_dims": List of (height, width) tuples for each decoded image.

        Raises
        ------
        KeyError
            If neither 'base64_image' nor 'base64_images' is found in `data`.
        ValueError
            If 'base64_images' is present but is not a list.
        """
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")

            images: List[np.ndarray] = []
            for b64 in base64_list:
                img = base64_to_numpy(b64)
                images.append(img)

            data["images"] = images

        elif "base64_image" in data:
            # Single-image fallback
            img = base64_to_numpy(data["base64_image"])
            data["images"] = [img]

        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol ("grpc" or "http"), supporting batched data.

        Parameters
        ----------
        data : dict of str -> Any
            The input data dictionary, expected to contain "images" (list of np.ndarray)
            and "image_dims" (list of (height, width) tuples), as produced by prepare_data_for_inference.
        protocol : str
            The inference protocol, either "grpc" or "http".
        max_batch_size : int
            The maximum batch size for batching.

        Returns
        -------
        tuple
            A tuple (formatted_batches, formatted_batch_data) where:
              - formatted_batches is a list of batches ready for inference.
              - formatted_batch_data is a list of scratch-pad dictionaries corresponding to each batch,
                containing the keys "images" and "image_dims" for later post-processing.

        Raises
        ------
        KeyError
            If either "images" or "image_dims" is not found in `data`.
        ValueError
            If an invalid protocol is specified.
        """

        images = data["images"]

        dims: List[Dict[str, Any]] = []
        data["image_dims"] = dims

        # Helper function to split a list into chunks of size up to chunk_size.
        def chunk_list(lst, chunk_size):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if "images" not in data or "image_dims" not in data:
            raise KeyError("Expected 'images' and 'image_dims' in data. Call prepare_data_for_inference first.")

        images = data["images"]
        dims = data["image_dims"]

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC OCR model (batched).")
            processed: List[np.ndarray] = []

            for img in images:
                arr, _dims = preprocess_image_for_paddle(img)
                dims.append(_dims)
                arr = arr.astype(np.float32)
                arr = np.expand_dims(arr, axis=0)
                processed.append(arr)

            batches = []
            batch_data_list = []
            for proc_chunk, orig_chunk, dims_chunk in zip(
                chunk_list(processed, max_batch_size),
                chunk_list(images, max_batch_size),
                chunk_list(dims, max_batch_size),
            ):
                batched_input = np.concatenate(proc_chunk, axis=0)
                batches.append(batched_input)
                batch_data_list.append({"images": orig_chunk, "image_dims": dims_chunk})
            return batches, batch_data_list

        elif protocol == "http":
            logger.debug("Formatting input for HTTP OCR model (batched).")
            if "base64_images" in data:
                base64_list = data["base64_images"]
            else:
                base64_list = [data["base64_image"]]

            input_list: List[Dict[str, Any]] = []
            for b64, img in zip(base64_list, images):
                image_url = f"data:image/png;base64,{b64}"
                image_obj = {"type": "image_url", "url": image_url}
                input_list.append(image_obj)
                _dims = {"new_width": img.shape[1], "new_height": img.shape[0]}
                dims.append(_dims)

            batches = []
            batch_data_list = []
            for input_chunk, orig_chunk, dims_chunk in zip(
                chunk_list(input_list, max_batch_size),
                chunk_list(images, max_batch_size),
                chunk_list(dims, max_batch_size),
            ):
                payload = {"input": input_chunk}
                batches.append(payload)
                batch_data_list.append({"images": orig_chunk, "image_dims": dims_chunk})

            return batches, batch_data_list

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")


class NemoRetrieverOCRModelInterface(OCRModelInterfaceBase):
    """
    An interface for handling inference with NemoRetrieverOCR model, supporting both gRPC and HTTP protocols.
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        return "NemoRetrieverOCR"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode one or more base64-encoded images into NumPy arrays, storing them
        alongside their dimensions in `data`.

        Parameters
        ----------
        data : dict of str -> Any
            The input data containing either:
             - 'base64_image': a single base64-encoded image, or
             - 'base64_images': a list of base64-encoded images.

        Returns
        -------
        dict of str -> Any
            The updated data dictionary with the following keys added:
            - "images": List of decoded NumPy arrays of shape (H, W, C).
            - "image_dims": List of (height, width) tuples for each decoded image.

        Raises
        ------
        KeyError
            If neither 'base64_image' nor 'base64_images' is found in `data`.
        ValueError
            If 'base64_images' is present but is not a list.
        """
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")

            images: List[np.ndarray] = []
            for b64 in base64_list:
                img = base64_to_numpy(b64)
                images.append(img)

            data["images"] = images

        elif "base64_image" in data:
            # Single-image fallback
            img = base64_to_numpy(data["base64_image"])
            data["images"] = [img]

        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        return data

    def coalesce_requests_to_batch(
        self,
        requests: List[np.ndarray],
        original_image_shapes: List[Tuple[int, int]],
        protocol: str,
        **kwargs,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Takes a list of individual data items (NumPy image arrays) and combines them
        into a single formatted batch ready for inference.

        This method mirrors the logic of `format_input` but operates on an already-formed
        batch from the dynamic batcher, so it does not perform any chunking.

        Parameters
        ----------
        requests : List[np.ndarray]
            A list of single data items, which are NumPy arrays representing images.
        protocol : str
            The inference protocol, either "grpc" or "http".
        **kwargs : Any
            Additional keyword arguments, such as `model_name` and `merge_level`.

        Returns
        -------
        Tuple[List[Any], List[Dict[str, Any]]]
            A tuple containing two lists, each with a single element:
            - The first list contains the single formatted batch.
            - The second list contains the single scratch-pad dictionary for that batch.
        """
        if not requests:
            return None, {}

        return self._format_single_batch(requests, original_image_shapes, protocol, **kwargs)

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol ("grpc" or "http"), supporting batched data.

        Parameters
        ----------
        data : dict of str -> Any
            The input data dictionary, expected to contain "images" (list of np.ndarray)
            and "image_dims" (list of (height, width) tuples), as produced by prepare_data_for_inference.
        protocol : str
            The inference protocol, either "grpc" or "http".
        max_batch_size : int
            The maximum batch size for batching.

        Returns
        -------
        tuple
            A tuple (formatted_batches, formatted_batch_data) where:
              - formatted_batches is a list of batches ready for inference.
              - formatted_batch_data is a list of scratch-pad dictionaries corresponding to each batch,
                containing the keys "images" and "image_dims" for later post-processing.

        Raises
        ------
        KeyError
            If either "images" or "image_dims" is not found in `data`.
        ValueError
            If an invalid protocol is specified.
        """

        # Helper function to split a list into chunks of size up to chunk_size.
        def chunk_list(lst, chunk_size):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if "images" not in data:
            raise KeyError("Expected 'images' in data. Call prepare_data_for_inference first.")

        images = data["base64_images"]
        dims = [img.shape[:2] for img in data["images"]]

        formatted_batches = []
        formatted_batch_data = []

        image_chunks = chunk_list(images, max_batch_size)
        dims_chunks = chunk_list(dims, max_batch_size)
        for image_chunk, dims_chunk in zip(image_chunks, dims_chunks):
            final_batch, batch_data = self._format_single_batch(image_chunk, dims_chunk, protocol, **kwargs)
            formatted_batches.append(final_batch)
            formatted_batch_data.append(batch_data)

        all_dims = [item for d in formatted_batch_data for item in d.get("image_dims", [])]
        data["image_dims"] = all_dims

        return formatted_batches, formatted_batch_data

    def _format_single_batch(
        self,
        batch_images: List[str],
        batch_dims: List[Tuple[int, int]],
        protocol: str,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        dims: List[Dict[str, Any]] = []

        merge_level = kwargs.get("merge_level", "paragraph")

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC OCR model (batched).")
            processed: List[np.ndarray] = []

            for img, shape in zip(batch_images, batch_dims):
                _dims = {"new_width": shape[1], "new_height": shape[0]}
                dims.append(_dims)

                arr = np.array([img], dtype=np.object_)
                arr = np.expand_dims(arr, axis=0)
                processed.append(arr)

            batched_input = np.concatenate(processed, axis=0)

            batch_size = batched_input.shape[0]

            merge_levels_list = [[merge_level] for _ in range(batch_size)]
            merge_levels = np.array(merge_levels_list, dtype="object")

            final_batch = [batched_input, merge_levels]
            batch_data = {"image_dims": dims}

            return final_batch, batch_data

        elif protocol == "http":
            logger.debug("Formatting input for HTTP OCR model (batched).")

            input_list: List[Dict[str, Any]] = []
            for b64, shape in zip(batch_images, batch_dims):
                image_url = f"data:image/png;base64,{b64}"
                image_obj = {"type": "image_url", "url": image_url}
                input_list.append(image_obj)
                _dims = {"new_width": shape[1], "new_height": shape[0]}
                dims.append(_dims)

            payload = {
                "input": input_list,
                "merge_levels": [merge_level] * len(input_list),
            }

            batch_data = {"image_dims": dims}

            return payload, batch_data

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_ocr_model_name(ocr_grpc_endpoint=None, default_model_name=DEFAULT_OCR_MODEL_NAME):
    """
    Determines the OCR model name by checking the environment, querying the gRPC endpoint,
    or falling back to a default.
    """
    # 1. Check for an explicit override from the environment variable first.
    ocr_model_name = os.getenv("OCR_MODEL_NAME", None)
    if ocr_model_name is not None:
        return ocr_model_name

    # 2. If no gRPC endpoint is provided or the endpoint is a NVCF endpoint, fall back to the default immediately.
    if (not ocr_grpc_endpoint) or ("grpc.nvcf.nvidia.com" in ocr_grpc_endpoint):
        logger.debug(f"No OCR gRPC endpoint provided. Falling back to default model name '{default_model_name}'.")
        return default_model_name

    # 3. Attempt to query the gRPC endpoint to discover the model name.
    try:
        client = grpcclient.InferenceServerClient(ocr_grpc_endpoint)
        model_index = client.get_model_repository_index(as_json=True)
        model_names = [x["name"] for x in model_index.get("models", [])]
        ocr_model_name = model_names[0]
    except Exception:
        logger.warning(f"Failed to get ocr model name after 30 seconds. Falling back to '{default_model_name}'.")
        ocr_model_name = default_model_name

    return ocr_model_name
