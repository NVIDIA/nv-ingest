# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nemo_retriever.models.nim.primitives import ModelInterface
from nemo_retriever.common.api.util.image_processing.transforms import numpy_to_base64

ACCEPTED_TEXT_CLASSES = set(
    [
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "TOC",
        "Bibliography",
        "Formula",
        "Page-header",
        "Page-footer",
        "Caption",
        "Footnote",
        "Floating-text",
    ]
)
ACCEPTED_TABLE_CLASSES = set(
    [
        "Table",
    ]
)
ACCEPTED_IMAGE_CLASSES = set(
    [
        "Picture",
    ]
)
ACCEPTED_CLASSES = ACCEPTED_TEXT_CLASSES | ACCEPTED_TABLE_CLASSES | ACCEPTED_IMAGE_CLASSES

_RE_EXTRACT_CLASS_BBOX = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)

# Internal resolution the NIM model uses for processing.
_NIM_TARGET_WIDTH = 1664
_NIM_TARGET_HEIGHT = 2048

logger = logging.getLogger(__name__)


def remove_nemotron_formatting(text: str) -> str:
    """Remove special NIM formatting tokens from text."""
    text = text.replace("<tbc>", "")
    text = text.replace("\\<|unk|\\>", "")
    text = text.replace("\\unknown", "")
    return text


def transform_bbox_to_original(
    bbox: Dict[str, float],
    original_width: int,
    original_height: int,
    target_w: int = _NIM_TARGET_WIDTH,
    target_h: int = _NIM_TARGET_HEIGHT,
) -> Dict[str, float]:
    """Transform bbox from NIM's internal padded coordinate space back to the
    original image dimensions.

    The NIM resizes the input image (preserving aspect ratio) to fit within
    *target_w* x *target_h* and then center-pads.  The returned bbox
    coordinates are normalised to [0, 1] relative to this padded space.  This
    function reverses that transform so the coordinates are relative to the
    original image instead.
    """
    aspect_ratio = original_width / original_height
    new_height = original_height
    new_width = original_width

    if original_height > target_h:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)

    if new_width > target_w:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)

    resized_width = new_width
    resized_height = new_height

    pad_left = (target_w - resized_width) // 2
    pad_top = (target_h - resized_height) // 2

    xmin = ((bbox["xmin"] * target_w) - pad_left) / resized_width
    xmax = ((bbox["xmax"] * target_w) - pad_left) / resized_width
    ymin = ((bbox["ymin"] * target_h) - pad_top) / resized_height
    ymax = ((bbox["ymax"] * target_h) - pad_top) / resized_height

    return {
        "xmin": max(0.0, xmin),
        "ymin": max(0.0, ymin),
        "xmax": min(1.0, xmax),
        "ymax": min(1.0, ymax),
    }


class NemotronParseModelInterface(ModelInterface):
    """
    An interface for handling inference with a Nemotron Parse model.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-parse",
        image_width: int = None,
        image_height: int = None,
        temperature: float = 0.0,
        repetition_penalty: float = 1.1,
    ):
        """
        Initialize the instance with a specified model name.
        Parameters
        ----------
        model_name : str, optional
            The name of the model to be used, by default "nvidia/nemotron-parse".
        image_width : int, optional
            Width of images sent to the NIM (after pre-scaling/padding).
            Used to reverse the NIM's internal bbox coordinate transform.
        image_height : int, optional
            Height of images sent to the NIM (after pre-scaling/padding).
        temperature : float, optional
            Sampling temperature for the NIM, by default 0.0 (deterministic).
        repetition_penalty : float, optional
            Repetition penalty for the NIM, by default 1.1.
        """
        self.model_name = model_name
        self._image_width = image_width
        self._image_height = image_height
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        return "nemotron_parse"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for inference by resizing images and storing their original shapes.

        Parameters
        ----------
        data : dict
            The input data containing a list of images.

        Returns
        -------
        dict
            The updated data dictionary with resized images and original image shapes.
        """

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to use ("grpc" or "http").
        **kwargs : dict
            Additional parameters for HTTP payload formatting.

        Returns
        -------
        Any
            The formatted input data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """

        # Helper function: chunk a list into sublists of length <= chunk_size.
        def chunk_list(lst: list, chunk_size: int) -> List[list]:
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if protocol == "grpc":
            raise ValueError("gRPC protocol is not supported for Nemotron Parse.")
        elif protocol == "http":
            logger.debug("Formatting input for HTTP Nemotron Parse model")
            # Prepare payload for HTTP request

            ## TODO: Ask @Edward Kim if we want to switch to JPEG/PNG here
            if "images" in data:
                base64_list = [numpy_to_base64(img) for img in data["images"]]
            else:
                base64_list = [numpy_to_base64(data["image"])]

            formatted_batches = []
            formatted_batch_data = []
            b64_chunks = chunk_list(base64_list, max_batch_size)

            for b64_chunk in b64_chunks:
                payload = self._prepare_nemotron_parse_payload(b64_chunk)
                formatted_batches.append(payload)
                formatted_batch_data.append({})
            return formatted_batches, formatted_batch_data

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
            raise ValueError("gRPC protocol is not supported for Nemotron Parse.")
        elif protocol == "http":
            logger.debug("Parsing output from HTTP Nemotron Parse model")
            return self._extract_content_from_nemotron_parse_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs) -> Any:
        """
        Process inference results for the Nemotron Parse model.

        Each call corresponds to one page (one batch/request). The caller
        (NimClient.infer) uses ``extend`` on the return value, so we wrap
        the per-page list of elements in an outer list to keep it as a
        single entry when coalesced.

        Parameters
        ----------
        output : Any
            The raw output from the model (list of bbox dicts for one page).

        Returns
        -------
        list
            A single-element list containing the per-page output.
        """

        return [output]

    # Prompt that instructs the v1.2+ model to predict bounding boxes,
    # classes, output markdown, and suppress text inside pictures.
    _NEMOTRON_PARSE_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

    @property
    def _uses_tagged_prompt_contract(self) -> bool:
        """Return whether the configured model accepts Parse task-control tokens."""
        normalized = self.model_name.lower().rstrip("/")
        return normalized != "nvidia/nemotron-parse"

    def _prepare_nemotron_parse_payload(self, base64_list: List[str]) -> Dict[str, Any]:
        messages = []

        for b64_img in base64_list:
            content = []
            if self._uses_tagged_prompt_contract:
                content.append({"type": "text", "text": self._NEMOTRON_PARSE_PROMPT})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                    },
                }
            )
            messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self._temperature,
        }
        normalized = self.model_name.lower().rstrip("/")
        if normalized == "nvidia/nemotron-parse-2.0":
            payload["max_tokens"] = 8192
        else:
            payload["repetition_penalty"] = self._repetition_penalty
            if normalized == "nvidia/nemotron-parse-v2.0":
                # The NIM context includes the task prompt; reserve headroom.
                payload.update(max_tokens=8192, top_k=1, skip_special_tokens=False)

        return payload

    def _extract_content_from_nemotron_parse_response(self, json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a Nemotron Parse HTTP API request.

        Supports two response formats:
        - Legacy tool_calls format: structured JSON in tool_calls[0]["function"]["arguments"]
        - Content tag format (v1.2+): tagged text in message["content"] with
          ``<x_...><y_...>text<x_...><y_...><class_...>`` markers.

        Parameters
        ----------
        json_response : dict
            The JSON response from the Nemotron Parse API.

        Returns
        -------
        list[dict]
            A list of dicts, each with keys ``type``, ``bbox`` (dict with
            ``xmin``, ``ymin``, ``xmax``, ``ymax``), and ``text``.

        Raises
        ------
        RuntimeError
            If the response does not contain the expected "choices" key or if it is empty.
        """

        if "choices" not in json_response or not json_response["choices"]:
            raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")

        message = json_response["choices"][0]["message"]

        # Legacy format: structured data via tool_calls.
        # The arguments JSON is [[page1_elems], [page2_elems], ...] — a list
        # of per-page lists.  Since the NimClient sends one request per page
        # (NIM_MAX_IMAGES_PER_PROMPT=1), we flatten to a single list of
        # element dicts so the shape matches the new content-tag format.
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tool_call = tool_calls[0]
            parsed = json.loads(tool_call["function"]["arguments"])
            # The NIM returns [[page1_elems], [page2_elems], ...].
            # Flatten to [elem, elem, ...] so each batch yields a flat
            # list of element dicts, matching the new content-tag format.
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                return [elem for page in parsed for elem in page]
            return parsed

        # v1.2+ format: tagged content string.
        # The NIM returns raw model output; we need to apply the same
        # postprocessing that the old NIM did server-side: parse tags,
        # clean formatting tokens, and map bboxes back to the input
        # image coordinate space.
        content = message.get("content", "")
        image_width = self._image_width
        image_height = self._image_height
        results = []
        for m in _RE_EXTRACT_CLASS_BBOX.finditer(content):
            x1, y1, text, x2, y2, cls = m.groups()
            # Normalize Inline-formula to Formula to match accepted classes
            if cls == "Inline-formula":
                cls = "Formula"

            text = remove_nemotron_formatting(text)

            bbox = {
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
            }
            if image_width is not None and image_height is not None:
                bbox = transform_bbox_to_original(bbox, image_width, image_height)

            results.append({"type": cls, "bbox": bbox, "text": text})
        return results
