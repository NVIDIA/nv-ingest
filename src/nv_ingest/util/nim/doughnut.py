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
from typing import Tuple

from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import ModelInterface

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

_re_extract_class_bbox = re.compile(
    r"<x_(\d+)><y_(\d+)>((?:|.(?:(?<!<x_\d)(?<!<y_\d)(?<!<class_[A-Za-z0-9]).)*))<x_(\d+)><y_(\d+)><class_([A-Za-z0-9\-]+)>",  # noqa: E501
    re.MULTILINE | re.DOTALL,
)

logger = logging.getLogger(__name__)


class DoughnutModelInterface(ModelInterface):
    """
    An interface for handling inference with a Doughnut model.
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        return "doughnut"

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

    def format_input(self, data: Dict[str, Any], protocol: str, **kwargs) -> Any:
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

        if protocol == "grpc":
            raise ValueError("gRPC protocol is not supported for Doughnut.")
        elif protocol == "http":
            logger.debug("Formatting input for HTTP Doughnut model")
            # Prepare payload for HTTP request
            base64_img = numpy_to_base64(data["image"])
            payload = self._prepare_doughnut_payload(base64_img)
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
            raise ValueError("gRPC protocol is not supported for Doughnut.")
        elif protocol == "http":
            logger.debug("Parsing output from HTTP Doughnut model")
            return self._extract_content_from_doughnut_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs) -> Any:
        """
        Process inference results for the Doughnut model.

        Parameters
        ----------
        output : Any
            The raw output from the model.

        Returns
        -------
        Any
            The processed inference results.
        """

        return output

    def _prepare_doughnut_payload(self, base64_img: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}",
                        },
                    }
                ],
            }
        ]
        payload = {
            "model": "nvidia/eclair",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "markdown_bbox",
                    },
                }
            ],
        }

        return payload

    def _extract_content_from_doughnut_response(self, json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a Deplot HTTP API request.

        Parameters
        ----------
        json_response : dict
            The JSON response from the Deplot API.

        Returns
        -------
        Any
            The extracted content from the response.

        Raises
        ------
        RuntimeError
            If the response does not contain the expected "choices" key or if it is empty.
        """

        if "choices" not in json_response or not json_response["choices"]:
            raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")

        tool_call = json_response["choices"][0]["message"]["tool_calls"][0]
        return json.loads(tool_call["function"]["arguments"])[0]


def extract_classes_bboxes(text: str) -> Tuple[List[str], List[Tuple[int, int, int, int]], List[str]]:
    classes: List[str] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    texts: List[str] = []

    last_end = 0

    for m in _re_extract_class_bbox.finditer(text):
        start, end = m.span()

        # [Bad box] Add the non-match chunk (text between the last match and the current match)
        if start > last_end:
            bad_text = text[last_end:start].strip()
            classes.append("Bad-box")
            bboxes.append((0, 0, 0, 0))
            texts.append(bad_text)

        last_end = end

        x1, y1, text, x2, y2, cls = m.groups()

        bbox = tuple(map(int, (x1, y1, x2, y2)))

        # [Bad box] check if the class is a valid class.
        if cls not in ACCEPTED_CLASSES:
            logger.debug(f"Dropped a bad box: invalid class {cls} at {bbox}.")
            classes.append("Bad-box")
            bboxes.append(bbox)
            texts.append(text)
            continue

        # Drop bad box: drop if the box is invalid.
        if (bbox[0] >= bbox[2]) or (bbox[1] >= bbox[3]):
            logger.debug(f"Dropped a bad box: invalid box {cls} at {bbox}.")
            classes.append("Bad-box")
            bboxes.append(bbox)
            texts.append(text)
            continue

        classes.append(cls)
        bboxes.append(bbox)
        texts.append(text)

    if last_end < len(text):
        bad_text = text[last_end:].strip()
        if len(bad_text) > 0:
            classes.append("Bad-box")
            bboxes.append((0, 0, 0, 0))
            texts.append(bad_text)

    return classes, bboxes, texts


def _fix_dots(m):
    # Remove spaces between dots.
    s = m.group(0)
    return s.startswith(" ") * " " + min(5, s.count(".")) * "." + s.endswith(" ") * " "


def strip_markdown_formatting(text):
    # Remove headers (e.g., # Header, ## Header, ### Header)
    text = re.sub(r"^(#+)\s*(.*)", r"\2", text, flags=re.MULTILINE)

    # Remove bold formatting (e.g., **bold text** or __bold text__)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)

    # Remove italic formatting (e.g., *italic text* or _italic text_)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Remove strikethrough formatting (e.g., ~~strikethrough~~)
    text = re.sub(r"~~(.*?)~~", r"\1", text)

    # Remove list items (e.g., - item, * item, 1. item)
    text = re.sub(r"^\s*([-*+]|[0-9]+\.)\s+", "", text, flags=re.MULTILINE)

    # Remove hyperlinks (e.g., [link text](http://example.com))
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # Remove inline code (e.g., `code`)
    text = re.sub(r"`(.*?)`", r"\1", text)

    # Remove blockquotes (e.g., > quote)
    text = re.sub(r"^\s*>\s*(.*)", r"\1", text, flags=re.MULTILINE)

    # Remove multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Limit dots sequences to max 5 dots
    text = re.sub(r"(?:\s*\.\s*){3,}", _fix_dots, text, flags=re.DOTALL)

    return text


def reverse_transform_bbox(
    bbox: Tuple[int, int, int, int],
    bbox_offset: Tuple[int, int],
    original_width: int,
    original_height: int,
) -> Tuple[int, int, int, int]:
    width_ratio = (original_width - 2 * bbox_offset[0]) / original_width
    height_ratio = (original_height - 2 * bbox_offset[1]) / original_height
    w1, h1, w2, h2 = bbox
    w1 = int((w1 - bbox_offset[0]) / width_ratio)
    h1 = int((h1 - bbox_offset[1]) / height_ratio)
    w2 = int((w2 - bbox_offset[0]) / width_ratio)
    h2 = int((h2 - bbox_offset[1]) / height_ratio)

    return (w1, h1, w2, h2)


def postprocess_text(txt: str, cls: str):
    if cls in ACCEPTED_CLASSES:
        txt = txt.replace("<tbc>", "").strip()  # remove <tbc> tokens (continued paragraphs)
        txt = strip_markdown_formatting(txt)
    else:
        txt = ""

    return txt
