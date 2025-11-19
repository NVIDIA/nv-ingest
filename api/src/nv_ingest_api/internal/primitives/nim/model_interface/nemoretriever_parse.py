# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nv_ingest_api.internal.primitives.nim import ModelInterface
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64

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

logger = logging.getLogger(__name__)


class NemoRetrieverParseModelInterface(ModelInterface):
    """
    An interface for handling inference with a NemoRetrieverParse model.
    """

    def __init__(self, model_name: str = "nvidia/nemoretriever-parse"):
        """
        Initialize the instance with a specified model name.
        Parameters
        ----------
        model_name : str, optional
            The name of the model to be used, by default "nvidia/nemoretriever-parse".
        """
        self.model_name = model_name

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        return "nemoretriever_parse"

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
            raise ValueError("gRPC protocol is not supported for NemoRetrieverParse.")
        elif protocol == "http":
            logger.debug("Formatting input for HTTP NemoRetrieverParse model")
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
                payload = self._prepare_nemoretriever_parse_payload(b64_chunk)
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
            raise ValueError("gRPC protocol is not supported for NemoRetrieverParse.")
        elif protocol == "http":
            logger.debug("Parsing output from HTTP NemoRetrieverParse model")
            return self._extract_content_from_nemoretriever_parse_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs) -> Any:
        """
        Process inference results for the NemoRetrieverParse model.

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

    def _prepare_nemoretriever_parse_payload(self, base64_list: List[str]) -> Dict[str, Any]:
        messages = []

        for b64_img in base64_list:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}",
                            },
                        }
                    ],
                }
            )
        payload = {
            "model": self.model_name,
            "messages": messages,
        }

        return payload

    def _extract_content_from_nemoretriever_parse_response(self, json_response: Dict[str, Any]) -> Any:
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
        return json.loads(tool_call["function"]["arguments"])
