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
    def name(self):
        return "Cached"

    def prepare_data_for_inference(self, data):
        # Expecting base64_image in data
        base64_image = data['base64_image']
        data['image_array'] = base64_to_numpy(base64_image)
        return data

    def format_input(self, data, protocol: str):
        if protocol == 'grpc':
            logger.debug("Formatting input for gRPC Cached model")
            # Convert image array to expected format
            image_data = data['image_array']
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)
            image_data = image_data.astype(np.float32)
            return image_data
        elif protocol == 'http':
            logger.debug("Formatting input for HTTP Cached model")
            # Prepare payload for HTTP request
            base64_img = data['base64_image']
            payload = self._prepare_nim_payload(base64_img)
            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str, data: Optional[Dict[str, Any]] = None):
        if protocol == 'grpc':
            logger.debug("Parsing output from gRPC Cached model")
            # Convert bytes output to string
            return " ".join([output[0].decode("utf-8") for output in response])
        elif protocol == 'http':
            logger.debug("Parsing output from HTTP Cached model")
            return self._extract_content_from_nim_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output, **kwargs):
        # For Cached model, the output is the chart content as a string
        return output

    def _prepare_nim_payload(self, base64_img: str) -> Dict[str, Any]:
        image_url = f"data:image/png;base64,{base64_img}"
        image = {"type": "image_url", "image_url": {"url": image_url}}

        message = {"content": [image]}
        payload = {"messages": [message]}

        return payload

    def _extract_content_from_nim_response(self, json_response):
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        return json_response["data"][0]["content"]
