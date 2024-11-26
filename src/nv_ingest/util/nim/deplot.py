# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any

import numpy as np
import logging

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class DeplotModelInterface(ModelInterface):
    def prepare_data_for_inference(self, data):
        # Expecting base64_image in data
        base64_image = data['base64_image']
        data['image_array'] = base64_to_numpy(base64_image)
        return data

    def format_input(self, data, protocol: str, **kwargs):
        if protocol == 'grpc':
            logger.debug("Formatting input for gRPC Deplot model")
            # Convert image array to expected format
            image_data = data['image_array']
            if image_data.ndim == 3:
                image_data = np.expand_dims(image_data, axis=0)
            # Convert to float32 and normalize if required
            image_data = image_data.astype(np.float32)
            # Normalize pixel values to [0, 1] if needed
            image_data /= 255.0
            return image_data
        elif protocol == 'http':
            logger.debug("Formatting input for HTTP Deplot model")
            # Prepare payload for HTTP request
            base64_img = data['base64_image']
            payload = self._prepare_deplot_payload(
                base64_img,
                max_tokens=kwargs.get('max_tokens', 500),
                temperature=kwargs.get('temperature', 0.5),
                top_p=kwargs.get('top_p', 0.9)
            )
            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str):
        if protocol == 'grpc':
            logger.debug("Parsing output from gRPC Deplot model")
            # Convert bytes output to string
            return " ".join([output[0].decode("utf-8") for output in response])
        elif protocol == 'http':
            logger.debug("Parsing output from HTTP Deplot model")
            return self._extract_content_from_deplot_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output, **kwargs):
        # For Deplot, the output is the chart content as a string
        return output

    def _prepare_deplot_payload(
            self,
            base64_img: str,
            max_tokens: int = 500,
            temperature: float = 0.5,
            top_p: float = 0.9
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": f"Generate the underlying data table of the figure below: "
                           f'<img src="data:image/png;base64,{base64_img}" />',
            }
        ]
        payload = {
            "model": "google/deplot",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
        }

        return payload

    def _extract_content_from_deplot_response(self, json_response):
        if "choices" not in json_response or not json_response["choices"]:
            raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")

        return json_response["choices"][0]["message"]["content"]
