from typing import Dict, Any

import numpy as np

from nv_ingest.util.image_processing.transforms import numpy_to_base64, base64_to_numpy
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle, logger, ModelInterface


class PaddleOCRModelInterface(ModelInterface):
    def prepare_data_for_inference(self, data):
        # Expecting base64_image in data
        base64_image = data['base64_image']
        image_array = base64_to_numpy(base64_image)
        data['image_array'] = image_array
        return data

    def format_input(self, data, protocol: str, **kwargs):
        image_array = data['image_array']
        if protocol == 'grpc':
            logger.debug("Formatting input for gRPC PaddleOCR model")
            # Preprocess image if necessary
            image_array = preprocess_image_for_paddle(image_array)
            if image_array.ndim == 3:
                image_array = np.expand_dims(image_array, axis=0)
            return image_array
        elif protocol == 'http':
            logger.debug("Formatting input for HTTP PaddleOCR model")
            base64_img = numpy_to_base64(image_array)
            payload = self._prepare_paddle_payload(base64_img)
            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str):
        if protocol == 'grpc':
            logger.debug("Parsing output from gRPC PaddleOCR model")
            # Convert bytes output to string
            return " ".join([output.decode("utf-8") for output in response])
        elif protocol == 'http':
            logger.debug("Parsing output from HTTP PaddleOCR model")
            return self._extract_content_from_paddle_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output, **kwargs):
        # For PaddleOCR, the output is the table content as a string
        return output

    def _prepare_paddle_payload(self, base64_img: str) -> Dict[str, Any]:
        image_url = f"data:image/png;base64,{base64_img}"
        image = {"type": "image_url", "image_url": {"url": image_url}}
        message = {"content": [image]}
        payload = {"messages": [message]}
        return payload

    def _extract_content_from_paddle_response(self, json_response):
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")
        return json_response["data"][0]["content"]
