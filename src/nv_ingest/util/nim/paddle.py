from typing import Dict, Any, Optional
import numpy as np
import logging

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface, preprocess_image_for_paddle

logger = logging.getLogger(__name__)


class PaddleOCRModelInterface(ModelInterface):
    def __init__(self, paddle_version: Optional[str] = None):
        self.paddle_version = paddle_version

    def prepare_data_for_inference(self, data):
        # Expecting base64_image in data
        base64_image = data['base64_image']
        image_array = base64_to_numpy(base64_image)
        data['image_array'] = image_array
        return data

    def format_input(self, data, protocol: str, **kwargs):
        image_data = data['image_array']
        if protocol == 'grpc':
            logger.debug("Formatting input for gRPC PaddleOCR model")
            image_data = preprocess_image_for_paddle(image_data, self.paddle_version)
            image_data = image_data.astype(np.float32)
            image_data = np.expand_dims(image_data, axis=0)

            return image_data
        elif protocol == 'http':
            logger.debug("Formatting input for HTTP PaddleOCR model")
            # For HTTP, preprocessing is not necessary
            base64_img = data['base64_image']
            payload = self._prepare_paddle_payload(base64_img)

            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response, protocol: str):
        if protocol == 'grpc':
            logger.debug("Parsing output from gRPC PaddleOCR model")
            # Convert bytes output to string
            if isinstance(response, np.ndarray):
                if response.dtype.type is np.object_ or response.dtype.type is np.bytes_:
                    output_strings = [
                        output[0].decode('utf-8') if isinstance(output, bytes) else str(output) for output in response
                    ]
                    return " ".join(output_strings)
                else:
                    output_bytes = response.tobytes()
                    output_str = output_bytes.decode('utf-8')
                    return output_str
            else:
                raise ValueError("Unexpected response format: response is not a NumPy array.")
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
