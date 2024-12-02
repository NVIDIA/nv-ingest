import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import packaging

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle

logger = logging.getLogger(__name__)


class PaddleOCRModelInterface(ModelInterface):
    """
    An interface for handling inference with a PaddleOCR model, supporting both gRPC and HTTP protocols.
    """

    def __init__(self, paddle_version: Optional[str] = None):
        """
        Initialize the PaddleOCR model interface.

        Parameters
        ----------
        paddle_version : str, optional
            The version of the PaddleOCR model (default: None).
        """
        self.paddle_version = paddle_version

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface, including the PaddleOCR version.
        """
        return f"PaddleOCR - {self.paddle_version}"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for inference by decoding the base64 image into a numpy array.

        Parameters
        ----------
        data : dict
            The input data containing a base64-encoded image.

        Returns
        -------
        dict
            The updated data dictionary with the decoded image array.
        """

        # Expecting base64_image in data
        base64_image = data["base64_image"]
        image_array = base64_to_numpy(base64_image)
        data["image_array"] = image_array
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
            Additional parameters for formatting.

        Returns
        -------
        Any
            The formatted input data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """

        image_data = data['image_array']
        if (protocol == 'grpc'):
            logger.debug("Formatting input for gRPC PaddleOCR model")
            image_data = preprocess_image_for_paddle(image_data, self.paddle_version)
            image_data = image_data.astype(np.float32)
            image_data = np.expand_dims(image_data, axis=0)

            return image_data
        elif (protocol == 'http'):
            logger.debug("Formatting input for HTTP PaddleOCR model")
            # For HTTP, preprocessing is not necessary
            base64_img = data["base64_image"]
            payload = self._prepare_paddle_payload(base64_img)

            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None) -> Any:
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
            If an invalid protocol is specified or the response format is unexpected.
        """

        if (protocol == 'grpc'):
            logger.debug("Parsing output from gRPC PaddleOCR model")
            return self._extract_content_from_paddle_grpc_response(response)
        elif (protocol == 'http'):
            logger.debug("Parsing output from HTTP PaddleOCR model")
            return self._extract_content_from_paddle_http_response(response)
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs) -> Any:
        """
        Process inference results for the PaddleOCR model.

        Parameters
        ----------
        output : Any
            The raw output from the model.
        kwargs : dict
            Additional parameters for processing.

        Returns
        -------
        Any
            The processed inference results.
        """

        # For PaddleOCR, the output is the table content as a string
        return output

    def _is_version_early_access_legacy_api(self):
        return self.paddle_version and (
            packaging.version.parse(self.paddle_version) < packaging.version.parse("0.2.1-rc2")
        )

    def _prepare_paddle_payload(self, base64_img: str) -> Dict[str, Any]:
        """
        Prepare a payload for the PaddleOCR HTTP API using a base64-encoded image.

        Parameters
        ----------
        base64_img : str
            The base64-encoded image string.

        Returns
        -------
        dict
            The formatted payload for the PaddleOCR API.
        """

        image_url = f"data:image/png;base64,{base64_img}"

        if self._is_version_early_access_legacy_api():
            image = {"type": "image_url", "image_url": {"url": image_url}}
            message = {"content": [image]}
            payload = {"messages": [message]}
        else:
            image = {"type": "image_url", "url": image_url}
            payload = {"input": [image]}

        return payload

    def _extract_content_from_paddle_response(self, json_response: Dict[str, Any]) -> Any:
        """
        Extract content from the JSON response of a PaddleOCR HTTP API request.

        Parameters
        ----------
        json_response : dict
            The JSON response from the PaddleOCR API.

        Returns
        -------
        Any
            The extracted content from the response.

        Raises
        ------
        RuntimeError
            If the response does not contain the expected "data" key or if it is empty.
        """

        if ("data" not in json_response or not json_response["data"]):
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        if self._is_version_early_access_legacy_api():
            content = json_response["data"][0]["content"]
        else:
            text_detections = json_response["data"][0]["text_detections"]
            # "simple" format
            content = " ".join(t["text_prediction"]["text"] for t in text_detections)

        return content

    def _extract_content_from_paddle_grpc_response(self, response):
        # Convert bytes output to string
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        if self._is_version_early_access_legacy_api():
            if response.dtype.type is np.object_ or response.dtype.type is np.bytes_:
                output_strings = [
                    output[0].decode("utf-8") if isinstance(output, bytes) else str(output) for output in response
                ]
                content = " ".join(output_strings)
            else:
                output_bytes = response.tobytes()
                output_str = output_bytes.decode("utf-8")
                content = output_str
        else:
            bboxes_bytestr, texts_bytestr, _ = response
            bboxes_list = json.loads(bboxes_bytestr.decode("utf8"))[0]
            texts_list = json.loads(texts_bytestr.decode("utf8"))[0]
            # "simple" format
            content = " ".join(texts_list)

        return content
