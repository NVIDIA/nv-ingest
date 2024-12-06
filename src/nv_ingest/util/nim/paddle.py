from typing import Dict, Any, Optional
import numpy as np
import logging

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface, preprocess_image_for_paddle

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
        base64_image = data['base64_image']
        image_array = base64_to_numpy(base64_image)
        data['image_array'] = image_array
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
            base64_img = data['base64_image']
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
        elif (protocol == 'http'):
            logger.debug("Parsing output from HTTP PaddleOCR model")
            return self._extract_content_from_paddle_response(response)
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
        image = {"type": "image_url", "image_url": {"url": image_url}}

        message = {"content": [image]}
        payload = {"messages": [message]}

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

        return json_response["data"][0]["content"]
