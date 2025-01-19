import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np

from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle

logger = logging.getLogger(__name__)


class PaddleOCRModelInterface(ModelInterface):
    """
    An interface for handling inference with a PaddleOCR model, supporting both gRPC and HTTP protocols.
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

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC PaddleOCR model")
            image_data = data["image_array"]

            image_data, metadata = preprocess_image_for_paddle(image_data)

            # Cache image dimensions for computing bounding boxes.
            self._orig_height = metadata["original_height"]
            self._orig_width = metadata["original_width"]
            self._scale_factor = metadata["scale_factor"]
            self._max_height = metadata["new_height"]
            self._max_width = metadata["new_width"]
            self._pad_height = metadata["pad_height"]
            self._pad_width = metadata["pad_width"]

            image_data = image_data.astype(np.float32)
            image_data = np.expand_dims(image_data, axis=0)

            return image_data
        elif protocol == "http":
            logger.debug("Formatting input for HTTP PaddleOCR model")
            # For HTTP, preprocessing is not necessary
            base64_img = data["base64_image"]
            payload = self._prepare_paddle_payload(base64_img)

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
            If an invalid protocol is specified or the response format is unexpected.
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC PaddleOCR model")

            return self._extract_content_from_paddle_grpc_response(response, data)
        elif protocol == "http":
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

        image = {"type": "image_url", "url": image_url}
        payload = {"input": [image]}

        return payload

    def _extract_content_from_paddle_http_response(
        self,
        json_response: Dict[str, Any],
    ) -> Any:
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

        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        text_detections = json_response["data"][0]["text_detections"]

        text_predictions = []
        bounding_boxes = []
        for text_detection in text_detections:
            text_predictions.append(text_detection["text_prediction"]["text"])
            bounding_boxes.append([(point["x"], point["y"]) for point in text_detection["bounding_box"]["points"]])

        text_predictions, bounding_boxes = self._postprocess_paddle_response(
            text_predictions, bounding_boxes, scale_factor=1.0
        )

        return text_predictions, bounding_boxes

    def _extract_content_from_paddle_grpc_response(self, response, data):
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        bboxes_bytestr, texts_bytestr, _ = response
        bounding_boxes = json.loads(bboxes_bytestr.decode("utf8"))[0]
        text_predictions = json.loads(texts_bytestr.decode("utf8"))[0]

        text_predictions, bounding_boxes = self._postprocess_paddle_response(
            text_predictions, bounding_boxes, scale_factor=self._scale_factor
        )

        return text_predictions, bounding_boxes

    def _postprocess_paddle_response(self, text_predictions, bounding_boxes, scale_factor):
        bboxes = []
        texts = []

        for box, txt in zip(bounding_boxes, text_predictions):
            if box == "nan":
                continue
            points = []
            for point in box:
                # The coordinates from Paddle are normlized. Convert them back to integers,
                # and scale or shift them back to their original positions if padded or scaled.
                x_pixels = float(point[0]) * self._max_width - self._pad_width
                y_pixels = float(point[1]) * self._max_height - self._pad_height
                x_original = x_pixels / scale_factor
                y_original = y_pixels / scale_factor
                points.append([x_original, y_original])
            bboxes.append(points)
            texts.append(txt)

        return texts, bboxes
