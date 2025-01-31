import json
import logging
from typing import Any, List, Tuple
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
        Prepare input data by decoding one or more base64-encoded images into numpy arrays.

        Parameters
        ----------
        data : dict
            The input data containing either:
             - 'base64_image': a single base64-encoded image, OR
             - 'base64_images': a list of base64-encoded images.

        Returns
        -------
        dict
            The updated data dictionary with "image_arrays", a list of decoded image arrays.
        """
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")

            image_arrays = []
            for b64 in base64_list:
                img = base64_to_numpy(b64)
                image_arrays.append(img)
            data["image_arrays"] = image_arrays

        elif "base64_image" in data:
            # Single-image fallback
            img = base64_to_numpy(data["base64_image"])
            data["image_arrays"] = [img]

        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, **kwargs) -> Any:
        """
        Format input data for the specified protocol ("grpc" or "http"),
        now capable of batching multiple images.
        """
        if "image_arrays" not in data:
            raise KeyError("Expected 'image_arrays' in data. Call prepare_data_for_inference first.")

        images = data["image_arrays"]

        data["_dimensions"] = []  # Cache image dimensions information for scale/shift.

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC PaddleOCR model (batched).")

            # For each image in the batch:
            # 1) Preprocess (if needed)
            # 2) Cast to float32
            # 3) Expand dims so shape => (1, H, W, C)
            processed = []
            self._dims = []
            for img in images:
                arr, _dims = preprocess_image_for_paddle(img)
                data["_dimensions"].append(_dims)
                arr = arr.astype(np.float32)
                arr = np.expand_dims(arr, axis=0)  # => shape (1, H, W, C)
                processed.append(arr)

            # Check that all images (beyond the batch dimension) have the same shape
            # If not, raise an error
            shapes = [p.shape[1:] for p in processed]  # List of (H, W, C) shapes
            if not all(s == shapes[0] for s in shapes[1:]):
                raise ValueError(f"All images must have the same dimensions for gRPC batching. " f"Found: {shapes}")

            # Concatenate along the batch dimension => shape (B, H, W, C)
            batched_input = np.concatenate(processed, axis=0)
            return batched_input

        elif protocol == "http":
            logger.debug("Formatting input for HTTP PaddleOCR model (batched).")

            # For HTTP, we build a single payload that includes ALL images.
            # New => {"input":[ {"type":"image_url","url":...}, {"type":"image_url","url":...}, ... ]}
            input_list = []
            base64_list = data.get("base64_images")
            if base64_list is None and "base64_image" in data:
                # fallback to single
                base64_list = [data["base64_image"]]

            for b64 in base64_list:
                image_url = f"data:image/png;base64,{b64}"
                image_obj = {"type": "image_url", "url": image_url}
                input_list.append(image_obj)

            payload = {"input": input_list}

            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the output from the model's inference response. For multi-image gRPC or HTTP,
        """
        if protocol == "grpc":
            logger.debug("Parsing output from gRPC PaddleOCR model (batched).")
            return self._extract_content_from_paddle_grpc_response(response, data["_dimensions"])

        elif protocol == "http":
            logger.debug("Parsing output from HTTP PaddleOCR model (batched).")
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
        DEPRECATED by batch logic in format_input.
        (Kept here if you need single-image direct calls.)
        """
        image_url = f"data:image/png;base64,{base64_img}"

        image = {"type": "image_url", "url": image_url}
        payload = {"input": [image]}

        return payload

    def _extract_content_from_paddle_http_response(
        self,
        json_response: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """
        Extract content from the JSON response of a PaddleOCR HTTP API request.
        Always return a list of (content, table_content_format) tuples.
        """
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        results = []
        for item_idx, item in enumerate(json_response["data"]):
            text_detections = item.get("text_detections", [])
            text_predictions = []
            bounding_boxes = []
            for td in text_detections:
                text_predictions.append(td["text_prediction"]["text"])
                bounding_boxes.append([[pt["x"], pt["y"]] for pt in td["bounding_box"]["points"]])

            results.append([bounding_boxes, text_predictions])

        return results

    def _extract_content_from_paddle_grpc_response(
        self,
        response: np.ndarray,
        dimensions: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        """
        Parses a gRPC response for one or more images.

        The response can have two possible shapes:
          - (3,) for batch_size=1
          - (3, n) for batch_size=n

        In either case:
          response[0, i]: byte string containing bounding box data
          response[1, i]: byte string containing text prediction data
          response[2, i]: (Optional) additional data/metadata (ignored or logged here)

        Returns a list of (content, table_content_format) of length n.
        """
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        # If we have shape (3,), convert to (3,1) so we can handle everything uniformly
        if response.ndim == 1 and response.shape == (3,):
            response = response.reshape(3, 1)
        elif response.ndim != 2 or response.shape[0] != 3:
            raise ValueError(f"Unexpected response shape: {response.shape}. " "Expecting (3,) or (3, n).")

        batch_size = response.shape[1]
        results = []

        for i in range(batch_size):
            # 1) Parse bounding boxes
            bboxes_bytestr = response[0, i]
            bounding_boxes = json.loads(bboxes_bytestr.decode("utf8"))

            # 2) Parse text predictions
            texts_bytestr = response[1, i]
            text_predictions = json.loads(texts_bytestr.decode("utf8"))

            # 3) Optionally handle or log the third element (extra data/metadata)
            extra_data_bytestr = response[2, i]
            logger.debug(f"Ignoring extra_data for image {i}: {extra_data_bytestr}")

            if isinstance(bounding_boxes, list) and len(bounding_boxes) == 1:
                bounding_boxes = bounding_boxes[0]
            if isinstance(text_predictions, list) and len(text_predictions) == 1:
                text_predictions = text_predictions[0]

            bounding_boxes, text_predictions = self._postprocess_paddle_response(
                bounding_boxes,
                text_predictions,
                dimensions,
                img_index=i,
            )

            results.append([bounding_boxes, text_predictions])

        return results

    def _postprocess_paddle_response(
        self,
        bounding_boxes: List[Any],
        text_predictions: List[str],
        dimensions: List[Dict[str, Any]],
        img_index: int = 0,
    ) -> str:
        """
        Convert bounding boxes & text to pseudo-markdown. For multiple images,
        we use self._dims[img_index] to recover the correct height/width.
        """
        max_width = dimensions[img_index]["new_width"]
        max_height = dimensions[img_index]["new_height"]
        pad_width = dimensions[img_index]["pad_width"]
        pad_height = dimensions[img_index]["pad_height"]
        scale_factor = dimensions[img_index]["scale_factor"]

        bboxes = []
        texts = []
        for box, txt in zip(bounding_boxes, text_predictions):
            if box == "nan":
                continue
            points = []
            for point in box:
                # Convert normalized coords back to actual pixel coords,
                # and shift them back to their original positions if padded.
                x_pixels = float(point[0]) * max_width - pad_width
                y_pixels = float(point[1]) * max_height - pad_height
                x_original = x_pixels / scale_factor
                y_original = y_pixels / scale_factor
                points.append([x_original, y_original])
            bboxes.append(points)
            texts.append(txt)

        return bboxes, texts
