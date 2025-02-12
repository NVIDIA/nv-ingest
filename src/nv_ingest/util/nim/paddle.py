import json
import logging
from typing import Any, List, Tuple
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from packaging import version as pkgversion
from sklearn.cluster import DBSCAN

from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import ModelInterface
from nv_ingest.util.nim.helpers import preprocess_image_for_paddle

logger = logging.getLogger(__name__)


class PaddleOCRModelInterface(ModelInterface):
    """
    An interface for handling inference with a PaddleOCR model, supporting both gRPC and HTTP protocols.
    """

    def __init__(self, paddle_version: Optional[str] = None) -> None:
        """
        Initialize the PaddleOCR model interface.

        Parameters
        ----------
        paddle_version : str, optional
            The version of the PaddleOCR model (default is None).
        """
        self.paddle_version: Optional[str] = paddle_version

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
        Decode one or more base64-encoded images into NumPy arrays, storing them
        alongside their dimensions in `data`.

        Parameters
        ----------
        data : dict of str -> Any
            The input data containing either:
             - 'base64_image': a single base64-encoded image, or
             - 'base64_images': a list of base64-encoded images.

        Returns
        -------
        dict of str -> Any
            The updated data dictionary with the following keys added:
            - "image_arrays": List of decoded NumPy arrays of shape (H, W, C).
            - "image_dims": List of (height, width) tuples for each decoded image.

        Raises
        ------
        KeyError
            If neither 'base64_image' nor 'base64_images' is found in `data`.
        ValueError
            If 'base64_images' is present but is not a list.
        """
        if "base64_images" in data:
            base64_list = data["base64_images"]
            if not isinstance(base64_list, list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")

            image_arrays: List[np.ndarray] = []
            dims: List[Tuple[int, int]] = []
            for b64 in base64_list:
                img = base64_to_numpy(b64)
                image_arrays.append(img)
                dims.append((img.shape[0], img.shape[1]))

            data["image_arrays"] = image_arrays
            data["image_dims"] = dims

        elif "base64_image" in data:
            # Single-image fallback
            img = base64_to_numpy(data["base64_image"])
            data["image_arrays"] = [img]
            data["image_dims"] = [(img.shape[0], img.shape[1])]

        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format input data for the specified protocol ("grpc" or "http"), supporting batched data.

        Parameters
        ----------
        data : dict of str -> Any
            The input data dictionary, expected to contain "image_arrays" (list of np.ndarray).
        protocol : str
            The inference protocol, either "grpc" or "http".
        max_batch_size : int
            The maximum batch size batching.

        Returns
        -------
        Any
            A list of formatted batches. For gRPC, each item is a batched NumPy array of shape (B, H, W, C)
            where B <= max_batch_size. For HTTP, each item is a JSON-serializable payload containing the
            base64 images in the format required by the PaddleOCR endpoint.

        Raises
        ------
        KeyError
            If "image_arrays" is not found in `data`.
        ValueError
            If an invalid protocol is specified, or if the image shapes are inconsistent for gRPC batching.
        """
        if "image_arrays" not in data:
            raise KeyError("Expected 'image_arrays' in data. Call prepare_data_for_inference first.")

        images = data["image_arrays"]

        # Helper function to split a list into chunks of size up to chunk_size.
        def chunk_list(lst, chunk_size):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC PaddleOCR model (batched).")

            processed: List[np.ndarray] = []
            for img in images:
                arr = preprocess_image_for_paddle(img, self.paddle_version).astype(np.float32)
                arr = np.expand_dims(arr, axis=0)  # => shape (1, H, W, C)
                processed.append(arr)

            batches = []
            for chunk in chunk_list(processed, max_batch_size):
                # Concatenate arrays in the chunk along the batch dimension => shape (B, H, W, C)
                batched_input = np.concatenate(chunk, axis=0)
                batches.append(batched_input)

            return batches

        elif protocol == "http":
            logger.debug("Formatting input for HTTP PaddleOCR model (batched).")

            # Use legacy or new API based on the PaddleOCR version
            if self._is_version_early_access_legacy_api():
                content_list: List[Dict[str, Any]] = []
                base64_list = data.get("base64_images")
                if base64_list is None and "base64_image" in data:
                    # fallback to single image
                    base64_list = [data["base64_image"]]

                for b64 in base64_list:
                    image_url = f"data:image/png;base64,{b64}"
                    image_obj = {"type": "image_url", "image_url": {"url": image_url}}
                    content_list.append(image_obj)

                batches = []
                for chunk in chunk_list(content_list, max_batch_size):
                    message = {"content": chunk}
                    payload = {"messages": [message]}
                    batches.append(payload)
                return batches

            else:
                input_list: List[Dict[str, Any]] = []
                base64_list = data.get("base64_images")
                if base64_list is None and "base64_image" in data:
                    # fallback to single image
                    base64_list = [data["base64_image"]]

                for b64 in base64_list:
                    image_url = f"data:image/png;base64,{b64}"
                    image_obj = {"type": "image_url", "url": image_url}
                    input_list.append(image_obj)

                batches = []
                for chunk in chunk_list(input_list, max_batch_size):
                    payload = {"input": chunk}
                    batches.append(payload)
                return batches

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Parse the model's inference response for the given protocol. The parsing
        may handle batched outputs for multiple images.

        Parameters
        ----------
        response : Any
            The raw response from the PaddleOCR model.
        protocol : str
            The protocol used for inference, "grpc" or "http".
        data : dict of str -> Any, optional
            Additional data dictionary that may include "image_dims" for bounding box scaling.
        **kwargs : Any
            Additional keyword arguments, such as custom `table_content_format`.

        Returns
        -------
        Any
            The parsed output, typically a list of (content, table_content_format) tuples.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        default_table_content_format = (
            TableFormatEnum.SIMPLE if self._is_version_early_access_legacy_api() else TableFormatEnum.PSEUDO_MARKDOWN
        )
        table_content_format = kwargs.get("table_content_format", default_table_content_format)

        # Enforce legacy constraints
        if self._is_version_early_access_legacy_api() and table_content_format != TableFormatEnum.SIMPLE:
            logger.warning(
                f"Paddle version {self.paddle_version} does not support {table_content_format} format. "
                "The table content will be in `simple` format."
            )
            table_content_format = TableFormatEnum.SIMPLE

        # Retrieve image dimensions if available
        dims: Optional[List[Tuple[int, int]]] = data.get("image_dims") if data else None

        if protocol == "grpc":
            logger.debug("Parsing output from gRPC PaddleOCR model (batched).")
            return self._extract_content_from_paddle_grpc_response(response, table_content_format, dims)

        elif protocol == "http":
            logger.debug("Parsing output from HTTP PaddleOCR model (batched).")
            return self._extract_content_from_paddle_http_response(response, table_content_format, dims)

        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, **kwargs: Any) -> Any:
        """
        Process inference results for the PaddleOCR model.

        Parameters
        ----------
        output : Any
            The raw output parsed from the PaddleOCR model.
        **kwargs : Any
            Additional keyword arguments for customization.

        Returns
        -------
        Any
            The post-processed inference results. By default, this simply returns the output
            as the table content (or content list).
        """
        return output

    def _is_version_early_access_legacy_api(self) -> bool:
        """
        Determine if the current PaddleOCR version is considered "early access" and thus uses
        the legacy API format.

        Returns
        -------
        bool
            True if the version is < 0.2.1-rc2; False otherwise.
        """
        return self.paddle_version is not None and pkgversion.parse(self.paddle_version) < pkgversion.parse("0.2.1-rc2")

    def _prepare_paddle_payload(self, base64_img: str) -> Dict[str, Any]:
        """
        DEPRECATED by batch logic in format_input. Kept here if you need single-image direct calls.

        Parameters
        ----------
        base64_img : str
            A single base64-encoded image string.

        Returns
        -------
        dict of str -> Any
            The payload in either legacy or new format for PaddleOCR's HTTP endpoint.
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

    def _extract_content_from_paddle_http_response(
        self, json_response: Dict[str, Any], table_content_format: Optional[str], dims: Optional[List[Tuple[int, int]]]
    ) -> List[Tuple[str, str]]:
        """
        Extract content from the JSON response of a PaddleOCR HTTP API request.

        Parameters
        ----------
        json_response : dict of str -> Any
            The JSON response returned by the PaddleOCR endpoint.
        table_content_format : str or None
            The specified format for table content (e.g., 'simple' or 'pseudo_markdown').
        dims : list of (int, int), optional
            A list of (height, width) for each corresponding image, used for bounding box
            scaling if not None.

        Returns
        -------
        list of (str, str)
            A list of (content, table_content_format) tuples, one for each image result.

        Raises
        ------
        RuntimeError
            If the response format is missing or invalid.
        ValueError
            If the `table_content_format` is unrecognized.
        """
        if "data" not in json_response or not json_response["data"]:
            raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")

        results: List[str] = []
        for item_idx, item in enumerate(json_response["data"]):
            if self._is_version_early_access_legacy_api():
                content = item.get("content", "")
            else:
                text_detections = item.get("text_detections", [])
                text_predictions: List[str] = []
                bounding_boxes: List[List[Tuple[float, float]]] = []

                for td in text_detections:
                    text_predictions.append(td["text_prediction"]["text"])
                    bounding_boxes.append([(pt["x"], pt["y"]) for pt in td["bounding_box"]["points"]])

                if table_content_format == TableFormatEnum.SIMPLE:
                    content = " ".join(text_predictions)
                elif table_content_format == TableFormatEnum.PSEUDO_MARKDOWN:
                    content = self._convert_paddle_response_to_psuedo_markdown(
                        bounding_boxes, text_predictions, img_index=item_idx, dims=dims
                    )
                else:
                    raise ValueError(f"Unexpected table format: {table_content_format}")

            results.append(content)

        # Convert each content into a tuple (content, format).
        return [(content, table_content_format) for content in results]

    def _extract_content_from_paddle_grpc_response(
        self, response: np.ndarray, table_content_format: str, dims: Optional[List[Tuple[int, int]]]
    ) -> List[Tuple[str, str]]:
        """
        Parse a gRPC response for one or more images. The response can have two possible shapes:
          - (3,) for batch_size=1
          - (3, n) for batch_size=n

        In either case:
          response[0, i]: byte string containing bounding box data
          response[1, i]: byte string containing text prediction data
          response[2, i]: (Optional) additional data/metadata (ignored here)

        Parameters
        ----------
        response : np.ndarray
            The raw NumPy array from gRPC. Expected shape: (3,) or (3, n).
        table_content_format : str
            The format of the output text content, e.g. 'simple' or 'pseudo_markdown'.
        dims : list of (int, int), optional
            A list of (height, width) for each corresponding image, used for bounding box scaling.

        Returns
        -------
        list of (str, str)
            A list of (content, table_content_format) for each image.

        Raises
        ------
        ValueError
            If the response is not a NumPy array or has an unexpected shape,
            or if the `table_content_format` is unrecognized.
        """
        if not isinstance(response, np.ndarray):
            raise ValueError("Unexpected response format: response is not a NumPy array.")

        # If we have shape (3,), convert to (3, 1)
        if response.ndim == 1 and response.shape == (3,):
            response = response.reshape(3, 1)
        elif response.ndim != 2 or response.shape[0] != 3:
            raise ValueError(f"Unexpected response shape: {response.shape}. Expecting (3,) or (3, n).")

        batch_size = response.shape[1]
        results: List[Tuple[str, str]] = []

        for i in range(batch_size):
            # 1) Parse bounding boxes
            bboxes_bytestr: bytes = response[0, i]
            bounding_boxes = json.loads(bboxes_bytestr.decode("utf8"))

            # 2) Parse text predictions
            texts_bytestr: bytes = response[1, i]
            text_predictions = json.loads(texts_bytestr.decode("utf8"))

            # 3) Log the third element (extra data/metadata) if needed
            extra_data_bytestr: bytes = response[2, i]
            logger.debug(f"Ignoring extra_data for image {i}: {extra_data_bytestr}")

            # Some gRPC responses nest single-item lists; flatten them if needed
            if isinstance(bounding_boxes, list) and len(bounding_boxes) == 1:
                bounding_boxes = bounding_boxes[0]
            if isinstance(text_predictions, list) and len(text_predictions) == 1:
                text_predictions = text_predictions[0]

            # Construct the content string
            if table_content_format == TableFormatEnum.SIMPLE:
                content = " ".join(text_predictions)
            elif table_content_format == TableFormatEnum.PSEUDO_MARKDOWN:
                content = self._convert_paddle_response_to_psuedo_markdown(
                    bounding_boxes, text_predictions, img_index=i, dims=dims
                )
            else:
                raise ValueError(f"Unexpected table format: {table_content_format}")

            results.append((content, table_content_format))

        return results

    @staticmethod
    def _convert_paddle_response_to_psuedo_markdown(
        bounding_boxes: List[Any],
        text_predictions: List[str],
        img_index: int = 0,
        dims: Optional[List[Tuple[int, int]]] = None,
    ) -> str:
        """
        Convert bounding boxes & text to pseudo-markdown format. For multiple images,
        the correct image dimensions (height, width) are retrieved from `dims[img_index]`.

        Parameters
        ----------
        bounding_boxes : list of Any
            A list (per line of text) of bounding boxes, each a list of (x, y) points.
        text_predictions : list of str
            A list of text predictions, one for each bounding box.
        img_index : int, optional
            The index of the image for which bounding boxes are being converted. Default is 0.
        dims : list of (int, int), optional
            A list of (height, width) for each corresponding image.

        Returns
        -------
        str
            The pseudo-markdown representation of detected text lines and bounding boxes.
            Each cluster of text is placed on its own line, with text columns separated by '|'.

        Notes
        -----
        - If `dims` is None or `img_index` is out of range, bounding boxes will not be scaled properly.
        """
        # Default to no scaling if dims are missing or out of range
        if not dims:
            logger.warning("No image_dims provided; bounding boxes will not be scaled.")
            target_h, target_w = 1, 1
        else:
            if img_index >= len(dims):
                logger.warning("Image index out of range for stored dimensions. Using first image dims by default.")
                target_h, target_w = dims[0]
            else:
                target_h, target_w = dims[img_index]

        scaled_boxes: List[List[float]] = []
        texts: List[str] = []

        # Convert normalized coords back to actual pixel coords
        for box, txt in zip(bounding_boxes, text_predictions):
            if box == "nan":
                continue
            points: List[List[float]] = []
            for point in box:
                x = float(point[0]) * target_w
                y = float(point[1]) * target_h
                points.append([x, y])
            scaled_boxes.append(points)
            texts.append(txt)

        if not scaled_boxes or not texts:
            return ""

        # Convert bounding boxes to a simplified (x0, y0, x1, y1) representation
        # by taking only the top-left and bottom-right corners
        bboxes_array = np.array(scaled_boxes).astype(int)
        # Reshape => (N, 4) by taking [0,1, 2,3] from the original (N, 4, 2)
        # but we have 4 corners => 8 values. So shape => (N, 8). Then keep indices [0, 1, 2, 7].
        bboxes_array = bboxes_array.reshape(-1, 8)[:, [0, 1, 2, -1]]

        preds_df = pd.DataFrame(
            {
                "x0": bboxes_array[:, 0],
                "y0": bboxes_array[:, 1],
                "x1": bboxes_array[:, 2],
                "y1": bboxes_array[:, 3],
                "text": texts,
            }
        )
        # Sort by top position
        preds_df = preds_df.sort_values("y0")

        dbscan = DBSCAN(eps=10, min_samples=1)
        dbscan.fit(preds_df["y0"].values[:, None])

        preds_df["cluster"] = dbscan.labels_
        # Sort by cluster and then by x0 to group text lines from left to right
        preds_df = preds_df.sort_values(["cluster", "x0"])

        lines = []
        for _, dfg in preds_df.groupby("cluster"):
            lines.append("| " + " | ".join(dfg["text"].values.tolist()) + " |")

        return "\n".join(lines)
