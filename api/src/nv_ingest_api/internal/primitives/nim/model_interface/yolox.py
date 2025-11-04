# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import warnings
from math import log
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import backoff
import numpy as np
import json
import pandas as pd

from nv_ingest_api.internal.primitives.nim import ModelInterface
import tritonclient.grpc as grpcclient
from nv_ingest_api.internal.primitives.nim.model_interface.decorators import multiprocessing_cache
from nv_ingest_api.internal.primitives.nim.model_interface.helpers import get_model_name
from nv_ingest_api.util.image_processing import scale_image_to_encoding_size
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64

logger = logging.getLogger(__name__)

YOLOX_PAGE_DEFAULT_VERSION = "nemoretriever-page-elements-v2"

# yolox-page-elements-v2 and v3 common contants
YOLOX_PAGE_CONF_THRESHOLD = 0.01
YOLOX_PAGE_IOU_THRESHOLD = 0.5
YOLOX_PAGE_MIN_SCORE = 0.1
YOLOX_PAGE_NIM_MAX_IMAGE_SIZE = 512_000
YOLOX_PAGE_IMAGE_PREPROC_HEIGHT = 1024
YOLOX_PAGE_IMAGE_PREPROC_WIDTH = 1024
YOLOX_PAGE_IMAGE_FORMAT = os.getenv("YOLOX_PAGE_IMAGE_FORMAT", "PNG")

# yolox-page-elements-v3 contants
YOLOX_PAGE_FINAL_SCORE = YOLOX_PAGE_V3_FINAL_SCORE = {
    "table": 0.1,
    "chart": 0.01,
    "title": 0.1,
    "infographic": 0.01,
    "paragraph": 0.1,
    "header_footer": 0.1,
}
YOLOX_PAGE_CLASS_LABELS = YOLOX_PAGE_V3_CLASS_LABELS = [
    "table",
    "chart",
    "title",
    "infographic",
    "paragraph",
    "header_footer",
]

# yolox-page-elements-v2 contants
YOLOX_PAGE_V2_FINAL_SCORE = {"table": 0.1, "chart": 0.01, "infographic": 0.01}
YOLOX_PAGE_V2_CLASS_LABELS = [
    "table",
    "chart",
    "title",
    "infographic",
]


# yolox-graphic-elements-v1 contants
YOLOX_GRAPHIC_CONF_THRESHOLD = 0.01
YOLOX_GRAPHIC_IOU_THRESHOLD = 0.25
YOLOX_GRAPHIC_MIN_SCORE = 0.1
YOLOX_GRAPHIC_NIM_MAX_IMAGE_SIZE = 512_000


YOLOX_GRAPHIC_CLASS_LABELS = [
    "chart_title",
    "x_title",
    "y_title",
    "xlabel",
    "ylabel",
    "other",
    "legend_label",
    "legend_title",
    "mark_label",
    "value_label",
]


# yolox-table-structure-v1 contants
YOLOX_TABLE_CONF_THRESHOLD = 0.01
YOLOX_TABLE_IOU_THRESHOLD = 0.25
YOLOX_TABLE_MIN_SCORE = 0.1
YOLOX_TABLE_NIM_MAX_IMAGE_SIZE = 512_000

YOLOX_TABLE_IMAGE_PREPROC_HEIGHT = 1024
YOLOX_TABLE_IMAGE_PREPROC_WIDTH = 1024

YOLOX_TABLE_CLASS_LABELS = [
    "border",
    "cell",
    "row",
    "column",
    "header",
]


# YoloxModelInterfaceBase implements methods that are common to yolox-page-elements and yolox-graphic-elements
class YoloxModelInterfaceBase(ModelInterface):
    """
    An interface for handling inference with a Yolox object detection model, supporting both gRPC and HTTP protocols.
    """

    def __init__(
        self,
        nim_max_image_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
    ):
        """
        Initialize the YOLOX model interface.
        Parameters
        ----------
        """
        self.nim_max_image_size = nim_max_image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.class_labels = class_labels

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
        if (not isinstance(data, dict)) or ("images" not in data):
            raise KeyError("Input data must be a dictionary containing an 'images' key with a list of images.")

        if not all(isinstance(x, np.ndarray) for x in data["images"]):
            raise ValueError("All elements in the 'images' list must be numpy.ndarray objects.")

        original_images = data["images"]
        data["original_image_shapes"] = [image.shape for image in original_images]

        return data

    def format_input(
        self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Format input data for the specified protocol, returning a tuple of:
          (formatted_batches, formatted_batch_data)
        where:
          - For gRPC: formatted_batches is a list of NumPy arrays, each of shape (B, H, W, C)
            with B <= max_batch_size.
          - For HTTP: formatted_batches is a list of JSON-serializable dict payloads.
          - In both cases, formatted_batch_data is a list of dicts that coalesce the original
            images and their original shapes in the same order as provided.

        Parameters
        ----------
        data : dict
            The input data to format. Must include:
              - "images": a list of numpy.ndarray images.
              - "original_image_shapes": a list of tuples with each image's (height, width),
                as set by prepare_data_for_inference.
        protocol : str
            The protocol to use ("grpc" or "http").
        max_batch_size : int
            The maximum number of images per batch.

        Returns
        -------
        tuple
            A tuple (formatted_batches, formatted_batch_data).

        Raises
        ------
        ValueError
            If the protocol is invalid.
        """

        # Helper functions to chunk a list into sublists of length up to chunk_size.
        def chunk_list(lst: list, chunk_size: int) -> List[list]:
            chunk_size = max(1, chunk_size)
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        def chunk_list_geometrically(lst: list, max_size: int) -> List[list]:
            # TRT engine in Yolox NIM (gRPC) only allows a batch size in powers of 2.
            chunks = []
            i = 0
            while i < len(lst):
                chunk_size = max(1, min(2 ** int(log(len(lst) - i, 2)), max_size))
                chunks.append(lst[i : i + chunk_size])
                i += chunk_size
            return chunks

        if protocol == "grpc":
            logger.debug("Formatting input for gRPC Yolox Ensemble model")
            b64_images = [numpy_to_base64(image, format=YOLOX_PAGE_IMAGE_FORMAT) for image in data["images"]]
            b64_chunks = chunk_list_geometrically(b64_images, max_batch_size)
            original_chunks = chunk_list_geometrically(data["images"], max_batch_size)
            shape_chunks = chunk_list_geometrically(data["original_image_shapes"], max_batch_size)

            batched_inputs = []
            formatted_batch_data = []
            for b64_chunk, orig_chunk, shapes in zip(b64_chunks, original_chunks, shape_chunks):
                input_array = np.array(b64_chunk, dtype=np.object_)
                current_batch_size = input_array.shape[0]
                single_threshold_pair = [self.conf_threshold, self.iou_threshold]
                thresholds = np.tile(single_threshold_pair, (current_batch_size, 1)).astype(np.float32)
                batched_inputs.append([input_array, thresholds])
                formatted_batch_data.append({"images": orig_chunk, "original_image_shapes": shapes})

            return batched_inputs, formatted_batch_data

        elif protocol == "http":
            logger.debug("Formatting input for HTTP Yolox model")
            content_list: List[Dict[str, Any]] = []
            for image in data["images"]:
                # Convert to uint8 if needed.
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)

                # Get original size directly from numpy array (width, height)
                original_size = (image.shape[1], image.shape[0])
                # Convert numpy array directly to base64 using OpenCV
                image_b64 = numpy_to_base64(image, format=YOLOX_PAGE_IMAGE_FORMAT)
                # Scale the image if necessary.
                scaled_image_b64, new_size = scale_image_to_encoding_size(
                    image_b64, max_base64_size=self.nim_max_image_size
                )
                if new_size != original_size:
                    logger.debug(f"Image was scaled from {original_size} to {new_size}.")

                content_list.append({"type": "image_url", "url": f"data:image/png;base64,{scaled_image_b64}"})

            # Chunk the payload content, the original images, and their shapes.
            content_chunks = chunk_list(content_list, max_batch_size)
            original_chunks = chunk_list(data["images"], max_batch_size)
            shape_chunks = chunk_list(data["original_image_shapes"], max_batch_size)

            payload_batches = []
            formatted_batch_data = []
            for chunk, orig_chunk, shapes in zip(content_chunks, original_chunks, shape_chunks):
                payload = {"input": chunk}
                payload_batches.append(payload)
                formatted_batch_data.append({"images": orig_chunk, "original_image_shapes": shapes})
            return payload_batches, formatted_batch_data

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
            logger.debug("Parsing output from gRPC Yolox model")
            return response  # For gRPC, response is already a numpy array
        elif protocol == "http":
            logger.debug("Parsing output from HTTP Yolox model")

            processed_outputs = []

            batch_results = response.get("data", [])
            for detections in batch_results:
                new_bounding_boxes = {label: [] for label in self.class_labels}

                bounding_boxes = detections.get("bounding_boxes", [])
                for obj_type, bboxes in bounding_boxes.items():
                    for bbox in bboxes:
                        xmin = bbox["x_min"]
                        ymin = bbox["y_min"]
                        xmax = bbox["x_max"]
                        ymax = bbox["y_max"]
                        confidence = bbox["confidence"]

                        new_bounding_boxes[obj_type].append([xmin, ymin, xmax, ymax, confidence])

                processed_outputs.append(new_bounding_boxes)

            return processed_outputs
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Process the results of the Yolox model inference and return the final annotations.

        Parameters
        ----------
        output_array : np.ndarray
            The raw output from the Yolox model.
        kwargs : dict
            Additional parameters for processing, including thresholds and number of classes.

        Returns
        -------
        list[dict]
            A list of annotation dictionaries for each image in the batch.
        """
        if protocol == "http":
            # For http, the output already has postprocessing applied. Skip to table/chart expansion.
            results = output

        elif protocol == "grpc":
            results = []
            # For grpc, apply the same NIM postprocessing.
            for out in output:
                if isinstance(out, bytes):
                    out = out.decode("utf-8")
                if isinstance(out, dict):
                    continue
                results.append(json.loads(out))
        inference_results = self.postprocess_annotations(results, **kwargs)
        return inference_results

    def postprocess_annotations(self, annotation_dicts, **kwargs):
        raise NotImplementedError()

    def transform_normalized_coordinates_to_original(self, results, original_image_shapes):
        """ """
        transformed_results = []

        for annotation_dict, shape in zip(results, original_image_shapes):
            new_dict = {}
            for label, bboxes_and_scores in annotation_dict.items():
                new_dict[label] = []
                for bbox_and_score in bboxes_and_scores:
                    bbox = bbox_and_score[:4]
                    transformed_bbox = [
                        bbox[0] * shape[1],
                        bbox[1] * shape[0],
                        bbox[2] * shape[1],
                        bbox[3] * shape[0],
                    ]
                    transformed_bbox += bbox_and_score[4:]
                    new_dict[label].append(transformed_bbox)
            transformed_results.append(new_dict)

        return transformed_results


class YoloxPageElementsModelInterface(YoloxModelInterfaceBase):
    """
    An interface for handling inference with yolox-page-elements model, supporting both gRPC and HTTP protocols.
    """

    def __init__(self, version: str = YOLOX_PAGE_DEFAULT_VERSION):
        """
        Initialize the yolox-page-elements model interface.
        """
        if version.endswith("-v3"):
            class_labels = YOLOX_PAGE_V3_CLASS_LABELS
        else:
            class_labels = YOLOX_PAGE_V2_CLASS_LABELS

        super().__init__(
            nim_max_image_size=YOLOX_PAGE_NIM_MAX_IMAGE_SIZE,
            conf_threshold=YOLOX_PAGE_CONF_THRESHOLD,
            iou_threshold=YOLOX_PAGE_IOU_THRESHOLD,
            min_score=YOLOX_PAGE_MIN_SCORE,
            class_labels=class_labels,
        )

    def name(
        self,
    ) -> str:
        """
        Returns the name of the Yolox model interface.

        Returns
        -------
        str
            The name of the model interface.
        """

        return "yolox-page-elements"

    def postprocess_annotations(self, annotation_dicts, final_score=None, **kwargs):
        original_image_shapes = kwargs.get("original_image_shapes", [])

        running_v3 = annotation_dicts and set(YOLOX_PAGE_V3_CLASS_LABELS) <= annotation_dicts[0].keys()

        if not final_score:
            if running_v3:
                final_score = YOLOX_PAGE_V3_FINAL_SCORE
            else:
                final_score = YOLOX_PAGE_V2_FINAL_SCORE

        if running_v3:
            expected_final_score_keys = YOLOX_PAGE_V3_FINAL_SCORE
        else:
            expected_final_score_keys = [x for x in YOLOX_PAGE_V2_FINAL_SCORE if x != "title"]

        if (not isinstance(final_score, dict)) or (sorted(final_score.keys()) != sorted(expected_final_score_keys)):
            raise ValueError(
                "yolox-page-elements requires a dictionary of thresholds per each class: "
                f"{expected_final_score_keys}"
            )

        if annotation_dicts and running_v3:
            annotation_dicts = [
                postprocess_page_elements_v3(annotation_dict, labels=YOLOX_PAGE_V3_CLASS_LABELS)
                for annotation_dict in annotation_dicts
            ]
        else:
            # Table/chart expansion is "business logic" specific to nv-ingest
            annotation_dicts = [expand_table_bboxes(annotation_dict) for annotation_dict in annotation_dicts]
            annotation_dicts = [expand_chart_bboxes(annotation_dict) for annotation_dict in annotation_dicts]

        inference_results = []

        # Filter out bounding boxes below the final threshold
        # This final thresholding is "business logic" specific to nv-ingest
        for annotation_dict in annotation_dicts:
            new_dict = {}
            if running_v3:
                for label in YOLOX_PAGE_V3_CLASS_LABELS:
                    if label in annotation_dict and label in final_score:
                        threshold = final_score[label]
                        new_dict[label] = [bb for bb in annotation_dict[label] if bb[4] >= threshold]
            else:
                for label in YOLOX_PAGE_V2_CLASS_LABELS:
                    if label in annotation_dict:
                        if label == "title":
                            new_dict[label] = annotation_dict[label]
                        elif label in final_score:
                            threshold = final_score[label]
                            new_dict[label] = [bb for bb in annotation_dict[label] if bb[4] >= threshold]

            inference_results.append(new_dict)

        inference_results = self.transform_normalized_coordinates_to_original(inference_results, original_image_shapes)

        return inference_results


class YoloxGraphicElementsModelInterface(YoloxModelInterfaceBase):
    """
    An interface for handling inference with yolox-graphic-elemenents model, supporting both gRPC and HTTP protocols.
    """

    def __init__(self):
        """
        Initialize the yolox-graphic-elements model interface.
        """
        super().__init__(
            nim_max_image_size=YOLOX_GRAPHIC_NIM_MAX_IMAGE_SIZE,
            conf_threshold=YOLOX_GRAPHIC_CONF_THRESHOLD,
            iou_threshold=YOLOX_GRAPHIC_IOU_THRESHOLD,
            min_score=YOLOX_GRAPHIC_MIN_SCORE,
            class_labels=YOLOX_GRAPHIC_CLASS_LABELS,
        )

    def name(
        self,
    ) -> str:
        """
        Returns the name of the Yolox model interface.

        Returns
        -------
        str
            The name of the model interface.
        """

        return "yolox-graphic-elements"

    def postprocess_annotations(self, annotation_dicts, **kwargs):
        original_image_shapes = kwargs.get("original_image_shapes", [])

        annotation_dicts = self.transform_normalized_coordinates_to_original(annotation_dicts, original_image_shapes)

        inference_results = []

        # bbox extraction: additional postprocessing speicifc to nv-ingest
        for pred, shape in zip(annotation_dicts, original_image_shapes):
            bbox_dict = get_bbox_dict_yolox_graphic(
                pred,
                shape,
                self.class_labels,
                self.min_score,
            )
            # convert numpy arrays to list
            bbox_dict = {
                label: array.tolist() if isinstance(array, np.ndarray) else array for label, array in bbox_dict.items()
            }
            inference_results.append(bbox_dict)

        return inference_results


class YoloxTableStructureModelInterface(YoloxModelInterfaceBase):
    """
    An interface for handling inference with yolox-graphic-elemenents model, supporting both gRPC and HTTP protocols.
    """

    def __init__(self):
        """
        Initialize the yolox-graphic-elements model interface.
        """
        super().__init__(
            nim_max_image_size=YOLOX_TABLE_NIM_MAX_IMAGE_SIZE,
            conf_threshold=YOLOX_TABLE_CONF_THRESHOLD,
            iou_threshold=YOLOX_TABLE_IOU_THRESHOLD,
            min_score=YOLOX_TABLE_MIN_SCORE,
            class_labels=YOLOX_TABLE_CLASS_LABELS,
        )

    def name(
        self,
    ) -> str:
        """
        Returns the name of the Yolox model interface.

        Returns
        -------
        str
            The name of the model interface.
        """

        return "yolox-table-structure"

    def postprocess_annotations(self, annotation_dicts, **kwargs):
        original_image_shapes = kwargs.get("original_image_shapes", [])

        annotation_dicts = self.transform_normalized_coordinates_to_original(annotation_dicts, original_image_shapes)

        inference_results = []

        # bbox extraction: additional postprocessing speicifc to nv-ingest
        for pred, shape in zip(annotation_dicts, original_image_shapes):
            bbox_dict = get_bbox_dict_yolox_table(
                pred,
                shape,
                self.class_labels,
                self.min_score,
            )
            # convert numpy arrays to list
            bbox_dict = {
                label: array.tolist() if isinstance(array, np.ndarray) else array for label, array in bbox_dict.items()
            }
            inference_results.append(bbox_dict)

        return inference_results


def expand_table_bboxes(annotation_dict, labels=None):
    """
    Additional preprocessing for tables: extend the upper bounds to capture titles if any.
    Args:
        annotation_dict: output of postprocess_results, a dictionary with keys "table", "figure", "title"

    Returns:
        annotation_dict: same as input, with expanded bboxes for charts

    """
    if not labels:
        labels = list(annotation_dict.keys())

    if not annotation_dict or len(annotation_dict["table"]) == 0:
        return annotation_dict

    new_annotation_dict = {label: [] for label in labels}

    for label, bboxes in annotation_dict.items():
        for bbox_and_score in bboxes:
            bbox, score = bbox_and_score[:4], bbox_and_score[4]

            if label == "table":
                height = bbox[3] - bbox[1]
                bbox[1] = max(0.0, min(1.0, bbox[1] - height * 0.2))

            new_annotation_dict[label].append([round(float(x), 4) for x in bbox + [score]])

    return new_annotation_dict


def expand_chart_bboxes(annotation_dict, labels=None):
    """
    Expand bounding boxes of charts and titles based on the bounding boxes of the other class.
    Args:
        annotation_dict: output of postprocess_results, a dictionary with keys "table", "figure", "title"

    Returns:
        annotation_dict: same as input, with expanded bboxes for charts

    """
    if not labels:
        labels = list(annotation_dict.keys())

    if not annotation_dict or len(annotation_dict["chart"]) == 0:
        return annotation_dict

    bboxes = []
    confidences = []
    label_idxs = []
    for i, label in enumerate(labels):
        label_annotations = np.array(annotation_dict[label])

        if len(label_annotations) > 0:
            bboxes.append(label_annotations[:, :4])
            confidences.append(label_annotations[:, 4])
            label_idxs.append(np.full(len(label_annotations), i))
    bboxes = np.concatenate(bboxes)
    confidences = np.concatenate(confidences)
    label_idxs = np.concatenate(label_idxs)

    pred_wbf, confidences_wbf, labels_wbf = weighted_boxes_fusion(
        bboxes[:, None],
        confidences[:, None],
        label_idxs[:, None],
        merge_type="biggest",
        conf_type="max",
        iou_thr=0.01,
        class_agnostic=False,
    )

    chart_bboxes = pred_wbf[labels_wbf == 1]
    chart_confidences = confidences_wbf[labels_wbf == 1]
    title_bboxes = pred_wbf[labels_wbf == 2]

    found_title_idxs, no_found_title_idxs = [], []
    for i in range(len(chart_bboxes)):
        match = match_with_title_v1(chart_bboxes[i], title_bboxes, iou_th=0.01)
        if match is not None:
            chart_bboxes[i] = match[0]
            title_bboxes = match[1]
            found_title_idxs.append(i)
        else:
            no_found_title_idxs.append(i)

    chart_bboxes[found_title_idxs] = expand_boxes_v1(chart_bboxes[found_title_idxs], r_x=1.05, r_y=1.1)
    chart_bboxes[no_found_title_idxs] = expand_boxes_v1(chart_bboxes[no_found_title_idxs], r_x=1.1, r_y=1.25)

    annotation_dict = {
        "table": annotation_dict["table"],
        "chart": np.concatenate([chart_bboxes, chart_confidences[:, None]], axis=1).tolist(),
        "title": annotation_dict["title"],
    }

    return annotation_dict


def postprocess_page_elements_v3(annotation_dict, labels=None):
    """
    Expand bounding boxes of tables/charts/infographics and titles based on the bounding boxes of the other class.
    Args:
        annotation_dict: output of postprocess_results, a dictionary with keys:
        "table", "chart", "infographics", "title", "paragraph", "header_footer".

    Returns:
        annotation_dict: same as input, with expanded bboxes for page elements.

    """
    if not labels:
        labels = list(annotation_dict.keys())

    if not annotation_dict:
        return annotation_dict

    bboxes = []
    confidences = []
    label_idxs = []

    for i, label in enumerate(labels):
        if label not in annotation_dict:
            continue

        label_annotations = np.array(annotation_dict[label])

        if len(label_annotations) > 0:
            bboxes.append(label_annotations[:, :4])
            confidences.append(label_annotations[:, 4])
            label_idxs.append(np.full(len(label_annotations), i))

    if not bboxes:
        return annotation_dict

    bboxes = np.concatenate(bboxes)
    confidences = np.concatenate(confidences)
    label_idxs = np.concatenate(label_idxs)

    bboxes, confidences, label_idxs = remove_overlapping_boxes_using_wbf(bboxes, confidences, label_idxs)
    bboxes, confidences, label_idxs, found_title = match_structured_boxes_with_title(
        bboxes, confidences, label_idxs, labels
    )
    bboxes, confidences, label_idxs = expand_tables_and_charts(bboxes, confidences, label_idxs, labels, found_title)
    bboxes, confidences, label_idxs = postprocess_included_texts(bboxes, confidences, label_idxs, labels)

    order = np.argsort(bboxes[:, 1] * 10 + bboxes[:, 0])
    bboxes, confidences, label_idxs = bboxes[order], confidences[order], label_idxs[order]

    new_annotation_dict = {}
    for i, label in enumerate(labels):
        selected_bboxes = bboxes[label_idxs == i]
        selected_confidences = confidences[label_idxs == i]
        new_annotation_dict[label] = np.concatenate([selected_bboxes, selected_confidences[:, None]], axis=1).tolist()

    return new_annotation_dict


def weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    iou_thr=0.5,
    skip_box_thr=0.0,
    conf_type="avg",
    merge_type="weighted",
    class_agnostic=False,
):
    """
    Custom wbf implementation that supports a class_agnostic mode and a biggest box fusion.
    Boxes are expected to be in normalized (x0, y0, x1, y1) format.

    Args:
        boxes_list (list[np array[n x 4]]): List of boxes. One list per model.
        scores_list (list[np array[n]]): List of confidences.
        labels_list (list[np array[n]]): List of labels
        iou_thr (float, optional): IoU threshold for matching. Defaults to 0.55.
        skip_box_thr (float, optional): Exclude boxes with score < skip_box_thr. Defaults to 0.0.
        conf_type (str, optional): Confidence merging type. Defaults to "avg".
        merge_type (str, optional): Merge type "weighted" or "biggest". Defaults to "weighted".
        class_agnostic (bool, optional): If True, merge boxes from different classes. Defaults to False.

    Returns:
        np array[N x 4]: Merged boxes,
        np array[N]: Merged confidences,
        np array[N]: Merged labels.
    """
    weights = np.ones(len(boxes_list))

    assert conf_type in ["avg", "max"], 'Conf type must be "avg" or "max"'
    assert merge_type in [
        "weighted",
        "biggest",
    ], 'Conf type must be "weighted" or "biggest"'

    filtered_boxes = prefilter_boxes(
        boxes_list,
        scores_list,
        labels_list,
        weights,
        skip_box_thr,
        class_agnostic=class_agnostic,
    )
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        np.empty((0, 8))

        clusters = []

        # Clusterize boxes
        for j in range(len(boxes)):
            ids = [i for i in range(len(boxes)) if i != j]
            index, best_iou = find_matching_box_fast(boxes[ids], boxes[j], iou_thr)

            if index != -1:
                index = ids[index]
                cluster_idx = [clust_idx for clust_idx, clust in enumerate(clusters) if (j in clust or index in clust)]
                if len(cluster_idx):
                    cluster_idx = cluster_idx[0]
                    clusters[cluster_idx] = list(set(clusters[cluster_idx] + [index, j]))
                else:
                    clusters.append([index, j])
            else:
                clusters.append([j])

        for j, c in enumerate(clusters):
            if merge_type == "weighted":
                weighted_box = get_weighted_box(boxes[c], conf_type)
            elif merge_type == "biggest":
                weighted_box = get_biggest_box(boxes[c], conf_type)

            if conf_type == "max":
                weighted_box[1] = weighted_box[1] / weights.max()
            else:  # avg
                weighted_box[1] = weighted_box[1] * len(c) / weights.sum()
            overall_boxes.append(weighted_box)

    overall_boxes = np.array(overall_boxes)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


def prefilter_boxes(boxes, scores, labels, weights, thr, class_agnostic=False):
    """
    Reformats and filters boxes.
    Output is a dict of boxes to merge separately.

    Args:
        boxes (list[np array[n x 4]]): List of boxes. One list per model.
        scores (list[np array[n]]): List of confidences.
        labels (list[np array[n]]): List of labels.
        weights (list): Model weights.
        thr (float): Confidence threshold
        class_agnostic (bool, optional): If True, merge boxes from different classes. Defaults to False.

    Returns:
        dict[np array [? x 8]]: Filtered boxes.
    """
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):
        if len(boxes[t]) != len(scores[t]):
            print(
                "Error. Length of boxes arrays not equal to length of scores array: {} != {}".format(
                    len(boxes[t]), len(scores[t])
                )
            )
            exit()

        if len(boxes[t]) != len(labels[t]):
            print(
                "Error. Length of boxes arrays not equal to length of labels array: {} != {}".format(
                    len(boxes[t]), len(labels[t])
                )
            )
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn("X2 < X1 value in box. Swap them.")
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn("Y2 < Y1 value in box. Swap them.")
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn("X1 < 0 in box. Set it to 0.")
                x1 = 0
            if x1 > 1:
                warnings.warn("X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                x1 = 1
            if x2 < 0:
                warnings.warn("X2 < 0 in box. Set it to 0.")
                x2 = 0
            if x2 > 1:
                warnings.warn("X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                x2 = 1
            if y1 < 0:
                warnings.warn("Y1 < 0 in box. Set it to 0.")
                y1 = 0
            if y1 > 1:
                warnings.warn("Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                y1 = 1
            if y2 < 0:
                warnings.warn("Y2 < 0 in box. Set it to 0.")
                y2 = 0
            if y2 > 1:
                warnings.warn("Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.")
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]

            label_k = "*" if class_agnostic else label
            if label_k not in new_boxes:
                new_boxes[label_k] = []
            new_boxes[label_k].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
    Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
    (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """

    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    ious = bb_iou_array(boxes_list[:, 4:], new_box[4:])
    # ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def get_biggest_box(boxes, conf_type="avg"):
    """
    Merges boxes by using the biggest box.

    Args:
        boxes (np array [n x 8]): Boxes to merge.
        conf_type (str, optional): Confidence merging type. Defaults to "avg".

    Returns:
        np array [8]: Merged box.
    """
    box = np.zeros(8, dtype=np.float32)
    box[4:] = boxes[0][4:]
    conf_list = []
    w = 0
    for b in boxes:
        box[4] = min(box[4], b[4])
        box[5] = min(box[5], b[5])
        box[6] = max(box[6], b[6])
        box[7] = max(box[7], b[7])
        conf_list.append(b[1])
        w += b[2]

    box[0] = merge_labels(np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]))
    #     print(box[0], np.array([b[0] for b in boxes]))

    box[1] = np.max(conf_list) if conf_type == "max" else np.mean(conf_list)
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    return box


def merge_labels(labels, confs):
    """
    Custom function for merging labels.
    If all labels are the same, return the unique value.
    Else, return the label of the most confident non-title (class 2) box.

    Args:
        labels (np array [n]): Labels.
        confs (np array [n]): Confidence.

    Returns:
        int: Label.
    """
    if len(np.unique(labels)) == 1:
        return labels[0]
    else:  # Most confident and not a title
        confs = confs[confs != 2]
        labels = labels[labels != 2]
        return labels[np.argmax(confs)]


def match_with_title_v1(chart_bbox, title_bboxes, iou_th=0.01):
    if not len(title_bboxes):
        return None

    dist_above = np.abs(title_bboxes[:, 3] - chart_bbox[1])
    dist_below = np.abs(chart_bbox[3] - title_bboxes[:, 1])

    dist_left = np.abs(title_bboxes[:, 0] - chart_bbox[0])

    ious = bb_iou_array(title_bboxes, chart_bbox)

    matches = None
    if np.max(ious) > iou_th:
        matches = np.where(ious > iou_th)[0]
    else:
        dists = np.min([dist_above, dist_below], 0)
        dists += dist_left
        #         print(dists)
        if np.min(dists) < 0.1:
            matches = [np.argmin(dists)]

    if matches is not None:
        new_bbox = chart_bbox
        for match in matches:
            new_bbox = merge_boxes(new_bbox, title_bboxes[match])
        title_bboxes = title_bboxes[[i for i in range(len(title_bboxes)) if i not in matches]]
        return new_bbox, title_bboxes

    else:
        return None


def bb_iou_array(boxes, new_box):
    # bb interesection over union
    xA = np.maximum(boxes[:, 0], new_box[0])
    yA = np.maximum(boxes[:, 1], new_box[1])
    xB = np.minimum(boxes[:, 2], new_box[2])
    yB = np.minimum(boxes[:, 3], new_box[3])

    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


def merge_boxes(b1, b2):
    b = b1.copy()
    b[0] = min(b1[0], b2[0])
    b[1] = min(b1[1], b2[1])
    b[2] = max(b1[2], b2[2])
    b[3] = max(b1[3], b2[3])
    return b


def expand_boxes_v1(boxes, r_x=1, r_y=1):
    dw = (boxes[:, 2] - boxes[:, 0]) / 2 * (r_x - 1)
    boxes[:, 0] -= dw
    boxes[:, 2] += dw

    dh = (boxes[:, 3] - boxes[:, 1]) / 2 * (r_y - 1)
    boxes[:, 1] -= dh
    boxes[:, 3] += dh

    boxes = np.clip(boxes, 0, 1)
    return boxes


def get_weighted_box(boxes, conf_type="avg"):
    """
    Merges boxes by using the weighted fusion.

    Args:
        boxes (np array [n x 8]): Boxes to merge.
        conf_type (str, optional): Confidence merging type. Defaults to "avg".

    Returns:
        np array [8]: Merged box.
    """
    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += b[1] * b[4:]
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]

    box[0] = merge_labels(np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]))

    box[1] = np.max(conf_list) if conf_type == "max" else np.mean(conf_list)
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def batched_overlaps(A, B):
    """
    Calculate the Intersection over Union (IoU) between
    two sets of bounding boxes in a batched manner.
    Normalization is modified to only use the area of A boxes, hence computing the overlaps.
    Args:
        A (ndarray): Array of bounding boxes of shape (N, 4) in format [x1, y1, x2, y2].
        B (ndarray): Array of bounding boxes of shape (M, 4) in format [x1, y1, x2, y2].
    Returns:
        ndarray: Array of IoU values of shape (N, M) representing the overlaps
         between each pair of bounding boxes.
    """
    A = A.copy()
    B = B.copy()

    A = A[None].repeat(B.shape[0], 0)
    B = B[:, None].repeat(A.shape[1], 1)

    low = np.s_[..., :2]
    high = np.s_[..., 2:]

    A, B = A.copy(), B.copy()
    A[high] += 1
    B[high] += 1

    intrs = (np.maximum(0, np.minimum(A[high], B[high]) - np.maximum(A[low], B[low]))).prod(-1)
    ious = intrs / (A[high] - A[low]).prod(-1)

    return ious


def find_boxes_inside(boxes, boxes_to_check, threshold=0.9):
    """
    Find all boxes that are inside another box based on
    the intersection area divided by the area of the smaller box,
    and removes them.
    """
    overlaps = batched_overlaps(boxes_to_check, boxes)
    to_keep = (overlaps >= threshold).sum(0) <= 1
    return boxes_to_check[to_keep]


def get_bbox_dict_yolox_graphic(preds, shape, class_labels, threshold_=0.1) -> Dict[str, np.ndarray]:
    """
    Extracts bounding boxes from YOLOX model predictions:
    - Applies thresholding
    - Reformats boxes
    - Cleans the `other` detections: removes the ones that are included  in other detections.
    - If no title is found, the biggest `other` box is used if it is larger than 0.3*img_w.
    Args:
        preds (np.ndarray): YOLOX model predictions including bounding boxes, scores, and labels.
        shape (tuple): Original image shape.
        threshold_ (float): Score threshold to filter bounding boxes.
    Returns:
        Dict[str, np.ndarray]: Dictionary of bounding boxes, organized by class.
    """
    bbox_dict = {label: np.array([]) for label in class_labels}

    for i, label in enumerate(class_labels):
        if label not in preds:
            continue

        bboxes_class = np.array(preds[label])

        if bboxes_class.size == 0:
            continue

        # Try to find a chart_title box
        threshold = threshold_ if label != "chart_title" else min(threshold_, bboxes_class[:, -1].max())
        bboxes_class = bboxes_class[bboxes_class[:, -1] >= threshold][:, :4].astype(int)

        sort = ["x0", "y0"] if label != "ylabel" else ["y0", "x0"]
        idxs = (
            pd.DataFrame(
                {
                    "y0": bboxes_class[:, 1],
                    "x0": bboxes_class[:, 0],
                }
            )
            .sort_values(sort, ascending=label != "ylabel")
            .index
        )
        bboxes_class = bboxes_class[idxs]
        bbox_dict[label] = bboxes_class

    # Remove other included
    if len(bbox_dict.get("other", [])):
        other = find_boxes_inside(
            np.concatenate(list([v for v in bbox_dict.values() if len(v)])), bbox_dict["other"], threshold=0.7
        )
        del bbox_dict["other"]
        if len(other):
            bbox_dict["other"] = other

    # Biggest other is title if no title
    if not len(bbox_dict.get("chart_title", [])) and len(bbox_dict.get("other", [])):
        boxes = bbox_dict["other"]
        ws = boxes[:, 2] - boxes[:, 0]
        if np.max(ws) > shape[1] * 0.3:
            bbox_dict["chart_title"] = boxes[np.argmax(ws)][None].copy()
            bbox_dict["other"] = np.delete(boxes, (np.argmax(ws)), axis=0)

    # Make sure other key not lost
    bbox_dict["other"] = bbox_dict.get("other", [])

    return bbox_dict


def get_bbox_dict_yolox_table(preds, shape, class_labels, threshold=0.1, delta=0.0):
    """
    Extracts bounding boxes from YOLOX model predictions:
    - Applies thresholding
    - Reformats boxes
    - Reorders predictions

    Args:
        preds (np.ndarray): YOLOX model predictions including bounding boxes, scores, and labels.
        shape (tuple): Original image shape.
        config: Model configuration, including size for bounding box adjustment.
        threshold (float): Score threshold to filter bounding boxes.
        delta (float): How much the table was cropped upwards.

    Returns:
        dict[str, np.ndarray]: Dictionary of bounding boxes, organized by class.
    """
    bbox_dict = {label: np.array([]) for label in class_labels}

    for i, label in enumerate(class_labels):
        if label not in ["cell", "row", "column"]:
            continue  # Ignore useless classes

        bboxes_class = np.array(preds[label])

        if bboxes_class.size == 0:
            continue

        # Threshold and clip
        bboxes_class = bboxes_class[bboxes_class[:, -1] >= threshold][:, :4].astype(int)
        bboxes_class[:, [0, 2]] = np.clip(bboxes_class[:, [0, 2]], 0, shape[1])
        bboxes_class[:, [1, 3]] = np.clip(bboxes_class[:, [1, 3]], 0, shape[0])

        # Reorder
        sort = ["x0", "y0"] if label != "row" else ["y0", "x0"]
        df = pd.DataFrame(
            {
                "y0": (bboxes_class[:, 1] + bboxes_class[:, 3]) / 2,
                "x0": (bboxes_class[:, 0] + bboxes_class[:, 2]) / 2,
            }
        )
        idxs = df.sort_values(sort).index
        bboxes_class = bboxes_class[idxs]

        bbox_dict[label] = bboxes_class

    # Enforce spanning the entire table
    if len(bbox_dict["row"]):
        bbox_dict["row"][:, 0] = 0
        bbox_dict["row"][:, 2] = shape[1]
    if len(bbox_dict["column"]):
        bbox_dict["column"][:, 1] = 0
        bbox_dict["column"][:, 3] = shape[0]

    # Shift back if cropped
    for k in bbox_dict:
        if len(bbox_dict[k]):
            bbox_dict[k][:, [1, 3]] = np.add(bbox_dict[k][:, [1, 3]], delta, casting="unsafe")

    return bbox_dict


def match_with_title_v3(bbox, title_bboxes, match_dist=0.1, delta=1.5, already_matched=[]):
    """
    Matches a bounding box with a title bounding box based on IoU or proximity.

    Args:
        bbox (numpy.ndarray): Bounding box to match with title [x_min, y_min, x_max, y_max].
        title_bboxes (numpy.ndarray): Array of title bounding boxes with shape (N, 4).
        match_dist (float, optional): Maximum distance for matching. Defaults to 0.1.
        delta (float, optional): Multiplier for matching several titles. Defaults to 1.5.
        already_matched (list, optional): List of already matched title indices. Defaults to [].

    Returns:
        tuple or None: If matched, returns a tuple of (merged_bbox, updated_title_bboxes).
                       If no match is found, returns None, None.
    """
    if not len(title_bboxes):
        return None, None

    dist_above = np.abs(title_bboxes[:, 3] - bbox[1])
    dist_below = np.abs(bbox[3] - title_bboxes[:, 1])

    dist_left = np.abs(title_bboxes[:, 0] - bbox[0])
    dist_center = np.abs(title_bboxes[:, 0] + title_bboxes[:, 2] - bbox[0] - bbox[2]) / 2

    dists = np.min([dist_above, dist_below], 0)
    dists += np.min([dist_left, dist_center], 0) / 2

    ious = bb_iou_array(title_bboxes, bbox)
    dists = np.where(ious > 0, min(match_dist, np.min(dists)), dists)

    if len(already_matched):
        dists[already_matched] = match_dist * 10  # Remove already matched titles

    # print(dists)
    matches = None  # noqa
    if np.min(dists) <= match_dist:
        matches = np.where(dists <= min(match_dist, np.min(dists) * delta))[0]

    if matches is not None:
        new_bbox = bbox
        for match in matches:
            new_bbox = merge_boxes(new_bbox, title_bboxes[match])
        return new_bbox, list(matches)
    else:
        return None, None


def match_boxes_with_title(boxes, confs, labels, classes, to_match_labels=["chart"], remove_matched_titles=False):
    """
    Matches charts with title.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        confs (numpy.ndarray): Array of confidence scores with shape (N,).
        labels (numpy.ndarray): Array of labels with shape (N,).
        classes (list): List of class names.
        to_match_labels (list): List of class names to match with titles.
        remove_matched_titles (bool): Whether to remove matched titles from the boxes.

    Returns:
        boxes (numpy.ndarray): Array of bounding boxes with shape (M, 4).
        confs (numpy.ndarray): Array of confidence scores with shape (M,).
        labels (numpy.ndarray): Array of labels with shape (M,).
        found_title (list): List of indices of matched titles.
        no_found_title (list): List of indices of unmatched titles.
    """
    # Put titles at the end
    title_ids = np.where(labels == classes.index("title"))[0]
    order = np.concatenate([np.delete(np.arange(len(boxes)), title_ids), title_ids])
    boxes = boxes[order]
    confs = confs[order]
    labels = labels[order]

    # Ids
    title_ids = np.where(labels == classes.index("title"))[0]
    to_match = np.where(np.isin(labels, [classes.index(c) for c in to_match_labels]))[0]

    # Matching
    found_title, already_matched = [], []
    for i in range(len(boxes)):
        if i not in to_match:
            continue
        merged_box, matched_title_ids = match_with_title_v3(
            boxes[i],
            boxes[title_ids],
            already_matched=already_matched,
        )
        if matched_title_ids is not None:
            # print(f'Merged {classes[int(labels[i])]} at idx #{i} with title {matched_title_ids[-1]}')  # noqa
            boxes[i] = merged_box
            already_matched += matched_title_ids
            found_title.append(i)

    if remove_matched_titles and len(already_matched):
        boxes = np.delete(boxes, title_ids[already_matched], axis=0)
        confs = np.delete(confs, title_ids[already_matched], axis=0)
        labels = np.delete(labels, title_ids[already_matched], axis=0)

    return boxes, confs, labels, found_title


def expand_boxes_v3(boxes, r_x=(1, 1), r_y=(1, 1), size_agnostic=True):
    """
    Expands bounding boxes by a specified ratio.
    Expected box format is normalized [x_min, y_min, x_max, y_max].

    Args:
        boxes (numpy.ndarray): Array of bounding boxes with shape (N, 4).
        r_x (tuple, optional): Left, right expansion ratios. Defaults to (1, 1) (no expansion).
        r_y (tuple, optional): Up, down expansion ratios. Defaults to (1, 1) (no expansion).
        size_agnostic (bool, optional): Expand independently of the bbox shape. Defaults to True.

    Returns:
        numpy.ndarray: Adjusted bounding boxes clipped to the [0, 1] range.
    """
    old_boxes = boxes.copy()

    if not size_agnostic:
        h = boxes[:, 3] - boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
    else:
        h, w = 1, 1

    boxes[:, 0] -= w * (r_x[0] - 1)  # left
    boxes[:, 2] += w * (r_x[1] - 1)  # right
    boxes[:, 1] -= h * (r_y[0] - 1)  # up
    boxes[:, 3] += h * (r_y[1] - 1)  # down

    boxes = np.clip(boxes, 0, 1)

    # Enforce non-overlapping boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = bb_iou_array(boxes[i][None], boxes[j])[0]
            old_iou = bb_iou_array(old_boxes[i][None], old_boxes[j])[0]
            # print(iou, old_iou)
            if iou > 0.05 and old_iou < 0.1:
                if boxes[i, 1] < boxes[j, 1]:  # i above j
                    boxes[j, 1] = min(old_boxes[j, 1], boxes[i, 3])
                    if old_iou > 0:
                        boxes[i, 3] = max(old_boxes[i, 3], boxes[j, 1])
                else:
                    boxes[i, 1] = min(old_boxes[i, 1], boxes[j, 3])
                    if old_iou > 0:
                        boxes[j, 3] = max(old_boxes[j, 3], boxes[i, 1])

    return boxes


def get_overlaps(boxes, other_boxes, normalize="box_only"):
    """
    Checks if a box overlaps with any other box.
    Boxes are expeceted in format (x0, y0, x1, y1)

    Args:
        boxes (np array [4] or [n x 4]): Boxes.
        other_boxes (np array [m x 4]): Other boxes.

    Returns:
        np array [n x m]: Overlaps.
    """
    if boxes.ndim == 1:
        boxes = boxes[None, :]

    x0, y0, x1, y1 = (boxes[:, 0][:, None], boxes[:, 1][:, None], boxes[:, 2][:, None], boxes[:, 3][:, None])
    areas = (y1 - y0) * (x1 - x0)

    x0_other, y0_other, x1_other, y1_other = (
        other_boxes[:, 0][None, :],
        other_boxes[:, 1][None, :],
        other_boxes[:, 2][None, :],
        other_boxes[:, 3][None, :],
    )
    areas_other = (y1_other - y0_other) * (x1_other - x0_other)

    # Intersection
    inter_y0 = np.maximum(y0, y0_other)
    inter_y1 = np.minimum(y1, y1_other)
    inter_x0 = np.maximum(x0, x0_other)
    inter_x1 = np.minimum(x1, x1_other)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    # Overlap
    if normalize == "box_only":  # Only consider box included in other box
        overlaps = inter_area / areas
    elif normalize == "all":  # Consider box included in other box and other box included in box
        overlaps = inter_area / np.minimum(areas, areas_other[:, None])
    else:
        raise ValueError(f"Invalid normalization: {normalize}")
    return overlaps


def postprocess_included(boxes, labels, confs, class_="title", classes=["table", "chart", "title", "infographic"]):
    """
    Post process title predictions.
    - Remove titles that are included in other boxes

    Args:
        boxes (numpy.ndarray [N, 4]): Array of bounding boxes.
        labels (numpy.ndarray [N]): Array of labels.
        confs (numpy.ndarray [N]): Array of confidences.
        class_ (str, optional): Class to postprocess. Defaults to "title".
        classes (list, optional): Classes. Defaults to ["table", "chart", "title", "infographic"].

    Returns:
        boxes (numpy.ndarray): Array of bounding boxes.
        labels (numpy.ndarray): Array of labels.
        confs (numpy.ndarray): Array of confidences.
    """
    boxes_to_pp = boxes[labels == classes.index(class_)]
    confs_to_pp = confs[labels == classes.index(class_)]

    order = np.argsort(confs_to_pp)  # least to most confident for NMS
    boxes_to_pp, confs_to_pp = boxes_to_pp[order], confs_to_pp[order]

    if len(boxes_to_pp) == 0:
        return boxes, labels, confs

    # other_boxes = boxes[labels != classes.index("title")]

    inclusion_classes = ["table", "infographic", "chart"]
    if class_ in ["header_footer", "title"]:
        inclusion_classes.append("paragraph")

    other_boxes = boxes[np.isin(labels, [classes.index(c) for c in inclusion_classes])]

    # Remove boxes included in other_boxes
    kept_boxes, kept_confs = [], []
    for i, b in enumerate(boxes_to_pp):
        if len(other_boxes) > 0:
            overlaps = get_overlaps(b, other_boxes, normalize="box_only")
            if overlaps.max() > 0.9:
                continue

        kept_boxes.append(b)
        kept_confs.append(confs_to_pp[i])

    # Aggregate
    kept_boxes = np.stack(kept_boxes) if len(kept_boxes) else np.empty((0, 4))
    kept_confs = np.stack(kept_confs) if len(kept_confs) else np.empty(0)

    boxes_pp = np.concatenate([boxes[labels != classes.index(class_)], kept_boxes])
    confs_pp = np.concatenate([confs[labels != classes.index(class_)], kept_confs])
    labels_pp = np.concatenate(
        [labels[labels != classes.index(class_)], np.ones(len(kept_boxes)) * classes.index(class_)]
    )

    return boxes_pp, labels_pp, confs_pp


def remove_overlapping_boxes_using_wbf(boxes, confs, labels):
    """
    Remove overlapping boxes using WBF
    """
    # Applied twice because once is not enough in some rare cases
    for _ in range(2):
        boxes, confs, labels = weighted_boxes_fusion(
            boxes[:, None],
            confs[:, None],
            labels[:, None],
            merge_type="biggest",
            conf_type="max",
            iou_thr=0.01,
            class_agnostic=False,
        )

    return boxes, confs, labels


def match_structured_boxes_with_title(boxes, confs, labels, classes):
    # Reorder by y, x
    order = np.argsort(boxes[:, 1] * 10 + boxes[:, 0])
    boxes, confs, labels = boxes[order], confs[order], labels[order]

    # Match with title
    # Although the model should detect titles, additional post-processing helps retrieve FNs
    found_title = []
    boxes, confs, labels, found_title = match_boxes_with_title(
        boxes,
        confs,
        labels,
        classes,
        to_match_labels=["chart", "table", "infographic"],
        remove_matched_titles=True,
    )

    return boxes, confs, labels, found_title


def expand_tables_and_charts(boxes, confs, labels, classes, found_title):
    # This is mostly to retrieve titles, but this also helps when YOLOX boxes are too tight.
    # Boxes with titles matched are expanded less.
    # Expansion is different for tables and charts
    no_found_title = [i for i in range(len(boxes)) if i not in found_title]
    ids = np.arange(len(boxes))

    if len(found_title):  # Boxes with title matched are expanded less
        ids_ = ids[found_title][labels[found_title] == classes.index("chart")]
        boxes[ids_] = expand_boxes_v3(
            boxes[ids_],
            r_x=(1.025, 1.025),
            r_y=(1.05, 1.05),
            size_agnostic=False,
        )
        ids_ = ids[found_title][labels[found_title] == classes.index("table")]
        boxes[ids_] = expand_boxes_v3(
            boxes[ids_],
            r_x=(1.01, 1.01),
            r_y=(1.05, 1.01),
        )

    ids_ = ids[no_found_title][labels[no_found_title] == classes.index("chart")]
    boxes[ids_] = expand_boxes_v3(
        boxes[ids_],
        r_x=(1.05, 1.05),
        r_y=(1.125, 1.125),
        size_agnostic=False,
    )

    ids_ = ids[no_found_title][labels[no_found_title] == classes.index("table")]
    boxes[ids_] = expand_boxes_v3(
        boxes[ids_],
        r_x=(1.02, 1.02),
        r_y=(1.05, 1.05),
    )

    order = np.argsort(boxes[:, 1] * 10 + boxes[:, 0])
    boxes, labels, confs = boxes[order], labels[order], confs[order]

    return boxes, labels, confs


def postprocess_included_texts(boxes, confs, labels, classes):
    for c in ["title", "paragraph", "header_footer"]:
        boxes, labels, confs = postprocess_included(boxes, labels, confs, c, classes)
    return boxes, labels, confs


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_yolox_model_name(yolox_grpc_endpoint, default_model_name="yolox"):
    try:
        client = grpcclient.InferenceServerClient(yolox_grpc_endpoint)
        model_index = client.get_model_repository_index(as_json=True)
        model_names = [x["name"] for x in model_index.get("models", [])]
        if "yolox_ensemble" in model_names:
            yolox_model_name = "yolox_ensemble"
        else:
            yolox_model_name = default_model_name
    except Exception:
        logger.warning(
            f"Failed to get yolox-page-elements version after 30 seconds. Falling back to '{default_model_name}'."
        )
        yolox_model_name = default_model_name

    return yolox_model_name


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_yolox_page_version(yolox_http_endpoint, default_version=YOLOX_PAGE_DEFAULT_VERSION):
    """
    Determines the YOLOX page elements model version by querying the endpoint.
    Falls back to a default version on failure.
    """
    try:
        yolox_version = get_model_name(yolox_http_endpoint, default_version)
        if not yolox_version:
            logger.warning(
                "Failed to obtain yolox-page-elements version from the endpoint. "
                f"Falling back to '{default_version}'."
            )
            return default_version

        return yolox_version
    except Exception:
        logger.warning(
            f"Failed to get yolox-page-elements version after 30 seconds. Falling back to '{default_version}'."
        )
        return default_version
