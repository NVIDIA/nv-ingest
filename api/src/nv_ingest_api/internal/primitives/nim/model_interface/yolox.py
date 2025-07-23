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
from nv_ingest_api.util.image_processing import scale_image_to_encoding_size
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64

logger = logging.getLogger(__name__)

# yolox-page-elements-v1 and v2 common contants
YOLOX_PAGE_CONF_THRESHOLD = 0.01
YOLOX_PAGE_IOU_THRESHOLD = 0.5
YOLOX_PAGE_MIN_SCORE = 0.1
YOLOX_PAGE_NIM_MAX_IMAGE_SIZE = 512_000
YOLOX_PAGE_IMAGE_PREPROC_HEIGHT = 1024
YOLOX_PAGE_IMAGE_PREPROC_WIDTH = 1024
YOLOX_PAGE_IMAGE_FORMAT = os.getenv("YOLOX_PAGE_IMAGE_FORMAT", "PNG")

# yolox-page-elements-v2 contants
YOLOX_PAGE_V2_NUM_CLASSES = 4
YOLOX_PAGE_V2_FINAL_SCORE = {"table": 0.1, "chart": 0.01, "infographic": 0.01}
YOLOX_PAGE_V2_CLASS_LABELS = [
    "table",
    "chart",
    "title",
    "infographic",
]


# yolox-graphic-elements-v1 contants
YOLOX_GRAPHIC_NUM_CLASSES = 10
YOLOX_GRAPHIC_CONF_THRESHOLD = 0.01
YOLOX_GRAPHIC_IOU_THRESHOLD = 0.25
YOLOX_GRAPHIC_MIN_SCORE = 0.1
YOLOX_GRAPHIC_FINAL_SCORE = 0.0
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
YOLOX_TABLE_NUM_CLASSES = 5
YOLOX_TABLE_CONF_THRESHOLD = 0.01
YOLOX_TABLE_IOU_THRESHOLD = 0.25
YOLOX_TABLE_MIN_SCORE = 0.1
YOLOX_TABLE_FINAL_SCORE = 0.0
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
        num_classes: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
        final_score: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
    ):
        """
        Initialize the YOLOX model interface.
        Parameters
        ----------
        """
        self.nim_max_image_size = nim_max_image_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.final_score = final_score
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

    def __init__(self):
        """
        Initialize the yolox-page-elements model interface.
        """
        num_classes = YOLOX_PAGE_V2_NUM_CLASSES
        final_score = YOLOX_PAGE_V2_FINAL_SCORE
        class_labels = YOLOX_PAGE_V2_CLASS_LABELS

        super().__init__(
            nim_max_image_size=YOLOX_PAGE_NIM_MAX_IMAGE_SIZE,
            num_classes=num_classes,
            conf_threshold=YOLOX_PAGE_CONF_THRESHOLD,
            iou_threshold=YOLOX_PAGE_IOU_THRESHOLD,
            min_score=YOLOX_PAGE_MIN_SCORE,
            final_score=final_score,
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

    def postprocess_annotations(self, annotation_dicts, **kwargs):
        original_image_shapes = kwargs.get("original_image_shapes", [])

        expected_final_score_keys = [x for x in self.class_labels if x != "title"]
        if (not isinstance(self.final_score, dict)) or (
            sorted(self.final_score.keys()) != sorted(expected_final_score_keys)
        ):
            raise ValueError(
                "yolox-page-elements-v2 requires a dictionary of thresholds per each class: "
                f"{expected_final_score_keys}"
            )

        # Table/chart expansion is "business logic" specific to nv-ingest
        annotation_dicts = [expand_table_bboxes(annotation_dict) for annotation_dict in annotation_dicts]
        annotation_dicts = [expand_chart_bboxes(annotation_dict) for annotation_dict in annotation_dicts]
        inference_results = []

        # Filter out bounding boxes below the final threshold
        # This final thresholding is "business logic" specific to nv-ingest
        for annotation_dict in annotation_dicts:
            new_dict = {}
            if "table" in annotation_dict:
                new_dict["table"] = [bb for bb in annotation_dict["table"] if bb[4] >= self.final_score["table"]]
            if "chart" in annotation_dict:
                new_dict["chart"] = [bb for bb in annotation_dict["chart"] if bb[4] >= self.final_score["chart"]]
            if "infographic" in annotation_dict:
                new_dict["infographic"] = [
                    bb for bb in annotation_dict["infographic"] if bb[4] >= self.final_score["infographic"]
                ]
            if "title" in annotation_dict:
                new_dict["title"] = annotation_dict["title"]
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
            num_classes=YOLOX_GRAPHIC_NUM_CLASSES,
            conf_threshold=YOLOX_GRAPHIC_CONF_THRESHOLD,
            iou_threshold=YOLOX_GRAPHIC_IOU_THRESHOLD,
            min_score=YOLOX_GRAPHIC_MIN_SCORE,
            final_score=YOLOX_GRAPHIC_FINAL_SCORE,
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
            num_classes=YOLOX_TABLE_NUM_CLASSES,
            conf_threshold=YOLOX_TABLE_CONF_THRESHOLD,
            iou_threshold=YOLOX_TABLE_IOU_THRESHOLD,
            min_score=YOLOX_TABLE_MIN_SCORE,
            final_score=YOLOX_TABLE_FINAL_SCORE,
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
        match = match_with_title(chart_bboxes[i], title_bboxes, iou_th=0.01)
        if match is not None:
            chart_bboxes[i] = match[0]
            title_bboxes = match[1]
            found_title_idxs.append(i)
        else:
            no_found_title_idxs.append(i)

    chart_bboxes[found_title_idxs] = expand_boxes(chart_bboxes[found_title_idxs], r_x=1.05, r_y=1.1)
    chart_bboxes[no_found_title_idxs] = expand_boxes(chart_bboxes[no_found_title_idxs], r_x=1.1, r_y=1.25)

    annotation_dict = {
        "table": annotation_dict["table"],
        "chart": np.concatenate([chart_bboxes, chart_confidences[:, None]], axis=1).tolist(),
        "title": annotation_dict["title"],
    }
    return annotation_dict


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


def match_with_title(chart_bbox, title_bboxes, iou_th=0.01):
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


def expand_boxes(boxes, r_x=1, r_y=1):
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
            "Failed to get yolox-page-elements version after 30 seconds. " f"Falling back to '{default_model_name}'."
        )
        yolox_model_name = default_model_name

    return yolox_model_name
