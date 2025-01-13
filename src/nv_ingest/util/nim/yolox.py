# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import io
import logging
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from packaging import version as pkgversion
from PIL import Image

from nv_ingest.util.image_processing.transforms import scale_image_to_encoding_size
from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)

YOLOX_MAX_BATCH_SIZE = 8
YOLOX_MAX_WIDTH = 1536
YOLOX_MAX_HEIGHT = 1536
YOLOX_NUM_CLASSES = 3
YOLOX_CONF_THRESHOLD = 0.01
YOLOX_IOU_THRESHOLD = 0.5
YOLOX_MIN_SCORE = 0.1
YOLOX_FINAL_SCORE = 0.48
YOLOX_NIM_MAX_IMAGE_SIZE = 360_000

YOLOX_IMAGE_PREPROC_HEIGHT = 1024
YOLOX_IMAGE_PREPROC_WIDTH = 1024


# Implementing YoloxPageElemenetsModelInterface with required methods
class YoloxPageElementsModelInterface(ModelInterface):
    """
    An interface for handling inference with a Yolox object detection model, supporting both gRPC and HTTP protocols.
    """

    def __init__(
        self,
        yolox_version: Optional[str] = None,
    ):
        """
        Initialize the YOLOX model interface.

        Parameters
        ----------
        yolox_version : str, optional
            The version of the YOLOX model (default: None).
        """
        self.yolox_version = yolox_version

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

        return f"yolox-page-elements (version {self.yolox_version})"

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

        original_images = data["images"]
        # Our yolox model expects images to be resized to 1024x1024
        resized_images = [
            resize_image(image, (YOLOX_IMAGE_PREPROC_WIDTH, YOLOX_IMAGE_PREPROC_HEIGHT)) for image in original_images
        ]
        data["original_image_shapes"] = [image.shape for image in original_images]
        data["resized_images"] = resized_images

        return data  # Return data with added 'resized_images' key

    def format_input(self, data: Dict[str, Any], protocol: str) -> Any:
        """
        Format input data for the specified protocol.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to use ("grpc" or "http").

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
            logger.debug("Formatting input for gRPC Yolox model")
            # Reorder axes to match model input (batch, channels, height, width)
            input_array = np.einsum("bijk->bkij", data["resized_images"]).astype(np.float32)
            return input_array

        elif protocol == "http":
            logger.debug("Formatting input for HTTP Yolox model")
            # Additional lists to keep track of scaling factors and new sizes
            scaling_factors = []
            content_list = []
            for image in data["resized_images"]:
                # Convert numpy array to PIL Image
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
                original_size = image_pil.size  # Should be (1024, 1024)

                # Save image to buffer
                buffered = io.BytesIO()
                image_pil.save(buffered, format="PNG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Now scale the image if necessary
                scaled_image_b64, new_size = scale_image_to_encoding_size(
                    image_b64, max_base64_size=YOLOX_NIM_MAX_IMAGE_SIZE
                )

                if new_size != original_size:
                    logger.warning(f"Image was scaled from {original_size} to {new_size} to meet size constraints.")

                # Compute scaling factor
                scaling_factor_x = new_size[0] / YOLOX_IMAGE_PREPROC_WIDTH
                scaling_factor_y = new_size[1] / YOLOX_IMAGE_PREPROC_HEIGHT
                scaling_factors.append((scaling_factor_x, scaling_factor_y))

                # Add to content_list
                if self._is_version_early_access_legacy_api():
                    content = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{scaled_image_b64}"}}
                else:
                    content = {"type": "image_url", "url": f"data:image/png;base64,{scaled_image_b64}"}

                content_list.append(content)

            # Store scaling factors in data
            data["scaling_factors"] = scaling_factors

            if self._is_version_early_access_legacy_api():
                payload = {"messages": [{"content": content_list}]}
            else:
                payload = {"input": content_list}

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
            logger.debug("Parsing output from gRPC Yolox model")
            return response  # For gRPC, response is already a numpy array
        elif protocol == "http":
            logger.debug("Parsing output from HTTP Yolox model")

            is_legacy_version = self._is_version_early_access_legacy_api()

            # Convert JSON response to numpy array similar to gRPC response
            if is_legacy_version:
                # Convert response data to GA API format.
                response_data = response.get("data", [])
                batch_results = []
                for idx, detections in enumerate(response_data):
                    curr_batch = {"index": idx, "bounding_boxes": {}}
                    for obj in detections:
                        obj_type = obj.get("type", "")
                        bboxes = obj.get("bboxes", [])
                        if not obj_type:
                            continue
                        if obj_type not in curr_batch:
                            curr_batch["bounding_boxes"][obj_type] = []
                        curr_batch["bounding_boxes"][obj_type].extend(bboxes)
                    batch_results.append(curr_batch)
            else:
                batch_results = response.get("data", [])

            batch_size = len(batch_results)
            processed_outputs = []

            scaling_factors = data.get("scaling_factors", [(1.0, 1.0)] * batch_size)

            x_min_label = "xmin" if is_legacy_version else "x_min"
            y_min_label = "ymin" if is_legacy_version else "y_min"
            x_max_label = "xmax" if is_legacy_version else "x_max"
            y_max_label = "ymax" if is_legacy_version else "y_max"
            confidence_label = "confidence"

            for detections in batch_results:
                idx = int(detections["index"])
                scale_factor_x, scale_factor_y = scaling_factors[idx]
                image_width = YOLOX_IMAGE_PREPROC_WIDTH
                image_height = YOLOX_IMAGE_PREPROC_HEIGHT

                # Initialize an empty tensor for detections
                max_detections = 100
                detection_tensor = np.zeros((max_detections, 85), dtype=np.float32)

                index = 0
                bounding_boxes = detections.get("bounding_boxes", [])
                for obj_type, bboxes in bounding_boxes.items():
                    for bbox in bboxes:
                        if index >= max_detections:
                            break
                        xmin_norm = bbox[x_min_label]
                        ymin_norm = bbox[y_min_label]
                        xmax_norm = bbox[x_max_label]
                        ymax_norm = bbox[y_max_label]
                        confidence = bbox[confidence_label]

                        # Convert normalized coordinates to absolute pixel values in scaled image
                        xmin_scaled = xmin_norm * image_width * scale_factor_x
                        ymin_scaled = ymin_norm * image_height * scale_factor_y
                        xmax_scaled = xmax_norm * image_width * scale_factor_x
                        ymax_scaled = ymax_norm * image_height * scale_factor_y

                        # Adjust coordinates back to 1024x1024 image space
                        xmin = xmin_scaled / scale_factor_x
                        ymin = ymin_scaled / scale_factor_y
                        xmax = xmax_scaled / scale_factor_x
                        ymax = ymax_scaled / scale_factor_y

                        # YOLOX expects bbox format: center_x, center_y, width, height
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        width = xmax - xmin
                        height = ymax - ymin

                        # Set the bbox coordinates
                        detection_tensor[index, 0] = center_x
                        detection_tensor[index, 1] = center_y
                        detection_tensor[index, 2] = width
                        detection_tensor[index, 3] = height

                        # Objectness score
                        detection_tensor[index, 4] = confidence

                        class_index = {"table": 0, "chart": 1, "title": 2}.get(obj_type, -1)
                        if class_index >= 0:
                            detection_tensor[index, 5 + class_index] = 1.0

                        index += 1

                # Trim the detection tensor to the actual number of detections
                detection_tensor = detection_tensor[:index, :]
                processed_outputs.append(detection_tensor)

            # Pad batch if necessary
            max_detections_in_batch = max([output.shape[0] for output in processed_outputs]) if processed_outputs else 0
            batch_output_array = np.zeros((batch_size, max_detections_in_batch, 85), dtype=np.float32)
            for i, output in enumerate(processed_outputs):
                batch_output_array[i, : output.shape[0], :] = output

            return batch_output_array
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def process_inference_results(self, output_array: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
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

        original_image_shapes = kwargs.get("original_image_shapes", [])
        num_classes = kwargs.get("num_classes", YOLOX_NUM_CLASSES)
        conf_thresh = kwargs.get("conf_thresh", YOLOX_CONF_THRESHOLD)
        iou_thresh = kwargs.get("iou_thresh", YOLOX_IOU_THRESHOLD)
        min_score = kwargs.get("min_score", YOLOX_MIN_SCORE)
        final_thresh = kwargs.get("final_thresh", YOLOX_FINAL_SCORE)

        pred = postprocess_model_prediction(output_array, num_classes, conf_thresh, iou_thresh, class_agnostic=True)

        results = postprocess_results(pred, original_image_shapes, min_score=min_score)

        annotation_dicts = [expand_chart_bboxes(annotation_dict) for annotation_dict in results]
        inference_results = []

        # Filter out bounding boxes below the final threshold
        for annotation_dict in annotation_dicts:
            new_dict = {}
            if "table" in annotation_dict:
                new_dict["table"] = [bb for bb in annotation_dict["table"] if bb[4] >= final_thresh]
            if "chart" in annotation_dict:
                new_dict["chart"] = [bb for bb in annotation_dict["chart"] if bb[4] >= final_thresh]
            if "title" in annotation_dict:
                new_dict["title"] = annotation_dict["title"]
            inference_results.append(new_dict)

        return inference_results

    def _is_version_early_access_legacy_api(self):
        return self.yolox_version and (pkgversion.parse(self.yolox_version) < pkgversion.parse("1.0.0-rc0"))


def postprocess_model_prediction(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # Convert numpy array to torch tensor
    prediction = torch.from_numpy(prediction.copy())

    # Compute box corners
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # If no detections, continue to the next image
        if not image_pred.size(0):
            continue

        # Ensure image_pred is 2D
        if image_pred.ndim == 1:
            image_pred = image_pred.unsqueeze(0)

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)

        # Confidence mask
        squeezed_conf = class_conf.squeeze(dim=1)
        conf_mask = image_pred[:, 4] * squeezed_conf >= conf_thre

        # Apply confidence mask
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        if not detections.size(0):
            continue

        # Apply Non-Maximum Suppression (NMS)
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
        detections = detections[nms_out_index]

        # Append detections to output
        output[i] = detections

    return output


def postprocess_results(results, original_image_shapes, min_score=0.0):
    """
    For each item (==image) in results, computes annotations in the form

     {"table": [[0.0107, 0.0859, 0.7537, 0.1219, 0.9861], ...],
      "figure": [...],
      "title": [...]
      }
    where each list of 5 floats represents a bounding box in the format [x1, y1, x2, y2, confidence]

    Keep only bboxes with high enough confidence.
    """
    labels = ["table", "chart", "title"]
    out = []

    for original_image_shape, result in zip(original_image_shapes, results):
        if result is None:
            out.append({})
            continue

        try:
            result = result.cpu().numpy()
            scores = result[:, 4] * result[:, 5]
            result = result[scores > min_score]

            # ratio is used when image was padded
            ratio = min(
                YOLOX_IMAGE_PREPROC_WIDTH / original_image_shape[0],
                YOLOX_IMAGE_PREPROC_HEIGHT / original_image_shape[1],
            )
            bboxes = result[:, :4] / ratio

            bboxes[:, [0, 2]] /= original_image_shape[1]
            bboxes[:, [1, 3]] /= original_image_shape[0]
            bboxes = np.clip(bboxes, 0.0, 1.0)

            label_idxs = result[:, 6]
            scores = scores[scores > min_score]
        except Exception as e:
            raise ValueError(f"Error in postprocessing {result.shape} and {original_image_shape}: {e}")

        annotation_dict = {label: [] for label in labels}

        # bboxes are in format [x_min, y_min, x_max, y_max]
        for j in range(len(bboxes)):
            label = labels[int(label_idxs[j])]
            bbox = bboxes[j]
            score = scores[j]

            # additional preprocessing for tables: extend the upper bounds to capture titles if any.
            if label == "table":
                height = bbox[3] - bbox[1]
                bbox[1] = (bbox[1] - height * 0.2).clip(0.0, 1.0)

            annotation_dict[label].append([round(float(x), 4) for x in np.concatenate((bbox, [score]))])

        out.append(annotation_dict)

    # {label: [[x1, y1, x2, y2, confidence], ...], ...}
    return out


def resize_image(image, target_img_size):
    w, h, _ = np.array(image).shape

    if target_img_size is not None:  # Resize + Pad
        r = min(target_img_size[0] / w, target_img_size[1] / h)
        image = cv2.resize(
            image,
            (int(h * r), int(w * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        image = np.pad(
            image,
            ((0, target_img_size[0] - image.shape[0]), (0, target_img_size[1] - image.shape[1]), (0, 0)),
            mode="constant",
            constant_values=114,
        )

    return image


def expand_chart_bboxes(annotation_dict, labels=None):
    """
    Expand bounding boxes of charts and titles based on the bounding boxes of the other class.
    Args:
        annotation_dict: output of postprocess_results, a dictionary with keys "table", "figure", "title"

    Returns:
        annotation_dict: same as input, with expanded bboxes for charts

    """
    if not labels:
        labels = ["table", "chart", "title"]

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
