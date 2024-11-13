# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List
import numpy as np
import warnings

import cv2
import numpy as np
import torch
import torchvision


def postprocess_model_prediction(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    prediction = torch.from_numpy(prediction.copy())
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)  # noqa: E203

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

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
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

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
            ratio = min(1024 / original_image_shape[0], 1024 / original_image_shape[1])
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


def prepare_images_for_inference(images: List[np.ndarray]) -> np.ndarray:
    """
    Prepare a list of images for model inference by resizing and reordering axes.

    Parameters
    ----------
    images : List[np.ndarray]
        A list of image arrays to be prepared for inference.

    Returns
    -------
    np.ndarray
        A numpy array suitable for model input, with the shape reordered to match the expected input format.

    Notes
    -----
    The images are resized to 1024x1024 pixels and the axes are reordered to match the expected input shape for
    the model (batch, channels, height, width).

    Examples
    --------
    >>> images = [np.random.rand(1536, 1536, 3) for _ in range(2)]
    >>> input_array = prepare_images_for_inference(images)
    >>> input_array.shape
    (2, 3, 1024, 1024)
    """

    resized_images = [resize_image(image, (1024, 1024)) for image in images]

    return np.einsum("bijk->bkij", resized_images).astype(np.float32)
