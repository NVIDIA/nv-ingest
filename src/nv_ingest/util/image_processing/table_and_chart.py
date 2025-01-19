# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import re

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


logger = logging.getLogger(__name__)


def process_yolox_graphic_elements(yolox_text_dict):
    """
    Process the inference results from yolox-graphic-elements model.

    Parameters
    ----------
    yolox_text : str
        The result from the yolox model inference.

    Returns
    -------
    str
        The concatenated and processed chart content as a string.
    """
    chart_content = ""

    chart_content += yolox_text_dict.get("chart_title", "")

    chart_content += " " + yolox_text_dict.get("caption", "")
    chart_content += " " + yolox_text_dict.get("x_title", "")
    chart_content += " " + yolox_text_dict.get("xlabel", "")
    chart_content += " " + yolox_text_dict.get("y_title", "")
    chart_content += " " + yolox_text_dict.get("ylabel", "")
    chart_content += " " + yolox_text_dict.get("legend_label", "")
    chart_content += " " + yolox_text_dict.get("legend_title", "")
    chart_content += " " + yolox_text_dict.get("mark_label", "")
    chart_content += " " + yolox_text_dict.get("value_label", "")
    chart_content += " " + yolox_text_dict.get("other", "")

    return chart_content


def match_bboxes(yolox_box, paddle_ocr_boxes, already_matched=None, delta=2.0):
    """
    Associates a yolox-graphic-elements box to PaddleOCR bboxes, by taking overlapping boxes.
    Criterion is iou > max_iou / delta where max_iou is the biggest found overlap.
    Boxes are expeceted in format (x0, y0, x1, y1)
    Args:
        yolox_box (np array [4]): Cached Bbox.
        paddle_ocr_boxes (np array [n x 4]): PaddleOCR boxes
        already_matched (list or None, Optional): Already matched ids to ignore.
        delta (float, Optional): IoU delta for considering several boxes. Defaults to 2..
    Returns:
        np array or list: Indices of the match bboxes
    """
    x0_1, y0_1, x1_1, y1_1 = yolox_box
    x0_2, y0_2, x1_2, y1_2 = (
        paddle_ocr_boxes[:, 0],
        paddle_ocr_boxes[:, 1],
        paddle_ocr_boxes[:, 2],
        paddle_ocr_boxes[:, 3],
    )

    # Intersection
    inter_y0 = np.maximum(y0_1, y0_2)
    inter_y1 = np.minimum(y1_1, y1_2)
    inter_x0 = np.maximum(x0_1, x0_2)
    inter_x1 = np.minimum(x1_1, x1_2)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    # Union
    area_1 = (y1_1 - y0_1) * (x1_1 - x0_1)
    area_2 = (y1_2 - y0_2) * (x1_2 - x0_2)
    union_area = area_1 + area_2 - inter_area

    # IoU
    ious = inter_area / union_area

    max_iou = np.max(ious)
    if max_iou <= 0.01:
        return []

    matches = np.where(ious > (max_iou / delta))[0]
    if already_matched is not None:
        matches = np.array([m for m in matches if m not in already_matched])
    return matches


def join_yolox_and_paddle_output(yolox_output, paddle_txts, paddle_boxes):
    """
    Matching boxes
    We need to associate a text to the paddle detections.
    For each class and for each CACHED detections, we look for overlapping text bboxes
    with  IoU > max_iou / delta where max_iou is the biggest found overlap.
    Found texts are added to the class representation, and removed from the texts to match
    """
    KEPT_CLASSES = [  # Used CACHED classes, corresponds to YoloX classes
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

    paddle_txts = np.array(paddle_txts)
    paddle_boxes = np.array(paddle_boxes)

    if (paddle_txts.size == 0) or (paddle_boxes.size == 0):
        return {}

    paddle_boxes = np.array(
        [
            paddle_boxes[:, :, 0].min(-1),
            paddle_boxes[:, :, 1].min(-1),
            paddle_boxes[:, :, 0].max(-1),
            paddle_boxes[:, :, 1].max(-1),
        ]
    ).T

    already_matched = []
    results = {}

    for k in KEPT_CLASSES:
        if not len(yolox_output.get(k, [])):  # No bounding boxes
            continue

        texts = []
        for yolox_box in yolox_output[k]:
            # if there's a score at the end, drop the score.
            yolox_box = yolox_box[:4]
            paddle_ids = match_bboxes(yolox_box, paddle_boxes, already_matched=already_matched, delta=4)

            if len(paddle_ids) > 0:
                text = " ".join(paddle_txts[paddle_ids].tolist())
                texts.append(text)

        processed_texts = []
        for t in texts:
            t = re.sub(r"\s+", " ", t)
            t = re.sub(r"\.+", ".", t)
            processed_texts.append(t)

        if "title" in k:
            processed_texts = " ".join(processed_texts)
        else:
            processed_texts = " - ".join(processed_texts)  # Space ?

        results[k] = processed_texts

    return results


def convert_paddle_response_to_psuedo_markdown(texts, bboxes):
    if (not bboxes) or (not texts):
        return ""

    bboxes = np.array(bboxes).astype(int)
    bboxes = bboxes.reshape(-1, 8)[:, [0, 1, 2, -1]]

    preds_df = pd.DataFrame(
        {"x0": bboxes[:, 0], "y0": bboxes[:, 1], "x1": bboxes[:, 2], "y1": bboxes[:, 3], "text": texts}
    )
    preds_df = preds_df.sort_values("y0")

    dbscan = DBSCAN(eps=10, min_samples=1)
    dbscan.fit(preds_df["y0"].values[:, None])

    preds_df["cluster"] = dbscan.labels_
    preds_df = preds_df.sort_values(["cluster", "x0"])

    results = ""
    for _, dfg in preds_df.groupby("cluster"):
        results += "| " + " | ".join(dfg["text"].values.tolist()) + " |\n"

    return results
