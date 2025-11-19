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

    return chart_content.strip()


def match_bboxes(yolox_box, ocr_boxes, already_matched=None, delta=2.0):
    """
    Associates a yolox-graphic-elements box to PaddleOCR bboxes, by taking overlapping boxes.
    Criterion is iou > max_iou / delta where max_iou is the biggest found overlap.
    Boxes are expeceted in format (x0, y0, x1, y1)
    Args:
        yolox_box (np array [4]): Cached Bbox.
        ocr_boxes (np array [n x 4]): PaddleOCR boxes
        already_matched (list or None, Optional): Already matched ids to ignore.
        delta (float, Optional): IoU delta for considering several boxes. Defaults to 2..
    Returns:
        np array or list: Indices of the match bboxes
    """
    x0_1, y0_1, x1_1, y1_1 = yolox_box
    x0_2, y0_2, x1_2, y1_2 = (
        ocr_boxes[:, 0],
        ocr_boxes[:, 1],
        ocr_boxes[:, 2],
        ocr_boxes[:, 3],
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


def join_yolox_graphic_elements_and_ocr_output(yolox_output, ocr_boxes, ocr_txts):
    """
    Matching boxes
    We need to associate a text to the ocr detections.
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

    ocr_txts = np.array(ocr_txts)
    ocr_boxes = np.array(ocr_boxes)

    if (ocr_txts.size == 0) or (ocr_boxes.size == 0):
        return {}

    ocr_boxes = np.array(
        [
            ocr_boxes[:, :, 0].min(-1),
            ocr_boxes[:, :, 1].min(-1),
            ocr_boxes[:, :, 0].max(-1),
            ocr_boxes[:, :, 1].max(-1),
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
            ocr_ids = match_bboxes(yolox_box, ocr_boxes, already_matched=already_matched, delta=4)

            if len(ocr_ids) > 0:
                text = " ".join(ocr_txts[ocr_ids].tolist())
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


def convert_ocr_response_to_psuedo_markdown(bboxes, texts):
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


def join_yolox_table_structure_and_ocr_output(yolox_cell_preds, ocr_boxes, ocr_txts):
    if (not ocr_boxes) or (not ocr_txts):
        return ""

    ocr_boxes = np.array(ocr_boxes)
    ocr_boxes_ = np.array(
        [
            ocr_boxes[:, :, 0].min(-1),
            ocr_boxes[:, :, 1].min(-1),
            ocr_boxes[:, :, 0].max(-1),
            ocr_boxes[:, :, 1].max(-1),
        ]
    ).T

    assignments = []
    for i, (b, t) in enumerate(zip(ocr_boxes_, ocr_txts)):
        # Find a cell
        matches_cell = assign_boxes(b, yolox_cell_preds["cell"], delta=1)
        cell = yolox_cell_preds["cell"][matches_cell[0]] if len(matches_cell) else b

        # Find a row
        matches_row = assign_boxes(cell, yolox_cell_preds["row"], delta=1)
        row_ids = matches_row if len(matches_row) else -1

        # Find a column - or more if if it is the first row
        if isinstance(row_ids, np.ndarray):
            delta = 2 if row_ids.min() == 0 else 1  # delta=2 if header column
        else:
            delta = 1
        matches_col = assign_boxes(cell, yolox_cell_preds["column"], delta=delta)
        col_ids = matches_col if len(matches_col) else -1

        assignments.append(
            {
                "index": i,
                "ocr_box": b,
                "is_table": isinstance(col_ids, np.ndarray) and isinstance(row_ids, np.ndarray),
                "cell_id": matches_cell[0] if len(matches_cell) else -1,
                "cell": cell,
                "col_ids": col_ids,
                "row_ids": row_ids,
                "text": t,
            }
        )
        # break
    df_assign = pd.DataFrame(assignments)

    # Merge cells with several assigned texts
    dfs = []
    for cell_id, df_cell in df_assign.groupby("cell_id"):
        if len(df_cell) > 1 and cell_id > -1:
            df_cell = merge_text_in_cell(df_cell)
        dfs.append(df_cell)
    df_assign = pd.concat(dfs)

    df_text = df_assign[~df_assign["is_table"]].reset_index(drop=True)

    # Table to text
    df_table = df_assign[df_assign["is_table"]].reset_index(drop=True)
    if len(df_table):
        mat = build_markdown(df_table)
        markdown_table = display_markdown(mat, use_header=False)

        all_boxes = np.stack(df_table.ocr_box.values)
        table_box = np.concatenate([all_boxes[:, [0, 1]].min(0), all_boxes[:, [2, 3]].max(0)])

        df_table_to_text = pd.DataFrame(
            [
                {
                    "ocr_box": table_box,
                    "text": markdown_table,
                    "is_table": True,
                }
            ]
        )
        # Final text representations dataframe
        df_text = pd.concat([df_text, df_table_to_text], ignore_index=True)

    df_text = df_text.rename(columns={"ocr_box": "box"})

    # Sort by y and x
    df_text["x"] = df_text["box"].apply(lambda x: (x[0] + x[2]) / 2)
    df_text["y"] = df_text["box"].apply(lambda x: (x[1] + x[3]) / 2)
    df_text["x"] = (df_text["x"] - df_text["x"].min()) // 10
    df_text["y"] = (df_text["y"] - df_text["y"].min()) // 20
    df_text = df_text.sort_values(["y", "x"], ignore_index=True)

    # Loop over lines
    rows_list = []
    for r, df_row in df_text.groupby("y"):
        if df_row["is_table"].values.any():  # Add table
            table = df_row[df_row["is_table"]]
            df_row = df_row[~df_row["is_table"]]
        else:
            table = None

        if len(df_row) > 1:  # Add text
            df_row = df_row.reset_index(drop=True)
            df_row["text"] = "\n".join(df_row["text"].values.tolist())

        rows_list.append(df_row.head(1))

        if table is not None:
            rows_list.append(table)

    df_display = pd.concat(rows_list, ignore_index=True)
    result = "\n".join(df_display.text.values.tolist())

    return result


def assign_boxes(ocr_box, boxes, delta=2.0, min_overlap=0.25):
    """
    Assigns the closest bounding boxes to a reference `ocr_box` based on overlap.

    Args:
        ocr_box (list or numpy.ndarray): Reference bounding box [x_min, y_min, x_max, y_max].
        boxes (numpy.ndarray): Array of candidate bounding boxes with shape (N, 4).
        delta (float, optional): Factor for matches relative to the best overlap. Defaults to 2.0.
        min_overlap (float, optional): Minimum required overlap for a match. Defaults to 0.25.

    Returns:
        list: Indices of the matched boxes sorted by decreasing overlap.
              Returns an empty list if no matches are found.
    """
    if not len(boxes):
        return []

    boxes = np.array(boxes)

    x0_1, y0_1, x1_1, y1_1 = ocr_box
    x0_2, y0_2, x1_2, y1_2 = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )

    # Intersection
    inter_y0 = np.maximum(y0_1, y0_2)
    inter_y1 = np.minimum(y1_1, y1_2)
    inter_x0 = np.maximum(x0_1, x0_2)
    inter_x1 = np.minimum(x1_1, x1_2)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    # Normalize by ocr_box size
    area_1 = (y1_1 - y0_1) * (x1_1 - x0_1)
    ious = inter_area / (area_1 + 1e-6)

    max_iou = np.max(ious)
    if max_iou <= min_overlap:  # No match
        return []

    n = len(np.where(ious >= (max_iou / delta))[0])
    matches = np.argsort(-ious)[:n]
    return matches


def build_markdown(df):
    """
    Convert a dataframe into a markdown table.

    Args:
        df (pandas DataFrame): The dataframe to convert.

    Returns:
        list[list]: A list of lists representing the markdown table.
    """
    df = df.reset_index(drop=True)
    n_cols = max([np.max(c) for c in df["col_ids"].values])
    n_rows = max([np.max(c) for c in df["row_ids"].values])

    mat = np.empty((n_rows + 1, n_cols + 1), dtype=str).tolist()

    for i in range(len(df)):
        if isinstance(df["row_ids"][i], int) or isinstance(df["col_ids"][i], int):
            continue
        for r in df["row_ids"][i]:
            for c in df["col_ids"][i]:
                mat[r][c] = (mat[r][c] + " " + df["text"][i]).strip()

    # Remove empty rows & columns
    mat = remove_empty_row(mat)
    mat = np.array(remove_empty_row(np.array(mat).T.tolist())).T.tolist()

    return mat


def merge_text_in_cell(df_cell):
    """
    Merges text from multiple rows into a single cell and recalculates its bounding box.
    Values are sorted by rounded (y, x) coordinates.

    Args:
        df_cell (pandas.DataFrame): DataFrame containing cells to merge.

    Returns:
        pandas.DataFrame: Updated DataFrame with merged text and a single bounding box.
    """
    ocr_boxes = np.stack(df_cell["ocr_box"].values)

    df_cell["x"] = (ocr_boxes[:, 0] - ocr_boxes[:, 0].min()) // 10
    df_cell["y"] = (ocr_boxes[:, 1] - ocr_boxes[:, 1].min()) // 10
    df_cell = df_cell.sort_values(["y", "x"])

    text = " ".join(df_cell["text"].values.tolist())
    df_cell["text"] = text
    df_cell = df_cell.head(1)
    df_cell["ocr_box"] = df_cell["cell"]
    df_cell.drop(["x", "y"], axis=1, inplace=True)

    return df_cell


def remove_empty_row(mat):
    """
    Remove empty rows from a matrix.

    Args:
        mat (list[list]): The matrix to remove empty rows from.

    Returns:
        list[list]: The matrix with empty rows removed.
    """
    mat_filter = []
    for row in mat:
        if max([len(c) for c in row]):
            mat_filter.append(row)
    return mat_filter


def display_markdown(
    data: list[list[str]],
    use_header: bool = False,
) -> str:
    """
    Convert a list of lists of strings into a markdown table.

    Parameters:
        data (list[list[str]]): The table data. The first sublist should contain headers.
        use_header (bool, optional): Whether to use the first sublist as headers. Defaults to True.

    Returns:
        str: A markdown-formatted table as a string.
    """
    if not len(data):
        return "EMPTY TABLE"

    max_cols = max(len(row) for row in data)
    data = [row + [""] * (max_cols - len(row)) for row in data]

    if use_header:
        header = "| " + " | ".join(data[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in data[1:])
        markdown_table = f"{header}\n{separator}\n{body}" if body else f"{header}\n{separator}"

    else:
        markdown_table = "\n".join("| " + " | ".join(row) + " |" for row in data)

    return markdown_table


def reorder_boxes(boxes, texts, confs, mode="top_left", dbscan_eps=10):
    """
    Reorders the boxes in reading order.
    If mode is "center", the boxes are reordered using bbox center.
    If mode is "top_left", the boxes are reordered using the top left corner.
    If dbscan_eps is not 0, the boxes are reordered using DBSCAN clustering.

    Args:
        boxes (np array [n x 4 x 2]): The bounding boxes of the OCR results.
        texts (np array [n]): The text of the OCR results.
        confs (np array [n]): The confidence scores of the OCR results.
        mode (str, optional): The mode to reorder the boxes. Defaults to "center".
        dbscan_eps (float, optional): The epsilon parameter for DBSCAN. Defaults to 10.

    Returns:
        List[List[int, ...]]: The reordered bounding boxes.
        List[str]: The reordered texts.
        List[float]: The reordered confidence scores.
    """
    df = pd.DataFrame(
        [[b, t, c] for b, t, c in zip(boxes, texts, confs)],
        columns=["bbox", "text", "conf"],
    )

    if mode == "center":
        df["x"] = df["bbox"].apply(lambda box: (box[0][0] + box[2][0]) / 2)
        df["y"] = df["bbox"].apply(lambda box: (box[0][1] + box[2][1]) / 2)
    elif mode == "top_left":
        df["x"] = df["bbox"].apply(lambda box: (box[0][0]))
        df["y"] = df["bbox"].apply(lambda box: (box[0][1]))

    if dbscan_eps:
        do_naive_sorting = False
        try:
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=1)
            dbscan.fit(df["y"].values[:, None])
            df["cluster"] = dbscan.labels_
            df["cluster_centers"] = df.groupby("cluster")["y"].transform("mean").astype(int)
            df = df.sort_values(["cluster_centers", "x"], ascending=[True, True], ignore_index=True)
        except ValueError:
            do_naive_sorting = True
    else:
        do_naive_sorting = True

    if do_naive_sorting:
        df["y"] = np.round((df["y"] - df["y"].min()) // 5, 0)
        df = df.sort_values(["y", "x"], ascending=[True, True], ignore_index=True)

    bboxes = df["bbox"].values.tolist()
    texts = df["text"].values.tolist()
    confs = df["conf"].values.tolist()

    return bboxes, texts, confs
