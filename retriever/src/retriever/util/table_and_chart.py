"""
Table/chart/infographic content reconstruction utilities.

Ports bbox-matching and content-reconstruction algorithms from
``nv_ingest_api.util.image_processing.table_and_chart`` and adds adapter
functions that convert the retriever's detection/OCR formats into the
pixel-coordinate representations expected by the core joining routines.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core algorithms ported from nv-ingest
# ---------------------------------------------------------------------------


def match_bboxes(
    yolox_box: np.ndarray,
    ocr_boxes: np.ndarray,
    already_matched: Optional[list] = None,
    delta: float = 2.0,
) -> np.ndarray:
    """Union-based IoU matching for chart graphic elements."""
    x0_1, y0_1, x1_1, y1_1 = yolox_box
    x0_2, y0_2, x1_2, y1_2 = (
        ocr_boxes[:, 0],
        ocr_boxes[:, 1],
        ocr_boxes[:, 2],
        ocr_boxes[:, 3],
    )

    inter_y0 = np.maximum(y0_1, y0_2)
    inter_y1 = np.minimum(y1_1, y1_2)
    inter_x0 = np.maximum(x0_1, x0_2)
    inter_x1 = np.minimum(x1_1, x1_2)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    area_1 = (y1_1 - y0_1) * (x1_1 - x0_1)
    area_2 = (y1_2 - y0_2) * (x1_2 - x0_2)
    union_area = area_1 + area_2 - inter_area

    ious = inter_area / union_area
    max_iou = np.max(ious)
    if max_iou <= 0.01:
        return []

    matches = np.where(ious > (max_iou / delta))[0]
    if already_matched is not None:
        matches = np.array([m for m in matches if m not in already_matched])
    return matches


def assign_boxes(
    ocr_box: np.ndarray,
    boxes: np.ndarray,
    delta: float = 2.0,
    min_overlap: float = 0.25,
) -> np.ndarray:
    """Area-normalized overlap matching for table structure."""
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

    inter_y0 = np.maximum(y0_1, y0_2)
    inter_y1 = np.minimum(y1_1, y1_2)
    inter_x0 = np.maximum(x0_1, x0_2)
    inter_x1 = np.minimum(x1_1, x1_2)
    inter_area = np.maximum(0, inter_y1 - inter_y0) * np.maximum(0, inter_x1 - inter_x0)

    area_1 = (y1_1 - y0_1) * (x1_1 - x0_1)
    ious = inter_area / (area_1 + 1e-6)

    max_iou = np.max(ious)
    if max_iou <= min_overlap:
        return []

    n = len(np.where(ious >= (max_iou / delta))[0])
    matches = np.argsort(-ious)[:n]
    return matches


def _join_yolox_graphic_elements_and_ocr_output(
    yolox_output: Dict[str, list],
    ocr_boxes: np.ndarray,
    ocr_txts: list,
) -> Dict[str, str]:
    """Match graphic-element detections to OCR text via IoU."""
    KEPT_CLASSES = [
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

    if ocr_txts.size == 0 or ocr_boxes.size == 0:
        return {}

    # Convert quadrilateral (N,4,2) → xyxy (N,4).
    ocr_boxes = np.array(
        [
            ocr_boxes[:, :, 0].min(-1),
            ocr_boxes[:, :, 1].min(-1),
            ocr_boxes[:, :, 0].max(-1),
            ocr_boxes[:, :, 1].max(-1),
        ]
    ).T

    already_matched: list = []
    results: Dict[str, str] = {}

    for k in KEPT_CLASSES:
        if not len(yolox_output.get(k, [])):
            continue

        texts = []
        for yolox_box in yolox_output[k]:
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
            processed_texts = " - ".join(processed_texts)

        results[k] = processed_texts

    return results


def process_yolox_graphic_elements(yolox_text_dict: Dict[str, str]) -> str:
    """Concatenate chart text by semantic region."""
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


def _join_yolox_table_structure_and_ocr_output(
    yolox_cell_preds: Dict[str, np.ndarray],
    ocr_boxes: list,
    ocr_txts: list,
) -> str:
    """Combine table-structure cell/row/column predictions with OCR text."""
    if not ocr_boxes or not ocr_txts:
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
        matches_cell = assign_boxes(b, yolox_cell_preds["cell"], delta=1)
        cell = yolox_cell_preds["cell"][matches_cell[0]] if len(matches_cell) else b

        matches_row = assign_boxes(cell, yolox_cell_preds["row"], delta=1)
        row_ids = matches_row if len(matches_row) else -1

        if isinstance(row_ids, np.ndarray):
            delta = 2 if row_ids.min() == 0 else 1
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

    df_assign = pd.DataFrame(assignments)

    dfs = []
    for cell_id, df_cell in df_assign.groupby("cell_id"):
        if len(df_cell) > 1 and cell_id > -1:
            df_cell = merge_text_in_cell(df_cell)
        dfs.append(df_cell)
    df_assign = pd.concat(dfs)

    df_text = df_assign[~df_assign["is_table"]].reset_index(drop=True)

    df_table = df_assign[df_assign["is_table"]].reset_index(drop=True)
    if len(df_table):
        mat = build_markdown(df_table)
        markdown_table = display_markdown(mat, use_header=True)

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
        df_text = pd.concat([df_text, df_table_to_text], ignore_index=True)

    df_text = df_text.rename(columns={"ocr_box": "box"})

    df_text["x"] = df_text["box"].apply(lambda x: (x[0] + x[2]) / 2)
    df_text["y"] = df_text["box"].apply(lambda x: (x[1] + x[3]) / 2)
    df_text["x"] = (df_text["x"] - df_text["x"].min()) // 10
    df_text["y"] = (df_text["y"] - df_text["y"].min()) // 20
    df_text = df_text.sort_values(["y", "x"], ignore_index=True)

    rows_list = []
    for r, df_row in df_text.groupby("y"):
        if df_row["is_table"].values.any():
            table = df_row[df_row["is_table"]]
            df_row = df_row[~df_row["is_table"]]
        else:
            table = None

        if len(df_row) > 1:
            df_row = df_row.reset_index(drop=True)
            df_row["text"] = "\n".join(df_row["text"].values.tolist())

        rows_list.append(df_row.head(1))

        if table is not None:
            rows_list.append(table)

    df_display = pd.concat(rows_list, ignore_index=True)
    result = "\n".join(df_display.text.values.tolist())
    return result


def build_markdown(df: pd.DataFrame) -> list:
    """Convert a dataframe with row_ids/col_ids/text into a markdown matrix."""
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

    mat = remove_empty_row(mat)
    mat = np.array(remove_empty_row(np.array(mat).T.tolist())).T.tolist()
    return mat


def display_markdown(data: list, use_header: bool = False) -> str:
    """Convert a list-of-lists into a markdown table string."""
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


def merge_text_in_cell(df_cell: pd.DataFrame) -> pd.DataFrame:
    """Merge text from multiple OCR items inside one table cell."""
    ocr_boxes = np.stack(df_cell["ocr_box"].values)

    df_cell = df_cell.copy()
    df_cell["x"] = (ocr_boxes[:, 0] - ocr_boxes[:, 0].min()) // 10
    df_cell["y"] = (ocr_boxes[:, 1] - ocr_boxes[:, 1].min()) // 10
    df_cell = df_cell.sort_values(["y", "x"])

    text = " ".join(df_cell["text"].values.tolist())
    df_cell["text"] = text
    df_cell = df_cell.head(1)
    df_cell["ocr_box"] = df_cell["cell"]
    df_cell = df_cell.drop(["x", "y"], axis=1)
    return df_cell


def remove_empty_row(mat: list) -> list:
    """Remove empty rows from a matrix."""
    mat_filter = []
    for row in mat:
        if max([len(c) for c in row]):
            mat_filter.append(row)
    return mat_filter


def reorder_boxes(
    boxes: np.ndarray,
    texts: list,
    confs: list,
    mode: str = "top_left",
    dbscan_eps: float = 10,
) -> Tuple[list, list, list]:
    """Reorder OCR boxes in reading order using DBSCAN clustering."""
    df = pd.DataFrame(
        [[b, t, c] for b, t, c in zip(boxes, texts, confs)],
        columns=["bbox", "text", "conf"],
    )

    if mode == "center":
        df["x"] = df["bbox"].apply(lambda box: (box[0][0] + box[2][0]) / 2)
        df["y"] = df["bbox"].apply(lambda box: (box[0][1] + box[2][1]) / 2)
    elif mode == "top_left":
        df["x"] = df["bbox"].apply(lambda box: box[0][0])
        df["y"] = df["bbox"].apply(lambda box: box[0][1])

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


# ---------------------------------------------------------------------------
# Adapter functions  (retriever formats → nv-ingest formats)
# ---------------------------------------------------------------------------


def _normalize_ocr_items(preds: Any) -> List[Dict[str, Any]]:
    """Normalize any OCR output format to ``[{"left", "right", "upper", "lower", "text"}, ...]``.

    Handles both list-of-dict (Nemotron OCR normalized-coord form) and
    dict with ``boxes``/``texts`` keys (packed form).
    """
    items: List[Dict[str, Any]] = []

    if isinstance(preds, list):
        for item in preds:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("left", "right", "upper", "lower")):
                continue
            txt = str(item.get("text") or "").strip()
            if not txt or txt == "nan":
                continue
            items.append(
                {
                    "left": float(item["left"]),
                    "right": float(item["right"]),
                    "upper": float(item["upper"]),
                    "lower": float(item["lower"]),
                    "text": txt,
                }
            )
    elif isinstance(preds, dict):
        pb = preds.get("boxes") or preds.get("bboxes") or []
        pt = preds.get("texts") or preds.get("text") or []
        if isinstance(pb, list) and isinstance(pt, list):
            for b, txt in zip(pb, pt):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                        xs = [float(p[0]) for p in b]
                        ys = [float(p[1]) for p in b]
                        items.append(
                            {
                                "left": min(xs),
                                "right": max(xs),
                                "upper": min(ys),
                                "lower": max(ys),
                                "text": txt.strip(),
                            }
                        )
                    elif all(isinstance(v, (int, float)) for v in b):
                        items.append(
                            {
                                "left": float(b[0]),
                                "upper": float(b[1]),
                                "right": float(b[2]),
                                "lower": float(b[3]),
                                "text": txt.strip(),
                            }
                        )

    return items


def _ocr_items_to_pixel_quad_boxes(
    ocr_items: List[Dict[str, Any]],
    crop_hw: Tuple[int, int],
) -> Tuple[list, list]:
    """Convert normalized OCR items to pixel-coordinate quadrilateral boxes.

    Returns ``(quad_boxes, texts)`` where *quad_boxes* is a list of
    ``[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]`` arrays (pixel coords) and
    *texts* is the corresponding list of strings.
    """
    H, W = crop_hw
    quad_boxes: list = []
    texts: list = []
    for item in ocr_items:
        left = float(item["left"]) * W
        right = float(item["right"]) * W
        upper = float(item["upper"]) * H
        lower = float(item["lower"]) * H
        quad_boxes.append([[left, upper], [right, upper], [right, lower], [left, lower]])
        texts.append(item["text"])
    return quad_boxes, texts


def _structure_dets_to_class_boxes(
    dets: List[Dict[str, Any]],
    crop_hw: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Group structure-model detections by label_name and scale to pixel coords.

    Parameters
    ----------
    dets : list[dict]
        Output of ``_prediction_to_detections()`` — each dict has
        ``bbox_xyxy_norm`` (normalized [0,1]) and ``label_name``.
    crop_hw : (int, int)
        ``(H, W)`` of the crop image.

    Returns
    -------
    dict[str, ndarray]
        ``{label_name: array_of_shape_(N, 4)}`` in pixel coordinates.
    """
    H, W = crop_hw
    grouped: Dict[str, list] = {}
    for d in dets:
        name = d.get("label_name", "")
        bbox = d.get("bbox_xyxy_norm")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = float(bbox[0]) * W, float(bbox[1]) * H, float(bbox[2]) * W, float(bbox[3]) * H
        grouped.setdefault(name, []).append([x1, y1, x2, y2])

    return {k: np.array(v) for k, v in grouped.items()}


def join_table_structure_and_ocr_output(
    structure_dets: List[Dict[str, Any]],
    ocr_preds: Any,
    crop_hw: Tuple[int, int],
) -> str:
    """Adapter: convert retriever table-structure detections + OCR items,
    then call the core joining function.

    Parameters
    ----------
    structure_dets : list[dict]
        From ``_prediction_to_detections()`` with label_names cell/row/column
        and ``bbox_xyxy_norm`` in [0, 1].
    ocr_preds : list | dict
        Raw OCR output from ``NemotronOCRV1.invoke()``.
    crop_hw : (int, int)
        ``(H, W)`` of the crop image.
    """
    ocr_items = _normalize_ocr_items(ocr_preds)
    if not ocr_items:
        return ""

    class_boxes = _structure_dets_to_class_boxes(structure_dets, crop_hw)

    # Ensure all three required keys exist.
    cell_preds: Dict[str, np.ndarray] = {
        "cell": class_boxes.get("cell", np.empty((0, 4))),
        "row": class_boxes.get("row", np.empty((0, 4))),
        "column": class_boxes.get("column", np.empty((0, 4))),
    }

    if cell_preds["cell"].shape[0] == 0:
        return ""

    quad_boxes, texts = _ocr_items_to_pixel_quad_boxes(ocr_items, crop_hw)
    return _join_yolox_table_structure_and_ocr_output(cell_preds, quad_boxes, texts)


def join_graphic_elements_and_ocr_output(
    ge_dets: List[Dict[str, Any]],
    ocr_preds: Any,
    crop_hw: Tuple[int, int],
) -> str:
    """Adapter: convert retriever graphic-elements detections + OCR items,
    then call the core joining + concatenation functions.

    Parameters
    ----------
    ge_dets : list[dict]
        From ``_prediction_to_detections()`` with chart-element label_names
        and ``bbox_xyxy_norm`` in [0, 1].
    ocr_preds : list | dict
        Raw OCR output from ``NemotronOCRV1.invoke()``.
    crop_hw : (int, int)
        ``(H, W)`` of the crop image.
    """
    ocr_items = _normalize_ocr_items(ocr_preds)
    if not ocr_items:
        return ""

    class_boxes = _structure_dets_to_class_boxes(ge_dets, crop_hw)
    if not class_boxes:
        return ""

    # Convert class_boxes values from (N,4) arrays to list-of-arrays (one per detection).
    yolox_output: Dict[str, list] = {}
    for k, arr in class_boxes.items():
        yolox_output[k] = [arr[i] for i in range(arr.shape[0])]

    quad_boxes, texts = _ocr_items_to_pixel_quad_boxes(ocr_items, crop_hw)

    matched = _join_yolox_graphic_elements_and_ocr_output(yolox_output, quad_boxes, texts)
    if not matched:
        return ""

    return process_yolox_graphic_elements(matched)


def reorder_ocr_for_infographic(
    ocr_preds: Any,
    crop_hw: Tuple[int, int],
) -> str:
    """Adapter: convert OCR items to pixel-coord quad boxes, reorder in
    reading order, and return joined text.
    """
    ocr_items = _normalize_ocr_items(ocr_preds)
    if not ocr_items:
        return ""

    quad_boxes, texts = _ocr_items_to_pixel_quad_boxes(ocr_items, crop_hw)
    if not quad_boxes:
        return ""

    confs = [1.0] * len(texts)
    _, reordered_texts, _ = reorder_boxes(
        np.array(quad_boxes), texts, confs, mode="top_left", dbscan_eps=10,
    )
    return "\n".join(t for t in reordered_texts if t)
