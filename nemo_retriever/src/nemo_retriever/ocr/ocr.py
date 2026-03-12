# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Crop-based OCR stage.

Runs Nemotron OCR v1 on table / chart / infographic regions detected by
PageElements v3. Text extraction for the full page is handled upstream
by PDFium in the PDF extraction stage.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import time
import traceback

import numpy as np
import pandas as pd
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.nim.nim import invoke_image_inference_batches
from nemo_retriever.utils.table_and_chart import join_graphic_elements_and_ocr_output

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Page-element labels that carry running text (as opposed to structured
# content like tables/charts/infographics).  Used by the OCR stage to
# decide which detections contribute to the page's ``text`` column.
_TEXT_LABELS: frozenset[str] = frozenset({"text", "title", "header_footer"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """
    Crop a base64-encoded RGB image by a normalized xyxy bbox.

    Returns
    -------
    cropped_image_b64 : str | None
        Base64-encoded cropped image (PNG), or *None* on failure.
    cropped_shape_hw : tuple[int, int] | None
        (H, W) of the crop, or *None* on failure.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return None, None
    try:
        x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    except Exception:
        return None, None

    try:
        raw = base64.b64decode(page_image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            if w <= 1 or h <= 1:
                return None, None

            def _clamp_int(v: float, lo: int, hi: int) -> int:
                if v != v:  # NaN
                    return lo
                return int(min(max(v, float(lo)), float(hi)))

            x1 = _clamp_int(x1n * w, 0, w)
            x2 = _clamp_int(x2n * w, 0, w)
            y1 = _clamp_int(y1n * h, 0, h)
            y2 = _clamp_int(y2n * h, 0, h)

            if x2 <= x1 or y2 <= y1:
                return None, None

            crop = im.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 1 or ch <= 1:
                return None, None

            buf = io.BytesIO()
            fmt = str(image_format or "png").lower()
            if fmt not in {"png"}:
                fmt = "png"
            crop.save(buf, format=fmt.upper())
            return base64.b64encode(buf.getvalue()).decode("ascii"), (int(ch), int(cw))
    except Exception:
        return None, None


def _crop_all_from_page(
    page_image_b64: str,
    detections: List[Dict[str, Any]],
    wanted_labels: set,
    *,
    as_b64: bool = False,
) -> List[Tuple[str, List[float], Any]]:
    """
    Decode the page image **once** and crop all matching detections.

    Returns a list of ``(label_name, bbox_xyxy_norm, value)`` tuples for
    detections whose ``label_name`` is in *wanted_labels* and whose crop is
    valid.  Skips detections that fail to crop (bad bbox, tiny region, etc.).

    When *as_b64* is ``False`` (default), *value* is an HWC uint8 numpy array
    suitable for local model inference.  When ``True``, *value* is a base64-
    encoded PNG string — this avoids a wasteful numpy→PIL→PNG round-trip on
    the remote inference path.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return []

    try:
        raw = base64.b64decode(page_image_b64)
        im0 = Image.open(io.BytesIO(raw))
        im = im0.convert("RGB")
        im0.close()
    except Exception:
        return []

    w, h = im.size
    if w <= 1 or h <= 1:
        im.close()
        return []

    def _clamp_int(v: float, lo: int, hi: int) -> int:
        if v != v:  # NaN
            return lo
        return int(min(max(v, float(lo)), float(hi)))

    results: List[Tuple[str, List[float], Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label_name = str(det.get("label_name") or "").strip()
        if label_name not in wanted_labels:
            continue

        bbox = det.get("bbox_xyxy_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1n, y1n, x2n, y2n = [float(x) for x in bbox]
        except Exception:
            continue

        x1 = _clamp_int(x1n * w, 0, w)
        x2 = _clamp_int(x2n * w, 0, w)
        y1 = _clamp_int(y1n * h, 0, h)
        y2 = _clamp_int(y2n * h, 0, h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = im.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        if cw <= 1 or ch <= 1:
            crop.close()
            continue

        if as_b64:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop.close()
            value = base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            value = np.asarray(crop, dtype=np.uint8).copy()
            crop.close()
        results.append((label_name, [float(x) for x in bbox], value))

    im.close()
    return results


def _np_rgb_to_b64_png(crop_array: np.ndarray) -> str:
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding.")
    img = Image.fromarray(crop_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_remote_ocr_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        # NIM text_detections format: return full list (not v[0])
        td = response_item.get("text_detections")
        if isinstance(td, list) and td:
            return td
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


def _parse_ocr_result(preds: Any) -> List[Dict[str, Any]]:
    """
    Parse the output of ``NemotronOCRV1.invoke()`` into a flat list of
    ``{"text": str, "sort_y": float, "sort_x": float}`` blocks.

    The model may return:
    * ``dict`` with ``boxes`` / ``texts`` keys (packed form), or
    * ``list[dict]`` with ``left``/``right``/``upper``/``lower``/``text`` keys
      (Nemotron OCR normalized-coord form), or
    * ``list[dict]`` with generic ``text`` + ``box``/``bbox`` keys.
    """
    blocks: List[Dict[str, Any]] = []

    # ---- dict form: {"boxes": [...], "texts": [...]} ----
    if isinstance(preds, dict):
        pb = preds.get("boxes") or preds.get("bboxes") or preds.get("bounding_boxes")
        pt = preds.get("texts") or preds.get("text_predictions") or preds.get("text")
        if isinstance(pb, list) and isinstance(pt, list):
            for b, txt in zip(pb, pt):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                sort_y, sort_x = 0.0, 0.0
                if isinstance(b, list):
                    if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                        # quadrilateral [[x1,y1], ...]
                        sort_y = float(b[0][1])
                        sort_x = float(b[0][0])
                    elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        # xyxy [x1, y1, x2, y2]
                        sort_y = float(b[1])
                        sort_x = float(b[0])
                blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})
        return blocks

    # ---- list form: list[dict] with various key conventions ----
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"text": item.strip(), "sort_y": 0.0, "sort_x": 0.0})
                continue
            if not isinstance(item, dict):
                continue

            # NIM text_detections format:
            # {"text_prediction": {"text": "...", "confidence": ...},
            #  "bounding_box": {"points": [{"x": ..., "y": ...}, ...]}}
            tp = item.get("text_prediction")
            if isinstance(tp, dict):
                txt0 = str(tp.get("text") or "").strip()
                if txt0 and txt0 != "nan":
                    sort_y, sort_x = 0.0, 0.0
                    bb = item.get("bounding_box")
                    if isinstance(bb, dict):
                        pts = bb.get("points")
                        if isinstance(pts, list) and pts:
                            try:
                                sort_x = float(pts[0].get("x", 0.0))
                                sort_y = float(pts[0].get("y", 0.0))
                            except Exception:
                                pass
                    blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            # Nemotron OCR normalized-coord form
            if all(k in item for k in ("left", "right", "upper", "lower")) and isinstance(item.get("text"), str):
                txt0 = str(item.get("text") or "").strip()
                if not txt0 or txt0 == "nan":
                    continue
                try:
                    sort_x = float(item["left"])
                    sort_y = float(item["lower"])
                except Exception:
                    sort_x, sort_y = 0.0, 0.0
                blocks.append({"text": txt0, "sort_y": sort_y, "sort_x": sort_x})
                continue

            # Generic text + box fallback
            txt = item.get("text") or item.get("ocr_text") or item.get("generated_text") or item.get("output_text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            sort_y, sort_x = 0.0, 0.0
            b = item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("bbox_points")
            if isinstance(b, list):
                if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                    sort_y = float(b[0][1])
                    sort_x = float(b[0][0])
                elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    sort_y = float(b[1])
                    sort_x = float(b[0])
            blocks.append({"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x})

    # ---- last-resort stringify ----
    if not blocks and preds is not None:
        s = ""
        try:
            s = str(preds).strip()
        except Exception:
            s = ""
        if s and s.lower() not in {"none", "null", "[]", "{}"}:
            blocks.append({"text": s, "sort_y": 0.0, "sort_x": 0.0})

    return blocks


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Sort text blocks by reading order (y then x) and join with whitespace."""
    blocks.sort(key=lambda b: (b.get("sort_y", 0.0), b.get("sort_x", 0.0)))
    return " ".join(b["text"] for b in blocks if b.get("text"))


def _blocks_to_pseudo_markdown(
    blocks: List[Dict[str, Any]],
    crop_hw: Tuple[int, int] = (0, 0),
) -> str:
    """Convert OCR text blocks into pseudo-markdown table format.

    Uses DBSCAN clustering on pixel y-coordinates to identify rows, then
    sorts within each row by x-coordinate and joins with pipe separators.

    Parameters
    ----------
    blocks : list of dict
        OCR text blocks with ``sort_y`` (normalised [0,1]) and ``sort_x``.
    crop_hw : (height, width)
        Pixel dimensions of the crop image.  When provided the normalised
        ``sort_y`` values are scaled to pixels and clustered with
        ``eps=10`` (matching nv-ingest behaviour).  Falls back to the old
        normalised-space heuristic when the height is unavailable.
    """
    if not blocks:
        return ""

    valid = [b for b in blocks if b.get("text")]
    if not valid:
        return ""

    from sklearn.cluster import DBSCAN

    df = pd.DataFrame(valid)
    df = df.sort_values("sort_y")

    y_vals = df["sort_y"].values
    crop_h = crop_hw[0] if crop_hw else 0

    if crop_h > 0:
        # Pixel-space clustering (matches nv-ingest eps=10).
        y_pixels = (y_vals * crop_h).astype(int)
        eps = 10
    else:
        # Fallback: normalise to [0,1] when pixel dims are unknown.
        y_range = y_vals.max() - y_vals.min()
        if y_range > 0:
            y_pixels = (y_vals - y_vals.min()) / y_range
            eps = 0.03
        else:
            y_pixels = y_vals
            eps = 0.1

    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(y_pixels.reshape(-1, 1))
    df["cluster"] = dbscan.labels_

    df = df.sort_values(["cluster", "sort_x"])

    rows = []
    for _, grp in df.groupby("cluster", sort=True):
        rows.append("| " + " | ".join(grp["text"].tolist()) + " |")
    return "\n".join(rows)


def _bboxes_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    """Check if two normalized bboxes are approximately equal."""
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(4))


def _find_ge_detections_for_bbox(
    row: Any,
    chart_bbox: Sequence[float],
) -> Optional[List[Dict[str, Any]]]:
    """Find graphic element detections for a chart bbox.

    Reads the ``graphic_elements_v1`` column from *row* and returns the
    detections list for the region whose ``bbox_xyxy_norm`` matches
    *chart_bbox*, or ``None`` if no match is found.
    """
    ge_col = getattr(row, "graphic_elements_v1", None)
    if not isinstance(ge_col, dict):
        return None
    regions = ge_col.get("regions")
    if not isinstance(regions, list):
        return None

    for region in regions:
        if not isinstance(region, dict):
            continue
        region_bbox = region.get("bbox_xyxy_norm")
        if not isinstance(region_bbox, (list, tuple)) or len(region_bbox) != 4:
            continue
        if _bboxes_close(chart_bbox, region_bbox):
            dets = region.get("detections")
            if isinstance(dets, list) and dets:
                return dets
    return None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def ocr_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    use_graphic_elements: bool = False,
    inference_batch_size: int = 8,
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run Nemotron OCR v1 on cropped regions detected by PageElements v3.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. For each detection whose ``label_name`` is a requested type, crop the
       page image, invoke OCR, parse the result, and collect text.
    3. Write per-type content lists and timing metadata to output columns.

    Parameters
    ----------
    batch_df : pandas.DataFrame
        Ray Data batch with ``page_elements_v3`` and ``page_image`` columns.
    model : NemotronOCRV1
        Initialised OCR model.
    extract_tables, extract_charts, extract_infographics : bool
        Which element types to OCR.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``table``, ``chart``,
        ``infographic``, and ``ocr_v1``.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("ocr_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Determine which labels we need to process.
    # Text/title labels are added per-row based on needs_ocr_for_text metadata.
    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    # Per-row accumulators.
    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_text: List[str] = []
    all_ocr_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        row_ocr_text_blocks: List[Dict[str, Any]] = []
        row_error: Any = None

        try:
            # --- get page elements detections ---
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            # --- get page image ---
            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None

            if not isinstance(page_image_b64, str) or not page_image_b64:
                # No image available — nothing to crop/OCR.
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_text.append(None)
                all_ocr_meta.append({"timing": None, "error": None})
                continue

            # --- determine per-row labels (text/title only for pages needing OCR) ---
            row_wanted = wanted_labels
            if extract_text:
                meta = getattr(row, "metadata", None) or {}
                needs_ocr = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
                if needs_ocr:
                    row_wanted = wanted_labels | _TEXT_LABELS

            # --- decode page image once, crop all matching detections ---
            if use_remote:
                crops = _crop_all_from_page(page_image_b64, dets, row_wanted, as_b64=True)
                crop_b64s: List[str] = [b64 for _label, _bbox, b64 in crops]
                crop_meta: List[Tuple[str, List[float]]] = [(label, bbox) for label, bbox, _b64 in crops]

                if crop_b64s:
                    response_items = invoke_image_inference_batches(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_pool_workers=int(retry.remote_max_pool_workers),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} OCR responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        preds = _extract_remote_ocr_item(response_items[i])

                        if label_name == "chart" and use_graphic_elements:
                            ge_dets = _find_ge_detections_for_bbox(row, bbox)
                            if ge_dets:
                                # Decode crop dimensions from the b64 PNG for graphic element joining.
                                crop_hw = (0, 0)
                                try:
                                    _raw = base64.b64decode(crop_b64s[i])
                                    with Image.open(io.BytesIO(_raw)) as _cim:
                                        _cw, _ch = _cim.size
                                        crop_hw = (_ch, _cw)
                                except Exception:
                                    pass
                                text = join_graphic_elements_and_ocr_output(ge_dets, preds, crop_hw)
                                if text:
                                    chart_items.append({"bbox_xyxy_norm": bbox, "text": text})
                                    continue

                        blocks = _parse_ocr_result(preds)
                        if label_name == "table":
                            crop_hw_table: Tuple[int, int] = (0, 0)
                            try:
                                _raw = base64.b64decode(crop_b64s[i])
                                with Image.open(io.BytesIO(_raw)) as _cim:
                                    _cw, _ch = _cim.size
                                    crop_hw_table = (_ch, _cw)
                            except Exception:
                                pass
                            text = _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw_table) or _blocks_to_text(blocks)
                        else:
                            text = _blocks_to_text(blocks)
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name in _TEXT_LABELS:
                            row_ocr_text_blocks.extend(blocks)
            else:
                crops = _crop_all_from_page(page_image_b64, dets, row_wanted)

                if inference_batch_size is None or inference_batch_size < 1:
                    raise ValueError(
                        f"inference_batch_size must be set and greater than 0. Value: {inference_batch_size}"
                    )

                local_batch_size = max(1, int(inference_batch_size))

                # Tables require word-level merging; charts/infographics use paragraph-level.
                # Group by merge level so each batched invoke uses one consistent setting.
                local_jobs: Dict[str, List[Tuple[str, List[float], np.ndarray]]] = {"word": [], "paragraph": []}
                for label_name, bbox, crop_array in crops:
                    ml = "word" if label_name == "table" else "paragraph"
                    local_jobs[ml].append((label_name, bbox, crop_array))

                def _append_local_result(
                    label_name: str, bbox: List[float], preds: Any, crop_hw: Tuple[int, int] = (0, 0)
                ) -> None:
                    if label_name == "chart" and use_graphic_elements:
                        ge_dets = _find_ge_detections_for_bbox(row, bbox)
                        if ge_dets:
                            text = join_graphic_elements_and_ocr_output(ge_dets, preds, crop_hw)
                            if text:
                                chart_items.append({"bbox_xyxy_norm": bbox, "text": text})
                                return
                    blocks = _parse_ocr_result(preds)
                    if label_name == "table":
                        text = _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw)
                        if not text:
                            text = _blocks_to_text(blocks)
                    else:
                        text = _blocks_to_text(blocks)
                    entry = {"bbox_xyxy_norm": bbox, "text": text}
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name in _TEXT_LABELS:
                        row_ocr_text_blocks.extend(blocks)

                for ml, jobs in local_jobs.items():
                    if not jobs:
                        continue
                    for start in range(0, len(jobs), local_batch_size):
                        batch_jobs = jobs[start : start + local_batch_size]
                        batch_crops = [crop_array for _, _, crop_array in batch_jobs]

                        # Try batched invoke first; if backend does not return one response
                        # per input, fall back to per-item to preserve correctness.
                        try:
                            batch_preds = model.invoke(batch_crops, merge_level=ml)
                        except Exception:
                            batch_preds = None

                        if isinstance(batch_preds, list) and len(batch_preds) == len(batch_jobs):
                            for (label_name, bbox, crop_array), preds in zip(batch_jobs, batch_preds):
                                _append_local_result(
                                    label_name, bbox, preds, crop_hw=(crop_array.shape[0], crop_array.shape[1])
                                )
                        else:
                            for label_name, bbox, crop_array in batch_jobs:
                                preds = model.invoke(crop_array, merge_level=ml)
                                _append_local_result(
                                    label_name, bbox, preds, crop_hw=(crop_array.shape[0], crop_array.shape[1])
                                )

        except BaseException as e:
            print(f"Warning: OCR failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        # Assemble OCR'd text from text/title detections for this row.
        # Use None as sentinel for "keep existing native text".
        if extract_text and row_ocr_text_blocks:
            all_text.append(_blocks_to_text(row_ocr_text_blocks))
        else:
            all_text.append(None)

        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_ocr_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total

    # Fill timing into metadata.
    for meta in all_ocr_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    # TODO: Is this actually a necessary copy?
    out = batch_df.copy()
    # Only overwrite content columns that this call is responsible for.
    # When extract_tables=False, preserve any existing `table` column
    # (e.g. populated by an upstream table-structure+OCR stage).
    if extract_tables or "table" not in out.columns:
        out["table"] = all_table
    if extract_charts or "chart" not in out.columns:
        out["chart"] = all_chart
    if extract_infographics or "infographic" not in out.columns:
        out["infographic"] = all_infographic
    if extract_text and "text" in out.columns:
        # Only overwrite rows where OCR produced text; preserve native text otherwise.
        for i, ocr_text in enumerate(all_text):
            if ocr_text is not None:
                out.iat[i, out.columns.get_loc("text")] = ocr_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["ocr_v1"] = all_ocr_meta
    return out


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------


class OCRActor:
    """
    Ray-friendly callable that initializes Nemotron OCR v1 once per actor.

    Usage with Ray Data::

        ds = ds.map_batches(
            OCRActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=8),
            fn_constructor_kwargs={
                "extract_tables": True,
                "extract_charts": True,
                "extract_infographics": False,
            },
        )
    """

    __slots__ = ("ocr_kwargs", "_model", "_remote_retry")

    def __init__(self, **ocr_kwargs: Any) -> None:
        import warnings

        if Image is not None:
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

        self.ocr_kwargs = dict(ocr_kwargs)
        invoke_url = str(self.ocr_kwargs.get("ocr_invoke_url") or self.ocr_kwargs.get("invoke_url") or "").strip()
        if invoke_url and "invoke_url" not in self.ocr_kwargs:
            self.ocr_kwargs["invoke_url"] = invoke_url

        # Normalize common constructor kwargs to expected runtime types/defaults.
        self.ocr_kwargs["extract_text"] = bool(self.ocr_kwargs.get("extract_text", False))
        self.ocr_kwargs["extract_tables"] = bool(self.ocr_kwargs.get("extract_tables", False))
        self.ocr_kwargs["extract_charts"] = bool(self.ocr_kwargs.get("extract_charts", False))
        self.ocr_kwargs["extract_infographics"] = bool(self.ocr_kwargs.get("extract_infographics", False))
        self.ocr_kwargs["use_graphic_elements"] = bool(self.ocr_kwargs.get("use_graphic_elements", False))
        self.ocr_kwargs["request_timeout_s"] = float(self.ocr_kwargs.get("request_timeout_s", 120.0))
        self.ocr_kwargs["inference_batch_size"] = int(self.ocr_kwargs.get("inference_batch_size", 8))

        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(self.ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(self.ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(self.ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        if invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._model = NemotronOCRV1()

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return ocr_page_elements(
                batch_df,
                model=self._model,
                remote_retry=self._remote_retry,
                **self.ocr_kwargs,
                **override_kwargs,
            )
        except BaseException as e:
            # Never let the Ray UDF raise — return a DataFrame with error metadata.
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["ocr_v1"] = [payload for _ in range(n)]
                return out
            return [{"ocr_v1": _error_payload(stage="actor_call", exc=e)}]


# ---------------------------------------------------------------------------
# Nemotron Parse v1.2
# ---------------------------------------------------------------------------


def _extract_parse_text(response_item: Any) -> str:
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def nemotron_parse_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run Nemotron Parse v1.2 on cropped page elements.

    Emits OCR-compatible content columns (``table``, ``chart``, ``infographic``)
    so this stage can replace the page-elements + OCR pair in pipeline wiring.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_text: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        row_text: Optional[str] = None
        row_error: Any = None

        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = pe.get("detections") or []
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
            if not isinstance(page_image_b64, str) or not page_image_b64:
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_text.append(None)
                all_meta.append({"timing": None, "error": None})
                continue

            if use_remote:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels, as_b64=True)
                # Parse-only mode may skip page-elements detection entirely. In that
                # case, parse the full page once and fan out the text to enabled
                # content channels.  The image is already base64 — pass it through.
                if not crops and wanted_labels:
                    crops = [("full_page", [0.0, 0.0, 1.0, 1.0], page_image_b64)]

                crop_b64s: List[str] = [b64 for _label, _bbox, b64 in crops]
                crop_meta: List[Tuple[str, List[float]]] = [(label, bbox) for label, bbox, _b64 in crops]

                if crop_b64s:
                    response_items = invoke_image_inference_batches(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_pool_workers=int(retry.remote_max_pool_workers),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} Parse responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        text = _extract_parse_text(response_items[i])
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name == "full_page":
                            if extract_tables:
                                table_items.append(dict(entry))
                            if extract_charts:
                                chart_items.append(dict(entry))
                            if extract_infographics:
                                infographic_items.append(dict(entry))
            else:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)
                if not crops and wanted_labels:
                    try:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        crops = [("full_page", [0.0, 0.0, 1.0, 1.0], full_crop)]
                    except Exception:
                        crops = []
                for label_name, bbox, crop_array in crops:
                    text = str(model.invoke(crop_array, task_prompt=task_prompt) or "").strip()
                    entry = {"bbox_xyxy_norm": bbox, "text": text}
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name == "full_page":
                        if extract_tables:
                            table_items.append(dict(entry))
                        if extract_charts:
                            chart_items.append(dict(entry))
                        if extract_infographics:
                            infographic_items.append(dict(entry))

            # When extract_text is requested, parse the full page for text
            # (only for pages that need OCR-based text extraction).
            meta = getattr(row, "metadata", None) or {}
            needs_ocr = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
            if extract_text and needs_ocr:
                try:
                    if use_remote:
                        resp = invoke_image_inference_batches(
                            invoke_url=invoke_url,
                            image_b64_list=[page_image_b64],
                            api_key=api_key,
                            timeout_s=float(request_timeout_s),
                            max_batch_size=1,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                            max_retries=int(retry.remote_max_retries),
                            max_429_retries=int(retry.remote_max_429_retries),
                        )
                        row_text = _extract_parse_text(resp[0]) if resp else ""
                    else:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        row_text = str(model.invoke(full_crop, task_prompt=task_prompt) or "").strip()
                except Exception:
                    row_text = ""

        except BaseException as e:
            print(f"Warning: Nemotron Parse failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "nemotron_parse_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_text.append(row_text)
        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_text and "text" in out.columns:
        # Only overwrite rows where parse produced text; preserve native text otherwise.
        for i, parse_text in enumerate(all_text):
            if parse_text is not None:
                out.iat[i, out.columns.get_loc("text")] = parse_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    # Aliases retained for experiments that read parse-specific columns.
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out


class NemotronParseActor:
    """
    Ray-friendly callable that initializes Nemotron Parse v1.2 once per actor.

    This actor is a drop-in map-batches stage intended for future pipeline
    wiring in batch/inprocess ingest modes.
    """

    __slots__ = (
        "_model",
        "_extract_tables",
        "_extract_charts",
        "_extract_infographics",
        "_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_task_prompt",
        "_remote_retry",
    )

    def __init__(
        self,
        *,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        self._invoke_url = (nemotron_parse_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        if self._invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronParseV12

            self._model = NemotronParseV12(task_prompt=self._task_prompt)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return nemotron_parse_page_elements(
                batch_df,
                model=self._model,
                invoke_url=self._invoke_url,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                task_prompt=self._task_prompt,
                extract_tables=self._extract_tables,
                extract_charts=self._extract_charts,
                extract_infographics=self._extract_infographics,
                remote_retry=self._remote_retry,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=e)}]
