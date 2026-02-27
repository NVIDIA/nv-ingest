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

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


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
) -> List[Tuple[str, List[float], np.ndarray]]:
    """
    Decode the page image **once** and crop all matching detections.

    Returns a list of ``(label_name, bbox_xyxy_norm, crop_array)`` tuples for
    detections whose ``label_name`` is in *wanted_labels* and whose crop is
    valid.  Skips detections that fail to crop (bad bbox, tiny region, etc.).

    Crops are returned as HWC uint8 numpy arrays so they can be passed
    directly to ``NemotronOCRV1.invoke()`` without a PNG/base64 round-trip.
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

    results: List[Tuple[str, List[float], np.ndarray]] = []
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

        crop_array = np.asarray(crop, dtype=np.uint8).copy()
        crop.close()
        results.append((label_name, [float(x) for x in bbox], crop_array))

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
    """Sort text blocks by reading order (y then x) and join with newlines."""
    blocks.sort(key=lambda b: (b.get("sort_y", 0.0), b.get("sort_x", 0.0)))
    return "\n".join(b["text"] for b in blocks if b.get("text"))


def _blocks_to_pseudo_markdown(blocks: List[Dict[str, Any]]) -> str:
    """Convert OCR text blocks into pseudo-markdown table format.

    Uses DBSCAN clustering on y-coordinates to identify rows, then
    sorts within each row by x-coordinate and joins with pipe separators.
    """
    if not blocks:
        return ""

    valid = [b for b in blocks if b.get("text")]
    if not valid:
        return ""

    from sklearn.cluster import DBSCAN

    df = pd.DataFrame(valid)

    # Normalize y-coordinates to [0,1] for scale-invariant clustering.
    y_vals = df["sort_y"].values
    y_range = y_vals.max() - y_vals.min()
    if y_range > 0:
        y_norm = (y_vals - y_vals.min()) / y_range
        eps = 0.03  # ~3% of bbox height ≈ one text line
    else:
        y_norm = y_vals
        eps = 0.1

    dbscan = DBSCAN(eps=eps, min_samples=1)
    dbscan.fit(y_norm.reshape(-1, 1))
    df["cluster"] = dbscan.labels_

    df = df.sort_values(["cluster", "sort_x"])

    rows = []
    for _, grp in df.groupby("cluster", sort=True):
        rows.append("| " + " | ".join(grp["text"].tolist()) + " |")
    return "\n".join(rows)


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
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
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
    all_ocr_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
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
                all_ocr_meta.append({"timing": None, "error": None})
                continue

            # --- decode page image once, crop all matching detections ---
            crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)

            if use_remote:
                crop_b64s: List[str] = []
                crop_meta: List[Tuple[str, List[float]]] = []
                for label_name, bbox, crop_array in crops:
                    crop_b64s.append(_np_rgb_to_b64_png(crop_array))
                    crop_meta.append((label_name, bbox))

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
                        blocks = _parse_ocr_result(preds)
                        if label_name == "table":
                            text = _blocks_to_pseudo_markdown(blocks) or _blocks_to_text(blocks)
                        else:
                            text = _blocks_to_text(blocks)
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
            else:
                for label_name, bbox, crop_array in crops:
                    # Use word-level merging for tables to preserve cell boundaries;
                    # paragraph-level for charts/infographics where structure matters less.
                    ml = "word" if label_name == "table" else "paragraph"
                    preds = model.invoke(crop_array, merge_level=ml)

                    # Parse and assemble text.
                    blocks = _parse_ocr_result(preds)
                    if label_name == "table":
                        text = _blocks_to_pseudo_markdown(blocks)
                        if not text:
                            text = _blocks_to_text(blocks)  # fallback
                    else:
                        text = _blocks_to_text(blocks)

                    entry = {
                        "bbox_xyxy_norm": bbox,
                        "text": text,
                    }

                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)

        except BaseException as e:
            print(f"Warning: OCR failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_ocr_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total

    # Fill timing into metadata.
    for meta in all_ocr_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
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

    __slots__ = (
        "_model",
        "_extract_tables",
        "_extract_charts",
        "_extract_infographics",
        "_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_remote_retry",
    )

    def __init__(
        self,
        *,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        self._invoke_url = (ocr_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        if self._invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._model = NemotronOCRV1()
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return ocr_page_elements(
                batch_df,
                model=self._model,
                invoke_url=self._invoke_url,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                extract_tables=self._extract_tables,
                extract_charts=self._extract_charts,
                extract_infographics=self._extract_infographics,
                remote_retry=self._remote_retry,
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
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
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
                all_meta.append({"timing": None, "error": None})
                continue

            crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)
            # Parse-only mode may skip page-elements detection entirely. In that
            # case, parse the full page once and fan out the text to enabled
            # content channels.
            if not crops and wanted_labels:
                try:
                    raw = base64.b64decode(page_image_b64)
                    with Image.open(io.BytesIO(raw)) as im0:
                        full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                    crops = [("full_page", [0.0, 0.0, 1.0, 1.0], full_crop)]
                except Exception:
                    crops = []

            if use_remote:
                crop_b64s: List[str] = []
                crop_meta: List[Tuple[str, List[float]]] = []
                for label_name, bbox, crop_array in crops:
                    crop_b64s.append(_np_rgb_to_b64_png(crop_array))
                    crop_meta.append((label_name, bbox))

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

        except BaseException as e:
            print(f"Warning: Nemotron Parse failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "nemotron_parse_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
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
