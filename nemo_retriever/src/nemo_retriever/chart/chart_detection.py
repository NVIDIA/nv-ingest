# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import time
import traceback

import pandas as pd
from nemo_retriever.nim.nim import invoke_image_inference_batches
from nemo_retriever.params import RemoteRetryParams

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _decode_b64_image_to_chw_tensor(image_b64: str) -> Tuple["torch.Tensor", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError("chart detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32)
    return t, (int(h), int(w))


def _crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """
    Crop a base64-encoded RGB image by a normalized xyxy bbox.

    Returns:
      - cropped_image_b64 (png) or None
      - cropped_shape_hw (H,W) or None
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


def _labels_from_model(model: Any) -> List[str]:
    try:
        labels = getattr(getattr(model, "_model", None), "labels", None)
        if isinstance(labels, (list, tuple)) and all(isinstance(x, str) for x in labels):
            return [str(x) for x in labels]
    except Exception:
        pass

    try:
        out = getattr(model, "output", None)
        if isinstance(out, dict):
            classes = out.get("classes")
            if isinstance(classes, (list, tuple)) and all(isinstance(x, str) for x in classes):
                return [str(x) for x in classes]
    except Exception:
        pass

    return []


def _prediction_to_detections(pred: Any, *, label_names: List[str]) -> List[Dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise ImportError("torch required for prediction parsing.")

    boxes = labels = scores = None
    if isinstance(pred, dict):
        # IMPORTANT: do not use `or` chains here. torch.Tensor truthiness is ambiguous and raises.
        def _get_any(d: Dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in d:
                    v = d.get(k)
                    if v is not None:
                        return v
            return None

        boxes = _get_any(pred, "boxes", "bboxes", "bbox", "box")
        labels = _get_any(pred, "labels", "classes", "class_ids", "class")
        scores = _get_any(pred, "scores", "conf", "confidences", "score")
    elif isinstance(pred, (list, tuple)) and len(pred) >= 3:
        boxes, labels, scores = pred[0], pred[1], pred[2]

    if boxes is None or labels is None:
        return []

    def _to_tensor(x: Any) -> Optional["torch.Tensor"]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        try:
            return torch.as_tensor(x).detach().cpu()
        except Exception:
            return None

    # Handle string labels (e.g. NIM returns ["chart_title", "xlabel", ...]).
    # torch.as_tensor cannot convert strings, so handle them before tensor conversion.
    _string_labels: Optional[List[str]] = None
    if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], str):
        _string_labels = [str(x) for x in labels]

    b = _to_tensor(boxes)
    labels_t = _to_tensor(labels) if _string_labels is None else None
    s = _to_tensor(scores) if scores is not None else None
    if b is None:
        return []
    if labels_t is None and _string_labels is None:
        return []

    if b.ndim != 2 or int(b.shape[-1]) != 4:
        return []
    if labels_t is not None:
        if labels_t.ndim == 2 and int(labels_t.shape[-1]) == 1:
            labels_t = labels_t.squeeze(-1)
        if labels_t.ndim != 1:
            return []

    n_labels = len(_string_labels) if _string_labels is not None else int(labels_t.shape[0])
    n = int(min(b.shape[0], n_labels))
    dets: List[Dict[str, Any]] = []
    for i in range(n):
        try:
            x1, y1, x2, y2 = [float(x) for x in b[i].tolist()]
        except Exception:
            continue

        if _string_labels is not None:
            label_i = i
            label_name = _string_labels[i]
        else:
            label_i: Optional[int]
            try:
                label_i = int(labels_t[i].item())
            except Exception:
                label_i = None

            label_name = None
            if label_i is not None and 0 <= label_i < len(label_names):
                label_name = label_names[label_i]
            if not label_name:
                label_name = f"label_{label_i}" if label_i is not None else "unknown"

        score_f: Optional[float]
        if s is not None and s.ndim >= 1 and int(s.shape[0]) > i:
            try:
                score_f = float(s[i].item())
            except Exception:
                score_f = None
        else:
            score_f = None

        dets.append(
            {
                "bbox_xyxy_norm": [x1, y1, x2, y2],
                "label": label_i,
                "label_name": str(label_name),
                "score": score_f,
            }
        )
    return dets


def _counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in detections:
        if not isinstance(d, dict):
            continue
        name = d.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{d.get('label')}"
        k = str(name)
        out[k] = int(out.get(k, 0) + 1)
    return out


def _remote_response_to_ge_detections(response_json: Any) -> List[Dict[str, Any]]:
    """Parse a NIM graphic-elements response into the standard detection list.

    The NIM returns either:
    * ``{"label_name": [[x0, y0, x1, y1, conf], ...], ...}`` (annotation dict), or
    * ``{"bounding_boxes": {"label_name": [{"x_min":..., ...}]}}`` (NIM v2), or
    * a dict with ``boxes``/``labels``/``scores`` tensors (model-pred style).
    """
    if not isinstance(response_json, dict):
        return []

    # Unwrap common NIM envelopes.
    candidates: List[Any] = [response_json]
    for key in ("data", "output", "predictions"):
        nested = response_json.get(key)
        if isinstance(nested, list) and nested:
            candidates.append(nested[0])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue

        # NIM v2 bounding_boxes format.
        bb = cand.get("bounding_boxes")
        if isinstance(bb, dict):
            dets: List[Dict[str, Any]] = []
            for label_name, items in bb.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        dets.append(
                            {
                                "bbox_xyxy_norm": [
                                    float(item.get("x_min", 0)),
                                    float(item.get("y_min", 0)),
                                    float(item.get("x_max", 0)),
                                    float(item.get("y_max", 0)),
                                ],
                                "label": None,
                                "label_name": str(label_name),
                                "score": float(item.get("confidence", 0)),
                            }
                        )
            if dets:
                return dets

        # Annotation dict: {"chart_title": [[x0, y0, x1, y1, conf], ...]}
        if all(isinstance(v, list) for v in cand.values()):
            dets = []
            for label_name, boxes in cand.items():
                if not isinstance(boxes, list):
                    continue
                for box in boxes:
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        dets.append(
                            {
                                "bbox_xyxy_norm": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                "label": None,
                                "label_name": str(label_name),
                                "score": float(box[4]) if len(box) > 4 else None,
                            }
                        )
            if dets:
                return dets

    return []


# ---------------------------------------------------------------------------
# Combined graphic-elements + OCR core function
# ---------------------------------------------------------------------------


def graphic_elements_ocr_page_elements(
    batch_df: Any,
    *,
    graphic_elements_model: Any = None,
    ocr_model: Any = None,
    graphic_elements_invoke_url: str = "",
    ocr_invoke_url: str = "",
    api_key: str = "",
    request_timeout_s: float = 120.0,
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run graphic-elements + OCR on chart crops and produce structure-aware text.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. Crop all chart detections from the page image.
    3. Run graphic-elements model on each crop to get element bboxes.
    4. Run OCR on each crop to get text with bboxes.
    5. Join the two outputs using ``join_graphic_elements_and_ocr_output()``
       to produce semantically structured chart text.
    6. Fall back to OCR-only text if graphic-elements returns no detections.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``chart`` and ``graphic_elements_ocr_v1``.
    """
    from nemo_retriever.ocr.ocr import (
        _blocks_to_text,
        _crop_all_from_page,
        _extract_remote_ocr_item,
        _np_rgb_to_b64_png,
        _parse_ocr_result,
    )
    from nemo_retriever.utils.table_and_chart import join_graphic_elements_and_ocr_output

    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )

    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("graphic_elements_ocr_page_elements currently only supports pandas.DataFrame input.")

    ge_url = (graphic_elements_invoke_url or kwargs.get("graphic_elements_invoke_url") or "").strip()
    ocr_url = (ocr_invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    use_remote_ge = bool(ge_url)
    use_remote_ocr = bool(ocr_url)

    if not use_remote_ge and graphic_elements_model is None:
        raise ValueError("A local `graphic_elements_model` is required when `graphic_elements_invoke_url` is not set.")
    if not use_remote_ocr and ocr_model is None:
        raise ValueError("A local `ocr_model` is required when `ocr_invoke_url` is not set.")

    label_names = _labels_from_model(graphic_elements_model) if graphic_elements_model is not None else []
    inference_batch_size = int(kwargs.get("inference_batch_size", 8))

    # Per-row accumulators.
    all_chart: List[List[Dict[str, Any]]] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        chart_items: List[Dict[str, Any]] = []
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
                all_chart.append(chart_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # --- Crop all chart detections ---
            crops = _crop_all_from_page(page_image_b64, dets, {"chart"})

            if not crops:
                all_chart.append(chart_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # Pre-compute base64 encodings once for remote paths.
            crop_b64s = (
                [_np_rgb_to_b64_png(crop_array) for _, _, crop_array in crops]
                if (use_remote_ge or use_remote_ocr)
                else []
            )

            # --- Run graphic-elements on all crops ---
            ge_results: List[List[Dict[str, Any]]] = []
            if use_remote_ge:
                response_items = invoke_image_inference_batches(
                    invoke_url=ge_url,
                    image_b64_list=crop_b64s,
                    api_key=api_key or None,
                    timeout_s=float(request_timeout_s),
                    max_batch_size=inference_batch_size,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                if len(response_items) != len(crops):
                    raise RuntimeError(f"Expected {len(crops)} GE responses, got {len(response_items)}")
                for resp in response_items:
                    ge_results.append(_remote_response_to_ge_detections(resp))
            else:
                # Local batched inference.
                for _, _, crop_array in crops:
                    chw = torch.from_numpy(crop_array).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
                    h, w = crop_array.shape[:2]
                    x = chw.unsqueeze(0)  # BCHW
                    try:
                        pre = graphic_elements_model.preprocess(x)
                    except Exception:
                        pre = x
                    if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                        pre = pre.unsqueeze(0)
                    pred = graphic_elements_model.invoke(pre, (h, w))
                    ge_dets = _prediction_to_detections(pred, label_names=label_names)
                    ge_results.append(ge_dets)

            # --- Run OCR on all crops ---
            ocr_results: List[Any] = []
            if use_remote_ocr:
                ocr_response_items = invoke_image_inference_batches(
                    invoke_url=ocr_url,
                    image_b64_list=crop_b64s,
                    api_key=api_key or None,
                    timeout_s=float(request_timeout_s),
                    max_batch_size=inference_batch_size,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                if len(ocr_response_items) != len(crops):
                    raise RuntimeError(f"Expected {len(crops)} OCR responses, got {len(ocr_response_items)}")
                for resp in ocr_response_items:
                    ocr_results.append(_extract_remote_ocr_item(resp))
            else:
                for _, _, crop_array in crops:
                    ocr_results.append(ocr_model.invoke(crop_array, merge_level="word"))

            # --- Join and build text per crop ---
            for crop_i, (label_name, bbox, crop_array) in enumerate(crops):
                crop_hw = (int(crop_array.shape[0]), int(crop_array.shape[1]))
                ge_dets = ge_results[crop_i]
                ocr_preds = ocr_results[crop_i]

                # Try structure-aware join first.
                text = join_graphic_elements_and_ocr_output(ge_dets, ocr_preds, crop_hw)

                # Fallback: if no GE detections matched, use OCR-only text.
                if not text:
                    blocks = _parse_ocr_result(ocr_preds)
                    text = _blocks_to_text(blocks)

                chart_items.append({"bbox_xyxy_norm": bbox, "text": text})

        except BaseException as e:
            print(f"Warning: graphic-elements+OCR failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "graphic_elements_ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_chart.append(chart_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["chart"] = all_chart
    out["graphic_elements_ocr_v1"] = all_meta
    return out


# ---------------------------------------------------------------------------
# Combined graphic-elements + OCR Ray Actor
# ---------------------------------------------------------------------------


class GraphicElementsActor:
    """
    Ray-friendly callable that initializes both graphic-elements and OCR
    models once per actor and runs the combined stage.
    """

    __slots__ = (
        "_graphic_elements_model",
        "_ocr_model",
        "_graphic_elements_invoke_url",
        "_ocr_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_remote_retry",
        "_inference_batch_size",
    )

    def __init__(
        self,
        *,
        graphic_elements_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        inference_batch_size: int = 8,
    ) -> None:
        self._graphic_elements_invoke_url = (graphic_elements_invoke_url or "").strip()
        self._ocr_invoke_url = (ocr_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._inference_batch_size = int(inference_batch_size)

        if self._graphic_elements_invoke_url:
            self._graphic_elements_model = None
        else:
            from nemo_retriever.model.local import NemotronGraphicElementsV1

            self._graphic_elements_model = NemotronGraphicElementsV1()

        if self._ocr_invoke_url:
            self._ocr_model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._ocr_model = NemotronOCRV1()

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return graphic_elements_ocr_page_elements(
                batch_df,
                graphic_elements_model=self._graphic_elements_model,
                ocr_model=self._ocr_model,
                graphic_elements_invoke_url=self._graphic_elements_invoke_url,
                ocr_invoke_url=self._ocr_invoke_url,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                remote_retry=self._remote_retry,
                inference_batch_size=self._inference_batch_size,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "chart_graphic_elements_ocr_actor_call",
                        "type": e.__class__.__name__,
                        "message": str(e),
                        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                    },
                }
                n = len(out.index)
                out["chart"] = [[] for _ in range(n)]
                out["graphic_elements_ocr_v1"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "graphic_elements_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "chart_graphic_elements_ocr_actor_call",
                            "type": e.__class__.__name__,
                            "message": str(e),
                            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                        },
                    }
                }
            ]
