# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Infographic detection (Nemotron Graphic Elements v1).

This mirrors `retriever.chart.chart_detection.detect_graphic_elements_v1`, but:
- prioritizes `infographics` crops when present
- uses infographic-specific output column defaults
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import base64
import io
import time
import traceback

import pandas as pd
from retriever.params import RemoteRetryParams
from retriever.nim.nim import invoke_image_inference_batches

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
        raise ImportError("infographic detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32) / 255.0
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

    b = _to_tensor(boxes)
    l = _to_tensor(labels)  # noqa: E741
    s = _to_tensor(scores) if scores is not None else None
    if b is None or l is None:
        return []

    if b.ndim != 2 or int(b.shape[-1]) != 4:
        return []
    if l.ndim == 2 and int(l.shape[-1]) == 1:
        l = l.squeeze(-1)  # noqa: E741
    if l.ndim != 1:
        return []

    n = int(min(b.shape[0], l.shape[0]))
    dets: List[Dict[str, Any]] = []
    for i in range(n):
        try:
            x1, y1, x2, y2 = [float(x) for x in b[i].tolist()]
        except Exception:
            continue

        label_i: Optional[int]
        try:
            label_i = int(l[i].item())
        except Exception:
            label_i = None

        score_f: Optional[float]
        if s is not None and s.ndim >= 1 and int(s.shape[0]) > i:
            try:
                score_f = float(s[i].item())
            except Exception:
                score_f = None
        else:
            score_f = None

        label_name = None
        if label_i is not None and 0 <= label_i < len(label_names):
            label_name = label_names[label_i]
        if not label_name:
            label_name = f"label_{label_i}" if label_i is not None else "unknown"

        dets.append(
            {
                "bbox_xyxy_norm": [x1, y1, x2, y2],
                "label": label_i,
                "label_name": str(label_name),
                "score": score_f,
            }
        )
    return dets


def _extract_remote_pred_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


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


def detect_infographic_elements_v1(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    output_column: str = "infographic_elements_v1",
    num_detections_column: str = "infographic_elements_v1_num_detections",
    counts_by_label_column: str = "infographic_elements_v1_counts_by_label",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run Nemotron Graphic Elements v1 on an infographic image source.

    Input:
      - pandas.DataFrame in/out
      - expects a usable base64 image in one of:
        - `image_b64` (direct column)
        - `page_image.image_b64` (pdf extraction output)
        - `infographics[0].image_b64` (preferred when available)
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("detect_infographic_elements_v1 currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    invoke_url = (invoke_url or kwargs.get("infographic_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    label_names = _labels_from_model(model) if model is not None else []

    tensors: List[Optional["torch.Tensor"]] = []
    shapes: List[Optional[Tuple[int, int]]] = []
    image_b64_list: List[Optional[str]] = []
    payloads: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        try:
            b64 = row.get("page_image", {}).get("image_b64", None)
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            image_b64_list.append(b64)
            if use_remote:
                tensors.append(None)
                shapes.append(None)
            else:
                t, orig_shape = _decode_b64_image_to_chw_tensor(b64)
                tensors.append(t)
                shapes.append(orig_shape)
            payloads.append({"detections": []})
        except BaseException as e:
            tensors.append(None)
            shapes.append(None)
            image_b64_list.append(None)
            payloads.append(_error_payload(stage="decode_image", exc=e))

    if use_remote:
        valid = [i for i, b64 in enumerate(image_b64_list) if isinstance(b64, str) and bool(b64)]
    else:
        valid = [i for i, t in enumerate(tensors) if t is not None and shapes[i] is not None]

    if use_remote and valid:
        valid_b64 = [image_b64_list[i] for i in valid if image_b64_list[i]]
        t0 = time.perf_counter()
        try:
            response_items = invoke_image_inference_batches(
                invoke_url=invoke_url,
                image_b64_list=cast(List[str], valid_b64),
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(inference_batch_size),
                max_pool_workers=int(retry.remote_max_pool_workers),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            elapsed = time.perf_counter() - t0
            if len(response_items) != len(valid):
                raise RuntimeError(f"Expected {len(valid)} remote predictions, got {len(response_items)}")
            for local_j, row_i in enumerate(valid):
                pred_item = _extract_remote_pred_item(response_items[local_j])
                dets = _prediction_to_detections(pred_item, label_names=label_names)
                payloads[row_i] = {"detections": dets, "timing": {"seconds": float(elapsed)}, "error": None}
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            for row_i in valid:
                payloads[row_i] = _error_payload(stage="remote_invoke", exc=e) | {"timing": {"seconds": float(elapsed)}}

    for chunk_start in range(0, len(valid), int(inference_batch_size)):
        if use_remote:
            break
        idxs = valid[chunk_start : chunk_start + int(inference_batch_size)]
        if not idxs:
            continue

        pre_list: List["torch.Tensor"] = []
        orig_shapes: List[Tuple[int, int]] = []
        for i in idxs:
            t = tensors[i]
            sh = shapes[i]
            if t is None or sh is None:
                continue
            orig_shapes.append(sh)
            x = t.unsqueeze(0)  # BCHW
            try:
                pre = model.preprocess(x)
            except Exception:
                pre = x
            if isinstance(pre, torch.Tensor) and pre.ndim == 4 and int(pre.shape[0]) == 1:
                pre_list.append(pre[0])
            elif isinstance(pre, torch.Tensor) and pre.ndim == 3:
                pre_list.append(pre)
            else:
                pre_list.append(t)

        if not pre_list:
            continue

        batch = torch.stack(pre_list, dim=0)
        t0 = time.perf_counter()
        try:
            preds = model.invoke(batch, orig_shapes)  # type: ignore[arg-type]
            elapsed = time.perf_counter() - t0
            if isinstance(preds, list):
                preds_list = preds
            else:
                preds_list = [preds]
            if len(preds_list) != len(idxs):
                raise RuntimeError("Batched invoke returned unexpected output shape; falling back to per-image calls.")
            for local_j, row_i in enumerate(idxs):
                dets = _prediction_to_detections(preds_list[local_j], label_names=label_names)
                payloads[row_i] = {"detections": dets, "timing": {"seconds": float(elapsed)}, "error": None}
        except BaseException:
            for local_j, row_i in enumerate(idxs):
                t = tensors[row_i]
                sh = shapes[row_i]
                if t is None or sh is None:
                    continue
                x = t.unsqueeze(0)
                t1 = time.perf_counter()
                try:
                    try:
                        pre = model.preprocess(x)
                    except Exception:
                        pre = x
                    if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                        pre = pre.unsqueeze(0)
                    pred = model.invoke(pre, sh)
                    dets = _prediction_to_detections(pred, label_names=label_names)
                    payloads[row_i] = {
                        "detections": dets,
                        "timing": {"seconds": float(time.perf_counter() - t1)},
                        "error": None,
                    }
                except BaseException as e:
                    payloads[row_i] = _error_payload(stage="invoke", exc=e) | {
                        "timing": {"seconds": float(time.perf_counter() - t1)}
                    }

    out = batch_df.copy()
    out[output_column] = payloads
    out[num_detections_column] = [int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in payloads]
    out[counts_by_label_column] = [
        _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in payloads
    ]
    return out


def detect_infographic_elements_v1_from_page_elements_v3(
    pages_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    page_elements_column: str = "page_elements_v3",
    page_elements_counts_by_label_column: str = "page_elements_v3_counts_by_label",
    page_image_column: str = "page_image",
    allowed_page_element_labels: Sequence[str] = ("infographic", "title"),
    output_column: str = "infographic_elements_v1",
    num_detections_column: str = "infographic_elements_v1_num_detections",
    counts_by_label_column: str = "infographic_elements_v1_counts_by_label",
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run Nemotron Graphic Elements v1 only on cropped infographic/title regions.

    Gate per page if any of `allowed_page_element_labels` has count > 0 in
    `page_elements_v3_counts_by_label`, then crop `page_image.image_b64` to each
    matching detection and run the model on those crops.

    Output payload shape:
      - `output_column`: {"regions": [...], "timing": {...}, "error": ...}
    """
    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError(
            "detect_infographic_elements_v1_from_page_elements_v3 currently only supports pandas.DataFrame input."
        )
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")
    invoke_url = (invoke_url or kwargs.get("infographic_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    allowed = {str(x).strip() for x in (allowed_page_element_labels or []) if str(x).strip()}
    if not allowed:
        allowed = {"infographic", "title"}

    out_payloads: List[Dict[str, Any]] = []
    out_total_dets: List[int] = []
    out_counts: List[Dict[str, int]] = []

    crop_b64s: List[str] = []
    crop_region_refs: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for _, row in pages_df.iterrows():
        page_payload: Dict[str, Any] = {"regions": [], "timing": {"seconds": 0.0}, "error": None}

        counts = row.get(page_elements_counts_by_label_column)
        has_any = False
        if isinstance(counts, dict):
            for k in allowed:
                try:
                    if int(counts.get(k) or 0) > 0:
                        has_any = True
                        break
                except Exception:
                    continue

        if not has_any:
            out_payloads.append(page_payload)
            out_total_dets.append(0)
            out_counts.append({})
            continue

        pe = row.get(page_elements_column)
        dets = pe.get("detections") if isinstance(pe, dict) else None
        if not isinstance(dets, list) or not dets:
            out_payloads.append(page_payload)
            out_total_dets.append(0)
            out_counts.append({})
            continue

        page_image = row.get(page_image_column) or {}
        page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
        if not isinstance(page_image_b64, str) or not page_image_b64:
            page_payload["error"] = {
                "stage": "crop",
                "type": "ValueError",
                "message": "page_image.image_b64 missing; cannot crop infographics for infographic_elements_v1.",
                "traceback": "",
            }
            out_payloads.append(page_payload)
            out_total_dets.append(0)
            out_counts.append({})
            continue

        regions: List[Dict[str, Any]] = []
        for det in dets:
            if not isinstance(det, dict):
                continue
            det_label = str(det.get("label_name") or "").strip()
            if det_label not in allowed:
                continue
            bbox = det.get("bbox_xyxy_norm")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            crop_b64, crop_shape_hw = _crop_b64_image_by_norm_bbox(
                page_image_b64, bbox_xyxy_norm=cast(Sequence[float], bbox)
            )
            if not crop_b64 or crop_shape_hw is None:
                continue

            region_payload: Dict[str, Any] = {
                "label_name": det_label,
                "bbox_xyxy_norm": [float(x) for x in bbox],
                "score": det.get("score"),
                "orig_shape_hw": crop_shape_hw,
                "detections": [],
                "timing": None,
                "error": None,
            }
            regions.append(region_payload)
            crop_b64s.append(crop_b64)
            crop_region_refs.append(region_payload)

        page_payload["regions"] = regions
        out_payloads.append(page_payload)
        out_total_dets.append(0)
        out_counts.append({})

    if crop_b64s:
        label_names = _labels_from_model(model) if model is not None else []

        crop_payloads: List[Dict[str, Any]] = []
        if use_remote:
            t0 = time.perf_counter()
            try:
                response_items = invoke_image_inference_batches(
                    invoke_url=invoke_url,
                    image_b64_list=crop_b64s,
                    api_key=api_key,
                    timeout_s=float(request_timeout_s),
                    max_batch_size=int(inference_batch_size),
                    max_pool_workers=int(retry.remote_max_pool_workers),
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                elapsed = time.perf_counter() - t0
                if len(response_items) != len(crop_b64s):
                    raise RuntimeError(f"Expected {len(crop_b64s)} remote predictions, got {len(response_items)}")
                for resp in response_items:
                    pred_item = _extract_remote_pred_item(resp)
                    dets = _prediction_to_detections(pred_item, label_names=label_names)
                    crop_payloads.append({"detections": dets, "timing": {"seconds": float(elapsed)}, "error": None})
            except BaseException as e:
                elapsed = time.perf_counter() - t0
                for _ in crop_b64s:
                    crop_payloads.append(
                        _error_payload(stage="remote_invoke", exc=e) | {"timing": {"seconds": float(elapsed)}}
                    )
        else:
            tensors: List[Optional["torch.Tensor"]] = []
            shapes: List[Optional[Tuple[int, int]]] = []
            for b64 in crop_b64s:
                try:
                    t, orig_shape = _decode_b64_image_to_chw_tensor(b64)
                    tensors.append(t)
                    shapes.append(orig_shape)
                    crop_payloads.append({"detections": []})
                except BaseException as e:
                    tensors.append(None)
                    shapes.append(None)
                    crop_payloads.append(_error_payload(stage="decode_image", exc=e))

            valid = [i for i, t in enumerate(tensors) if t is not None and shapes[i] is not None]

            for chunk_start in range(0, len(valid), int(inference_batch_size)):
                idxs = valid[chunk_start : chunk_start + int(inference_batch_size)]
                if not idxs:
                    continue

                pre_list: List["torch.Tensor"] = []
                orig_shapes: List[Tuple[int, int]] = []
                for i in idxs:
                    t = tensors[i]
                    sh = shapes[i]
                    if t is None or sh is None:
                        continue
                    orig_shapes.append(sh)
                    x = t.unsqueeze(0)
                    try:
                        pre = model.preprocess(x)
                    except Exception:
                        pre = x
                    if isinstance(pre, torch.Tensor) and pre.ndim == 4 and int(pre.shape[0]) == 1:
                        pre_list.append(pre[0])
                    elif isinstance(pre, torch.Tensor) and pre.ndim == 3:
                        pre_list.append(pre)
                    else:
                        pre_list.append(t)

                if not pre_list:
                    continue

                batch = torch.stack(pre_list, dim=0)
                t0 = time.perf_counter()
                try:
                    preds = model.invoke(batch, orig_shapes)  # type: ignore[arg-type]
                    elapsed = time.perf_counter() - t0
                    preds_list = preds if isinstance(preds, list) else [preds]
                    if len(preds_list) != len(idxs):
                        raise RuntimeError(
                            "Batched invoke returned unexpected output shape; falling back to per-image calls."
                        )
                    for local_j, crop_i in enumerate(idxs):
                        dets = _prediction_to_detections(preds_list[local_j], label_names=label_names)
                        crop_payloads[crop_i] = {
                            "detections": dets,
                            "timing": {"seconds": float(elapsed)},
                            "error": None,
                        }
                except BaseException:
                    for crop_i in idxs:
                        t = tensors[crop_i]
                        sh = shapes[crop_i]
                        if t is None or sh is None:
                            continue
                        x = t.unsqueeze(0)
                        t1 = time.perf_counter()
                        try:
                            try:
                                pre = model.preprocess(x)
                            except Exception:
                                pre = x
                            if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                                pre = pre.unsqueeze(0)
                            pred = model.invoke(pre, sh)
                            dets = _prediction_to_detections(pred, label_names=label_names)
                            crop_payloads[crop_i] = {
                                "detections": dets,
                                "timing": {"seconds": float(time.perf_counter() - t1)},
                                "error": None,
                            }
                        except BaseException as e:
                            crop_payloads[crop_i] = _error_payload(stage="invoke", exc=e) | {
                                "timing": {"seconds": float(time.perf_counter() - t1)}
                            }

        for crop_i, region_ref in enumerate(crop_region_refs):
            payload = crop_payloads[crop_i] if crop_i < len(crop_payloads) else {"detections": []}
            if isinstance(payload, dict):
                region_ref["detections"] = payload.get("detections") or []
                region_ref["timing"] = payload.get("timing")
                region_ref["error"] = payload.get("error")
            else:
                region_ref["detections"] = []
                region_ref["timing"] = None
                region_ref["error"] = {
                    "stage": "invoke",
                    "type": "TypeError",
                    "message": "Unexpected payload type",
                    "traceback": "",
                }

    # Aggregate counts per page.
    for i, page_payload in enumerate(out_payloads):
        regions = page_payload.get("regions") or []
        total_dets = 0
        agg_counts: Dict[str, int] = {}
        if isinstance(regions, list):
            for r in regions:
                if not isinstance(r, dict):
                    continue
                dets = r.get("detections") or []
                if isinstance(dets, list):
                    total_dets += int(len(dets))
                    for d in dets:
                        if not isinstance(d, dict):
                            continue
                        name = d.get("label_name")
                        if not isinstance(name, str) or not name.strip():
                            name = f"label_{d.get('label')}"
                        k = str(name)
                        agg_counts[k] = int(agg_counts.get(k, 0) + 1)
        out_total_dets[i] = int(total_dets)
        out_counts[i] = agg_counts

    elapsed_total = time.perf_counter() - t0_total
    for page_payload in out_payloads:
        if isinstance(page_payload, dict):
            page_payload["timing"] = {"seconds": float(elapsed_total)}

    out = pages_df.copy()
    out[output_column] = out_payloads
    out[num_detections_column] = out_total_dets
    out[counts_by_label_column] = out_counts
    return out


@dataclass(slots=True)
class InfographicDetectionActor:
    """
    Ray-friendly callable that initializes Nemotron Graphic Elements v1 once.
    """

    detect_kwargs: Dict[str, Any]

    def __init__(self, **detect_kwargs: Any) -> None:
        self.detect_kwargs = dict(detect_kwargs)
        invoke_url = str(
            self.detect_kwargs.get("infographic_invoke_url") or self.detect_kwargs.get("invoke_url") or ""
        ).strip()
        if invoke_url and "invoke_url" not in self.detect_kwargs:
            self.detect_kwargs["invoke_url"] = invoke_url
        if invoke_url:
            self._model = None
        else:
            from retriever.model.local import NemotronGraphicElementsV1

            self._model = NemotronGraphicElementsV1()

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            # Prefer crop-based execution when page-elements are present.
            if isinstance(batch_df, pd.DataFrame) and (
                "page_elements_v3" in batch_df.columns or "page_elements_v3_counts_by_label" in batch_df.columns
            ):
                return detect_infographic_elements_v1_from_page_elements_v3(
                    batch_df,
                    model=self._model,
                    **self.detect_kwargs,
                    **override_kwargs,
                )
            return detect_infographic_elements_v1(batch_df, model=self._model, **self.detect_kwargs, **override_kwargs)
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                out["infographic_elements_v1"] = [
                    {"regions": [], "timing": None, "error": payload.get("error")} for _ in range(len(out.index))
                ]
                out["infographic_elements_v1_num_detections"] = [0 for _ in range(len(out.index))]
                out["infographic_elements_v1_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"infographic_elements_v1": _error_payload(stage="actor_call", exc=e)}]
