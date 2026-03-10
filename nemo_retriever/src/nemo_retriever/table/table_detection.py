# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

import time
import traceback

import pandas as pd
from nemo_retriever.params import RemoteRetryParams

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_DEFAULT_TABLE_STRUCTURE_LABELS: List[str] = ["cell", "row", "column"]


def _labels_from_model(model: Any) -> List[str]:
    # Prefer underlying model labels if present.
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
    """
    Best-effort conversion of model output into a standard detection list.

    Produces dicts of the form:
      {"bbox_xyxy_norm": [...], "label": int|None, "label_name": str, "score": float|None}
    """
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

    # Normalize to torch tensors.
    def _to_tensor(x: Any) -> Optional["torch.Tensor"]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        try:
            return torch.as_tensor(x).detach().cpu()
        except Exception:
            return None

    # Handle string labels (e.g. NIM returns ["cell", "row", "column", ...]).
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

    # Expect boxes (N,4), labels (N,)
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


def _parse_nim_bounding_boxes(response_item: Any) -> List[Dict[str, Any]]:
    """Parse the ``bounding_boxes`` NIM response format.

    NIM table-structure endpoints return::

        {"index": 0, "bounding_boxes": {
            "cell": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}, ...],
            "row":  [...],
            "column": [...]
        }}

    Returns a flat list of detection dicts compatible with
    ``_structure_dets_to_class_boxes``.
    """
    bb = None
    if isinstance(response_item, dict):
        bb = response_item.get("bounding_boxes")
    if not isinstance(bb, dict):
        return []

    dets: List[Dict[str, Any]] = []
    for label_name, items in bb.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                bbox = [float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"])]
            except (KeyError, TypeError, ValueError):
                continue
            score = None
            try:
                score = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                pass
            dets.append(
                {
                    "bbox_xyxy_norm": bbox,
                    "label_name": str(label_name),
                    "score": score,
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


# ---------------------------------------------------------------------------
# Combined table-structure + OCR core function
# ---------------------------------------------------------------------------


def table_structure_ocr_page_elements(
    batch_df: Any,
    *,
    table_structure_model: Any = None,
    ocr_model: Any = None,
    table_structure_invoke_url: str = "",
    ocr_invoke_url: str = "",
    api_key: str = "",
    request_timeout_s: float = 120.0,
    remote_retry: RemoteRetryParams | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run table-structure + OCR on table crops and produce structure-aware markdown.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. Crop all table detections from the page image.
    3. Run table-structure model on each crop to get cell/row/column bboxes.
    4. Run OCR on each crop to get text with bboxes.
    5. Join the two outputs using ``join_table_structure_and_ocr_output()``
       to produce properly-structured markdown tables.
    6. Fall back to OCR-only pseudo-markdown if table-structure returns no cells.

    Parameters
    ----------
    batch_df : pandas.DataFrame
        Ray Data batch with ``page_elements_v3`` and ``page_image`` columns.
    table_structure_model : NemotronTableStructureV1 | None
        Local table-structure model, or None for remote inference.
    ocr_model : NemotronOCRV1 | None
        Local OCR model, or None for remote inference.
    table_structure_invoke_url : str
        Remote NIM endpoint for table-structure inference.
    ocr_invoke_url : str
        Remote NIM endpoint for OCR inference.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``table`` and ``table_structure_ocr_v1``.
    """
    from nemo_retriever.nim.nim import invoke_image_inference_batches
    from nemo_retriever.ocr.ocr import (
        _blocks_to_pseudo_markdown,
        _crop_all_from_page,
        _extract_remote_ocr_item,
        _np_rgb_to_b64_png,
        _parse_ocr_result,
    )
    from nemo_retriever.utils.table_and_chart import join_table_structure_and_ocr_output

    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )

    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("table_structure_ocr_page_elements currently only supports pandas.DataFrame input.")

    ts_url = (table_structure_invoke_url or kwargs.get("table_structure_invoke_url") or "").strip()
    ocr_url = (ocr_invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    use_remote_ts = bool(ts_url)
    use_remote_ocr = bool(ocr_url)

    if not use_remote_ts and table_structure_model is None:
        raise ValueError("A local `table_structure_model` is required when `table_structure_invoke_url` is not set.")
    if not use_remote_ocr and ocr_model is None:
        raise ValueError("A local `ocr_model` is required when `ocr_invoke_url` is not set.")

    label_names = _labels_from_model(table_structure_model) if table_structure_model is not None else []
    if not label_names:
        label_names = _DEFAULT_TABLE_STRUCTURE_LABELS
    inference_batch_size = int(kwargs.get("inference_batch_size", 8))

    # Per-row accumulators.
    all_table: List[List[Dict[str, Any]]] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
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
                all_table.append(table_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # --- Pass 1: Collect table crops ---
            crops = _crop_all_from_page(page_image_b64, dets, {"table"})

            if not crops:
                all_table.append(table_items)
                all_meta.append({"timing": None, "error": None})
                continue

            # Pre-compute base64 encodings once for remote paths.
            crop_b64s = (
                [_np_rgb_to_b64_png(crop_array) for _, _, crop_array in crops]
                if (use_remote_ts or use_remote_ocr)
                else []
            )

            # --- Pass 2: Run table-structure on all crops ---
            structure_results: List[List[Dict[str, Any]]] = []
            if use_remote_ts:
                response_items = invoke_image_inference_batches(
                    invoke_url=ts_url,
                    image_b64_list=crop_b64s,
                    api_key=api_key or None,
                    timeout_s=float(request_timeout_s),
                    max_batch_size=inference_batch_size,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                if len(response_items) != len(crops):
                    raise RuntimeError(f"Expected {len(crops)} table-structure responses, got {len(response_items)}")
                for resp in response_items:
                    # Try NIM bounding_boxes format first, fall back to generic parser.
                    parsed = _parse_nim_bounding_boxes(resp)
                    if not parsed:
                        pred_item = _extract_remote_pred_item(resp)
                        parsed = _prediction_to_detections(pred_item, label_names=label_names)
                    structure_results.append(parsed)
            else:
                # Local batched inference.
                for _, _, crop_array in crops:
                    chw = torch.from_numpy(crop_array).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
                    h, w = crop_array.shape[:2]
                    x = chw.unsqueeze(0)  # BCHW
                    try:
                        pre = table_structure_model.preprocess(x, (h, w))
                    except TypeError:
                        pre = table_structure_model.preprocess(x)
                    if isinstance(pre, torch.Tensor) and pre.ndim == 3:
                        pre = pre.unsqueeze(0)
                    pred = table_structure_model.invoke(pre, (h, w))
                    dets = _prediction_to_detections(pred, label_names=label_names)
                    structure_results.append(dets)

            # --- Pass 3: Run OCR on all crops ---
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

            # --- Pass 4: Match and build markdown per crop ---
            for crop_i, (label_name, bbox, crop_array) in enumerate(crops):
                crop_hw = (int(crop_array.shape[0]), int(crop_array.shape[1]))
                structure_dets = structure_results[crop_i]
                ocr_preds = ocr_results[crop_i]

                # Try structure-aware markdown first.
                markdown = join_table_structure_and_ocr_output(structure_dets, ocr_preds, crop_hw)

                # Fallback: if no cells were detected, use OCR-only pseudo-markdown.
                if not markdown:
                    blocks = _parse_ocr_result(ocr_preds)
                    markdown = _blocks_to_pseudo_markdown(blocks)
                    if not markdown:
                        # Last resort: plain text.
                        from nemo_retriever.ocr.ocr import _blocks_to_text

                        markdown = _blocks_to_text(blocks)

                table_items.append({"bbox_xyxy_norm": bbox, "text": markdown})

        except BaseException as e:
            print(f"Warning: table-structure+OCR failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "table_structure_ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_table.append(table_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    out["table"] = all_table
    out["table_structure_ocr_v1"] = all_meta
    return out


# ---------------------------------------------------------------------------
# Combined table-structure + OCR Ray Actor
# ---------------------------------------------------------------------------


class TableStructureActor:
    """
    Ray-friendly callable that initializes both table-structure and OCR
    models once per actor and runs the combined stage.

    Usage with Ray Data::

        ds = ds.map_batches(
            TableStructureActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=8),
            fn_constructor_kwargs={
                "table_structure_invoke_url": "...",
                "ocr_invoke_url": "...",
            },
        )
    """

    __slots__ = (
        "_table_structure_model",
        "_ocr_model",
        "_table_structure_invoke_url",
        "_ocr_invoke_url",
        "_api_key",
        "_request_timeout_s",
        "_remote_retry",
    )

    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        self._table_structure_invoke_url = (table_structure_invoke_url or "").strip()
        self._ocr_invoke_url = (ocr_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )

        if self._table_structure_invoke_url:
            self._table_structure_model = None
        else:
            from nemo_retriever.model.local import NemotronTableStructureV1

            self._table_structure_model = NemotronTableStructureV1()

        if self._ocr_invoke_url:
            self._ocr_model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._ocr_model = NemotronOCRV1()

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return table_structure_ocr_page_elements(
                batch_df,
                table_structure_model=self._table_structure_model,
                ocr_model=self._ocr_model,
                table_structure_invoke_url=self._table_structure_invoke_url,
                ocr_invoke_url=self._ocr_invoke_url,
                api_key=self._api_key,
                request_timeout_s=self._request_timeout_s,
                remote_retry=self._remote_retry,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "table_structure_actor_call",
                        "type": e.__class__.__name__,
                        "message": str(e),
                        "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                    },
                }
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["table_structure_ocr_v1"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "table_structure_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "table_structure_actor_call",
                            "type": e.__class__.__name__,
                            "message": str(e),
                            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                        },
                    }
                }
            ]
