from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import base64
import io
import time
import traceback

from nemotron_page_elements_v3.utils import postprocess_preds_page_element
import pandas as pd

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

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
        postprocess_page_elements_v3,
        YOLOX_PAGE_V3_CLASS_LABELS,
    )
except ImportError:
    postprocess_page_elements_v3 = None  # type: ignore[assignment,misc]
    YOLOX_PAGE_V3_CLASS_LABELS = None  # type: ignore[assignment]

from retriever.nim.nim import invoke_page_elements_batches


TensorOrArray = Union["torch.Tensor", "np.ndarray"]


def _ensure_chw_float_tensor(x: TensorOrArray) -> "torch.Tensor":
    """
    Normalize a single image into a CHW float32 torch.Tensor suitable for batching.

    Accepts either:
    - torch.Tensor in CHW or 1xCHW (or CHW-like) formats
    - np.ndarray in CHW or HWC (RGB) formats (optionally with leading batch dim=1)
    """
    if torch is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch and numpy.")

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        arr = x
        # Squeeze trivial batch dimension if present.
        if arr.ndim == 4 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {getattr(arr, 'shape', None)}")

        # Heuristic: HWC (RGB) -> CHW; otherwise assume already CHW-like.
        if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr))
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)!r}")

    # Squeeze trivial batch dimension if present.
    if t.ndim == 4 and int(t.shape[0]) == 1:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    # Keep 0-255 range: resize_pad pads with 114.0 (designed for 0-255),
    # and YoloXWrapper.forward() handles the 0-255 → model-input conversion.
    t = t.to(dtype=torch.float32)

    return t.contiguous()


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
        raise ImportError("page element detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32) / 255.0
    return t, (int(h), int(w))


def _decode_b64_image_to_np_array(image_b64: str) -> Tuple["np.array", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im)

    return arr, (int(h), int(w))


def _labels_from_model(_model: Any) -> List[str]:
    return [
        "table",
        "chart",
        "title",
        "infographic",
        "text",
        "header_footer",
    ]


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


def _postprocess_to_per_image_detections(
    *,
    boxes: Any,
    labels: Any,
    scores: Any,
    batch_size: int,
    label_names: List[str],
) -> List[List[Dict[str, Any]]]:
    """
    Convert model postprocess outputs into a list of per-image detection dicts.

    Expected detection format matches the "stage2 page_elements_v3 json" used by `retriever.image.render`.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection postprocess.")

    # Normalize to per-image tensors.
    def _as_list(x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return [x]

    # If tensors include a batch dimension, split them.
    if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
        boxes_list = [boxes[i] for i in range(int(boxes.shape[0]))]
    else:
        boxes_list = _as_list(boxes)

    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
        labels_list = [labels[i] for i in range(int(labels.shape[0]))]
    else:
        labels_list = _as_list(labels)

    if isinstance(scores, torch.Tensor) and scores.ndim == 2:
        scores_list = [scores[i] for i in range(int(scores.shape[0]))]
    else:
        scores_list = _as_list(scores)

    n = min(len(boxes_list), len(labels_list), len(scores_list), int(batch_size))
    out: List[List[Dict[str, Any]]] = []
    for i in range(n):
        bi = boxes_list[i]
        li = labels_list[i]
        si = scores_list[i]

        if not isinstance(bi, torch.Tensor) or not isinstance(li, torch.Tensor) or not isinstance(si, torch.Tensor):
            out.append([])
            continue

        # Move to CPU for safe conversion.
        bi = bi.detach().cpu()
        li = li.detach().cpu()
        si = si.detach().cpu()

        # Common shapes:
        # - boxes: (N,4)
        # - labels: (N,)
        # - scores: (N,)
        if bi.ndim != 2 or bi.shape[-1] != 4:
            out.append([])
            continue

        n_det = int(bi.shape[0])
        dets: List[Dict[str, Any]] = []
        for j in range(n_det):
            try:
                x1, y1, x2, y2 = [float(x) for x in bi[j].tolist()]
            except Exception:
                continue

            label_i: Optional[int]
            try:
                label_i = int(li[j].item())
            except Exception:
                label_i = None

            score_f: Optional[float]
            try:
                score_f = float(si[j].item())
            except Exception:
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
        out.append(dets)

    # If model returned fewer splits than requested, pad.
    while len(out) < int(batch_size):
        out.append([])
    return out[: int(batch_size)]


# -- Label mapping between retriever ("text") and API ("paragraph") --
_RETRIEVER_LABEL_NAMES = ["table", "chart", "title", "infographic", "text", "header_footer"]
_RETRIEVER_TO_API = {"text": "paragraph"}
_API_TO_RETRIEVER = {"paragraph": "text"}


def _detections_to_annotation_dict(
    dets: List[Dict[str, Any]],
) -> Dict[str, List[List[float]]]:
    """Convert a list of detection dicts into the annotation_dict format expected by
    ``postprocess_page_elements_v3``.

    Each detection dict has keys ``bbox_xyxy_norm``, ``label_name``, ``score``.
    The annotation_dict maps label names (using API naming, i.e. "paragraph") to
    ``[[x0, y0, x1, y1, confidence], ...]``.
    """
    ann: Dict[str, List[List[float]]] = {}
    for d in dets:
        name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        bbox = list(d["bbox_xyxy_norm"])  # [x0, y0, x1, y1]
        bbox.append(float(d["score"]) if d["score"] is not None else 0.0)
        ann.setdefault(name, []).append(bbox)
    return ann


def _annotation_dict_to_detections(
    ann_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert an annotation_dict back into a list of detection dicts.

    Maps API label names back to retriever names (e.g. "paragraph" -> "text")
    and assigns integer label IDs from the retriever label order.
    """
    dets: List[Dict[str, Any]] = []
    for api_name, entries in ann_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            # entry is [x0, y0, x1, y1, confidence]
            dets.append(
                {
                    "bbox_xyxy_norm": list(entry[:4]),
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry[4]) if len(entry) > 4 else 0.0,
                }
            )
    return dets


def _bounding_boxes_to_detections(
    bb_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a bounding_boxes dict (NIM API format) to detection dicts.

    Input format: {"label": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}, ...]}
    """
    dets: List[Dict[str, Any]] = []
    for api_name, entries in bb_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            dets.append(
                {
                    "bbox_xyxy_norm": [
                        float(entry["x_min"]),
                        float(entry["y_min"]),
                        float(entry["x_max"]),
                        float(entry["y_max"]),
                    ],
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry.get("confidence", 0.0)),
                }
            )
    return dets


def _apply_page_elements_v3_postprocess(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply ``postprocess_page_elements_v3`` (box fusion, title matching,
    expansion, overlap removal) to a single image's detection list.

    Returns the original detections unchanged if the API function is unavailable.
    """
    if postprocess_page_elements_v3 is None or not dets:
        return dets
    try:
        ann_dict = _detections_to_annotation_dict(dets)
        labels = YOLOX_PAGE_V3_CLASS_LABELS if YOLOX_PAGE_V3_CLASS_LABELS is not None else list(ann_dict.keys())
        result = postprocess_page_elements_v3(ann_dict, labels=labels)
        return _annotation_dict_to_detections(result)
    except Exception:
        return dets


def _remote_response_to_detections(
    *,
    response_json: Dict[str, Any],
    label_names: List[str],
    thresholds_per_class: Sequence[float],
) -> List[Dict[str, Any]]:
    # Try direct model-pred style payload first (or common wrappers around it).
    candidates: List[Any] = [response_json]
    data_list = response_json.get("data")
    if isinstance(data_list, list) and data_list:
        candidates.append(data_list[0])
    output_list = response_json.get("output")
    if isinstance(output_list, list) and output_list:
        candidates.append(output_list[0])
    pred_list = response_json.get("predictions")
    if isinstance(pred_list, list) and pred_list:
        candidates.append(pred_list[0])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            boxes, labels, scores = postprocess_preds_page_element(cand, list(thresholds_per_class), label_names)
            dets = _postprocess_to_per_image_detections(
                boxes=[boxes],
                labels=[labels],
                scores=[scores],
                batch_size=1,
                label_names=label_names,
            )[0]
            return _apply_page_elements_v3_postprocess(dets)
        except Exception:
            pass

    # NIM bounding_boxes format:
    # {"index": 0, "bounding_boxes": {"title": [{"x_min": ..., "y_min": ..., ...}]}}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        bb = cand.get("bounding_boxes")
        if isinstance(bb, dict):
            try:
                dets = _bounding_boxes_to_detections(bb)
                return _apply_page_elements_v3_postprocess(dets)
            except Exception:
                pass

    # Fall back to API-style annotation dict:
    # {"table": [[x0,y0,x1,y1,conf], ...], "paragraph": [...]}
    for cand in candidates:
        if not isinstance(cand, dict) or not cand:
            continue
        if all(isinstance(v, list) for v in cand.values()):
            try:
                dets = _annotation_dict_to_detections(cand)  # type: ignore[arg-type]
                return _apply_page_elements_v3_postprocess(dets)
            except Exception:
                pass

    raise RuntimeError(f"Unsupported remote response format (keys={list(response_json.keys())!r})")


def detect_page_elements_v3(
    pages_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    output_column: str = "page_elements_v3",
    num_detections_column: str = "page_elements_v3_num_detections",
    counts_by_label_column: str = "page_elements_v3_counts_by_label",
    **kwargs: Any,
) -> Any:
    """
    Run Nemotron Page Elements v3 on a pandas batch.

    Input:
      - `pages_df`: pandas.DataFrame (typical Ray Data `batch_format="pandas"`)
        Must contain an image base64 source either in `image_b64` or one of
        `images`/`tables`/`charts`/`infographics` (each as list[{"image_b64": ...}]).

    Output:
      - returns a pandas.DataFrame with original columns preserved, plus:
        - `output_column`: dict payload {"detections": [...], "timing": {...}, "error": {...?}}
        - `num_detections_column`: int
        - `counts_by_label_column`: dict[str,int]

    Notes:
      - This function internally batches model invocations in chunks of `inference_batch_size`
        to enforce batch=8 even if Ray provides larger `map_batches` frames.
    """
    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError("detect_page_elements_v3 currently only supports pandas.DataFrame input.")

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Working snippet for single image inference and debugging
    # breakpoint()
    # first_page = pages_df.iloc[0]
    # b64 = first_page.get("page_image")["image_b64"]

    # t, orig_shape = _decode_b64_image_to_np_array(b64)

    # # Inference
    # with torch.inference_mode():
    #     x = model.preprocess(t)
    #     preds = model(x, orig_shape)[0]

    # print(preds)
    # breakpoint()

    invoke_url = (invoke_url or kwargs.get("page_elements_invoke_url") or "").strip()
    use_remote = bool(invoke_url)

    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Prepare per-row decode artifacts (local mode), raw base64 (remote mode),
    # and placeholders for missing/errored rows.
    row_tensors: List[Optional[TensorOrArray]] = []
    row_shapes: List[Optional[Tuple[int, int]]] = []
    row_b64: List[Optional[str]] = []
    row_payloads: List[Dict[str, Any]] = []

    label_names = _labels_from_model(model) if model is not None else list(_RETRIEVER_LABEL_NAMES)
    if model is not None and hasattr(model, "thresholds_per_class"):
        thresholds_per_class = getattr(model, "thresholds_per_class")
    else:
        thresholds_per_class = [0.0 for _ in label_names]

    for _, row in pages_df.iterrows():
        try:
            b64 = row.get("page_image")["image_b64"]
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            row_b64.append(b64)
            if use_remote:
                row_tensors.append(None)
                row_shapes.append(None)
            else:
                t, orig_shape = _decode_b64_image_to_np_array(b64)
                row_tensors.append(t)
                row_shapes.append(orig_shape)
            row_payloads.append({"detections": []})
        except BaseException as e:
            row_tensors.append(None)
            row_shapes.append(None)
            row_b64.append(None)
            row_payloads.append(_error_payload(stage="decode_image", exc=e))

    # Run inference over only valid rows, but write results back in original order.
    if use_remote:
        valid_indices = [i for i, b64 in enumerate(row_b64) if b64]
    else:
        valid_indices = [i for i, t in enumerate(row_tensors) if t is not None and row_shapes[i] is not None]

    if (not use_remote) and valid_indices and torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection.")

    if use_remote and valid_indices:
        valid_b64: List[str] = []
        for row_i in valid_indices:
            b64 = row_b64[row_i]
            if b64:
                valid_b64.append(b64)

        t0 = time.perf_counter()
        try:
            response_items = invoke_page_elements_batches(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(inference_batch_size),
                max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
                max_retries=int(kwargs.get("remote_max_retries", 10)),
                max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
            )
            elapsed = time.perf_counter() - t0

            if len(response_items) != len(valid_indices):
                raise RuntimeError(
                    "Remote response count mismatch: " f"expected {len(valid_indices)}, got {len(response_items)}"
                )

            for local_i, row_i in enumerate(valid_indices):
                dets = _remote_response_to_detections(
                    response_json=response_items[local_i],
                    label_names=label_names,
                    thresholds_per_class=thresholds_per_class,
                )
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            print(f"Warning: page_elements remote inference failed: {type(e).__name__}: {e}")
            for row_i in valid_indices:
                row_payloads[row_i] = _error_payload(stage="remote_inference", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    for chunk_start in range(0, len(valid_indices), int(inference_batch_size)):
        chunk_idx = valid_indices[chunk_start : chunk_start + int(inference_batch_size)]
        if not chunk_idx:
            continue

        if use_remote:
            continue

        # Preprocess each image to a fixed shape so we can stack.
        pre_list: List[TensorOrArray] = []
        orig_shapes: List[Tuple[int, int]] = []
        for i in chunk_idx:
            t = row_tensors[i]
            sh = row_shapes[i]
            if t is None or sh is None:
                continue
            orig_shapes.append(sh)
            try:
                # `preprocess` may accept/return torch.Tensor or np.ndarray.
                pre = model.preprocess(t)  # type: ignore[arg-type]

                # Normalize to a single-image CHW-like item (torch or numpy); we'll convert to torch at stack time.
                if isinstance(pre, torch.Tensor):
                    if pre.ndim == 4 and int(pre.shape[0]) == 1:
                        pre_list.append(pre[0])
                    elif pre.ndim == 3:
                        pre_list.append(pre)
                    else:
                        pre_list.append(pre)
                elif isinstance(pre, np.ndarray):
                    if pre.ndim == 4 and int(pre.shape[0]) == 1:
                        pre_list.append(pre[0])
                    else:
                        pre_list.append(pre)
                else:
                    pre_list.append(t)
            except Exception:
                pre_list.append(t)

        if not pre_list:
            continue

        batch = torch.stack([_ensure_chw_float_tensor(x) for x in pre_list], dim=0)

        t0 = time.perf_counter()
        try:
            # Best-effort: pass list of shapes for batching; fall back to per-image if unsupported.
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    preds = model(batch, orig_shapes) if len(pre_list) > 1 else model(batch, orig_shapes[0])
            # Some local wrappers return only the first prediction dict even for batched inputs.
            # Detect that and force per-image invocation so every row gets its own detections.
            if len(pre_list) > 1:
                if isinstance(preds, dict):
                    raise RuntimeError("Model returned a single pred dict for batched input.")
                if isinstance(preds, list) and len(preds) != len(pre_list):
                    raise RuntimeError(
                        f"Model returned {len(preds)} preds for batch size {len(pre_list)}; falling back to per-image."
                    )
        except Exception as ex:
            print(f"Error invoking model: {ex}")
            preds_list: List[Any] = []
            for j in range(int(batch.shape[0])):
                preds_list.append(model(batch[j : j + 1], orig_shapes[j]))
            preds = preds_list
        elapsed = time.perf_counter() - t0

        # Normalize preds into a list of per-image prediction dicts.
        if isinstance(preds, dict):
            preds_list2 = [preds]
        elif isinstance(preds, list):
            preds_list2 = preds
        else:
            preds_list2 = [preds]  # type: ignore[list-item]

        try:
            # Preferred: allow model wrapper to handle batched postprocess.
            if hasattr(model, "postprocess"):
                boxes, labels, scores = model.postprocess(preds_list2)  # type: ignore[attr-defined]
            else:
                # Fallback: run upstream util per-image.
                # `postprocess_preds_page_element` expects a single pred dict and returns numpy arrays.
                boxes_list: List["torch.Tensor"] = []
                labels_list: List["torch.Tensor"] = []
                scores_list: List["torch.Tensor"] = []
                for p in preds_list2:
                    if not isinstance(p, dict):
                        boxes_list.append(torch.empty((0, 4), dtype=torch.float32))
                        labels_list.append(torch.empty((0,), dtype=torch.int64))
                        scores_list.append(torch.empty((0,), dtype=torch.float32))
                        continue
                    b_np, l_np, s_np = postprocess_preds_page_element(
                        p,
                        thresholds_per_class,
                        label_names,
                    )
                    boxes_list.append(torch.as_tensor(b_np, dtype=torch.float32))
                    labels_list.append(torch.as_tensor(l_np, dtype=torch.int64))
                    scores_list.append(torch.as_tensor(s_np, dtype=torch.float32))
                boxes, labels, scores = boxes_list, labels_list, scores_list
            per_image_dets = _postprocess_to_per_image_detections(
                boxes=boxes,
                labels=labels,
                scores=scores,
                batch_size=len(pre_list),
                label_names=label_names,
            )
            # Apply v3 postprocessing (box fusion, title matching, expansion, overlap removal)
            per_image_dets = [_apply_page_elements_v3_postprocess(dets) for dets in per_image_dets]
            for local_i, row_i in enumerate(chunk_idx):
                dets = per_image_dets[local_i] if local_i < len(per_image_dets) else []
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            # If postprocess fails, attach an error but keep job alive.
            for row_i in chunk_idx:
                row_payloads[row_i] = _error_payload(stage="postprocess", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    out = pages_df.copy()
    out[output_column] = row_payloads
    out[num_detections_column] = [
        int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads
    ]
    out[counts_by_label_column] = [
        _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in row_payloads
    ]
    return out


class PageElementDetectionActor:
    """
    Ray-friendly callable that initializes Nemotron Page Elements v3 once.

    Use with Ray Data:
      ds = ds.map_batches(PageElementDetectionActor, fn_constructor_kwargs={...}, batch_format="pandas")
    """

    __slots__ = ("detect_kwargs", "_model")

    def __init__(self, **detect_kwargs: Any) -> None:
        self.detect_kwargs = dict(detect_kwargs)
        invoke_url = str(
            self.detect_kwargs.get("page_elements_invoke_url") or self.detect_kwargs.get("invoke_url") or ""
        ).strip()
        if invoke_url and "invoke_url" not in self.detect_kwargs:
            self.detect_kwargs["invoke_url"] = invoke_url
        if invoke_url:
            self._model = None
        else:
            from retriever.model.local import NemotronPageElementsV3

            self._model = NemotronPageElementsV3()

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return detect_page_elements_v3(
                pages_df,
                model=self._model,
                **self.detect_kwargs,
                **override_kwargs,
            )
        except Exception as e:
            # As a last line of defense, never let the Ray UDF raise.
            if isinstance(pages_df, pd.DataFrame):
                out = pages_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                out["page_elements_v3"] = [payload for _ in range(len(out.index))]
                out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
                out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"page_elements_v3": _error_payload(stage="actor_call", exc=e)}]
