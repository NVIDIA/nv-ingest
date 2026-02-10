from __future__ import annotations

from dataclasses import dataclass
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

    # Normalize dtype/range.
    if t.dtype == torch.uint8:
        t = t.to(dtype=torch.float32) / 255.0
    else:
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


def _labels_from_model(model: Any) -> List[str]:
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


def detect_page_elements_v3(
    pages_df: Any,
    *,
    model: Any,
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

    # Prepare per-row image tensors (and placeholders for missing/errored rows).
    row_tensors: List[Optional[TensorOrArray]] = []
    row_shapes: List[Optional[Tuple[int, int]]] = []
    row_payloads: List[Dict[str, Any]] = []

    label_names = _labels_from_model(model)

    for _, row in pages_df.iterrows():
        try:
            b64 = row.get("page_image")["image_b64"]
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            t, orig_shape = _decode_b64_image_to_np_array(b64)
            row_tensors.append(t)
            row_shapes.append(orig_shape)
            row_payloads.append({"detections": []})
        except BaseException as e:
            row_tensors.append(None)
            row_shapes.append(None)
            row_payloads.append(_error_payload(stage="decode_image", exc=e))

    # Run inference over only valid rows, but write results back in original order.
    valid_indices = [i for i, t in enumerate(row_tensors) if t is not None and row_shapes[i] is not None]

    if valid_indices and torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection.")

    for chunk_start in range(0, len(valid_indices), int(inference_batch_size)):
        chunk_idx = valid_indices[chunk_start : chunk_start + int(inference_batch_size)]
        if not chunk_idx:
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
                    b_np, l_np, s_np = postprocess_preds_page_element(p, model.thresholds_per_class, model.labels)
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
                row_payloads[row_i] = _error_payload(stage="postprocess", exc=e) | {"timing": {"seconds": float(elapsed)}}

    out = pages_df.copy()
    out[output_column] = row_payloads
    out[num_detections_column] = [int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads]
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

