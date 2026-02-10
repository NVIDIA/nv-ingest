from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import time
import traceback

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
        raise ImportError("table structure detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32) / 255.0
    return t, (int(h), int(w))


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

    b = _to_tensor(boxes)
    l = _to_tensor(labels)
    s = _to_tensor(scores) if scores is not None else None
    if b is None or l is None:
        return []

    # Expect boxes (N,4), labels (N,)
    if b.ndim != 2 or int(b.shape[-1]) != 4:
        return []
    if l.ndim == 2 and int(l.shape[-1]) == 1:
        l = l.squeeze(-1)
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


def detect_table_structure_v1(
    batch_df: Any,
    *,
    model: Any,
    inference_batch_size: int = 8,
    output_column: str = "table_structure_v1",
    num_detections_column: str = "table_structure_v1_num_detections",
    counts_by_label_column: str = "table_structure_v1_counts_by_label",
) -> Any:
    """
    Run Nemotron Table Structure v1 on a pandas batch.

    Input:
      - `batch_df`: pandas.DataFrame (Ray Data `batch_format="pandas"`)
        Must contain a base64 image source (best-effort from `tables`/`image_b64`/etc).
    Output:
      - pandas.DataFrame with original columns preserved, plus:
        - `output_column`: {"detections": [...], "timing": {...}, "error": ...}
        - `num_detections_column`: int
        - `counts_by_label_column`: dict[str,int]

    Notes:
      - This function chunks work in `inference_batch_size` to bound per-call cost even if
        Ray provides larger batches.
      - If the underlying model doesn't support batched invocation, we fall back to per-image calls.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("detect_table_structure_v1 currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    label_names = _labels_from_model(model)

    # Decode inputs.
    tensors: List[Optional["torch.Tensor"]] = []
    shapes: List[Optional[Tuple[int, int]]] = []
    payloads: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        try:
            b64 = row.get("page_image", {}).get("image_b64", None)
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            t, orig_shape = _decode_b64_image_to_chw_tensor(b64)
            tensors.append(t)
            shapes.append(orig_shape)
            payloads.append({"detections": []})
        except BaseException as e:
            tensors.append(None)
            shapes.append(None)
            payloads.append(_error_payload(stage="decode_image", exc=e))

    valid = [i for i, t in enumerate(tensors) if t is not None and shapes[i] is not None]

    for chunk_start in range(0, len(valid), int(inference_batch_size)):
        idxs = valid[chunk_start : chunk_start + int(inference_batch_size)]
        if not idxs:
            continue

        # Attempt batched preprocess/invoke; fall back to per-image if unsupported.
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
                pre = model.preprocess(x, sh)
            except TypeError:
                pre = model.preprocess(x)
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
            # Normalize to per-image list
            if isinstance(preds, list):
                preds_list = preds
            else:
                preds_list = [preds]
            # If model returned single pred for whole batch, duplicate best-effort by slicing not possible -> fallback.
            if len(preds_list) != len(idxs):
                raise RuntimeError("Batched invoke returned unexpected output shape; falling back to per-image calls.")
            for local_j, row_i in enumerate(idxs):
                dets = _prediction_to_detections(preds_list[local_j], label_names=label_names)
                payloads[row_i] = {"detections": dets, "timing": {"seconds": float(elapsed)}, "error": None}
        except BaseException:
            # Per-image fallback
            for local_j, row_i in enumerate(idxs):
                t = tensors[row_i]
                sh = shapes[row_i]
                if t is None or sh is None:
                    continue
                x = t.unsqueeze(0)
                t1 = time.perf_counter()
                try:
                    try:
                        pre = model.preprocess(x, sh)
                    except TypeError:
                        pre = model.preprocess(x)
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


@dataclass(slots=True)
class TableStructureActor:
    """
    Ray-friendly callable that initializes Nemotron Table Structure v1 once.
    """

    detect_kwargs: Dict[str, Any]

    def __init__(self, **detect_kwargs: Any) -> None:
        self.detect_kwargs = dict(detect_kwargs)
        from retriever.model.local import NemotronTableStructureV1

        self._model = NemotronTableStructureV1()

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return detect_table_structure_v1(
                batch_df,
                model=self._model,
                **self.detect_kwargs,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                out["table_structure_v1"] = [payload for _ in range(len(out.index))]
                out["table_structure_v1_num_detections"] = [0 for _ in range(len(out.index))]
                out["table_structure_v1_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"table_structure_v1": _error_payload(stage="actor_call", exc=e)}]

