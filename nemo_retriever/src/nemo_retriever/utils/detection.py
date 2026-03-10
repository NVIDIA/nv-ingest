# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def prediction_to_detections(pred: Any, *, label_names: List[str]) -> List[Dict[str, Any]]:
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
