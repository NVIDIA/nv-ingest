from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from PIL import Image
import torch


def normalize_endpoint(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if not ep:
        return ep
    if "://" not in ep:
        ep = "http://" + ep
    return ep


def default_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})

    # Auto-wire hosted NVIDIA API key if available. Local NIMs usually do not need it.
    if "Authorization" not in headers:
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def tensor_to_png_b64(img: torch.Tensor) -> str:
    """
    Convert a CHW tensor into a base64-encoded PNG.

    Accepts:
      - CHW (3,H,W) or (1,H,W)
    Returns:
      - base64 string (no data: prefix)
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")

    x = img.detach()
    if x.device.type != "cpu":
        x = x.cpu()

    # Convert to uint8 in [0,255]
    if x.dtype.is_floating_point:
        maxv = float(x.max().item()) if x.numel() else 1.0
        # Heuristic: treat [0,1] images as normalized.
        if maxv <= 1.5:
            x = x * 255.0
        x = x.clamp(0, 255).to(dtype=torch.uint8)
    else:
        x = x.clamp(0, 255).to(dtype=torch.uint8)

    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if c == 1:
        arr = x.squeeze(0).numpy()
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    elif c == 3:
        arr = x.permute(1, 2, 0).contiguous().numpy()
        pil = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {c}")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def post_batched_images(
    *,
    endpoint: str,
    images_chw: Sequence[torch.Tensor],
    headers: Dict[str, str],
    timeout_seconds: float,
) -> Any:
    """
    POST a list of CHW images to a NIM-style endpoint:

      {"input": [{"type": "image_url", "url": "data:image/png;base64,..."} ...]}
    """
    payload = {
        "input": [{"type": "image_url", "url": f"data:image/png;base64,{tensor_to_png_b64(img)}"} for img in images_chw]
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_per_item_list(obj: Any) -> Optional[List[Any]]:
    """
    Best-effort extraction of per-item outputs from common NIM-ish response shapes.
    """
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for k in ("data", "output", "outputs", "results", "result"):
        v = obj.get(k)
        if isinstance(v, list):
            return v
    return None


def _to_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def _to_long_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.tensor(x, dtype=torch.int64)


def coerce_boxes_labels_scores(item: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try hard to coerce a per-image detection payload into (boxes, labels, scores).

    Supported item shapes (best-effort):
      - {"boxes": ..., "labels": ..., "scores": ...}
      - {"detections": [{"bbox":..., "label":..., "score":...}, ...]}
      - {"objects": [{"box":..., "label":..., "score":...}, ...]}
    """
    boxes: List[Any] = []
    labels: List[Any] = []
    scores: List[Any] = []

    if isinstance(item, dict):
        if all(k in item for k in ("boxes", "labels", "scores")):
            return _to_float_tensor(item["boxes"]), _to_long_tensor(item["labels"]), _to_float_tensor(item["scores"])

        det_list = None
        for k in ("detections", "objects", "items"):
            v = item.get(k)
            if isinstance(v, list):
                det_list = v
                break
        if det_list is not None:
            for d in det_list:
                if not isinstance(d, dict):
                    continue
                b = d.get("bbox") or d.get("box") or d.get("boxes")
                l = d.get("label") or d.get("labels") or d.get("class_id") or d.get("class")
                s = d.get("score") or d.get("scores") or d.get("confidence")
                if b is not None:
                    boxes.append(b)
                if l is not None:
                    labels.append(l)
                if s is not None:
                    scores.append(s)

    # Empty-safe tensors
    if not boxes:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.float32),
        )
    return _to_float_tensor(boxes), _to_long_tensor(labels), _to_float_tensor(scores)
