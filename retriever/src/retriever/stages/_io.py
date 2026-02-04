from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg")

PAGE_ELEMENT_LABELS = {
    0: "table",
    1: "chart",
    2: "title",
    3: "infographic",
    4: "text",
    5: "header_footer",
}

def iter_images(input_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if not p.name.endswith("page_element_detections.png"):
                paths.append(p)
    return paths


def load_image_rgb_chw_u8(path: Path, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load an image as CHW uint8 tensor on device.
    Returns (tensor, (H, W)).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)  # HWC
    h, w = int(arr.shape[0]), int(arr.shape[1])
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    t = t.to(device=device, dtype=torch.uint8, non_blocking=(device.type == "cuda"))
    return t, (h, w)


def to_bbox_list(bbox: Any) -> List[float]:
    if isinstance(bbox, torch.Tensor):
        return [float(x) for x in bbox.detach().cpu().tolist()]
    if isinstance(bbox, np.ndarray):
        return [float(x) for x in bbox.tolist()]
    if isinstance(bbox, (list, tuple)):
        return [float(x) for x in bbox]
    return [float(x) for x in list(bbox)]


def to_scalar_int(v: Any) -> Any:
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return int(v.item())
    if isinstance(v, (np.integer, int)):
        return int(v)
    # Some model wrappers emit float labels (e.g. np.float32(2.0)); coerce safely.
    if isinstance(v, (np.floating, float)):
        try:
            return int(v)
        except Exception:
            return int(float(v))
    return v


def to_scalar_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.item())
    if isinstance(v, (np.floating, float, int)):
        return float(v)
    return None


def crop_tensor_normalized_xyxy(image_tensor: torch.Tensor, bbox_xyxy_norm: Sequence[float]) -> torch.Tensor:
    """
    Crop a CHW image tensor using normalized [xmin, ymin, xmax, ymax] in [0, 1].
    """
    xmin_n, ymin_n, xmax_n, ymax_n = [float(x) for x in bbox_xyxy_norm]
    _, H, W = image_tensor.shape
    x1 = int(xmin_n * W)
    y1 = int(ymin_n * H)
    x2 = int(xmax_n * W)
    y2 = int(ymax_n * H)

    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image_tensor[:, y1:y2, x1:x2]


def bbox_region_to_page(
    *,
    region_bbox_xyxy_norm_in_page: Sequence[float],
    det_bbox_xyxy_norm_in_region: Sequence[float],
) -> List[float]:
    """
    Convert a bbox normalized within a region-crop into a bbox normalized within the full page.
    """
    rx1, ry1, rx2, ry2 = [float(x) for x in region_bbox_xyxy_norm_in_page]
    dx1, dy1, dx2, dy2 = [float(x) for x in det_bbox_xyxy_norm_in_region]
    rw = max(1e-12, rx2 - rx1)
    rh = max(1e-12, ry2 - ry1)
    px1 = rx1 + dx1 * rw
    py1 = ry1 + dy1 * rh
    px2 = rx1 + dx2 * rw
    py2 = ry1 + dy2 * rh
    # clamp
    px1 = min(max(px1, 0.0), 1.0)
    py1 = min(max(py1, 0.0), 1.0)
    px2 = min(max(px2, 0.0), 1.0)
    py2 = min(max(py2, 0.0), 1.0)
    if px2 <= px1:
        px2 = min(1.0, px1 + 1e-6)
    if py2 <= py1:
        py2 = min(1.0, py1 + 1e-6)
    return [float(px1), float(py1), float(px2), float(py2)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """
    Atomically write JSON to disk (write temp file then replace).
    Prevents partially-written JSON if a process is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    tmp_path.replace(path)


def coerce_embedding_to_vector(obj: Any) -> torch.Tensor:
    """
    Mirrors slimgest.recall.topk._coerce_embedding_to_vector logic (duplicated to keep stages standalone).
    """
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        d: Dict[str, Any] = obj
        for k in ("pooler_output", "sentence_embedding", "embeddings", "embedding"):
            if k in d:
                return coerce_embedding_to_vector(d[k])
        for k in ("last_hidden_state", "hidden_states"):
            if k in d:
                return coerce_embedding_to_vector(d[k])
        raise ValueError(f"Unrecognized embedding dict keys: {list(d.keys())[:10]}")
    elif hasattr(obj, "pooler_output"):
        return coerce_embedding_to_vector(getattr(obj, "pooler_output"))
    elif hasattr(obj, "last_hidden_state"):
        return coerce_embedding_to_vector(getattr(obj, "last_hidden_state"))
    elif isinstance(obj, (tuple, list)) and len(obj) > 0:
        return coerce_embedding_to_vector(obj[0])
    else:
        raise TypeError(f"Unsupported embedding object type: {type(obj)}")

    if t.ndim == 1:
        return t
    if t.ndim == 2:
        return t[0]
    if t.ndim == 3:
        return t[0].mean(dim=0)
    raise ValueError(f"Unsupported embedding tensor shape: {tuple(t.shape)}")


def normalize_l2(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = v.float()
    return v / (v.norm(p=2) + eps)


def iter_detection_dicts(preds: Any) -> Iterable[Dict[str, Any]]:
    """
    Normalize model outputs into an iterable of dicts with 'boxes'/'labels'/'scores'.
    Many Nemotron wrappers return List[Dict[...]]; some return Dict[...].
    """
    if preds is None:
        return []
    if isinstance(preds, dict):
        return [preds]
    if isinstance(preds, list):
        return preds
    return []

