"""P0.4 — Raster parity metrics (pure numpy; no skimage/scipy dependency).

Used later to gate the CUDA rasterizer against the pypdfium baseline: bit-exact parity is
impossible (different AA/hinting), so we measure perceptual closeness instead.

Metrics: exact_match_frac, mae, max_abs_diff, psnr_db, ssim (uniform-window approximation).
A windowed SSIM via an integral-image box filter avoids a scipy/skimage dependency.
"""
from __future__ import annotations
import numpy as np


def _as_gray_f32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[:, :, :3].astype(np.float32)
        a = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return a.astype(np.float32)


def _box(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over (2r+1)^2 windows via summed-area table; 'same' size, edge-clamped counts."""
    H, W = img.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) + r + 1, 0, H)
    x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) + r + 1, 0, W)
    S = (ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0])
    cnt = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return (S / cnt).astype(np.float64)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7, L: float = 255.0) -> float:
    ga, gb = _as_gray_f32(a).astype(np.float64), _as_gray_f32(b).astype(np.float64)
    if ga.shape != gb.shape:
        return float("nan")
    r = win // 2
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a, mu_b = _box(ga, r), _box(gb, r)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = _box(ga * ga, r) - mu_a2
    vb = _box(gb * gb, r) - mu_b2
    vab = _box(ga * gb, r) - mu_ab
    s = ((2 * mu_ab + c1) * (2 * vab + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return float(np.clip(s.mean(), -1.0, 1.0))


def psnr(a: np.ndarray, b: np.ndarray, L: float = 255.0) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    if a32.shape != b32.shape:
        return float("nan")
    mse = float(np.mean((a32 - b32) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10((L * L) / mse))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Full parity report between two rasters of the same shape."""
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "shape_match": True,
        "exact_match_frac": round(float(np.mean(diff == 0)), 6),
        "mae": round(float(diff.mean()), 4),
        "max_abs_diff": int(diff.max()),
        "psnr_db": round(psnr(a, b), 3),
        "ssim": round(ssim(a, b), 5),
    }


if __name__ == "__main__":
    # self-test: identical, mildly noised, shifted
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    print("identical:", compare(x, x))
    y = np.clip(x.astype(np.int16) + rng.integers(-4, 5, x.shape), 0, 255).astype(np.uint8)
    print("noised   :", compare(x, y))
