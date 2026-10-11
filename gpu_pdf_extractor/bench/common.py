"""Shared helpers for the Phase-0 benchmark harness."""
from __future__ import annotations
import hashlib, json, platform, resource, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "gpu_pdf_extractor" / "corpus"
RESULTS = REPO / "gpu_pdf_extractor" / "results"


def load_manifest() -> dict:
    return json.loads((CORPUS / "manifest.json").read_text())


def percentiles(xs: list[float], ps=(50, 90, 99)) -> dict:
    if not xs:
        return {f"p{p}": None for p in ps}
    s = sorted(xs)
    out = {}
    for p in ps:
        k = (len(s) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        out[f"p{p}"] = round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)
    return out


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux; monotonic process high-water mark.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def raster_hash(arr) -> str:
    """Stable hash of a rendered raster (shape + dtype + bytes) for drift/parity tracking."""
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def env_block(extra: dict | None = None) -> dict:
    import numpy as np
    import pypdfium2 as pdfium
    b = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pypdfium2_pdfium_build": getattr(pdfium, "PDFIUM_INFO", None).__str__(),
        "wall_clock_note": "Date.now unavailable in-tool; timestamp stamped by caller if needed",
    }
    if extra:
        b.update(extra)
    return b


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(payload, indent=2))
    return p


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
