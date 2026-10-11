#!/usr/bin/env python3
"""P3'(a) — Real model end-to-end: prove the DLPack zero-copy transport is recall-NEUTRAL.

Holds the rendered raster constant and runs the REAL NemotronPageElementsV3 two ways:
  A) base64 transport (what crosses the current extract->page_elements Ray boundary)
  B) DLPack zero-copy transport (rasterizer device buffer -> torch.from_dlpack -> model)
then asserts the model's raw predictions are identical. Identical model inputs/outputs ⇒ identical
detections ⇒ no retrieval-recall change. (Full agent_eval retrieval needs the services stack; this
isolates the transport, which is the only thing the fusion changes.)

Env: HF_HOME pointed at cached weights, HF_HUB_OFFLINE=1. Run with .venv/bin/python.
"""
from __future__ import annotations
import base64, io, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "native" / "build"))
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "python"))
sys.path.insert(0, str(ROOT / "nemo_retriever" / "src"))
import _gpu_raster as R
import gpu_pdfium
from PIL import Image
from nemo_retriever.models.local import NemotronPageElementsV3


def render_rgb(pdf_bytes, target=1024):
    doc = gpu_pdfium.PdfDocument(pdf_bytes)
    try:
        p = doc.get_page(0)
        s = max(min(target / p.get_width(), target / p.get_height()), 1e-3)
        bgr = p.render(scale=s).to_numpy()
        p.close()
    finally:
        doc.close()
    return np.ascontiguousarray(bgr[:, :, ::-1])  # RGB HWC uint8


def preds_base64(model, rgb):
    """A) encode->decode (the current cross-stage transport), then model."""
    im = Image.fromarray(rgb); b = io.BytesIO(); im.save(b, format="PNG")
    b64 = base64.b64encode(b.getvalue()).decode()
    raw = base64.b64decode(b64)
    arr = np.array(Image.open(io.BytesIO(raw)))            # HWC RGB uint8 (host)
    H, W = arr.shape[:2]
    x = model.preprocess(arr)
    with torch.inference_mode():
        return model(x, (H, W))


def preds_dlpack(model, rgb):
    """B) upload once -> DeviceImage -> torch.from_dlpack (zero-copy) -> model, all on GPU."""
    dev = R.upload_to_device(rgb, 0)
    t = torch.from_dlpack(dev)                              # HWC RGB uint8 on CUDA, same buffer
    H, W = int(t.shape[0]), int(t.shape[1])
    chw = t.permute(2, 0, 1).contiguous().to(torch.float32)  # CHW, on GPU
    x = model.preprocess(chw)
    with torch.inference_mode():
        return model(x, (H, W)), dev.data_ptr, t.data_ptr()


def to_cpu(o):
    if isinstance(o, torch.Tensor): return o.detach().float().cpu()
    if isinstance(o, (list, tuple)): return [to_cpu(x) for x in o]
    return o


def max_abs_diff(a, b):
    a, b = to_cpu(a), to_cpu(b)
    if isinstance(a, list):
        return max((max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape: return float("inf")
        return float((a - b).abs().max()) if a.numel() else 0.0
    return 0.0


def main():
    pdfs = sorted((ROOT / "gpu_pdf_extractor" / "corpus" / "dense_vector").glob("*.pdf"))[:4]
    pdfs += sorted((ROOT / "gpu_pdf_extractor" / "corpus" / "born_digital").glob("*.pdf"))[:4]
    t0 = time.perf_counter()
    model = NemotronPageElementsV3()
    print(f"model loaded in {time.perf_counter()-t0:.1f}s\n")

    print(f"{'page':40s} {'maxΔ(A,B)':>10} {'ptr_match':>9} {'A_ms':>7} {'B_ms':>7}")
    worst = 0.0
    for p in pdfs:
        rgb = render_rgb(p.read_bytes())
        torch.cuda.synchronize(); ta = time.perf_counter()
        pa = preds_base64(model, rgb)
        torch.cuda.synchronize(); a_ms = (time.perf_counter() - ta) * 1000
        tb = time.perf_counter()
        pb, dptr, tptr = preds_dlpack(model, rgb)
        torch.cuda.synchronize(); b_ms = (time.perf_counter() - tb) * 1000
        d = max_abs_diff(pa, pb)
        worst = max(worst, d)
        print(f"{p.name[:40]:40s} {d:>10.2e} {str(dptr==tptr):>9} {a_ms:>7.1f} {b_ms:>7.1f}")
    print(f"\nworst max|Δ| across pages: {worst:.2e}  -> "
          f"{'IDENTICAL detections (recall-neutral)' if worst < 1e-3 else 'DIVERGENCE — investigate'}")


if __name__ == "__main__":
    main()
