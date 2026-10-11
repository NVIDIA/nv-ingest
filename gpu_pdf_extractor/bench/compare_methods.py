#!/usr/bin/env python3
"""Old pypdfium-staged method vs fused-operator method — fair, same pages, same real model.

For each page, runs the REAL NemotronPageElementsV3 two ways and times components:
  A) STAGED (current): pypdfium render -> base64 PNG encode -> decode -> model            (host transport)
  B) FUSED  (ours):    pypdfium render -> upload device -> torch.from_dlpack -> model      (zero-copy)
Render is identical (shared). Verifies prediction parity (Δ), reports ms/page by component and
pages/sec, and writes a chart to results/method_comparison.png.

Env: HF_HOME -> cached weights, HF_HUB_OFFLINE=1.  Run with .venv/bin/python.
"""
from __future__ import annotations
import base64, io, sys, time
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "gpu_pdf_extractor" / "native" / "build"),
                str(ROOT / "gpu_pdf_extractor" / "python"), str(ROOT / "nemo_retriever" / "src")]
import _gpu_raster as R, gpu_pdfium
from PIL import Image
from nemo_retriever.models.local import NemotronPageElementsV3

SYNC = torch.cuda.synchronize


def render_rgb(pdf_bytes, target=1024):
    doc = gpu_pdfium.PdfDocument(pdf_bytes)
    try:
        p = doc.get_page(0)
        s = max(min(target / p.get_width(), target / p.get_height()), 1e-3)
        rgb = np.ascontiguousarray(p.render(scale=s).to_numpy()[:, :, ::-1]); p.close()
    finally:
        doc.close()
    return rgb


def staged(model, rgb):
    t = time.perf_counter()
    im = Image.fromarray(rgb); b = io.BytesIO(); im.save(b, format="PNG")
    b64 = base64.b64encode(b.getvalue()).decode()
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    transport = time.perf_counter() - t
    H, W = arr.shape[:2]
    t = time.perf_counter(); x = model.preprocess(arr)
    with torch.inference_mode():
        preds = model(x, (H, W))
    SYNC(); model_t = time.perf_counter() - t
    return preds, transport, model_t


def fused(model, rgb):
    t = time.perf_counter()
    dev = R.upload_to_device(rgb, 0)
    tt = torch.from_dlpack(dev)
    chw = tt.permute(2, 0, 1).contiguous().to(torch.float32)
    SYNC(); transport = time.perf_counter() - t
    H, W = int(tt.shape[0]), int(tt.shape[1])
    t = time.perf_counter(); x = model.preprocess(chw)
    with torch.inference_mode():
        preds = model(x, (H, W))
    SYNC(); model_t = time.perf_counter() - t
    return preds, transport, model_t, dev.data_ptr == tt.data_ptr()


def maxdiff(a, b):
    if isinstance(a, torch.Tensor):
        return float((a.float() - b.float()).abs().max()) if a.shape == b.shape else float("inf")
    if isinstance(a, (list, tuple)):
        return max((maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def main():
    pdfs = (sorted((ROOT/"gpu_pdf_extractor"/"corpus"/"dense_vector").glob("*.pdf"))[:6] +
            sorted((ROOT/"gpu_pdf_extractor"/"corpus"/"born_digital").glob("*.pdf"))[:6])
    model = NemotronPageElementsV3()
    rasters = [render_rgb(p.read_bytes()) for p in pdfs]
    # render timing (shared by both)
    rt = []
    for p in pdfs:
        t = time.perf_counter(); render_rgb(p.read_bytes()); rt.append((time.perf_counter()-t)*1000)
    # warmup
    staged(model, rasters[0]); fused(model, rasters[0])

    A = {"transport": [], "model": []}; B = {"transport": [], "model": []}
    worst = 0.0; ptr_ok = True
    for rgb in rasters:
        pa, ta, ma = staged(model, rgb); A["transport"].append(ta*1000); A["model"].append(ma*1000)
        pb, tb, mb, ok = fused(model, rgb); B["transport"].append(tb*1000); B["model"].append(mb*1000)
        ptr_ok = ptr_ok and ok; worst = max(worst, maxdiff(pa, pb))

    mr = float(np.mean(rt))
    aA = (mr, float(np.mean(A["transport"])), float(np.mean(A["model"])))
    aB = (mr, float(np.mean(B["transport"])), float(np.mean(B["model"])))
    totA, totB = sum(aA), sum(aB)
    print(f"pages={len(rasters)}  parity max|Δ|={worst:.1e}  zero-copy_ptr_match={ptr_ok}\n")
    print(f"{'component':12s} {'STAGED(b64) ms':>16} {'FUSED(dlpack) ms':>18}")
    for i, name in enumerate(("render", "transport", "model")):
        print(f"{name:12s} {(aA[i]):>16.2f} {(aB[i]):>18.2f}")
    print(f"{'TOTAL':12s} {totA:>16.2f} {totB:>18.2f}")
    print(f"{'pages/sec':12s} {1000/totA:>16.1f} {1000/totB:>18.1f}")
    print(f"\ntransport speedup: {aA[1]/max(aB[1],1e-6):.1f}x   end-to-end per-page speedup: {totA/totB:.2f}x")

    # ---- chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["STAGED\n(pypdfium+base64)", "FUSED\n(DLPack zero-copy)"]
    rentd = [aA[0], aB[0]]; trans = [aA[1], aB[1]]; modl = [aA[2], aB[2]]
    ax1.bar(labels, rentd, label="render (pypdfium)", color="#8c9eff")
    ax1.bar(labels, trans, bottom=rentd, label="transport", color="#ff7043")
    ax1.bar(labels, modl, bottom=[rentd[i]+trans[i] for i in range(2)], label="model fwd", color="#66bb6a")
    ax1.set_ylabel("ms / page (mean)"); ax1.set_title("Per-page time by component"); ax1.legend()
    for i, tot in enumerate((totA, totB)):
        ax1.text(i, tot, f"{tot:.0f} ms", ha="center", va="bottom", fontweight="bold")
    pps = [1000/totA, 1000/totB]
    bars = ax2.bar(labels, pps, color=["#90a4ae", "#26a69a"])
    ax2.set_ylabel("pages / sec"); ax2.set_title(f"Throughput (per process)  •  parity Δ={worst:.0e}")
    for b, v in zip(bars, pps):
        ax2.text(b.get_x()+b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
    fig.suptitle("PDF page processing: staged pypdfium+base64 vs fused DLPack (same model, same pages)")
    fig.tight_layout()
    out = ROOT / "gpu_pdf_extractor" / "results" / "method_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"\nchart -> {out}")


if __name__ == "__main__":
    main()
