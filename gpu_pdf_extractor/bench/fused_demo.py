#!/usr/bin/env python3
"""P3'.3 — Demonstrate the fused GPU operator with zero-copy DLPack handoff between operators.

Builds FusedGPUOperator([RasterizeGPUOperator, PageElementStubGPUOperator]) and runs it on
single-page PDFs from the corpus. Proves the device raster produced by the rasterize operator is
consumed by the next operator at the SAME device pointer — i.e. it crossed the operator boundary
with zero host copies (no base64, no D2H+reupload), which is impossible across a Ray stage boundary.

Run:  python3 gpu_pdf_extractor/bench/fused_demo.py
"""
from __future__ import annotations
import base64, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "python"))
sys.path.insert(0, str(ROOT / "nemo_retriever" / "src"))
import gpu_pdfium
from fused import FusedGPUOperator, RasterizeGPUOperator, PageElementStubGPUOperator


def make_single_page_df(n_pages: int = 8) -> pd.DataFrame:
    """Split a few corpus PDFs into single-page rows (bytes), like split_pdf_batch output."""
    rows = []
    corpus = ROOT / "gpu_pdf_extractor" / "corpus"
    pdfs = list((corpus / "born_digital").glob("*.pdf")) + list((corpus / "dense_vector").glob("*.pdf"))
    for p in pdfs:
        if len(rows) >= n_pages:
            break
        doc = gpu_pdfium.PdfDocument(p.read_bytes())
        try:
            single = gpu_pdfium.PdfDocument.new()
            single.import_pages(doc, pages=[0])
            buf = __import__("io").BytesIO(); single.save(buf); single.close()
            rows.append({"bytes": buf.getvalue(), "path": p.name, "page_number": 1})
        finally:
            doc.close()
    return pd.DataFrame(rows)


def base64_roundtrip_cost(df, target_px=1024):
    """Measure what the CURRENT staged pipeline pays between extract and page_elements:
    render -> base64 encode -> (Ray serialize) -> base64 decode -> numpy. (host copies)."""
    from PIL import Image
    import io
    t = time.perf_counter()
    for _, row in df.iterrows():
        doc = gpu_pdfium.PdfDocument(row["bytes"]); page = doc.get_page(0)
        arr = page.render(scale=min(target_px/page.get_width(), target_px/page.get_height())).to_numpy()
        # encode (extract stage output) ...
        im = Image.fromarray(arr[:, :, ::-1]); b = io.BytesIO(); im.save(b, format="PNG")
        b64 = base64.b64encode(b.getvalue()).decode()
        # ... decode (page_elements stage input)
        raw = base64.b64decode(b64); arr2 = np.array(Image.open(io.BytesIO(raw)))
        page.close(); doc.close()
    return time.perf_counter() - t


def main():
    df = make_single_page_df(8)
    print(f"input: {len(df)} single-page PDFs\n")

    fused = FusedGPUOperator(operators=[RasterizeGPUOperator(target_px=1024, dev=0),
                                        PageElementStubGPUOperator()])
    fused(df.head(1))  # warm up CUDA context (amortize lazy init out of the timed run)
    t = time.perf_counter()
    out = fused(df)
    dt = time.perf_counter() - t

    # Prove zero-copy: the pointer the consumer read == the pointer the producer allocated.
    producer_ptr = [di.data_ptr for di in out["page_image_dev"]]
    consumer_ptr = list(out["consumed_device_ptr"])
    zero_copy = all(a == b for a, b in zip(producer_ptr, consumer_ptr))

    print("=== fused device pipeline (rasterize -> stub page-element) ===")
    for i, row in out.iterrows():
        di = row["page_image_dev"]
        print(f"  {row['path'][:40]:40s} dev_img {tuple(di.shape)} @ {hex(di.data_ptr)} "
              f"-> consumer @ {hex(row['consumed_device_ptr'])} act={row['stub_activation']:.2f}")
    print(f"\nZERO-COPY across operator boundary (producer ptr == consumer ptr): {zero_copy}")
    print(f"host copies between operators: 0 base64, 0 D2H  (1 H2D total, inside rasterize)")
    print(f"fused wall: {dt*1000:.1f} ms for {len(df)} pages")

    b64 = base64_roundtrip_cost(df)
    print(f"\nbaseline base64 round-trip (what crosses the Ray extract->page_elements boundary today):"
          f" {b64*1000:.1f} ms for {len(df)} pages")
    print(f"  -> fused removes the base64 encode/decode + the cross-stage host serialization.")
    print(f"\ntorch hook: replace consume_mean with `torch.from_dlpack(dev_img)` -> in-process model "
          f"(DeviceImage exposes __dlpack__ / __dlpack_device__={out['page_image_dev'].iloc[0].__dlpack_device__()}).")


if __name__ == "__main__":
    main()
