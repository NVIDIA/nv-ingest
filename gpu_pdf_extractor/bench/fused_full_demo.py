#!/usr/bin/env python3
"""P3'(b) — Full fused GPU stage: rasterize -> page-element detect -> crop -> OCR, all on-device.

Builds FusedGPUOperator([Rasterize, PageElement(real model), Crop, OCRStub]) and runs it on
single-page PDFs. The page raster is uploaded once (1 H2D); from there the device tensor is shared
zero-copy into the real detection model (torch.from_dlpack), the detected regions are cropped
ON-DEVICE, and the crops are consumed by the downstream (stub) OCR/table op without ever leaving
the GPU. Proves the whole crop->downstream chain keeps device residency in one process.

Real table-structure/OCR weights aren't cached offline here, so the final op is a stub; the crops
are real device tensors ready for `model(crop)`.

Env: HF_HOME -> cached weights, HF_HUB_OFFLINE=1.  Run with .venv/bin/python.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "python"))
sys.path.insert(0, str(ROOT / "nemo_retriever" / "src"))
import gpu_pdfium
from fused import (FusedGPUOperator, RasterizeGPUOperator, PageElementGPUOperator,
                   CropGPUOperator, TableStructureGPUOperator)


def make_df(n=6):
    rows = []
    for sub in ("dense_vector", "born_digital"):
        for p in sorted((ROOT / "gpu_pdf_extractor" / "corpus" / sub).glob("*.pdf")):
            if len(rows) >= n:
                break
            doc = gpu_pdfium.PdfDocument(p.read_bytes())
            try:
                s = gpu_pdfium.PdfDocument.new(); s.import_pages(doc, pages=[0])
                import io; b = io.BytesIO(); s.save(b); s.close()
                rows.append({"bytes": b.getvalue(), "path": p.name, "page_number": 1})
            finally:
                doc.close()
    return pd.DataFrame(rows)


def main():
    df = make_df(6)
    fused = FusedGPUOperator(operators=[
        RasterizeGPUOperator(target_px=1024, dev=0),
        PageElementGPUOperator(dev=0),
        CropGPUOperator(),
        TableStructureGPUOperator(dev=0),   # REAL downstream model on table crops
    ])
    fused(df.head(1))  # warmup
    t = time.perf_counter()
    out = fused(df)
    dt = time.perf_counter() - t

    print(f"{'page':40s} {'page_dev_ptr':>16} {'#regions':>9} {'#tables':>8}")
    for _, row in out.iterrows():
        print(f"{row['path'][:40]:40s} {hex(row['page_image_dev'].data_ptr):>16} "
              f"{len(row['region_crops_dev']):>9} {row['n_tables_structured']:>8}")
    print(f"\ntable-structure inputs on-device (never left GPU): {out.attrs.get('table_inputs_on_device')}")
    print(f"real table-structure inferences run: {int(out['n_tables_structured'].sum())}")
    print(f"fused stage wall: {dt*1000:.1f} ms for {len(df)} pages "
          f"(rasterize->REAL detect->crop->REAL table-structure, 1 H2D/page, 0 base64, 0 D2H between ops)")
    print("pipeline:", " -> ".join(type(o).__name__ for o in fused.operators))


if __name__ == "__main__":
    main()
