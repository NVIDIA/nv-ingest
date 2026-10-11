#!/usr/bin/env python3
"""Full fused GPU stage with ALL real downstream models, routed by page-element label.

Chain: Rasterize -> PageElement(real) -> Crop(on-device, label-tagged)
       -> TableStructure(real, 'table')  -> GraphicElements(real, 'chart')  -> OCR(real, 'text')

One uploaded device raster per page; every model consumes on-device crops with zero-copy handoff
between operators. Env: HF_HOME -> cached weights, HF_HUB_OFFLINE=1.  Run with .venv/bin/python.
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
                   CropGPUOperator, TableStructureGPUOperator, GraphicElementsGPUOperator,
                   OCRGPUOperator)


def make_df(n=6):
    rows = []
    for sub in ("dense_vector", "multi_image", "born_digital"):
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
        TableStructureGPUOperator(dev=0),
        GraphicElementsGPUOperator(dev=0),
        OCRGPUOperator(dev=0, max_regions_per_page=4),
    ])
    fused(df.head(1))  # warmup all models
    t = time.perf_counter()
    out = fused(df)
    dt = time.perf_counter() - t

    print(f"{'page':38s} {'regions':>7} {'tables':>6} {'charts':>6} {'text_ocr':>8} {'ocr_chars':>9}")
    for _, row in out.iterrows():
        print(f"{row['path'][:38]:38s} {len(row['region_crops_dev']):>7} "
              f"{row['n_tables_structured']:>6} {row['n_charts_structured']:>6} "
              f"{row['n_text_ocr']:>8} {row['ocr_chars']:>9}")
    print(f"\non-device: table_inputs={out.attrs.get('table_inputs_on_device')} "
          f"chart_inputs={out.attrs.get('chart_inputs_on_device')}")
    print(f"real inferences: tables={int(out['n_tables_structured'].sum())} "
          f"charts={int(out['n_charts_structured'].sum())} text_ocr={int(out['n_text_ocr'].sum())}")
    print(f"fused stage wall: {dt*1000:.0f} ms for {len(df)} pages "
          f"(1 H2D/page, 0 base64 / 0 D2H between operators)")
    print("pipeline:", " -> ".join(type(o).__name__ for o in fused.operators))


if __name__ == "__main__":
    main()
