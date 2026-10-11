#!/usr/bin/env python3
"""Wire the fused GPU stage as a REAL Ray Data GPU actor.

Runs FusedGPUOperator via ds.map_batches with num_gpus=1 and concurrency>1, so Ray spins up a pool
of GPU actors, each constructing the child operators (rasterize + real page-element + crop + real
table/graphic models) ON THE WORKER, and processes single-page-PDF batches. This is exactly how the
codebase's executor stages a GPUOperator node (operator_class + fn_constructor_kwargs + auto num_gpus).

Env handled via Ray runtime_env (PYTHONPATH + HF cache). Run with .venv/bin/python.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = str(ROOT / "gpu_pdf_extractor" / "python")
NATIVE = str(ROOT / "gpu_pdf_extractor" / "native" / "build")
SRC = str(ROOT / "nemo_retriever" / "src")
sys.path[:0] = [PKG, NATIVE, SRC]

import pandas as pd
import ray
import gpu_pdfium
from fused import (FusedGPUOperator, RasterizeGPUOperator, PageElementGPUOperator,
                   CropGPUOperator, TableStructureGPUOperator, GraphicElementsGPUOperator,
                   HostFinalizeOperator)

HF_HOME = "/raid/nemo-retriever-harbor-adapters/.cache/huggingface"

# Picklable spec: classes by reference + plain kwargs. Children built on the Ray worker.
OPERATOR_SPECS = [
    (RasterizeGPUOperator, {"target_px": 1024, "dev": 0}),
    (PageElementGPUOperator, {"dev": 0}),
    (CropGPUOperator, {}),
    (TableStructureGPUOperator, {"dev": 0}),
    (GraphicElementsGPUOperator, {"dev": 0}),
    (HostFinalizeOperator, {}),   # drop device-only columns -> host-serializable stage output
]


def make_rows(n=8):
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
    return rows


def main():
    rows = make_rows(8)
    ray.init(
        num_gpus=8,
        runtime_env={"env_vars": {
            "PYTHONPATH": f"{PKG}:{NATIVE}:{SRC}",
            "HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PATH": "/usr/local/cuda/bin:" + __import__("os").environ.get("PATH", ""),
        }},
        ignore_reinit_error=True,
    )
    print("ray:", ray.__version__, "| cluster GPUs:", ray.cluster_resources().get("GPU"))

    ds = ray.data.from_pandas(pd.DataFrame(rows))
    t = time.perf_counter()
    out_ds = ds.map_batches(
        FusedGPUOperator,
        fn_constructor_kwargs={"operator_specs": OPERATOR_SPECS},  # built on worker
        batch_format="pandas",
        num_gpus=1,          # each actor reserves 1 GPU (what the executor auto-sets for GPUOperator)
        concurrency=2,       # pool of 2 GPU actors
        batch_size=4,
    )
    res = out_ds.to_pandas()
    dt = time.perf_counter() - t

    show = [c for c in ("path", "n_region_crops", "n_tables_structured", "n_charts_structured")
            if c in res.columns]
    print(res[show].to_string(index=False))
    print(f"\nReal Ray GPU actors: 2 actors x 1 GPU, models built on workers.")
    print(f"processed {len(res)} pages in {dt:.1f}s via ds.map_batches(num_gpus=1, concurrency=2)")
    print(f"tables={int(res.get('n_tables_structured', pd.Series([0])).sum())} "
          f"charts={int(res.get('n_charts_structured', pd.Series([0])).sum())}")
    ray.shutdown()


if __name__ == "__main__":
    main()
