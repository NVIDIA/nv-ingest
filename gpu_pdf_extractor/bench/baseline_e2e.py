#!/usr/bin/env python3
"""P0.3 — End-to-end throughput baseline through the REAL nemo_retriever operators.

Runs split_pdf_batch -> pdf_extraction (the actual code paths we will replace) on the corpus,
CPU-only, with NIM/YOLOX-dependent flags OFF. Measures the gate metric: pages/sec and docs/sec.
Caps pages-per-doc and docs-per-bucket to bound wall time on huge PDFs (e.g. citigroup=963pg).

Single-thread by default; pass --workers N to measure throughput scaling (the GPU backend's
batching is what must beat the multi-worker CPU number). Writes results/baseline_e2e.json.

Run:  python3 gpu_pdf_extractor/bench/baseline_e2e.py [--pages-per-doc 8] [--docs-per-bucket 6] [--workers 1]
"""
from __future__ import annotations
import argparse, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nemo_retriever" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from common import CORPUS, load_manifest, percentiles, peak_rss_mb, env_block, write_result  # noqa
from nemo_retriever.operators.extract.pdf.split import split_pdf_batch
from nemo_retriever.operators.extract.pdf.extract import pdf_extraction
from nemo_retriever.common.params import PdfSplitParams


def run_doc(path: Path, pages_per_doc: int) -> dict:
    raw = path.read_bytes()
    t0 = time.perf_counter()
    df_in = pd.DataFrame([{"bytes": raw, "path": path.name}])
    pages = split_pdf_batch(df_in, params=PdfSplitParams(start_page=1, end_page=pages_per_doc))
    split_s = time.perf_counter() - t0
    n = len(pages)
    if n == 0:
        return {"name": path.name, "pages": 0, "wall_s": split_s, "split_s": split_s, "extract_s": 0.0}
    t0 = time.perf_counter()
    out = pdf_extraction(pages, extract_text=True, extract_images=True, extract_page_as_image=True,
                         text_extraction_method="pdfium", render_mode="fit_to_model")
    extract_s = time.perf_counter() - t0
    errs = sum(1 for _, r in out.iterrows() if r.get("metadata", {}).get("error"))
    return {"name": path.name, "pages": n, "wall_s": split_s + extract_s,
            "split_s": split_s, "extract_s": extract_s, "page_errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-per-doc", type=int, default=8)
    ap.add_argument("--docs-per-bucket", type=int, default=6)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    man = load_manifest()
    jobs = []  # (bucket, path)
    for bucket, names in man["buckets"].items():
        if bucket == "malformed":
            continue  # exercised by area bench; not a throughput signal
        for name in names[: args.docs_per_bucket]:
            p = CORPUS / bucket / name
            if p.exists():
                jobs.append((bucket, p))

    results: dict[str, list] = {}
    t_all = time.perf_counter()
    if args.workers == 1:
        ran = [(b, run_doc(p, args.pages_per_doc)) for b, p in jobs]
    else:
        # Process-based (matches production Ray model). pypdfium2/PDFium is not reliably
        # thread-safe for concurrent split+render, so threads are intentionally NOT used.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [(b, ex.submit(run_doc, p, args.pages_per_doc)) for b, p in jobs]
            ran = [(b, f.result()) for b, f in futs]
    wall_all = time.perf_counter() - t_all
    for b, r in ran:
        results.setdefault(b, []).append(r)

    out = {"config": {**vars(args), "backend": "pypdfium2",
                      "flags": "extract_text+images+page_as_image, method=pdfium (no NIM)"},
           "env": env_block(), "buckets": {}}
    tot_pages = tot_docs = 0
    tot_wall_serial = 0.0
    for b, rs in results.items():
        pages = sum(r["pages"] for r in rs)
        wall = sum(r["wall_s"] for r in rs)  # serial-equivalent (sum of per-doc walls)
        per_doc_ms = [1000 * r["wall_s"] for r in rs]
        tot_pages += pages; tot_docs += len(rs); tot_wall_serial += wall
        out["buckets"][b] = {
            "docs": len(rs), "pages": pages,
            "pages_per_sec_serial": round(pages / wall, 1) if wall else None,
            "docs_per_sec_serial": round(len(rs) / wall, 2) if wall else None,
            "ms_per_doc": {k: (round(v, 1) if v is not None else None)
                           for k, v in percentiles(per_doc_ms).items()},
            "split_frac": round(sum(r["split_s"] for r in rs) / wall, 3) if wall else None,
            "extract_frac": round(sum(r["extract_s"] for r in rs) / wall, 3) if wall else None,
        }
    out["overall"] = {
        "docs": tot_docs, "pages": tot_pages,
        "serial_pages_per_sec": round(tot_pages / tot_wall_serial, 1) if tot_wall_serial else None,
        "serial_docs_per_sec": round(tot_docs / tot_wall_serial, 2) if tot_wall_serial else None,
        "wallclock_pages_per_sec": round(tot_pages / wall_all, 1) if wall_all else None,
        "workers": args.workers,
        "wallclock_s": round(wall_all, 2),
    }
    out["peak_rss_mb"] = peak_rss_mb()
    p = write_result("baseline_e2e.json", out)

    print(f"{'bucket':14s} {'docs':>4} {'pages':>5} {'pg/s':>7} {'doc/s':>6} {'split%':>7} {'extract%':>8}")
    for b, r in out["buckets"].items():
        print(f"{b:14s} {r['docs']:>4} {r['pages']:>5} {str(r['pages_per_sec_serial']):>7} "
              f"{str(r['docs_per_sec_serial']):>6} {str(r['split_frac']):>7} {str(r['extract_frac']):>8}")
    ov = out["overall"]
    print(f"\nOVERALL serial: {ov['serial_pages_per_sec']} pg/s, {ov['serial_docs_per_sec']} doc/s "
          f"over {ov['pages']} pages / {ov['docs']} docs (workers={ov['workers']}, "
          f"wallclock {ov['wallclock_pages_per_sec']} pg/s)\npeak RSS {out['peak_rss_mb']} MB  -> {p}")


if __name__ == "__main__":
    main()
