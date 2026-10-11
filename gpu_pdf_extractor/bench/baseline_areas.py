#!/usr/bin/env python3
"""P0.2 — Baseline benchmark of the current pypdfium backend, per capability area.

For each corpus bucket, samples pages across its docs and measures per-page latency for each
of the 6 capability areas (open amortized, render, text, object-enum, image-decode, split),
plus p50/p90/p99 and pages/sec. Records a stable raster hash per sampled page (drift/parity
anchor for the future GPU backend). Writes results/baseline_areas.json.

Run:  python3 gpu_pdf_extractor/bench/baseline_areas.py [--dpi 300] [--max-pages-per-doc 6]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as praw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPUS, load_manifest, percentiles, peak_rss_mb, raster_hash, env_block, write_result  # noqa


def bench_doc(path: Path, dpi: int, max_pages: int):
    raw = path.read_bytes()
    t0 = time.perf_counter()
    doc = pdfium.PdfDocument(raw)
    open_s = time.perf_counter() - t0
    scale = dpi / 72.0
    per_area = {k: [] for k in ("render", "text", "objects", "image_decode")}
    hashes = []
    n = min(len(doc), max_pages)
    for i in range(n):
        page = doc.get_page(i)
        try:
            s = time.perf_counter(); arr = page.render(scale=scale).to_numpy(); per_area["render"].append(time.perf_counter() - s)
            if i == 0:
                hashes.append(raster_hash(arr))
            s = time.perf_counter(); page.get_textpage().get_text_bounded(); per_area["text"].append(time.perf_counter() - s)
            s = time.perf_counter()
            objs = list(page.get_objects(filter=(praw.FPDF_PAGEOBJ_IMAGE,), max_depth=1))
            per_area["objects"].append(time.perf_counter() - s)
            s = time.perf_counter()
            for o in objs:
                try:
                    b = o.get_bitmap(render=True)
                    if b is not None:
                        b.to_numpy()
                except Exception:
                    pass
            per_area["image_decode"].append(time.perf_counter() - s)
        finally:
            page.close()
    # split: import page 0 into a fresh doc and save
    t0 = time.perf_counter()
    try:
        single = pdfium.PdfDocument.new(); single.import_pages(doc, pages=[0])
        import io as _io; single.save(_io.BytesIO()); single.close()
        split_s = time.perf_counter() - t0
    except Exception:
        split_s = None
    doc.close()
    return {"pages_profiled": n, "open_s": open_s, "split_s_per_page": split_s,
            "per_area_s": per_area, "first_page_raster_sha1": hashes[0] if hashes else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-pages-per-doc", type=int, default=6)
    args = ap.parse_args()

    man = load_manifest()
    out = {"config": {"dpi": args.dpi, "max_pages_per_doc": args.max_pages_per_doc, "backend": "pypdfium2"},
           "env": env_block(), "buckets": {}}

    for bucket, names in man["buckets"].items():
        agg = {k: [] for k in ("render", "text", "objects", "image_decode")}
        opens, splits, docs_done, pages_done, doc_reports = [], [], 0, 0, []
        for name in names:
            p = CORPUS / bucket / name
            if not p.exists():
                continue
            try:
                r = bench_doc(p, args.dpi, args.max_pages_per_doc)
            except Exception as e:
                doc_reports.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                continue
            docs_done += 1; pages_done += r["pages_profiled"]
            opens.append(r["open_s"])
            if r["split_s_per_page"] is not None:
                splits.append(r["split_s_per_page"])
            for k in agg:
                agg[k].extend(r["per_area_s"][k])
            doc_reports.append({"name": name, "pages": r["pages_profiled"],
                                "first_page_raster_sha1": r["first_page_raster_sha1"]})
        render = agg["render"]
        out["buckets"][bucket] = {
            "docs": docs_done, "pages_profiled": pages_done,
            "ms_per_page": {k: round(1000 * sum(v) / len(v), 3) if v else None for k, v in agg.items()},
            "render_latency_ms": {k: (round(1000 * x, 3) if x is not None else None)
                                  for k, x in percentiles(render).items()},
            "render_pages_per_sec": round(len(render) / sum(render), 1) if render else None,
            "open_ms_mean": round(1000 * sum(opens) / len(opens), 3) if opens else None,
            "split_ms_per_page_mean": round(1000 * sum(splits) / len(splits), 3) if splits else None,
            "docs_detail": doc_reports,
        }
    out["peak_rss_mb"] = peak_rss_mb()
    p = write_result("baseline_areas.json", out)

    print(f"{'bucket':14s} {'docs':>4} {'pages':>5} {'render':>8} {'text':>7} {'objects':>8} {'imgdec':>7} {'pg/s':>7}")
    for b, r in out["buckets"].items():
        mp = r["ms_per_page"]
        print(f"{b:14s} {r['docs']:>4} {r['pages_profiled']:>5} "
              f"{str(mp['render']):>8} {str(mp['text']):>7} {str(mp['objects']):>8} "
              f"{str(mp['image_decode']):>7} {str(r['render_pages_per_sec']):>7}")
    print(f"\npeak RSS: {out['peak_rss_mb']} MB   ->  {p}")


if __name__ == "__main__":
    main()
