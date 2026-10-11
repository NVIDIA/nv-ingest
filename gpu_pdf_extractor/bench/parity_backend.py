#!/usr/bin/env python3
"""P1.4 — Corpus-wide parity: gpu_pdfium (native) vs pypdfium2, per capability area.

For each corpus doc, compares the two backends on: render (SSIM/exact-match over sampled pages),
text extraction (string equality), object enumeration (counts by type), and metadata. Aggregates
per bucket and writes results/parity_backend.json. This is the P1 acceptance evidence.

Run:  python3 gpu_pdf_extractor/bench/parity_backend.py [--dpi 200] [--max-pages-per-doc 4]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pypdfium2 as pp
import pypdfium2.raw as praw
import gpu_pdfium as gp
from common import CORPUS, load_manifest, write_result, env_block  # noqa
from parity import compare  # noqa

TYPES = [("text", praw.FPDF_PAGEOBJ_TEXT), ("path", praw.FPDF_PAGEOBJ_PATH),
         ("image", praw.FPDF_PAGEOBJ_IMAGE), ("form", praw.FPDF_PAGEOBJ_FORM)]


def cmp_doc(path: Path, dpi: float, max_pages: int) -> dict:
    raw = path.read_bytes()
    rec = {"name": path.name}
    try:
        pdoc, gdoc = pp.PdfDocument(raw), gp.PdfDocument(raw)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["pagecount_match"] = (len(pdoc) == len(gdoc))
    n = min(len(pdoc), max_pages)
    ssims, exacts, shape_ok, text_ok, obj_ok = [], [], True, 0, 0
    scale = dpi / 72.0
    for i in range(n):
        pa = pdoc[i].render(scale=scale).to_numpy()
        ga = gdoc[i].render(scale=scale).to_numpy()
        c = compare(pa, ga)
        shape_ok = shape_ok and c.get("shape_match", False)
        if c.get("shape_match"):
            ssims.append(c["ssim"]); exacts.append(c["exact_match_frac"])
        pt = pdoc[i].get_textpage().get_text_bounded() or ""
        gt = gdoc[i].get_textpage().get_text_bounded() or ""
        text_ok += int(pt.strip() == gt.strip())
        page_obj_ok = all(
            len(list(pdoc[i].get_objects(filter=(t,), max_depth=1)))
            == len(list(gdoc[i].get_objects(filter=(t,), max_depth=1)))
            for _, t in TYPES
        )
        obj_ok += int(page_obj_ok)
    rec.update({
        "pages_compared": n,
        "render_shape_match": shape_ok,
        "render_min_ssim": round(min(ssims), 5) if ssims else None,
        "render_mean_exact": round(sum(exacts) / len(exacts), 5) if exacts else None,
        "text_match_pages": f"{text_ok}/{n}",
        "objcount_match_pages": f"{obj_ok}/{n}",
        "metadata_match": (pdoc.get_metadata_dict().get("Producer")
                           == gdoc.get_metadata_dict().get("Producer")),
    })
    pdoc.close(); gdoc.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=float, default=200)
    ap.add_argument("--max-pages-per-doc", type=int, default=4)
    args = ap.parse_args()
    man = load_manifest()
    out = {"config": vars(args), "env": env_block({"gpu_backend": gp.PDFIUM_INFO}), "buckets": {}}
    g_min_ssim, g_text_pages, g_text_total, g_obj_pages, g_obj_total = 1.0, 0, 0, 0, 0
    docs_ok = docs_err = 0
    for bucket, names in man["buckets"].items():
        recs = []
        for name in names:
            p = CORPUS / bucket / name
            if not p.exists():
                continue
            r = cmp_doc(p, args.dpi, args.max_pages_per_doc)
            recs.append(r)
            if r.get("error"):
                docs_err += 1
                continue
            docs_ok += 1
            if r["render_min_ssim"] is not None:
                g_min_ssim = min(g_min_ssim, r["render_min_ssim"])
            to, tn = map(int, r["text_match_pages"].split("/"))
            oo, on = map(int, r["objcount_match_pages"].split("/"))
            g_text_pages += to; g_text_total += tn; g_obj_pages += oo; g_obj_total += on
        out["buckets"][bucket] = recs
    out["summary"] = {
        "docs_compared_ok": docs_ok, "docs_errored": docs_err,
        "global_min_render_ssim": round(g_min_ssim, 5),
        "text_match_rate": f"{g_text_pages}/{g_text_total}",
        "objcount_match_rate": f"{g_obj_pages}/{g_obj_total}",
    }
    p = write_result("parity_backend.json", out)
    print(f"{'bucket':14s} {'docs':>4} {'min_ssim':>9} {'text_ok':>9} {'obj_ok':>9} {'meta_ok':>8}")
    for b, recs in out["buckets"].items():
        ok = [r for r in recs if not r.get("error")]
        if not ok:
            print(f"{b:14s} {len(recs):>4}  (all errored/empty)"); continue
        mss = min((r["render_min_ssim"] for r in ok if r["render_min_ssim"] is not None), default=None)
        tt = sum(int(r['text_match_pages'].split('/')[0]) for r in ok)
        tn = sum(int(r['text_match_pages'].split('/')[1]) for r in ok)
        oo = sum(int(r['objcount_match_pages'].split('/')[0]) for r in ok)
        on = sum(int(r['objcount_match_pages'].split('/')[1]) for r in ok)
        mok = sum(int(r["metadata_match"]) for r in ok)
        print(f"{b:14s} {len(ok):>4} {str(mss):>9} {f'{tt}/{tn}':>9} {f'{oo}/{on}':>9} {f'{mok}/{len(ok)}':>8}")
    s = out["summary"]
    print(f"\nSUMMARY  docs_ok={s['docs_compared_ok']} errored={s['docs_errored']}  "
          f"min_render_ssim={s['global_min_render_ssim']}  text={s['text_match_rate']}  "
          f"objcounts={s['objcount_match_rate']}\n-> {p}")


if __name__ == "__main__":
    main()
