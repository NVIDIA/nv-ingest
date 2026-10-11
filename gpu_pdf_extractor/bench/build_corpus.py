#!/usr/bin/env python3
"""P0.1 — Assemble a bucketed golden corpus from repo PDFs + synthetic stressors.

Discovers PDFs repo-wide (excluding .venv and the corpus dir itself), dedupes by content
hash, probes each with pypdfium2 to measure features (pages, text density, image/path object
counts, rotation), classifies into buckets, symlinks them under corpus/<bucket>/, and writes
corpus/manifest.json. Also synthesizes a couple of malformed PDFs.

Buckets: born_digital, scanned, rotated, multi_image, dense_vector, malformed.
Run:  python3 gpu_pdf_extractor/bench/build_corpus.py
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as praw

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "gpu_pdf_extractor" / "corpus"
BUCKETS = ["born_digital", "scanned", "rotated", "multi_image", "dense_vector", "malformed"]
PROBE_PAGES = 8  # cap per-doc probing cost on huge PDFs


def discover() -> list[Path]:
    out = []
    for p in REPO.rglob("*.pdf"):
        s = str(p)
        if "/.venv/" in s or "/gpu_pdf_extractor/corpus/" in s:
            continue
        try:
            rp = p.resolve()
            if rp.is_file():
                out.append(rp)
        except Exception:
            continue
    return out


def dedupe(paths: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for p in paths:
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            continue
        if h not in seen:
            seen[h] = p
    return list(seen.values())


def probe(path: Path) -> dict | None:
    try:
        raw = path.read_bytes()
        doc = pdfium.PdfDocument(raw)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    try:
        npages = len(doc)
        probe_n = min(npages, PROBE_PAGES)
        tot_chars = tot_imgs = tot_paths = 0
        rotations = set()
        sizes = []
        for i in range(probe_n):
            page = doc.get_page(i)
            try:
                rotations.add(int(page.get_rotation()))
                sizes.append((round(page.get_width(), 1), round(page.get_height(), 1)))
                txt = page.get_textpage().get_text_bounded() or ""
                tot_chars += len(txt.strip())
                for obj in page.get_objects(filter=(praw.FPDF_PAGEOBJ_IMAGE,), max_depth=1):
                    tot_imgs += 1
                for obj in page.get_objects(filter=(praw.FPDF_PAGEOBJ_PATH,), max_depth=1):
                    tot_paths += 1
            finally:
                page.close()
        return {
            "pages": npages,
            "probed_pages": probe_n,
            "chars_per_page": round(tot_chars / max(probe_n, 1), 1),
            "images_per_page": round(tot_imgs / max(probe_n, 1), 2),
            "paths_per_page": round(tot_paths / max(probe_n, 1), 2),
            "rotations": sorted(rotations),
            "page_size_pt": sizes[0] if sizes else None,
            "size_bytes": len(raw),
        }
    finally:
        doc.close()


def classify(feat: dict) -> str:
    if feat.get("error"):
        return "malformed"
    if any(r % 360 != 0 for r in feat.get("rotations", [])):
        return "rotated"
    chars = feat.get("chars_per_page", 0)
    imgs = feat.get("images_per_page", 0)
    paths = feat.get("paths_per_page", 0)
    if chars < 50 and imgs >= 1:
        return "scanned"          # little/no text, image-backed → scanned-like
    if imgs >= 1.5:
        return "multi_image"
    if paths >= 50:
        return "dense_vector"     # vector-heavy (the object-enum-spike case)
    return "born_digital"


def synth_malformed(dst: Path, donor: Path | None):
    dst.mkdir(parents=True, exist_ok=True)
    # 1) garbage that claims to be a PDF
    (dst / "garbage_header.pdf").write_bytes(b"%PDF-1.7\n" + b"\xde\xad\xbe\xef" * 64 + b"\n%%EOF\n")
    # 2) truncated real PDF (first 40% of a donor)
    if donor is not None:
        b = donor.read_bytes()
        (dst / "truncated.pdf").write_bytes(b[: max(1024, len(b) * 4 // 10)])
    return ["garbage_header.pdf"] + (["truncated.pdf"] if donor else [])


def synth_scanned_rotated(scanned_dir: Path, rotated_dir: Path, donor: Path | None):
    """Synthesize an image-only ('scanned') PDF and a rotated-orientation variant via PIL+pdfium.

    No real scanned/rotated PDFs exist in the repo, so we render a donor page to a raster and
    wrap it as an image-only page (no text layer = scanned-like). The rotated variant bakes a
    90-degree orientation into the pixels (landscape). NOTE: PIL-saved PDFs do not set a page
    /Rotate entry, so the /Rotate *parse* path is exercised by the benchmark via render(rotation=)
    rather than by this file.
    """
    if donor is None:
        return [], []
    from PIL import Image
    scanned_dir.mkdir(parents=True, exist_ok=True)
    rotated_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(donor.read_bytes())
    try:
        page = doc.get_page(0)
        try:
            arr0 = page.render(scale=150 / 72).to_numpy()      # upright raster
            arr90 = page.render(scale=150 / 72, rotation=90).to_numpy()
        finally:
            page.close()
    finally:
        doc.close()

    def to_rgb(a):
        im = Image.fromarray(a)
        return im.convert("RGB")

    s_name, r_name = "synthetic_scanned.pdf", "synthetic_rotated.pdf"
    to_rgb(arr0).save(scanned_dir / s_name, "PDF", resolution=150.0)
    to_rgb(arr90).save(rotated_dir / r_name, "PDF", resolution=150.0)
    return [s_name], [r_name]


def main():
    for b in BUCKETS:
        d = CORPUS / b
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():          # clear stale links/synthetic files for a clean rebuild
            try:
                f.unlink()
            except Exception:
                pass
    cands = dedupe(discover())
    print(f"discovered {len(cands)} unique PDFs", file=sys.stderr)

    manifest = {"buckets": {b: [] for b in BUCKETS}, "entries": []}
    donor_for_trunc = None
    for p in sorted(cands, key=lambda x: x.name):
        feat = probe(p)
        bucket = classify(feat)
        if bucket == "born_digital" and donor_for_trunc is None and not feat.get("error"):
            donor_for_trunc = p
        link = CORPUS / bucket / p.name
        if not link.exists():
            try:
                link.symlink_to(p)
            except FileExistsError:
                pass
        entry = {"name": p.name, "bucket": bucket, "src": str(p), **feat}
        manifest["entries"].append(entry)
        manifest["buckets"][bucket].append(p.name)

    synth = synth_malformed(CORPUS / "malformed", donor_for_trunc)
    for name in synth:
        manifest["buckets"]["malformed"].append(name)
        manifest["entries"].append({"name": name, "bucket": "malformed", "synthetic": True})

    try:
        s_names, r_names = synth_scanned_rotated(
            CORPUS / "scanned", CORPUS / "rotated", donor_for_trunc
        )
        for name in s_names:
            manifest["buckets"]["scanned"].append(name)
            manifest["entries"].append({"name": name, "bucket": "scanned", "synthetic": True})
        for name in r_names:
            manifest["buckets"]["rotated"].append(name)
            manifest["entries"].append({"name": name, "bucket": "rotated", "synthetic": True})
    except Exception as e:
        print(f"WARN: scanned/rotated synthesis failed: {e}", file=sys.stderr)

    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n=== corpus buckets ===")
    for b in BUCKETS:
        print(f"  {b:14s}: {len(manifest['buckets'][b])}  {manifest['buckets'][b][:6]}")
    print(f"\nmanifest -> {CORPUS / 'manifest.json'}")


if __name__ == "__main__":
    main()
