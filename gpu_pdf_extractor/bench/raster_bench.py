#!/usr/bin/env python3
"""P2.1/P2.2 — Correctness + throughput benchmark for the CUDA AA polygon-fill rasterizer.

(1) Correctness: GPU output vs a numpy reference of the SAME supersampled even-odd fill, on a
    small scene (validates the CUDA kernel matches its spec — SSIM≈1, near-exact).
(2) Throughput: full-page-size rasterization across N pages, single GPU and all GPUs, measured
    against the P0 end-to-end gate (56.7 pages/sec, 8-process CPU pypdfium).

Run:  python3 gpu_pdf_extractor/bench/raster_bench.py [--pages 256] [--polys 200] [--ss 4]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gpu_pdf_extractor" / "native" / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gpu_raster as R
from parity import compare


def gen_scene(n_pages, W, H, polys_per_page, seed=0):
    """Random convex-ish quads/triangles per page → flat arrays for the kernel."""
    rng = np.random.default_rng(seed)
    edges, edge_start, edge_count, color, poly_page = [], [], [], [], []
    page_poly_start, page_poly_count = [], []
    pi = 0
    for pg in range(n_pages):
        page_poly_start.append(pi)
        for _ in range(polys_per_page):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            r = rng.uniform(20, 120)
            k = int(rng.integers(3, 6))  # 3..5 vertices
            angs = np.sort(rng.uniform(0, 2 * np.pi, k))
            vx = np.clip(cx + r * np.cos(angs), 0, W)
            vy = np.clip(cy + r * np.sin(angs), 0, H)
            edge_start.append(len(edges) // 4)
            for i in range(k):
                j = (i + 1) % k
                edges += [float(vx[i]), float(vy[i]), float(vx[j]), float(vy[j])]
            edge_count.append(k)
            color.append(int(rng.integers(0, 0xFFFFFF)))
            poly_page.append(pg)
            pi += 1
        page_poly_count.append(pi - page_poly_start[pg])
    return dict(W=W, H=H, n_pages=n_pages,
                edges=edges, edge_start=edge_start, edge_count=edge_count, color=color,
                page_poly_start=page_poly_start, page_poly_count=page_poly_count, poly_page=poly_page)


def ref_render(scene, SS):
    """Numpy reference: same supersampled even-odd painter-order fill (single page 0)."""
    W, H = scene["W"], scene["H"]
    img = np.full((H, W, 3), 255.0)
    ps, pc = scene["page_poly_start"][0], scene["page_poly_count"][0]
    ys, xs = np.mgrid[0:H, 0:W]
    for k in range(pc):
        pi = ps + k
        es, ec = scene["edge_start"][pi], scene["edge_count"][pi]
        cov = np.zeros((H, W))
        for sy in range(SS):
            for sx in range(SS):
                fx = xs + (sx + 0.5) / SS
                fy = ys + (sy + 0.5) / SS
                inside = np.zeros((H, W), bool)
                for i in range(ec):
                    ax, ay, bx, by = scene["edges"][(es + i) * 4:(es + i) * 4 + 4]
                    cond = (ay > fy) != (by > fy)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        xint = ax + (fy - ay) / (by - ay) * (bx - ax)
                    inside ^= cond & (fx < xint)
                cov += inside
        cov /= SS * SS
        c = scene["color"][pi]
        cb, cg, cr = c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF
        for ch, cc in enumerate((cb, cg, cr)):
            img[:, :, ch] = cc * cov + img[:, :, ch] * (1 - cov)
    return (img + 0.5).astype(np.uint8)


def render_gpu(scene, SS, dev=0):
    arr, kms = R.render(scene["W"], scene["H"], scene["n_pages"], SS, dev,
                        scene["edges"], scene["edge_start"], scene["edge_count"],
                        scene["color"], scene["page_poly_start"], scene["page_poly_count"],
                        scene["poly_page"])
    return np.asarray(arr), kms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=256)
    ap.add_argument("--polys", type=int, default=200)
    ap.add_argument("--ss", type=int, default=4)
    ap.add_argument("--w", type=int, default=2550)
    ap.add_argument("--h", type=int, default=3300)
    args = ap.parse_args()
    ndev = R.device_count()
    print(f"GPUs: {ndev}")

    # (1) correctness on a small scene
    small = gen_scene(1, 96, 96, 4, seed=1)
    g, _ = render_gpu(small, args.ss)
    ref = ref_render(small, args.ss)
    print("CORRECTNESS (GPU vs numpy ref, 96x96):", compare(ref, g[0]))

    # (2) throughput at full page size
    print(f"\nTHROUGHPUT  page={args.w}x{args.h}  polys/page={args.polys}  SS={args.ss}")
    scene = gen_scene(args.pages, args.w, args.h, args.polys)

    # single GPU: include H2D+kernel+D2H (wallclock) and kernel-only
    t = time.perf_counter(); _, kms = render_gpu(scene, args.ss, 0); wall = time.perf_counter() - t
    print(f"  1 GPU : {args.pages} pages in {wall*1000:.1f} ms wall  -> {args.pages/wall:,.0f} pg/s "
          f"(kernel {kms:.1f} ms -> {args.pages/(kms/1000):,.0f} pg/s kernel-only)")

    # breakdown: H2D / kernel / D2H, and the zero-copy ceiling (no D2H = P3 hand-off model)
    tb = R.time_render(scene["W"], scene["H"], scene["n_pages"], args.ss, 0, True,
                       scene["edges"], scene["edge_start"], scene["edge_count"], scene["color"],
                       scene["page_poly_start"], scene["page_poly_count"], scene["poly_page"])
    tz = R.time_render(scene["W"], scene["H"], scene["n_pages"], args.ss, 0, False,
                       scene["edges"], scene["edge_start"], scene["edge_count"], scene["color"],
                       scene["page_poly_start"], scene["page_poly_count"], scene["poly_page"])
    n = args.pages
    print(f"  breakdown (1 GPU): H2D={tb['h2d_ms']:.1f}ms kernel={tb['kernel_ms']:.1f}ms "
          f"D2H={tb['d2h_ms']:.1f}ms")
    print(f"  ZERO-COPY ceiling (H2D+kernel, no D2H): {(tz['h2d_ms']+tz['kernel_ms']):.1f} ms "
          f"-> {n/((tz['h2d_ms']+tz['kernel_ms'])/1000):,.0f} pg/s  [P3 hand-off model]")

    # all GPUs zero-copy ceiling: split pages across devices, kernel+H2D only, concurrent
    if ndev > 1:
        from concurrent.futures import ThreadPoolExecutor
        per = (n + ndev - 1) // ndev
        zslices = []
        for dv in range(ndev):
            cnt = min(per, n - dv * per)
            if cnt <= 0: break
            zslices.append((dv, gen_scene(cnt, args.w, args.h, args.polys, seed=200 + dv)))
        def zrun(item):
            dv, sc = item
            t = R.time_render(sc["W"], sc["H"], sc["n_pages"], args.ss, dv, False,
                              sc["edges"], sc["edge_start"], sc["edge_count"], sc["color"],
                              sc["page_poly_start"], sc["page_poly_count"], sc["poly_page"])
            return sc["n_pages"]
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(zslices)) as ex:
            cs = list(ex.map(zrun, zslices))
        wz = time.perf_counter() - t
        print(f"  {len(zslices)} GPUs zero-copy: {sum(cs)} pages in {wz*1000:.1f} ms -> {sum(cs)/wz:,.0f} pg/s")

    # all GPUs: split pages across devices, run, measure wallclock (each builds its own scene slice)
    if ndev > 1:
        from concurrent.futures import ThreadPoolExecutor
        per = (args.pages + ndev - 1) // ndev
        slices = []
        for d in range(ndev):
            n = min(per, args.pages - d * per)
            if n <= 0: break
            slices.append((d, gen_scene(n, args.w, args.h, args.polys, seed=100 + d)))
        def run(item):
            d, sc = item; _, kms = render_gpu(sc, args.ss, d); return sc["n_pages"]
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(slices)) as ex:
            counts = list(ex.map(run, slices))
        wall = time.perf_counter() - t
        tot = sum(counts)
        print(f"  {len(slices)} GPUs: {tot} pages in {wall*1000:.1f} ms wall -> {tot/wall:,.0f} pg/s")

    print(f"\nGATE: P0 end-to-end CPU baseline = 56.7 pg/s (8 proc). Target >=3x = 170 pg/s.")


if __name__ == "__main__":
    main()
