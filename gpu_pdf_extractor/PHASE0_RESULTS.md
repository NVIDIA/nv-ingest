# Phase 0 — Results (baseline, harness, toolchain)

Status: **COMPLETE.** Gate to enter P1 (baseline numbers + reproducible bench + building empty
extension + parity metric) is satisfied.

## Deliverables
| # | Item | Where |
|---|------|-------|
| P0.1 | Bucketed golden corpus (+ manifest) | `corpus/` , `corpus/manifest.json` |
| P0.2 | Per-area baseline benchmark | `bench/baseline_areas.py` → `results/baseline_areas.json` |
| P0.3 | End-to-end throughput baseline (real operators) | `bench/baseline_e2e.py` → `results/baseline_e2e.json` |
| P0.4 | Raster parity metric + quality-gate doc | `bench/parity.py`, `bench/QUALITY_GATE.md` |
| P0.5 | Build skeleton (nanobind+CUDA+PDFium link proof) | `native/` |

## Corpus (39 unique PDFs, deduped repo-wide; + synthetic)
born_digital 28 · dense_vector 9 · multi_image 2 · scanned 1\* · rotated 1\* · malformed 2\*
(\* synthetic: image-only "scanned", baked-orientation "rotated", garbage+truncated "malformed").

## Baseline numbers (this box; pypdfium2 4.30.0 / PDFium build 6462; single core unless noted)

### Per-area, ms/page @ 300 DPI (P0.2)
| bucket | render | text | objects | imgdec | render pg/s |
|--------|-------:|-----:|--------:|-------:|------------:|
| born_digital | 28.5 | 0.81 | 5.62 | 4.21 | 35.1 |
| dense_vector | 35.4 | 1.27 | **11.57** | 4.99 | 28.3 |
| multi_image | 34.2 | 0.42 | 0.68 | 2.11 | 29.3 |
| scanned (synth) | 87.0 | 0.12 | 0.04 | 19.3 | 11.5 |
| rotated (synth) | 76.7 | 0.12 | 0.04 | 16.4 | 13.0 |

**Render dominates** (confirms the earlier profile); object-enum spikes to ~12 ms on dense vector.

### End-to-end through real `split_pdf_batch → pdf_extraction` (P0.3, CPU, no NIM)
- **Serial: 13.0 pages/sec, 2.06 docs/sec.** extract/render = 81–99% of wall time.
- **8 processes: 56.7 pages/sec wallclock** (~4.4× scaling; sublinear from big-doc skew).
- Finding: pypdfium2 is **not reliably thread-safe** for concurrent split+render (a thread pool
  collapsed output to ~1 page/doc); scaling must be process-based (Ray), as production does.

## THE GATE (locked metric = end-to-end pipeline throughput)
The GPU backend must beat **56.7 pages/sec** (8-process CPU wallclock on this box) end-to-end,
target **≥3×**, AND show no recall regression per `bench/QUALITY_GATE.md`. The many-GPU batched
rasterizer + zero-copy hand-off (P2/P3) is how we expect to clear it.

## Toolchain (verified)
8× NVIDIA **H100 NVL** (sm_90), CUDA 13.0, cmake 3.22.1, ninja, nanobind 2.13.0, PDFium 151.0.7906.0.
`native/` builds a module where `cuda_add(2,40)→42`, device-info reports the 8 GPUs, and PDFium
initializes — i.e. all three pillars link and load together.

## Reproduce
```bash
python3 bench/build_corpus.py
python3 bench/baseline_areas.py --dpi 300 --max-pages-per-doc 6
python3 bench/baseline_e2e.py  --pages-per-doc 8 --docs-per-bucket 6 --workers 8
( . /etc/profile.d/cuda.sh; cd native && cmake -S . -B build -G Ninja \
   -DPython_EXECUTABLE=$(which python3) -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build build )
```

## Known caveats / P1 inputs
- PDFium version mismatch (link 7906 vs baseline 6462) — reconcile in P1 for true parity.
- scanned/rotated buckets are synthetic (no real samples in repo) — augment if real ones arrive.
- `/Rotate` parse path not covered by a file; exercised via render `rotation=` param.
- E2E baseline excludes NIM/YOLOX inference (downstream of our component) by design.
