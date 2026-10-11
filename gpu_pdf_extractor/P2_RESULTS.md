# Phase 2 — Results & Go/No-Go (CUDA rasterizer core)

Status: **core primitive built, validated, benchmarked.** Outcome reframes the program (below).

## What was built (`native/src/raster_kernels.cu` → module `_gpu_raster`)
A from-scratch **CUDA batched anti-aliased polygon-fill rasterizer**: supersampled coverage,
painter's-order src-over compositing over white, per-polygon bounding-box early-out, one thread
per output pixel, batched across pages and GPUs. Plus instrumented timing (`time_render`, H2D /
kernel / D2H split, optional D2H to model the P3 zero-copy hand-off). Benchmark + numpy reference:
`bench/raster_bench.py`.

## Measured (8× H100 NVL, page 2550×3300, 200 polys/page, SS=4)
| measurement | result | vs gate (170 pg/s) |
|---|---|---|
| Correctness (GPU vs numpy reference) | **exact match, SSIM 1.0** | — |
| Kernel only, 1 GPU | **712 pages/sec** | **4.2×** |
| H2D (geometry upload) | ~0.5–2 ms (negligible) | — |
| **D2H (raster copy-back)** | **~5560 ms / 512 pg ≈ 90% of wall** | the bottleneck |
| With D2H, 1 GPU (naive) | 67 pg/s | 0.4× |
| Zero-copy ceiling, 1 GPU (H2D+kernel) | **712 pg/s** | **4.2×** |
| Zero-copy, 8 GPUs (threads, warm-up not amortized) | 385 pg/s | 2.3× |

## The decisive finding
**GPU rasterization compute is NOT the bottleneck — data transfer is.** A single H100 fills pages
~12× faster than the 8-process CPU baseline (56.7 pg/s) and 4× the 3× gate. But copying full-res
rasters back to host (≈25 MB/page) erases it: D2H is ~90% of wall time, and naively it's *slower*
than CPU. This is exactly the regime the plan predicted and the reason **P3 (zero-copy device
hand-off to the YOLOX/OCR model) is the actual lever**, not more rasterization compute.

(8-GPU scaling here is dominated by lazy CUDA context creation + multi-GB `cudaMalloc` inside the
timed region — an engineering artifact, not a compute ceiling. Warm contexts + persistent buffers
would recover near-linear scaling. Not pursued because the single-GPU number already settles the
question.)

## Honest gap to a real PDFium-parity renderer
The benchmark proves the **fill primitive** on synthetic polygons. A drop-in `render()` replacement
must also reproduce, to SSIM ≥0.98 vs PDFium on **real** pages:
- **Text/glyph rasterization** — the dominant content on born_digital (text-heavy) pages; requires
  font loading, hinting, glyph outline fill, subpixel positioning. This is its own large subsystem
  and the biggest unknown for parity. **Not addressed by P2.**
- Clipping paths, stroke geometry, blend modes, shadings/gradients, image resampling, CMYK/ICC
  color management.
- Bit-/perceptual-parity with PDFium's AGG anti-aliasing.

Matching all of this is a multi-quarter effort with real parity risk, especially for text.

## Recommendation (go/no-go)
**Do NOT pursue a full from-scratch CUDA renderer next.** Rationale:
1. **P1 already ships value today** — a verified, byte-identical, zero-regression drop-in (no ctypes,
   explicit threading control). That is the safe default to deploy.
2. P2 shows the win is **transfer/pipeline, not compute** — so the high-ROI GPU work is **P3**:
   keep PDFium rasterization (or the CUDA fill where eligible) but eliminate the host round-trip —
   render into device buffers consumed directly by the downstream GPU model, with warm contexts,
   persistent allocations, and streams. The 712 pg/s ceiling says this can clear the gate.
3. Treat **full CUDA text rasterization as a separate, gated research spike** — first prove glyph
   rasterization can hit SSIM ≥0.98 vs PDFium on a text page sample *before* committing to it.

### Proposed next step (instead of "P2 full renderer")
**P3': zero-copy raster→model pipeline.** Wire `render()` to deliver device-resident rasters
(PDFium-rasterized, uploaded once; or CUDA-filled for vector-only pages) straight into the YOLOX/OCR
preprocessing, then run the agent_eval recall gate end-to-end. Decision to invest in a from-scratch
CUDA text renderer is deferred behind a glyph-parity spike.

## Reproduce
```bash
. /etc/profile.d/cuda.sh && cd gpu_pdf_extractor/native && cmake --build build
python3 gpu_pdf_extractor/bench/raster_bench.py --pages 512 --polys 200 --ss 4
```
