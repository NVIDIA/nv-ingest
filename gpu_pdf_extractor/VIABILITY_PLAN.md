# GPU PDF Extractor — Viability Plan

Goal (user's words): a C++/CUDA PDF read/extraction/rasterizer that is **faster, highly
parallelizable, and a near drop-in replacement** for the current `pypdfium2` integration,
with Python bindings, developed outside `nemo_retriever/src/` (this folder).

This document is for **review and green-light before any execution.**

---

## 1. The central design decision: full rewrite vs. hybrid

A from-scratch C++/CUDA engine covering the **entire** contract (parse + text + objects +
metadata + split/write + render) is **not viable** as a first deliverable:
- PDF *parsing* is sequential/branchy and gains ~nothing from CUDA (see ANALYSIS §1, §4).
- Reproducing PDFium's malformed-PDF robustness and security hardening is multi-engineer-year work.
- Only the **render path** (~1 of 6 capability areas) is GPU-suited — but it is the hot path.

**Recommended architecture — Hybrid:**

```
        ┌─────────────────────────────────────────────┐
        │  gpu_pdfium  (Python shim, drop-in API)       │  ← matches pypdfium2 contract
        └───────────────┬───────────────────────────────┘
                        │ pybind11 / nanobind
        ┌───────────────▼───────────────────────────────┐
        │  libgpu_pdf  (C++ core)                         │
        │  ┌──────────────────┐   ┌────────────────────┐ │
        │  │ Parse / structure│   │  CUDA rasterizer    │ │
        │  │ text / objects / │   │  (our from-scratch  │ │
        │  │ metadata / split │   │  C++/CUDA code)     │ │
        │  │  = PDFium or     │──▶│  display-list →     │ │
        │  │    MuPDF (C++)   │   │  batched GPU raster │ │
        │  └──────────────────┘   └────────────────────┘ │
        └─────────────────────────────────────────────────┘
```

- Keep a proven C++ parser (**PDFium** as a linked C++ lib, or **MuPDF**) for everything that
  isn't rendering. We compile/link it ourselves in C++ (no ctypes), which already removes the
  current concurrency wart and per-call ctypes overhead.
- Build **our own C++/CUDA rasterizer** that consumes the parser's display list and renders
  **batches of pages** on the GPU. This is where the "from scratch C++/CUDA" effort concentrates
  and where the speed/parallelism win actually lives.
- Expose a `gpu_pdfium` Python module whose classes/methods mirror the contract exactly, so
  integration is flipping a feature flag at the ~8 import sites.

This concentrates risk on the one part that can win, de-risks the 95% that can't, and still
delivers a genuine native C++/CUDA artifact with Python bindings.

> If a 100%-from-scratch parser is a hard requirement, that is a separate, much larger program
> (own track, own staffing) — I'll flag it as a decision rather than assume it.

---

## 2. Where the win comes from (and how we'll prove it)

- Current: per-page `page.render()` on CPU (AGG rasterizer), single-threaded per page, ctypes
  hop per call, BGRA→RGB on CPU via OpenCV.
- Proposed: parse N pages → upload display lists → **rasterize N pages in parallel on GPU** →
  channel-swap + scale + pad on GPU → return pinned/`numpy` (or device handle for zero-copy
  into YOLOX/OCR preprocessing).
- Speedup is expected on **batched, high-DPI, many-page** corpora; it can be *negative* for
  1–2 tiny pages due to H2D/D2H transfer. We will measure both regimes.

**Quantitative gate (must beat to proceed past each phase):**
- End-to-end render throughput (pages/sec, incl. transfers) **≥ 3×** batched multi-thread CPU
  PDFium on a representative corpus, OR
- Demonstrable pipeline win when fused zero-copy with downstream GPU preprocessing.
- **Quality:** downstream retrieval/extraction recall on the `agent_eval` corpus within an
  agreed tolerance of the pypdfium baseline (target: no statistically significant regression).
- If neither holds → stop at Phase 1 (still ships a faster native shim) or abandon.

---

## 3. Phased plan (each phase is independently shippable / killable)

### Phase 0 — Baseline, corpus, harness, toolchain *(prerequisite; no product code)*
- Provision a GPU build/bench box (CUDA toolkit, CMake ≥ 3.26, scikit-build-core). None exists
  in the current sandbox.
- Assemble a **golden corpus** (born-digital, scanned, rotated, multi-image, malformed PDFs).
- Build a benchmark harness measuring pypdfium today: pages/sec, p50/p99 latency, peak RAM,
  raster output hashes — across render-only, text, object-extraction, split.
- Define the **parity metric** (per-pixel SSIM/PSNR of rasters + downstream recall delta via
  the existing `agent_eval` harness).
- **Exit:** documented baseline numbers + reproducible bench + parity scoring.

### Phase 1 — Native C++ shim, drop-in, **CPU-only** *(de-risks the contract)*
- pybind11/nanobind bindings wrapping a linked C++ PDF parser implementing the full contract
  (§2 of ANALYSIS): `PdfDocument/PdfPage/PdfBitmap/PdfTextPage/PdfImage/PageObject`, constants,
  `PdfiumError`, numpy via buffer protocol.
- Build system: scikit-build-core + CMake, Python-3.12 CUDA-capable wheel.
- Wire a `NEMO_PDF_BACKEND={pypdfium,gpu}` flag; route the ~8 import sites through one adapter.
- **Exit:** full test corpus passes through `gpu_pdfium` with byte/recall parity to pypdfium;
  no CUDA yet. *Already a useful artifact* (no ctypes, fixes concurrency wart).

### Phase 2 — CUDA rasterizer (paths/fills/compositing), batched *(the core bet)*
- From-scratch C++/CUDA: display-list → GPU tessellation/scanline fill, anti-aliasing, alpha
  compositing; batch many pages per launch; on-GPU channel swap + scale + pad (replacing the
  CPU OpenCV steps).
- Plugs behind `page.render()` / `pdfium_pages_to_numpy`; CPU rasterizer remains fallback for
  unsupported constructs.
- **Exit:** meet the §2 quantitative + quality gate on the render path.

### Phase 3 — Glyph/text & image rasterization on GPU + zero-copy hand-off
- GPU glyph rasterization (text-heavy pages), GPU image resampling for embedded images.
- Return device buffers consumable directly by YOLOX/OCR preprocessing (eliminate D2H+H2D).
- **Exit:** end-to-end pipeline speedup (not just the render microbenchmark).

### Phase 4 — Integration, rollout, decision
- Default-off feature flag → canary on a pipeline → benchmark vs baseline → go/no-go.
- Packaging in the deployment CUDA image; fallback to pypdfium on any error.

---

## 4. Tech stack (proposed)
- **C++17/20**, **CUDA** (target the deployment image's CUDA arch); **CMake + scikit-build-core**.
- Bindings: **nanobind** (lighter/faster than pybind11; either works) returning zero-copy numpy.
- Parser core: **PDFium** linked as C++ (closest behavioral parity to today) — MuPDF is the
  alternative (cleaner API, AGPL/commercial licensing caveat to clear with legal).
- Wheel: scikit-build-core, manylinux+CUDA, py3.12 only.

## 5. Top risks & mitigations
| Risk | Mitigation |
|------|-----------|
| Rasterizer parity hurts downstream recall | Phase 0 parity gate via `agent_eval`; CPU fallback per-page on divergence |
| Transfer overhead kills small-batch wins | Batch-first API; zero-copy device output (Phase 3); keep CPU path for tiny inputs |
| Scope explosion (PDF spec edge cases) | Hybrid: reuse proven parser; only rasterize ourselves |
| No GPU/toolchain in dev env | Phase 0 provisions a GPU CI runner before any CUDA work |
| PDFium/MuPDF licensing & linking | Resolve in Phase 0 (PDFium = BSD-ish; MuPDF = AGPL/commercial) |
| Maintenance burden of a bespoke engine | Decision gate after Phase 2; ship Phase 1 even if CUDA is abandoned |

## 6. What I'd build first if green-lit
Phase 0 harness + baseline numbers, then the Phase 1 CPU drop-in shim — because it (a) proves
the contract end-to-end, (b) is independently valuable, and (c) is the foundation the CUDA
rasterizer slots into without touching `nemo_retriever/src` again.

## 7. Locked decisions (2026-06-22)
1. **Architecture: HYBRID** — linked C++ PDFium for the 5 non-render areas + our own from-scratch
   C++/CUDA batched rasterizer for the render hot path.
2. **v1 scope: FULL 6-area contract** — `gpu_pdfium` is a complete drop-in (render + text + objects +
   metadata + split/import/save), not render-only. CUDA acceleration lands behind `render()` in P2;
   P1 wires the entire contract on CPU first.
3. **Parser core: PDFium** — linked as a C++ library against the public `FPDF_*` headers (BSD-style,
   closest behavioral parity to today's backend, lowest parity risk).
4. **Gate metric: END-TO-END PIPELINE THROUGHPUT** — docs/sec through the real extraction pipeline
   (not an isolated render microbench), which rewards the P3 zero-copy device hand-off. Quality
   co-gate: no recall regression on the agent_eval corpus.

## 8. Confirmed build environment (2026-06-22)
- 8× NVIDIA sm_90 (Hopper) GPUs, driver 580.159.03; CUDA 13.0 toolkit (`/usr/local/cuda-13.0`,
  nvcc V13.0.88); cmake 3.22.1, ninja 1.10.1, gcc 14.2. Toolchain verified end-to-end (compiled+ran
  an sm_90 kernel). 8 GPUs strongly favor the batched multi-page rasterizer design.
- Target rasterizer arch: `sm_90`. (cmake 3.22 OK to start; upgrade to ≥3.27 via pip/Kitware if
  scikit-build-core + CUDA-13 needs it.)
