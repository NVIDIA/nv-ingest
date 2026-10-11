# GPU PDF Extraction — Findings Report

**Goal:** evaluate whether a C++/CUDA PDF extraction path (and a fused, zero-copy GPU operator)
can beat the current `pypdfium2`-based extraction in `nemo_retriever`, and quantify the real
end-to-end benefit in the production `retriever ingest` pipeline.

**Bottom line:** The component-level wins are real and large (bit-identical native drop-in;
712 pages/s GPU rasterization; 2.75× transport speedup via DLPack). **But they do not translate
into an end-to-end ingest speedup — and the fused operator *regresses* the default Ray pipeline
(0.44–0.60×).** The production pipeline is **inference-bound** and Ray already parallelizes stages
across GPUs by pipelining; fusing them into one actor sacrifices that. Recommendation: keep the
staged pipeline; do not ship the fused operator for the multi-GPU batch path.

All artifacts live under `gpu_pdf_extractor/` (benchmarks in `bench/`, results in `results/`,
native code in `native/`, the src integration in `nemo_retriever/src/.../operators/extract/fused/`).

---

## 1. Baseline & profiling (P0)
*Refs: `bench/profile_areas.py`, `bench/baseline_areas.py`, `bench/baseline_e2e.py`, `results/baseline_*.json`, `PHASE0_RESULTS.md`*

- Profiled the 6 pypdfium capability areas on a 39-PDF bucketed corpus. **Rendering dominates:
  85–94 %** of per-page time on typical pages (28–87 ms/page @300 DPI); parsing is ~free
  (~0.01 ms/page, lazy); object-enumeration spikes to ~50 % only on dense-vector pages.
- End-to-end CPU extraction throughput (real operators, no NIM): **13 pages/s serial, 56.7 pages/s
  on 8 processes**.
- Finding: pypdfium2 is **not thread-safe** for concurrent split+render → scaling must be
  process-based (which is what Ray does).

## 2. Native PDFium drop-in (P1)
*Refs: `native/src/pdf_bindings.cpp` (`_gpu_pdfium`), `python/gpu_pdfium/`, `bench/parity_backend.py`, `results/parity_backend.json`, `P1_RESULTS.md`*

- Built a nanobind C++ module over linked PDFium exposing the full pypdfium2 surface, plus a
  `gpu_pdfium` package that drops in via `sys.modules` injection (no source edits).
- **Parity across the corpus: render SSIM = 1.0 (bit-identical), text 108/108 pages, object counts
  108/108, metadata 41/41.** Malformed PDFs rejected by both.
- Value: removes per-call ctypes overhead + gives explicit thread/init control. Independently
  shippable; recall-neutral by construction.

## 3. From-scratch CUDA rasterizer (P2)
*Refs: `native/src/raster_kernels.cu` (`_gpu_raster`), `bench/raster_bench.py`, `P2_RESULTS.md`*

- Built a batched anti-aliased polygon-fill CUDA kernel (correctness-validated vs a numpy reference,
  exact match).
- **Kernel throughput: 712 pages/s on one H100 (4.2× the 170 pg/s "3× gate", 12× the CPU baseline).**
- **Decisive finding: compute is NOT the bottleneck — transfer is.** Copying full-res rasters back
  to host (D2H) is **~90 % of wall time**; naive single-GPU throughput collapses to 67 pages/s.
- Go/No-Go: a full from-scratch CUDA renderer (glyphs/clipping/blending/colour) is multi-quarter,
  high-risk, and unwarranted given the transfer-bound result. **Not pursued.**

## 4. Fused operator + DLPack zero-copy (P3′)
*Refs: `bench/compare_methods.py`, `results/method_comparison.png`, `bench/fused_*_demo.py`, `python/fused/`, `P3prime_RESULTS.md`*

- Built a `DeviceImage` DLPack handle so a CUDA raster hands off **zero-copy** to torch
  (`torch.from_dlpack`, pointer-identical) and to the real `NemotronPageElementsV3` /
  `TableStructureV1` / `GraphicElementsV1` / `OCRV2` models, chained on-device in one process.
- **Transport-layer micro-benchmark (same real model, same pages, base64 vs DLPack):**
  - transport **74.8 ms → 0.6 ms (125×)**; end-to-end per-page **122 ms → 44 ms = 2.75×
    (8.2 → 22.6 pages/s)**; model predictions **bit-identical (max|Δ| = 0)** → recall-neutral.
  - See chart: `results/method_comparison.png`.
- **Caveat that matters:** this 2.75× avoids *both* base64 encode and decode. The real ingest
  pipeline sets `extract_page_as_image=True`, so the page's base64 is a **required output artifact**
  — the encode cannot be removed, shrinking the realizable saving; and the model forward dominates.

## 5. Production integration + the decisive ingest benchmark
*Refs: `nemo_retriever/src/.../operators/extract/fused/fused_extract.py`, `graph/ingestor_runtime.py` (`NEMO_FUSED_EXTRACT` toggle), `tmp/ingest_bench/*.log`*

- Integrated `FusedExtractActor` into `nemo_retriever/src`: it composes the real
  `PDFExtraction → PageElement → OCR` operators in one actor (recall-identical by construction),
  toggled by `NEMO_FUSED_EXTRACT=1` in `build_graph`. Verified active on the default pdf path.
- Ran the **default `retriever ingest` pipeline** (local models + vLLM embed + LanceDB), baseline vs
  fused, on **jp20** (20 PDFs) and a **bo767 subset** (50 PDFs), in both run modes:

| dataset | mode | baseline | fused | fused speedup | rows (base/fused) |
|---|---|---:|---:|---:|---|
| jp20 | inprocess | 262 s | 225 s | 1.16× | 3147 / 3147 |
| jp20 | **batch (Ray)** | **197 s** | 329 s | **0.60×** | 3147 / 3148 |
| bo50 | inprocess | 342 s | 343 s | 1.00× | 3330 / 3329 |
| bo50 | **batch (Ray)** | **208 s** | 470 s | **0.44×** | 3329 / 3330 |

**Findings:**
1. **Batch (Ray) baseline beats inprocess** (jp20 197 vs 262 s; bo50 208 vs 342 s) — Ray Data
   pipelines the stages across the 8 GPUs and overlaps them.
2. **Inprocess fusion is neutral** (1.16× / 1.00×) — inprocess already runs stages in one process,
   so there is little serialization to remove.
3. **Batch fusion *regresses* throughput (0.44–0.60×), worse at scale** — collapsing the stages into
   one serial GPU actor destroys Ray's cross-stage, cross-GPU pipelining. The transport bytes saved
   are negligible next to model inference, which Ray already overlaps.
4. **Recall-neutral:** row counts differ by ±1 (0.03 %) in both directions — non-deterministic
   batch-boundary effects, not a fusion defect.

## 6. Why the 2.75× does not translate (the core lesson)
- The 2.75× is a **transport micro-benchmark**; the real pipeline is **inference-bound** (YOLOX/OCR +
  vLLM embed), so transport savings are a small fraction of wall time.
- **Zero-copy requires one process (fusion); pipelining requires separate actors.** These are
  mutually exclusive across a Ray boundary, and for this inference-bound workload **pipelining wins**.
- The page-image base64 is a required pipeline output, so the encode cannot be eliminated anyway.

## 7. Recommendations
1. **Keep the staged pipeline** for the default multi-GPU batch path — it is the fastest.
2. **Do not ship the fused operator** for batch/Ray. Its only plausible niche is single-GPU /
   inprocess / memory-constrained or genuinely transport-bound deployments.
3. **Do not pursue Phase I2** (DLPack inside the fused actor) for the batch pipeline: it still fuses
   to one actor (loses pipelining) and only attacks transport (not the bottleneck).
4. **Worth keeping:** the **P1 native PDFium drop-in** (recall-neutral, removes ctypes overhead) is a
   low-risk standalone win independent of fusion.
5. If zero-copy is ever revisited, the right architecture is **pipelined actors + device-handle
   transport (CUDA IPC / Triton shared memory)**, not single-actor fusion — and only if profiling
   shows transport is actually limiting (it is not, today).

## Appendix — reproduce
```bash
. /etc/profile.d/cuda.sh && (cd gpu_pdf_extractor/native && cmake --build build)   # native modules
export HF_HOME=/raid/nemo-retriever-harbor-adapters/.cache/huggingface
python3   gpu_pdf_extractor/bench/baseline_e2e.py            # P0 CPU baseline
.venv/bin/python gpu_pdf_extractor/bench/parity_backend.py   # P1 parity (needs torch venv)
.venv/bin/python gpu_pdf_extractor/bench/raster_bench.py     # P2 CUDA rasterizer
.venv/bin/python gpu_pdf_extractor/bench/compare_methods.py  # P3' 2.75x transport chart
# Ingest benchmark (default pipeline), baseline vs fused, both modes:
NEMO_FUSED_EXTRACT=0 .venv/bin/retriever ingest /raid/data/jp20 --run-mode inprocess --table-name jp20_base
NEMO_FUSED_EXTRACT=1 .venv/bin/retriever ingest /raid/data/jp20 --run-mode batch     --table-name jp20_bfused
```
