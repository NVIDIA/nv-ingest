# P3′ — Fused GPU Operator + DLPack Zero-Copy Handoff (prototype)

Status: **prototype built and proven.** Device rasters hand off between operators with zero host
copies, using the existing operator API. Validates the Mode-B fusion design from the integration
analysis.

## Question answered
"Can we fuse stages with the existing operator API, and does DLPack help?" — **Yes**, with the
precise boundaries from earlier:
- DLPack is the in-process framework seam (CUDA buffer → torch), **not** a cross-process / CPU↔GPU
  transfer. It works *inside* a fused operator; it cannot cross a Ray `map_batches` boundary.
- So the fused operator runs children **in one process**, and DLPack carries the device tensor
  between them.

## What was built
| Piece | Where |
|---|---|
| DLPack device handoff in CUDA module (`DeviceImage` w/ `__dlpack__`/`__dlpack_device__`, `upload_to_device`, `consume_mean`) | `native/src/raster_kernels.cu` → `_gpu_raster` |
| `FusedGPUOperator(operators=[...])` — runs each child's `preprocess/process/postprocess` in one process | `python/fused/operators.py` |
| `RasterizeGPUOperator` (PDFium render → `DeviceImage` on GPU) + `PageElementStubGPUOperator` (consumes device tensor zero-copy) | same |
| Demo / proof | `bench/fused_demo.py` |

`FusedGPUOperator`, `RasterizeGPUOperator`, `PageElementStubGPUOperator` are ordinary
`AbstractOperator`s — **same API as every existing operator** (`preprocess → process →
postprocess`, `run`, `__call__`). The fused operator just composes a list of them in-process and
lets a device-tensor column (`page_image_dev`, holding `DeviceImage` handles) flow between them.

## Proven results (8× H100, 8 single-page PDFs from the corpus)
- **Zero-copy across the operator boundary: the device pointer the rasterize operator allocated ==
  the pointer the next operator consumed, for every page.** No base64, no D2H; exactly **1 H2D**
  total (inside rasterize — unavoidable while PDFium renders on CPU).
- `DeviceImage.__dlpack__()` returns a `dltensor` capsule and `__dlpack_device__()` → `(2,0)`
  (kDLCUDA) — i.e. **`torch.from_dlpack(dev_img)` will wrap the same memory zero-copy** when the
  real model is plugged in.
- Timing (after CUDA warmup): fused device path **216 ms / 8 pages** vs the base64 round-trip
  **674 ms / 8 pages** that crosses the current extract→page_elements boundary — ~3×, and that
  baseline is conservative (it excludes Ray's cross-process serialization and the model-side
  re-upload the real pipeline also pays).

## How the real model plugs in (one line)
In `PageElementStubGPUOperator.process`, replace the stub kernel with:
```python
import torch
t = torch.from_dlpack(dev_img)          # zero-copy CUDA tensor, same buffer
detections = page_elements_model(t)     # in-process NemotronPageElementsV3, on-device
```
The in-process model already accepts torch CHW tensors (`_ensure_chw_float_tensor`), so a
`from_dlpack` tensor drops in. Requires the **local** model path (no `invoke_url`); a remote NIM
would force HTTP+base64 and break zero-copy (would need Triton CUDA shared memory instead).

## Honest caveats / next steps
- torch is **not installed** in this runtime, so the consumer is a native CUDA kernel
  (`consume_mean`) standing in for the model. The DLPack capsule + `__dlpack_device__` are present
  and standards-correct, so the torch path is wiring, not redesign — but it's **not yet executed
  end-to-end with a real model here.** Next: install a CUDA-matched torch, swap the stub for
  `NemotronPageElementsV3`, run the agent_eval recall gate.
- Per-page `cudaMalloc/cudaFree` dominates un-warmed runs (the P2 allocation-overhead artifact).
  Production should pool device buffers / use a caching allocator (torch's allocator if torch owns
  them) and batch pages per launch.
- This prototype fuses 2 operators; the full stage also needs **crop → table/chart/OCR** kept
  on-device in the same actor for end-to-end zero-copy.
- Crucially (from P2): even with PDFium CPU rasterization + this 1 H2D, the win is real because we
  eliminate base64 + the cross-stage host round-trip. The from-scratch CUDA rasterizer remains
  optional, only needed if render becomes the bottleneck after fusion.

## UPDATE — real model end-to-end (a) + full on-device chain (b)

Both validated with the **real `NemotronPageElementsV3`** (weights cached offline at
`/raid/nemo-retriever-harbor-adapters/.cache/huggingface`; runtime = the repaired `.venv`,
torch 2.11.0+cu130).

**(a) DLPack transport is recall-NEUTRAL — proven with the real model** (`bench/fused_model_e2e.py`).
Holding the rendered raster constant, ran the real model two ways — base64 transport vs
`torch.from_dlpack` zero-copy — and compared raw predictions:
- **worst max|Δ| across pages = 0.00e+00** → identical model outputs ⇒ identical detections ⇒ no
  retrieval-recall change. `torch.from_dlpack` pointer == producer pointer (zero-copy) on every page.
- DLPack path also faster per page (e.g. ~12 ms vs ~48 ms steady-state) — base64 encode/decode +
  host copies removed.
- This is the right-scoped recall gate: it isolates the *only* thing the fusion changes (transport).
  Full agent_eval retrieval needs the embedding+VDB+qrels services stack and isn't run here.

**(b) Full fused stage, crop→OCR kept on-device** (`bench/fused_full_demo.py`):
`FusedGPUOperator([RasterizeGPUOperator, PageElementGPUOperator(real), CropGPUOperator, OCRStubGPUOperator])`
- Real page-element detection runs on the zero-copy device tensor; detected regions (normalized
  boxes) are cropped **on-device** (torch slicing) and handed to the downstream op as CUDA tensors.
- Result: **61 regions across 6 pages, all crops on-device = True**, 1 H2D/page, 0 base64, 0 D2H
  between operators; full chain 207 ms / 6 pages.
**(b+) UPGRADE — real downstream model, weights now cached.** Cached the remaining model weights
(instantiating each online): `nvidia/nemotron-table-structure-v1`, `nemotron-graphic-elements-v1`,
`nemotron-ocr-v2` now sit alongside page-elements in the offline HF cache. Replaced the OCR stub
with the **real `NemotronTableStructureV1`** (`TableStructureGPUOperator`):
- Full chain `Rasterize → PageElement(real) → Crop → TableStructure(real)` runs end-to-end; on
  `embedded_table.pdf` it detected 18 regions incl. 4 tables → **4 real table-structure inferences**.
- `table-structure inputs on-device = True` — the table crops fed the model without leaving the GPU.
- Two real models chained in one process: 1 H2D/page, 0 base64, 0 D2H between operators; 415 ms / 6 pages.

This closes the loop: the fused operator runs **multiple real GPU models** over a single uploaded
device raster with zero-copy crop handoff between them — the Mode-B design, validated.

**(b++) ALL THREE downstream models wired** (`bench/fused_all_models_demo.py`):
`Rasterize → PageElement(real) → Crop → TableStructure(real) → GraphicElements(real) → OCR(real)`,
each routed by page-element label (`table`→table-structure, `chart`→graphic-elements, `text`→OCR).
- Over 6 pages: 4 real table-structure + 19 real OCR inferences; OCR returned real text (≈1–3.5k
  chars/page). `table_inputs_on_device=True`, `chart_inputs_on_device=True`. (charts=0 only because
  no `chart`-labeled regions in this sample; the path runs when present.)
- New ops: `GraphicElementsGPUOperator`, `OCRGPUOperator` (same `preprocess/process/postprocess` API).
- Honest nuances: **OCR-v2's wrapper re-encodes the crop to PNG internally** (its pipeline operates
  on image bytes), so OCR is zero-copy only *up to* the model, not through it — a property of that
  model, not the operator handoff (table + graphic stay on-GPU through inference). OCR dominates wall
  time (~3.2 s/6 pages) from that encode + detector+recognizer. OCR-v2 needs network on first
  construction (its `_download_checkpoints` isn't covered by the offline pin) — run once online to
  fully cache.

Net: the fused operator + DLPack design is validated with a real model — zero-copy across every
operator boundary, recall-neutral, faster. New operators live in `python/fused/operators.py`
(`PageElementGPUOperator`, `CropGPUOperator`, `OCRStubGPUOperator`).

## Ray GPU actor wiring (`bench/ray_fused_demo.py`)
The fused stage runs as a **real Ray Data GPU actor pool**, the same shape the codebase executor
uses for a `GPUOperator` node (`operator_class` + `fn_constructor_kwargs` + auto `num_gpus`):
- `FusedGPUOperator` now accepts a picklable `operator_specs` (list of `(OperatorClass, kwargs)`)
  and builds the children — which hold un-picklable torch models — **on the Ray worker**.
  `get_constructor_kwargs` is overridden to ship only `operator_specs` (never the built operators).
- `ds.map_batches(FusedGPUOperator, fn_constructor_kwargs={"operator_specs": [...]}, num_gpus=1,
  concurrency=2, batch_format="pandas")` → a pool of 2 GPU actors, each reserving 1 GPU, each
  constructing rasterize + real page-element + crop + real table + real graphic on the worker.
- **`HostFinalizeOperator`** (terminal, CPU) drops the device-only columns (`page_image_dev`,
  `region_crops_dev`) so the stage OUTPUT is host-serializable — device tensors live ONLY inside the
  actor and never cross the Ray boundary (the same constraint that motivated fusion).
- Verified: 8 pages through 2 GPU actors, 11.1 s (incl. per-actor model load), real inferences
  tables=5 / charts=2, results returned cleanly. Confirms zero-copy device handoff *within* each
  Ray actor + host-only data *between* Ray stages.

To slot into the real pipeline graph: wrap `FusedGPUOperator` in a `Node` with
`operator_class=FusedGPUOperator` and constructor kwargs `{"operator_specs": [...]}`; the executor
already auto-allocates `num_gpus` for `GPUOperator` and calls `run()` per batch.

## Reproduce
```bash
. /etc/profile.d/cuda.sh && cd gpu_pdf_extractor/native && cmake --build build      # native modules
# real-model runs use the repaired .venv (torch+cu130) and cached weights:
export HF_HOME=/raid/nemo-retriever-harbor-adapters/.cache/huggingface HF_HUB_OFFLINE=1
.venv/bin/python gpu_pdf_extractor/bench/fused_model_e2e.py     # (a) recall-neutral parity
.venv/bin/python gpu_pdf_extractor/bench/fused_full_demo.py     # (b) full on-device chain
.venv/bin/python gpu_pdf_extractor/bench/fused_all_models_demo.py  # table+graphic+OCR real models
.venv/bin/python gpu_pdf_extractor/bench/ray_fused_demo.py     # real Ray Data GPU actor pool
python3 gpu_pdf_extractor/bench/fused_demo.py                   # torch-free mechanism demo
```

