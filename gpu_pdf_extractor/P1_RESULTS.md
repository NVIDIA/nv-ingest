# Phase 1 — Results (native CPU drop-in over PDFium)

Status: **COMPLETE & VERIFIED.** Full 6-area `gpu_pdfium` drop-in is byte-for-byte equivalent to
pypdfium2 across the corpus, wired behind `NEMO_PDF_BACKEND` with zero source edits to nemo_retriever.

## Deliverables
| # | Item | Where |
|---|------|-------|
| P1.1 | Native binding classes (read path) | `native/src/pdf_bindings.cpp` → module `_gpu_pdfium` |
| P1.2 | Write path (new/import_pages/save) | same module (`FPDF_CreateNewDocument`/`FPDF_ImportPages`/`FPDF_SaveAsCopy`) |
| P1.3 | pypdfium2-compatible Python package + backend flag | `python/gpu_pdfium/` (`__init__.py`, `raw.py`) |
| P1.4 | Corpus-wide parity validation | `bench/parity_backend.py` → `results/parity_backend.json` |

## What was built
A nanobind module mirroring the exact pypdfium2 surface the codebase uses:
`PdfDocument` (open from bytes/BytesIO/path, `new`, `__len__`, `__getitem__`, `get_page`,
`get_metadata_dict`, `import_pages`, `save`, `close`), `PdfPage` (`get_width/height/size/rotation`,
`render(scale,rotation)`, `get_textpage`, `get_objects(filter,max_depth)`), `PdfTextPage`
(`get_text_bounded`), `PdfBitmap` (`to_numpy`, `mode`), `PdfPageObject`/`PdfImage`
(`type`, `get_pos`, `get_size`, `get_bitmap`), plus `raw.FPDF_PAGEOBJ_*` and `PdfiumError`.

Render matches pypdfium2 byte-for-byte by replicating its choices: default **BGR** 3-channel,
dims `ceil(size*scale)`, white fill, `draw_annots=True` (`FPDF_ANNOT`). Text matches by using
`FPDFText_GetBoundedText` over the page bbox (not raw char-index order).

## Drop-in mechanism (no edits to the 8 import sites)
`gpu_pdfium.activate()` injects the module as `pypdfium2` and `pypdfium2.raw` in `sys.modules`;
`maybe_activate_from_env()` does so iff `NEMO_PDF_BACKEND=gpu`. Verified: with the flag set, the
real `split_pdf_batch → pdf_extraction` pipeline ran entirely on the native backend, 0 errors,
producing identical text + page images + extracted images.

## Parity results (gpu_pdfium vs pypdfium2, `results/parity_backend.json`)
| bucket | docs | min render SSIM | text pages | objcount pages | metadata |
|--------|-----:|----------------:|-----------:|---------------:|---------:|
| born_digital | 28 | 1.0 | 70/70 | 70/70 | 28/28 |
| dense_vector | 9 | 1.0 | 28/28 | 28/28 | 9/9 |
| multi_image | 2 | 1.0 | 8/8 | 8/8 | 2/2 |
| scanned | 1 | 1.0 | 1/1 | 1/1 | 1/1 |
| rotated | 1 | 1.0 | 1/1 | 1/1 | 1/1 |
| malformed | 2 | — | rejected by both backends (graceful) | | |

**Global: min render SSIM = 1.0 (bit-identical), text 108/108, objcounts 108/108, metadata 41/41.**

## Quality co-gate (`bench/QUALITY_GATE.md`)
Because P1 rasters and text are **bit-identical** to pypdfium2, downstream YOLOX/OCR/embedding
inputs are unchanged, so retrieval recall is unchanged **by construction** — the agent_eval recall
run is not needed to clear P1. It becomes meaningful in P2, when the CUDA rasterizer produces
perceptually-close (not bit-identical) output.

## Why P1 already has value (independent of the CUDA bet)
- Links PDFium as C++ (no per-call ctypes overhead).
- Owns init/threading explicitly (foundation for fixing the concurrent-render thread-unsafety).
- Is the stable contract layer the P2 CUDA rasterizer slots behind `render()`.

## Build / run
```bash
. /etc/profile.d/cuda.sh
cd gpu_pdf_extractor/native && cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE=$(which python3) -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build build
# parity:
python3 gpu_pdf_extractor/bench/parity_backend.py
# drop-in:
NEMO_PDF_BACKEND=gpu PYTHONPATH=gpu_pdf_extractor/python python3 -c "import gpu_pdfium; gpu_pdfium.activate(); import pypdfium2; print(pypdfium2 is gpu_pdfium)"
```

## Caveats / P2 inputs
- PDFium build mismatch (link 7906 vs baseline pypdfium 6462) caused **no** observed divergence
  (all SSIM 1.0); revisit only if a divergence appears.
- PageObject handles require the owning page to stay alive during iteration (matches all call sites;
  no `keep_alive` on the list return because Python lists aren't weak-referenceable).
- Packaging is a CMake build, not yet a pip wheel — wheelization deferred to P1.5/P4.
