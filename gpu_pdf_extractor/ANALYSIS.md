# Current pypdfium2 Implementation — Analysis & Replacement Contract

Scope: every place `pypdfium2` (`pdfium` / `libpdfium`) is used under `nemo_retriever/src/`,
and the exact API surface a drop-in replacement must satisfy.

Dependency: `pypdfium2==4.30.0` (pinned in `nemo_retriever/pyproject.toml`). This is a
prebuilt wheel that bundles Google's PDFium (C++) binary plus a ctypes-based Python API.

---

## 1. Where it's used

### Two distinct work phases

PDF processing splits into two phases with **very different parallelism characteristics**:

| Phase | What it does | Parallelism | GPU fit |
|-------|--------------|-------------|---------|
| **Parse / interpret** | xref + object parsing, stream decompression (Flate/DCT/JBIG2), content-stream → display list, font/glyph resolution | branchy, pointer-chasing, sequential | **Poor** — already optimal in C++ |
| **Rasterize / render** | execute display list: path fill/stroke + anti-alias, alpha compositing, image resampling, glyph rasterization | pixel-parallel, embarrassingly so when batched across pages | **Good** — the real CUDA opportunity |

The codebase rasterizes **every page** to a numpy array at 200–300 DPI for downstream
YOLOX detection, OCR, and page-image storage. At Ray-Data scale this render path is the
dominant CPU cost. **That is the hot path worth accelerating.** Text/object/metadata/split
paths get little-to-no GPU benefit.

### Call sites (grouped by what they exercise)

**Core wrapper — `common/api/util/pdf/pdfium.py`** (the module almost everything funnels through):
- `pdfium_pages_to_numpy(pages, render_dpi=300, scale_tuple, padding_tuple, rotation)` → `(list[np.ndarray], list[(pad_w,pad_h)])` — **the render hot path**. Uses `page.render(scale, rotation)` + `bitmap.to_numpy()`.
- `convert_bitmap_to_corrected_numpy(bitmap)` — BGRA→RGB (note: deliberately avoids pdfium's `rev_byteorder` because it SIGTRAPs under concurrent rendering — a known thread-safety wart of the current backend).
- `is_scanned_page(page)` → bool — `get_textpage().get_text_bounded()` + count image objects.
- `extract_simple_images_from_pdfium_page` / `extract_image_like_objects_from_pdfium_page` / `extract_merged_{images,shapes}` / `extract_forms_from_pdfium_page` — `page.get_objects(filter=…, max_depth=…)`, `obj.get_pos()`, `obj.get_size()`, `obj.get_bitmap(render=)`.
- `convert_pdfium_position(pos, w, h)` — PDF bottom-left → top-left bbox.

**Extraction engines** (`common/api/internal/extract/pdf/engines/`):
- `pdfium.py::pdfium_extractor(...)` — full orchestrator: `PdfDocument(stream)`, `doc.get_page(i)`, `page.get_size()`, text, render, object extraction, `page.close()`, `doc.close()`. ThreadPool over pages.
- `nemotron_parse.py` — `PdfDocument`, per-page render via `pdfium_pages_to_numpy`, text fallback via `get_text_bounded()`.
- `adobe.py`, `unstructured_io.py` — pdfium used **only** for metadata.

**Operators** (`operators/extract/pdf/`):
- `extract.py::pdf_extraction(...)` — single-page (post-split) path: `PdfDocument(bytes)`, `doc.get_page(0)`, `_render_page_to_base64` (`page.render(scale)`), text, object images.
- `split.py::_split_pdf_to_single_page_bytes(...)` — `PdfDocument.new()`, `single.import_pages(doc, pages=[i])`, `single.save(buf)`. **Document-mutation path** (no rendering).

**Other**: `common/io/image_store.py::render_page_image_b64` (path → page → base64); `cli/pdf/stage.py::_safe_pdf_page_count` (`len(doc)`); `common/api/util/metadata/aggregators.py::extract_pdf_metadata` (`get_metadata_dict()`, `len(doc)`).

---

## 2. The drop-in contract (must be satisfied to swap in)

```python
class PdfDocument:
    def __init__(self, src: bytes | BytesIO | str | os.PathLike): ...
    @staticmethod
    def new() -> "PdfDocument": ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> "PdfPage": ...
    def get_page(self, i: int) -> "PdfPage": ...
    def get_page_count(self) -> int: ...
    def get_metadata_dict(self) -> dict: ...                  # CreationDate, ModDate, Keywords, …
    def import_pages(self, src: "PdfDocument", pages: list[int]) -> None: ...
    def save(self, buf: BytesIO) -> None: ...
    def close(self) -> None: ...

class PdfPage:
    def get_width(self) -> float; def get_height(self) -> float      # points (1/72")
    def get_size(self) -> tuple[float, float]
    def get_rotation(self) -> int                                    # 0/90/180/270
    def render(self, scale: float = 1.0, rotation: int = 0) -> "PdfBitmap"   # HOT PATH
    def get_textpage(self) -> "PdfTextPage"
    def get_objects(self, filter: tuple = (), max_depth: int = 0) -> Iterable["PageObject"]
    def close(self) -> None

class PdfTextPage:
    def get_text_bounded(self) -> str

class PdfBitmap:
    def to_numpy(self) -> np.ndarray
    mode: str                                                        # "BGRA"|"BGRX"|"BGR"|"Gray"…

class PageObject:
    type: int                                                        # FPDF_PAGEOBJ_* constant
    def get_pos(self) -> tuple[float,float,float,float]              # left,bottom,right,top
    def get_size(self) -> tuple[float,float]
    def get_bitmap(self, render: bool = True) -> "PdfBitmap | None"  # PdfImage objects

# Module-level constants consumed: pypdfium2.raw.FPDF_PAGEOBJ_{TEXT,PATH,IMAGE,SHADING,FORM}
# Exception type referenced: pdfium.PdfiumError
```

Observations that constrain the design:
- The contract is **broad**: rendering is only ~1 of 6 capability areas. Text extraction,
  object enumeration with positions/types, embedded-image bitmap decode, metadata, and
  page split/import/save are all load-bearing and are **pure parsing/structure work**.
- `import_pages` + `save` means we also need PDF **writing**, not just reading.
- Coordinate conventions (bottom-left origin), rotation handling, and bitmap channel
  order are relied on by downstream geometry.

---

## 3. Environment reality (this machine)

- **No CUDA toolchain**: `nvcc` absent. **No CMake.** `nvidia-smi` → `NVML: Unknown Error` (no usable GPU here).
- Deployment images do use CUDA (vLLM JIT-compiles kernels). Python pinned **3.12 only**.
- Conclusion: a GPU build + benchmark runner must be **provisioned** before Phase 2; it does
  not exist in the current dev sandbox.

---

## 4. Key technical risks (feed the plan)

1. **Reimplementing all of PDFium is not viable.** PDFium is hundreds of KLOC of
   battle-tested, security-hardened parsing across decades of malformed-PDF edge cases.
   A from-scratch parser would be a multi-engineer-year effort and a security liability.
2. **GPU helps rasterization, not parsing.** The 95% of the contract that is parsing/
   structure/writing gains ~nothing from CUDA.
3. **Bit-exact parity with PDFium's AGG rasterizer is unattainable** (anti-aliasing, font
   hinting, subpixel rounding differ). Downstream (YOLOX/OCR/embeddings) is tolerant, but
   this must be **measured**, not assumed — the existing `agent_eval` recall harness is the
   natural quality gate.
4. **Transfer overhead**: H2D upload of raster inputs / D2H of bitmaps can erase GPU wins
   for small/few pages. The win is **batched, high-DPI, many-page** workloads, ideally with
   zero-copy hand-off to the downstream GPU model preprocessing.
5. The current backend already has a **concurrency wart** (`rev_byteorder` SIGTRAP) — a
   correctly threaded/batched replacement is itself a reliability win independent of speed.
