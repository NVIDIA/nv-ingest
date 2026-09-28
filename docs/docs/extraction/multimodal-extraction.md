# Multimodal extraction

NeMo Retriever Library classifies and extracts text, tables, charts, infographics, and related layout from documents and media. This page groups formats, extraction modes, structured outputs, and throughput guidance in one place. Use the table of contents to jump to a topic.

## On this page

- [Supported file types and formats](#supported-file-types-and-formats)
- [Text and layout extraction](#text-and-layout-extraction)
- [PDF image extraction](#pdf-image-extraction)
- [Tables](#tables)
- [Charts and infographics](#charts-and-infographics)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Image captioning](#image-captioning)
- [Metadata and content schema](#metadata-and-content-schema)
- [Extraction limitations and quality](#extraction-limitations-and-quality)

## Supported file types and formats { #supported-file-types-and-formats }

NeMo Retriever Library accepts multiple document and media types. A current list (including PDF, Office formats, HTML, images, audio, and video, some early access) appears in [NeMo Retriever Library Overview](overview.md) under **NeMo Retriever Library supports the following file types**.

**Related**

- [Troubleshoot](troubleshoot.md) for format-specific issues
- [Speech and audio](audio-video.md)

## Text and layout extraction { #text-and-layout-extraction }

For PDFs, NeMo Retriever Library typically uses **pdfium**-based extraction with configurable depth and paths. Scanned or mixed pages may use hybrid, OCR-oriented, or Nemotron Parse methods. For `method` options such as `pdfium`, `pdfium_hybrid`, `ocr`, and `nemotron_parse`, refer to the [Python API reference](nemo-retriever-api-reference.md).

!!! note
    `method="nemotron_parse"` requires the Nemotron Parse NIM client dependencies. Install them with the `nemotron-parse` extra, for example `pip install "nemo-retriever[nemotron-parse]"`, before running PDF extraction through Nemotron Parse. This path does not produce chart modality rows; for chart detection, refer to [Charts and infographics](#charts-and-infographics).

**Related**

- [NeMo Retriever Library Overview](overview.md)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Chunking](concepts.md#chunking)

## PDF image extraction { #pdf-image-extraction }

For the PDFium path, `extract_images=True` extracts rendered page regions that resemble images. A region can represent a composite figure, including an entire PDF Form XObject. The output does not necessarily correspond one-to-one with the raster IMAGE XObjects stored in the PDF.

Set `extract_nested_images=True` when you also need the individual raster IMAGE objects nested in Form XObjects. This option applies only to PDFium extraction and requires `extract_images=True`. Configuration validation rejects `extract_nested_images=True` when `extract_images=False`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import ExtractParams

result = (
    create_ingestor(run_mode="inprocess")
    .files(["report.pdf"])
    .extract(
        ExtractParams(
            method="pdfium",
            extract_images=True,
            extract_nested_images=True,
        )
    )
    .ingest()
)
```

Both local and service CLI ingest expose the same image extraction options. Enable both options with the command for your ingest path.

```bash
retriever ingest report.pdf \
  --extract-images --extract-nested-images

retriever ingest service report.pdf \
  --extract-images --extract-nested-images
```

Nested image extraction has the following output behavior:

- It adds nested raster images to the default rendered regions. The output can contain both a composite Form region and its individual source images.
- It emits each occurrence of a decodable nested IMAGE object. Repeated placements of the same source remain separate output elements. Each occurrence is decoded independently. The extractor does not reuse encoded payloads across PDF image objects because their render state can differ.
- It returns each decoded source bitmap at its intrinsic pixel dimensions. The payload resolution is independent of the PDF page-render DPI.
- It reports each occurrence location as normalized page-space coordinates in `bbox_xyxy_norm`.

Nested image extraction enforces the following limits:

- A maximum of 256 nested image occurrences per page.
- A maximum of 50,000,000 decoded pixels, summed across all nested image placements on a page.
- A maximum of 64 MiB of raw source data, summed across all nested image placements on a page.

If supplemental nested image extraction fails or exceeds a limit, the extractor discards all nested image additions for that page. It preserves successful base results, including text, the rendered page image, and default rendered image regions. The extractor records the failure in `metadata.error` with the `nested_images` stage and reports it through the operator error output.

To remove repeated image content after extraction, add a `.dedup(...)` stage. For configuration details, refer to [Control image deduplication](#control-image-deduplication).

## Tables { #tables }

NeMo Retriever Library detects tables as structured page elements, processes them through the appropriate NIMs, and exports formats suitable for downstream RAG (including Markdown-oriented representations where configured). Availability depends on pipeline and model configuration; refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md).

**Related**

- [NeMo Retriever Library Overview](overview.md) for artifact classification
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) for advanced visual parsing
- [Metadata reference](content-metadata.md)

## Charts and infographics { #charts-and-infographics }

Charts and infographic regions are classified with other page layout elements (tables, text blocks, titles) and processed through layout detection and OCR. `extract_charts` and `extract_infographics` are enabled by default. Outputs use the same metadata schema as other extracted objects.

!!! important "Chart modality requires the default layout path"
    [Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) semantic classes do not include `Chart` or `Infographic`. The model labels regions as `Text`, `Table`, `Picture`, `Caption`, `List-item`, `Section-header`, and similar types instead.

    When you set `method="nemotron_parse"`:

    - The pipeline does not produce `chart` or `infographic` modality rows, even when `extract_charts=True` or `extract_infographics=True`.
    - Chart- and infographic-filtered retrieval (for example, queries scoped to figure or chart content) returns no hits.
    - Chart-heavy and infographic-heavy pages are typically emitted as `Picture` or other non-chart modalities.

    For chart and infographic detection and modality-specific retrieval, use the default **pdfium** layout path (page-elements detection and OCR), not `method="nemotron_parse"`.

Chart-labeled PDF regions are **not** routed through the Omni caption stage; they remain on the layout-and-OCR path. For scope and validation guidance, refer to [Image captioning](#image-captioning).

For natural-language infographic descriptions, optionally enable [image captioning](#image-captioning) and set `caption_infographics=True` when you need VLM captions on infographic regions.

**Related**

- [NeMo Retriever Library Overview](overview.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Multimodal embeddings (VLM)](embedding.md) when you treat graphics as images for embedding

## OCR and scanned documents { #ocr-and-scanned-documents }

Scanned PDFs and image-only pages rely on OCR and hybrid paths that combine native text extraction with OCR when needed. For extract methods such as `ocr` and `pdfium_hybrid`, refer to the [Python API reference](nemo-retriever-api-reference.md).

When you run extraction locally with Hugging Face weights, the default OCR engine is **Nemotron OCR v2**, which operates in **multilingual** mode by default. For CLI flags and API parameters, refer to [CLI — OCR language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#ocr-language-mode). For Kubernetes image pins and overrides, refer to [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#ocr-nim-configuration). For hosted OCR endpoints and the NVCF language-mode limitation, refer to [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints).

**Related**

- [Text and layout extraction](#text-and-layout-extraction)
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse)
- [Extraction limitations and quality](#extraction-limitations-and-quality)

## Image captioning { #image-captioning }

Image captioning generates natural-language descriptions for unstructured image content. Retrieval can then use text embeddings over captions and visual embeddings where you configure them.

**Captioning is optional** — enable it in your ingest configuration (for example, the `caption` API or pipeline flag) when you need natural-language descriptions of image content. Reasoning traces are disabled by default for captioning.

Direct local Hugging Face captioning defaults to `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`. Its BF16 weights are approximately 62 GiB. For this local vLLM profile, NeMo Retriever Library reserves `0.95` of a dedicated GPU's memory for the model and KV cache. An NVIDIA H100 with 80 GB of memory meets this local profile's minimum capacity when it is dedicated to captioning. The Nano caption profiles retain their `0.5` memory-utilization default and remain available through explicit model overrides. Hosted captioning defaults to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` at the NVIDIA API endpoint.

The 80 GB requirement for the self-hosted Omni NIM describes the NIM deployment. It does not by itself establish that local Hugging Face vLLM inference has enough memory for both weights and its KV cache. For local vLLM, use a dedicated GPU and retain the model-profile default unless you need to tune its memory reservation explicitly.

For example, to override the local vLLM memory reservation from the SDK, pass `CaptionParams` to `caption`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import CaptionParams, ExtractParams

result = (
    create_ingestor(run_mode="inprocess")
    .files(["multimodal_test.png"])
    .extract(ExtractParams(extract_images=True))
    .caption(
        CaptionParams(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
            gpu_memory_utilization=0.95,
        )
    )
    .ingest()
)
```

Chart-classified PDF regions stay on the layout/OCR path; only non-chart image regions and optional infographics (`caption_infographics=True`) receive Omni captions.

### Control image deduplication

Captioning a non-image document automatically enables image deduplication before the caption stage. The default removes duplicate image content. It also removes image crops that overlap table, chart, or infographic regions with an intersection over union (IoU) of `0.45` or greater. This behavior retains the structured region instead of sending the overlapping raw image crop for captioning. Image-only inputs do not enable deduplication automatically because the image is the document content.

For command-line ingest, use `--dedup` to enable the default behavior explicitly. Use `--no-dedup` to disable both content-hash and bounding-box deduplication and preserve all extracted image crops. Opting out can increase the number of caption requests, processing latency, and model cost.

Use `--dedup-iou-threshold` with an explicit `--dedup` to change the bounding-box threshold. You cannot combine the threshold option with `--no-dedup`.

```bash
retriever ingest ./data/multimodal_test.pdf \
  --caption \
  --no-dedup
```

Use `--dry-run` to inspect the effective deduplication configuration before ingest. The resolved plan includes automatically enabled deduplication for captioned non-image documents and reflects an explicit `--no-dedup` override.

The Python API provides finer-grained control with `DedupParams`. For example, retain content-hash deduplication while preserving crops that overlap structured regions.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import DedupParams, ExtractParams

result = (
    create_ingestor(run_mode="inprocess")
    .files(["report.pdf"])
    .extract(ExtractParams(extract_images=True))
    .dedup(DedupParams(content_hash=True, bbox_iou=False))
    .caption()
    .ingest()
)
```

To disable all image deduplication through the Python API, call `.dedup(DedupParams(content_hash=False, bbox_iou=False))` before `.caption()`.

**Related**

- [Multimodal embeddings (VLM)](embedding.md)
- [Metadata reference](content-metadata.md)
- [Image captioning](prerequisites-support-matrix.md#image-captioning)

## Metadata and content schema { #metadata-and-content-schema }

Extracted objects follow the schema and field descriptions in the [Metadata reference](content-metadata.md). Use that page for tables, types, and per-field notes.

With `method="nemotron_parse"`, each page row in the returned extraction `DataFrame` includes a `nemotron_parse_v1_2` metadata dictionary. Its `raw_output` field contains the model response passed to element routing. A value of `None` means that no model response was available; an empty response is stored as `""`. With a remote endpoint, `raw_output` is the stripped message text of the chat-completions response, and a response without message content is also stored as `""`. This additive metadata does not change routing into the existing `text`, `table`, `chart`, and `infographic` fields.

For bundled local v1.2 inference, any vLLM finish reason other than `stop` records a structured error in `nemotron_parse_v1_2.error`. Its `type` is `IncompleteModelOutputError`, and its `stage` is `nemotron_parse_pages_finish_reason`. NeMo Retriever Library still routes any returned prefix into content fields, and `get_error_rows()` and `return_failures=True` report these pages because `error` is populated. Remote responses do not report a finish reason, so this check applies only to local inference. Consumers that require complete pages must check `error`; they can use `raw_output` for diagnostics.

## Extraction limitations and quality { #extraction-limitations-and-quality }

Hosted Page Elements, Table Structure, and Graphic Elements NIM endpoints cap inline base64 image payloads at about **180,000 characters** (roughly 180 KB). The NeMo Retriever pipeline downscales large page renders before remote NIM calls. Direct API integrations must keep inline payloads under that cap. Hosted Page Elements does not accept NVCF Asset API references. For limits, plus `dpi` and `render_mode` tuning, refer to [Hosted Page Elements NIM image size limits](troubleshoot.md#hosted-page-elements-nim-image-size-limits).

Image payload limits are separate from the throughput metrics in the rest of this section.

A single headline metric can drastically misrepresent system efficiency. The amount of compute that you need to process a dataset depends far more on its content and how your pipeline operates than on its disk size. This section explains why, and offers better ways to measure and report throughput.

Some common throughput measures, and their problems, include the following:

- **TB/day, GB/hour, MB/s** – Useful for capacity planning for storage and network, and the cost of data movement or archival. A weak proxy for compute due to compression and encoding differences.
- **docs/min (documents per minute)** – Easy to understand, but documents vary wildly in length and complexity.
- **pages/sec (pages per second)** – Usually correlates with work batching (sets-of-pages from PDFs). Varies with per-page complexity and modality mix.
- **images/sec** – Relevant when image transforms dominate. Sensitive to resolution.
- **tokens/sec** – Useful for LLM/VLM text-heavy stages. Ignores non-text work.
- **elements/sec (tables/sec, charts/sec, OCR pages/sec)** – Stage-specific and informative. Must be paired with prevalence (how many elements per page).

### Summary

- Disk size is not a reflection of expected processing time. Content complexity and enabled tasks dominate actual compute cost.
- Pages/sec is generally better than data-size-over-time metrics because it correlates more with work units, but it is still imperfect.
- Report throughput alongside dataset characteristics and stage-level metrics for meaningful, reproducible comparisons.

### Example use cases

The following two datasets can yield the reverse ranking if you evaluate by data-size-over-time versus by pages/sec:

- **Complex-but-small** – A 1000-page PDF where each page contains dense tables and charts. The PDF may be small on disk (vector text, compressed graphics) yet very expensive to process (table detection, OCR, structure reconstruction, chart parsing).
- **Large-but-simple** – A 1000-page PDF with one large image per page. The file may be huge on disk (high-DPI scans) but comparatively fast to process if your pipeline mostly routes images without heavy analysis.

### What drives processing cost

The following factors drive processing cost.
!!! important
    None of the following factors correlate with file size.

- Content modality and tasks enabled
  - Text OCR vs. native text extraction
  - Table structure detection and reconstruction
  - Chart detection and text extraction
  - Image captioning or vision-language models
  - Embedding generation and vector storage

- Content density and complexity per page
  - Number of elements (tables, figures, charts, text blocks)
  - Layout complexity (nested tables, merged cells, multi-column text)
  - Languages, scripts, and fonts (OCR difficulty)

- Resolution and quality
  - DPI for scanned pages (I/O and pre-processing cost)
  - Compression artifacts vs. vector graphics

- Pipeline configuration
  - Which stages are turned on/off
  - Model choices (accuracy vs. speed trade-offs)
  - Batch sizes, concurrency, hardware placement

- System factors
  - Warm-up vs. steady state
  - I/O bandwidth and storage latency
  - Network latency to inference services

### Why data-size-over-time is misleading

Use data-size-over-time metrics for storage and network planning, not for compute efficiency.

The following are examples of why data-size-over-time metrics are misleading:

- Compression breaks the proxy
  - Highly compressible vector PDFs may be tiny yet compute-heavy.
  - Scanned images may be huge but require minimal analysis.

- Format dependency
  - Two datasets with identical content can have wildly different byte sizes due to encoding/format.

- Incentivizes the wrong optimizations
  - Encourages selecting “big-byte” but easy datasets to inflate data-size-over-time without improving true efficiency.

- Not portable across stages
  - Bytes are not additive across pipeline stages (and often increase or decrease as formats change).

- Hard to reproduce
  - Data-size-over-time varies wildly with dataset encoding choices, not just system performance.

### Why pages/sec is better (but imperfect)

When you report pages/sec, you should also report dataset characterization.

The following are some reasons why pages/sec is better than data-size-over-time metrics:

- Closer to the work unit
  - Pipelines commonly schedule and process sets-of-pages from PDFs to saturate pipeline resources.
- Normalizes away compression and file format
  - A page is a page regardless of on-disk bytes.

However, pages/sec is still imperfect because of the following:

- Page complexity varies
  - Pages with many tables/charts/figures or dense text cost more than blank or simple pages.
- Modality mix differs
  - OCR-heavy pages vs. native text pages drive very different compute paths.
- Resolution matters
  - High-DPI scans require more I/O and pre-processing.

### Example: interpreting the two 1000-page PDFs

The supposedly fast dataset by data-size-over-time can be the slow one by pages/sec, and vice versa. Only context-rich reporting avoids this trap.

The following are reasons why:

- Complex tables + charts per page (small file size)
  - Data-size-over-time appears low due to tiny bytes, but compute is high → pages/sec and stage-level metrics reveal true cost.
  - Expect lower pages/sec and lower tables/sec/charts/sec to dominate.
- Single large image per page (large file size)
  - Data-size-over-time appears high due to big bytes, but compute can be low → fast pages/sec.
  - If table/chart stages are skipped, stage-level numbers show negligible table/chart work.

### Practical tips for fair comparisons

The following are practical tips for fair comparisons:

- Separate warm-up from steady-state measurements.
- Fix the pipeline configuration and model versions for a given comparison.
- Keep concurrency and resource limits identical across runs.
- Provide dataset characterization alongside throughput numbers.
