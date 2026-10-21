# Multimodal extraction

NeMo Retriever Library classifies and extracts text, tables, charts, infographics, and related layout from documents and media. This page groups formats, extraction modes, structured outputs, and throughput guidance in one place. Use the table of contents to jump to a topic.

## On this page

- [Supported file types and formats](#supported-file-types-and-formats)
- [Text and layout extraction](#text-and-layout-extraction)
- [Serve Nemotron 3.5 Parse with vLLM](#served-nemotron-35-parse)
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
    `method="nemotron_parse"` requires the Nemotron Parse NIM client dependencies. Install them with the `nemotron-parse` extra, for example `pip install "nemo-retriever[nemotron-parse]"`, before running PDF extraction through Nemotron Parse. Available element classes depend on the model; refer to [Charts and infographics](#charts-and-infographics).

**Related**

- [NeMo Retriever Library Overview](overview.md)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Chunking](concepts.md#chunking)

## Serve Nemotron 3.5 Parse with vLLM { #served-nemotron-35-parse }

You can run Nemotron 3.5 Parse in a separate vLLM server and send page images
from an in-process or batch graph. NeMo Retriever Library keeps its existing
local inference dependencies. The server uses the model package's own runtime.

Install the optional model-team client package at the pinned revision in your
NeMo Retriever Library environment. `--no-deps` keeps the package's serving
runtime pins from replacing the library's dependencies.

```bash
pip install "nemo-retriever[nemotron-parse]"
pip install --no-deps \
  "git+https://huggingface.co/nvidia/Nemotron-3.5-Parse@e7ea79242a87ea0ad3919b672bfedb429a683206"
```

The optional `nemotron-parse` package is separate from the
`nemo-retriever[nemotron-parse]` extra. The extra installs the library's Parse
client dependencies; the package provides the model team's `DocumentParser`.
Refer to the [pinned model package and serving
instructions](https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/e7ea79242a87ea0ad3919b672bfedb429a683206)
when preparing the separate server environment.

The optional package also installs a vLLM plugin. Its version check requires
vLLM `0.26.0`, Transformers `5.14.1`, and Hugging Face Hub `1.24.0`, which differ
from the library's local runtime. Before starting NeMo Retriever Library
processes or their Ray workers, allow only the library's embedding plugin in
that environment.

```bash
export VLLM_PLUGINS=nrl_nemotron35_embedding
```

This setting prevents the model package's serving plugin from loading in the
library's local vLLM runtime. Apply it to the client environment only. The
separate Parse server must use its validated runtime and load its own Parse
plugin; do not give that server the client allowlist.

Native NIM `/v1/parse` requests and native-schema Parquet output do not require
this optional model-team package. However, interleaved export uses its geometry
helper to crop Parse 3.5 `Picture` elements in `normalized_page_content`
coordinates. Install the pinned package above and apply the client plugin
allowlist when you need those crops. If the package cannot be imported, the
writer records the affected page in `materialize_error` and continues exporting
other pages. Native NIM pages without these picture crops do not need the
package. Refer to [interleaved Parquet output](nemo-retriever-api-reference.md#interleaved-parquet-output)
for the page-validation contract.

Configure the server's OpenAI-compatible `/v1/chat/completions` endpoint and
set `nemotron_parse_model` explicitly. The following example assumes that the
server exposes `nvidia/Nemotron-3.5-Parse` as its served model name.

```python
from nemo_retriever import create_ingestor

result, failures = (
    create_ingestor(run_mode="batch")
    .files(["report.pdf"])
    .extract(
        method="nemotron_parse",
        nemotron_parse_invoke_url="http://localhost:8000/v1/chat/completions",
        nemotron_parse_model="nvidia/Nemotron-3.5-Parse",
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    .ingest(return_failures=True)
)
```

`Nemotron-3.5-Parse` is also accepted as the model name when it matches the
server configuration. These names select the model-team `DocumentParser`
single-pass path. That path uses the package's full-page prompt and sampling
settings; the library's legacy `task_prompt` does not override them. Other
model names do not select this adapter.

Remote requests use the existing NIM transport and retry configuration.
`remote_retry.remote_max_pool_workers` limits concurrent page requests per
actor. `BatchTuningParams.nemotron_parse_batches_in_flight` controls local
inference only and cannot be used with a remote endpoint. No local Parse model
is loaded for this remote route.

Inspect `failures` and the `nemotron_parse_v1_2` metadata in each page row.
A failed served 3.5 request retains its page identity and diagnostic fields;
successful sibling pages remain available. Missing optional client dependencies
produce an installation error instead of selecting another model.

The native Nemotron Parse NIM uses a different `/v1/parse` contract. Refer to
[Parse endpoint contracts](prerequisites-support-matrix.md#nemotron-parse-hosted-vs-self-hosted)
before changing an existing NIM endpoint.

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

!!! note "Parse element classes depend on the model"
    [Nemotron Parse v1.2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2)
    does not predict a `Chart` class. Its figures can be labeled `Picture`,
    which the tagged-content router places in the `infographic` channel when
    `extract_infographics=True`.

    Nemotron 3.5 Parse can predict `Chart` elements. The router places these in
    the `chart` channel when `extract_charts=True`. `Table` elements go to
    `table`; `Picture` and `Infographic` elements go to `infographic`. These
    channels contain the model's extracted text and normalized boxes.

    Enabling an extraction flag does not make a model predict additional
    classes. For a separate layout-detection and OCR workflow, use the default
    PDFium path.

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

With `method="nemotron_parse"`, each page row includes a
`nemotron_parse_v1_2` metadata dictionary. This column name remains the same
for local v1.2, remote chat, native NIM, and served 3.5 extraction.

The following fields support page-level diagnostics.

| Field | Meaning |
| --- | --- |
| `raw_output` | Model response before element routing. `None` means no response was available; `""` means an empty response. Whitespace normalization depends on the endpoint contract. |
| `finish_reason` | Completion reason when the backend provides it. Local v1.2 uses `unknown` when its completion reason is missing. |
| `usage` | Backend token usage when available. Local v1.2 does not populate usage. |
| `error` | Failure details, including `stage`, `type`, and `message`. Some failures also include a traceback. |
| `native_parse` | Native `/v1/parse` completion metadata, retained alongside the common fields for compatibility. |
| `stages` and `inference` | Raw stage responses and effective model-package settings for successful served 3.5 calls. |

Local v1.2 and chat responses with an available finish reason other than
`stop` or `eos` receive an `IncompleteModelOutputError` at
`nemotron_parse_pages_finish_reason`. The legacy router can retain a returned
prefix in content fields. Consumers that require complete pages must inspect
`error`, rather than treating nonempty text as success.

The served 3.5 adapter rejects empty or incomplete full-page responses before
element routing and records the error on the affected page. The native
`/v1/parse` client rejects `finish_reason="length"` and retains its existing
batch failure behavior. For that rejection, the failing page retains its raw
response and completion metadata. Both `get_error_rows()` and
`return_failures=True` report pages with populated error fields when you retain
result rows. To preserve native rows and diagnostic metadata on disk, use
[graph Parquet output](nemo-retriever-api-reference.md#graph-parquet-output).
Its summary mode avoids retaining the full result DataFrame; inspect the saved
error fields when reviewing failures. The optional
[interleaved output schema](nemo-retriever-api-reference.md#interleaved-parquet-output)
projects complete tagged pages into ordered text, table, and picture-crop rows.
It retains rejected pages as `materialize_error` rows and uses temporary disk
storage to order documents without collecting the full corpus in memory.

In a custom Ray graph, a node's `node_overrides` entry can set `target_num_rows_per_block` to a positive integer and `strict_rows_per_block=True` to request blocks of exactly that many rows, except the final block. Pair it with the same `batch_size` so the node receives full batches instead of a small remainder after each bundle. `strict_rows_per_block` defaults to `False`; strict blocks buffer up to one block of rows, which can delay the first call.

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
