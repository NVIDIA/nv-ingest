# Why Throughput Is Dataset-Dependent

A single headline metric like TB/day, GB/hour, MB/s, docs/min, pages/sec, images/sec, tokens/sec, or elements/sec (tables/sec, charts/sec) can drastically misrepresent system efficiency. The amount of compute needed to process a dataset depends far more on its content and how your pipeline operates than on its disk size. This page explains why and offers better ways to measure and report throughput.

## Common throughput measures you may see

- **TB/day, GB/hour, MB/s**
  - Useful for storage/network planning; weak proxy for compute due to compression and encoding differences.
- **docs/min (documents per minute)**
  - Easy to understand, but documents vary wildly in length and complexity.
- **pages/sec (pages per second)**
  - Usually correlates with work batching (sets-of-pages from PDFs); still varies with per-page complexity and modality mix.
- **images/sec**
  - Relevant when image transforms dominate; sensitive to resolution.
- **tokens/sec**
  - Useful for LLM/VLM text-heavy stages (embedding/generation); ignores non-text work.
- **elements/sec (tables/sec, charts/sec, OCR pages/sec)**
  - Stage-specific and informative; must be paired with prevalence (how many elements per page).

## TL;DR

- Disk size is not a reflection of expected processing time; content complexity and enabled tasks dominate actual compute cost.
- Pages/sec is generally better than data-size-over-time metrics (e.g., TB/day, GB/hour, MB/s) because it correlates more with work units, but it is still imperfect.
- Report throughput alongside dataset characteristics and stage-level metrics for meaningful, reproducible comparisons.

## Motivating Examples

- Complex-but-small: A 1000-page PDF where each page contains dense tables and charts. The PDF may be small on disk (vector text, compressed graphics) yet very expensive to process (table detection, OCR, structure reconstruction, chart parsing).
- Large-but-simple: A 1000-page PDF with one large image per page. The file may be huge on disk (high-DPI scans) but comparatively fast to process if your pipeline mostly routes images without heavy analysis.

These two datasets can yield the reverse ranking if you evaluate by data-size-over-time versus by pages/sec.

## What Actually Drives Processing Cost

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

None of the above are captured by file size.

## Why data-size-over-time Is Misleading

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

Use data-size-over-time metrics for storage and network planning, not for compute efficiency.

## Why Pages/sec Is Better (But Imperfect)

- Closer to the work unit
  - Pipelines commonly schedule and process sets-of-pages from PDFs to saturate pipeline resources.
- Normalizes away compression and file format
  - A page is a page regardless of on-disk bytes.

However, pages/sec is still imperfect because:

- Page complexity varies
  - Pages with many tables/charts/figures or dense text cost more than blank or simple pages.
- Modality mix differs
  - OCR-heavy pages vs. native text pages drive very different compute paths.
- Resolution matters
  - High-DPI scans require more I/O and pre-processing.

Therefore, pages/sec should be accompanied by dataset characterization.

## Example: Interpreting the Two 1000-Page PDFs

- Complex tables + charts per page (small file size)
  - Data-size-over-time appears low due to tiny bytes, but compute is high → pages/sec and stage-level metrics reveal true cost.
  - Expect lower pages/sec and lower tables/sec/charts/sec to dominate.
- Single large image per page (large file size)
  - Data-size-over-time appears high due to big bytes, but compute can be low → fast pages/sec.
  - If table/chart stages are skipped, stage-level numbers show negligible table/chart work.

The “fast” dataset by data-size-over-time can be the “slow” one by pages/sec, and vice versa. Only context-rich reporting avoids this trap.

## Practical Tips for Fair Comparisons

- Separate warm-up from steady-state measurements.
- Fix the pipeline configuration and model versions for a given comparison.
- Keep concurrency and resource limits identical across runs.
- Provide dataset characterization alongside throughput numbers.

## When data-size-over-time Metrics Are Still Useful

- Capacity planning for storage and network.
- Cost of data movement or archival.

For compute efficiency and pipeline scaling discussions, prefer pages/sec and stage-level metrics with proper dataset context.
