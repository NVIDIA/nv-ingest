# NeMo Retriever Library V2 API Guide: PDF Pre Splitting

> **TL;DR:** V2 API automatically splits large PDFs into chunks for faster parallel processing.
> 
> **Python:** Enable with `message_client_kwargs={"api_version": "v2"}` and configure chunk size with `.pdf_split_config(pages_per_chunk=64)`.
> 
> **CLI:** Use `--api_version v2 --pdf_split_page_count 64`

## Table of Contents

1. [Quick Start](#quick-start) - Get running in 5 minutes
2. [Configuration Guide](#configuration-guide) - All configuration options
3. [How It Works](#how-it-works) - Architecture overview
4. [Migration from V1](#migration-from-v1) - Upgrade existing code


---

## Quick Start

### What is V2 API?

The V2 API automatically splits large PDFs into smaller chunks before processing, enabling:

- **Higher throughput** - 1.3-1.5x faster for large documents
- **Better parallelization** - Distribute work across Ray workers
- **Configurable chunk sizes** - Tune to your infrastructure (1-128 pages)

### Minimal Example

```python
from nv_ingest_client.client import Ingestor

# Two-step configuration
ingestor = Ingestor(
    message_client_hostname="http://localhost",
    message_client_port=7670,
    message_client_kwargs={"api_version": "v2"}  # ← Step 1: Enable V2
)

# Run with optional chunk size override
results = ingestor.files(["large_document.pdf"]) \
    .extract(extract_text=True, extract_tables=True) \
    .pdf_split_config(pages_per_chunk=64) \  # ← Step 2: Configure splitting
    .ingest()

print(f"Processed {results['metadata']['total_pages']} pages")
```

### CLI Usage

```bash
nv-ingest-cli \
  --api_version v2 \
  --pdf_split_page_count 64 \
  --doc large_document.pdf \
  --task 'extract:{"document_type":"pdf", "extract_text":true}' \
  --output_directory ./results
```

**That's it!** PDFs larger than 64 pages will be automatically split and processed in parallel.

---

## Configuration Guide

### Two Required Settings

| Setting | Purpose | How to Set |
|---------|---------|------------|
| **API Version** | Route requests to V2 endpoints | `message_client_kwargs={"api_version": "v2"}` |
| **Chunk Size** | Pages per chunk (optional) | `.pdf_split_config(pages_per_chunk=N)` |

### Configuration Priority Chain

The chunk size is resolved in this order:

```
1. Client Override (HIGHEST)     → .pdf_split_config(pages_per_chunk=64)
2. Server Environment Variable   → PDF_SPLIT_PAGE_COUNT=64 in .env
3. Hardcoded Default (FALLBACK)  → 32 pages
```

**Client override always wins** - useful for per-request tuning.

### Option 1: Client-Side Configuration (Recommended)

```python
# Full control over chunk size per request
ingestor = Ingestor(
    message_client_kwargs={"api_version": "v2"}
).files(pdf_files) \
 .extract(...) \
 .pdf_split_config(pages_per_chunk=64)  # Client override
```

**Pros:**
- ✅ Different chunk sizes for different workloads
- ✅ No server config changes needed
- ✅ Clear intent in code

**Cons:**
- ❌ Must specify in every request

### Option 2: Server-Side Default

Set a cluster-wide default via Docker Compose `.env`:

```bash
# .env file
PDF_SPLIT_PAGE_COUNT=64
```

```yaml
# docker-compose.yaml (already configured)
services:
  nv-ingest-ms-runtime:
    environment:
      - PDF_SPLIT_PAGE_COUNT=${PDF_SPLIT_PAGE_COUNT:-32}
```

**Pros:**
- ✅ Set once, applies to all clients
- ✅ Different defaults per environment (dev/staging/prod)
- ✅ Clients don't need to specify

**Cons:**
- ❌ Requires server restart to change
- ❌ Less flexible than client override

### Option 3: Use the Default

Simply enable V2 without configuring chunk size:

```python
# Uses default 32 pages per chunk
ingestor = Ingestor(
    message_client_kwargs={"api_version": "v2"}
).files(pdf_files).extract(...).ingest()
```

### Configuration Matrix

| Client Config | Server Env Var | Effective Chunk Size | Use Case |
|---------------|----------------|---------------------|----------|
| `.pdf_split_config(64)` | Not set | **64** | Client controls everything |
| `.pdf_split_config(128)` | `PDF_SPLIT_PAGE_COUNT=32` | **128** | Client override wins |
| Not set | `PDF_SPLIT_PAGE_COUNT=48` | **48** | Server default applies |
| Not set | Not set | **32** | Hardcoded fallback |

### Choosing Chunk Size

> **Note:** We are developing an auto-tuning system that will automatically select chunk sizes based on document characteristics, available resources, and historical performance. This will eliminate manual tuning for most use cases.

**Smaller chunks (16-32 pages):**
- ✅ Maximum parallelism
- ✅ Lower GPU memory per worker
- ❌ More overhead from splitting/aggregation
- **Best for:** Limited GPU memory, many available workers

**Medium chunks (32-64 pages):**
- ✅ Balanced parallelism and overhead
- ✅ Good for most workloads
- **Best for:** General use (recommended starting point)

**Larger chunks (64-128 pages):**
- ✅ Minimal overhead
- ❌ Less parallelism
- **Best for:** Very large datasets, fewer workers

**Very large chunks (128+ pages):**
- ❌ Limited parallel benefits
- **Best for:** Testing or when splitting overhead is problematic

**Valid range:** 1-128 pages (server enforces with clamping)

---

## How It Works

### Architecture Flow

```
Client                    API Layer (V2)              Ray Workers
  │                            │                           │
  │   1. Submit PDF            │                           │
  ├──────────────────────────► │                           │
  │   (200 pages)              │                           │
  │                            │   2. Split into chunks    │
  │                            │   (64 pages each)         │
  │                            ├───────┐                   │
  │                            │       │ Chunk 1 (1-64)    │
  │                            │       │ Chunk 2 (65-128)  │
  │                            │       │ Chunk 3 (129-192) │
  │                            │       └ Chunk 4 (193-200) │
  │                            │                           │
  │                            │   3. Process in parallel  │
  │                            ├──────────────────────────►│
  │                            │                           │ Worker A → Chunk 1
  │                            │                           │ Worker B → Chunk 2
  │                            │                           │ Worker C → Chunk 3
  │                            │                           │ Worker D → Chunk 4
  │                            │                           │
  │   4. Fetch result          │   5. Aggregate all chunks │
  │ ◄──────────────────────────┼───────────────────────────┤
  │   (all chunks combined)    │   (ordered by page)       │
```

### Submission Phase

When you submit a PDF:

1. **Page Count Check:** Server reads PDF metadata to get total pages
2. **Split Decision:** If `page_count > pages_per_chunk`, trigger splitting
3. **Chunk Creation:** Use `pypdfium2` to split PDF into page ranges
4. **Subjob Generation:** Create subjobs with deterministic UUIDs
5. **Redis Storage:** Store parent→subjob mapping with metadata
6. **Queue Submission:** Submit all chunks to Ray task queue

**Example:**
```
Original: document.pdf (200 pages)
Config: pages_per_chunk=64

Chunks created:
- document.pdf#pages_1-64    (64 pages)
- document.pdf#pages_65-128  (64 pages)
- document.pdf#pages_129-192 (64 pages)
- document.pdf#pages_193-200 (8 pages)
```

### Processing Phase

Each chunk is processed independently:

- **Parallel execution** across available Ray workers
- **Full pipeline** runs on each chunk (extraction, embedding, etc.)
- **Per-chunk telemetry** emitted to trace/annotations
- **Results stored** in Redis with subjob IDs

### Fetch Phase

When you fetch results:

1. **Parent Check:** API checks if job has subjobs
2. **State Verification:** Checks all subjob states in parallel (batched)
3. **Wait if Needed:** Returns 202 if any chunks still processing
4. **Fetch All Results:** Fetches all subjob results in parallel (batched)
5. **Aggregate Data:** Combines all chunk data in original page order
6. **Compute Metrics:** Calculates parent-level trace aggregations
7. **Return Response:** Single unified response with all chunks

<!-- **Concurrency Control:**
- Fetches batched to avoid overwhelming Redis connection pool
- Default: `max(1, min(num_chunks, pool_size // 2))` parallel ops
- Typical pool size: 10 → ~5 parallel fetches -->

### Response Structure

**Small PDFs (≤ chunk size):**
```json
{
  "data": [...],
  "trace": {...},
  "annotations": {...},
  "metadata": {
    "total_pages": 15
  }
}
```

**Large PDFs (split into chunks):**
```json
{
  "data": [...],  // All chunks combined in page order
  "status": "success",
  "trace": {
    // Parent-level aggregated metrics
    "trace::entry::pdf_extractor": 1000,
    "trace::exit::pdf_extractor": 5000,
    "trace::resident_time::pdf_extractor": 800
  },
  "annotations": {...},  // Merged from all chunks
  "metadata": {
    "parent_job_id": "abc-123",
    "total_pages": 200,
    "pages_per_chunk": 64,
    "original_source_id": "document.pdf",
    "subjobs_failed": 0,
    "chunks": [
      {
        "job_id": "chunk-1-uuid",
        "chunk_index": 1,
        "start_page": 1,
        "end_page": 64,
        "page_count": 64
      }
      // ... more chunks
    ],
    "trace_segments": [
      // Per-chunk trace details (for debugging)
    ]
  }
}
```

### Trace Aggregation

Parent-level metrics computed from chunk traces:

- **`trace::entry::<stage>`** - Earliest entry across all chunks (when first chunk started)
- **`trace::exit::<stage>`** - Latest exit across all chunks (when last chunk finished)
- **`trace::resident_time::<stage>`** - Sum of all chunk durations (total compute time)

**Example:**
```
Chunk 1: entry=1000, exit=1100 → duration=100ms
Chunk 2: entry=2000, exit=2150 → duration=150ms
Chunk 3: entry=2100, exit=2300 → duration=200ms

Parent metrics:
entry = min(1000, 2000, 2100) = 1000  ← First chunk started
exit = max(1100, 2150, 2300) = 2300   ← Last chunk finished
resident_time = 100 + 150 + 200 = 450 ← Total compute

Wall-clock time = exit - entry = 1300ms (parallelization benefit!)
Compute time = resident_time = 450ms (actual work done)
```

---

## Migration from V1

### Minimal Migration (V2, No Splitting)

Smallest possible change - just route to V2 endpoints:

```python
# Before (V1)
ingestor = Ingestor(
    message_client_hostname="http://localhost",
    message_client_port=7670
)

# After (V2, identical behavior for PDFs ≤32 pages)
ingestor = Ingestor(
    message_client_hostname="http://localhost",
    message_client_port=7670,
    message_client_kwargs={"api_version": "v2"}  # Only change
)
```

**Behavior:** No splitting occurs, responses identical to V1.

### Full V2 with Splitting

Enable splitting for large PDFs:

```python
# V2 with PDF splitting
ingestor = Ingestor(
    message_client_hostname="http://localhost",
    message_client_port=7670,
    message_client_kwargs={"api_version": "v2"}
).files(pdf_files) \
 .extract(extract_text=True, extract_tables=True) \
 .pdf_split_config(pages_per_chunk=64) \
 .ingest()
```

### Test Script Pattern

For test scripts like `tools/harness/src/nv_ingest_harness/cases/e2e.py`:

```python
import os
from nv_ingest_client.client import Ingestor

# Read from environment
api_version = os.getenv("API_VERSION", "v1")
pdf_split_page_count = int(os.getenv("PDF_SPLIT_PAGE_COUNT", "32"))

# Build ingestor kwargs
ingestor_kwargs = {
    "message_client_hostname": f"http://{hostname}",
    "message_client_port": 7670
}

# Enable V2 if configured
if api_version == "v2":
    ingestor_kwargs["message_client_kwargs"] = {"api_version": "v2"}

# Create ingestor
ingestor = Ingestor(**ingestor_kwargs).files(data_dir)

# Configure splitting for V2
if api_version == "v2" and pdf_split_page_count:
    ingestor = ingestor.pdf_split_config(pages_per_chunk=pdf_split_page_count)

# Continue with pipeline
ingestor = ingestor.extract(...).ingest()
```

### Backward Compatibility

**V1 clients continue to work:**
- Still route to `/v1/submit_job` and `/v1/fetch_job`
- No changes required
- No splitting occurs

**V2 responses are V1-compatible:**
- Top-level `data`, `trace`, `annotations` have same structure
- Additional metadata in `metadata` object (ignored by V1 parsers)
- Existing response parsing code works unchanged

---

**HTTP status codes:**

| Code | Meaning | Action |
|------|---------|--------|
| 200 | All chunks complete | Parse results |
| 202 | Still processing | Poll again later |
| 404 | Job not found | Check job ID |
| 410 | Result consumed | Already fetched (destructive mode) |
| 500 | Server error | Check logs |
| 503 | Processing failed | Check failed_subjobs metadata |

---

#### Silent Clamping of Chunk Size

**Symptom:** Requested chunk size not used

**Cause:** Server clamps to valid range (1-128)

**Check server logs for:**
```
WARNING: Client requested split_page_count=1000; clamped to 128
```

**Solution:** Use values within 1-128 range

### Response Fields

**All PDFs:**
- `data` - Array of extracted content
- `trace` - Trace metrics
- `annotations` - Task annotations
- `metadata.total_pages` - Total page count

**Split PDFs only:**
- `metadata.parent_job_id` - Parent job UUID
- `metadata.pages_per_chunk` - Configured chunk size
- `metadata.chunks[]` - Chunk descriptors
- `metadata.trace_segments[]` - Per-chunk traces
- `metadata.failed_subjobs[]` - Failed chunk details

### Trace Metrics

**Parent-level (all jobs):**
- `trace::entry::<stage>` - Earliest start time
- `trace::exit::<stage>` - Latest finish time
- `trace::resident_time::<stage>` - Total compute time

**Chunk-level (split jobs only):**
- `metadata.trace_segments[].trace` - Per-chunk traces

### Key Files

**Server Implementation:**
- `src/nv_ingest/api/v2/ingest.py` - V2 endpoints
- `src/nv_ingest/framework/util/service/impl/ingest/redis_ingest_service.py` - Redis state management

**Client Implementation:**
- `client/src/nv_ingest_client/client/interface.py` - Ingestor class
- `client/src/nv_ingest_client/util/util.py` - Configuration utilities
- `client/src/nv_ingest_client/client/ingest_job_handler.py` - Job handling

**Schemas:**
- `api/src/nv_ingest_api/internal/schemas/meta/ingest_job_schema.py` - PdfConfigSchema

---

## FAQ

**Q: Do I need to specify chunk size every time?**  
A: No. If you don't call `.pdf_split_config()`, the server uses either the `PDF_SPLIT_PAGE_COUNT` env var or the hardcoded default (32 pages).

**Q: When does splitting actually occur?**  
A: Only when `page_count > pages_per_chunk`. Smaller PDFs are processed as single jobs (no overhead).

**Q: Will my V1 response parsing code work with V2?**  
A: Yes! Top-level `data`, `trace`, and `annotations` fields are identical. Additional metadata is added under `metadata` (which V1 parsers ignore).

**Q: How do I know if splitting occurred?**  
A: Check `len(result["metadata"].get("chunks", [])) > 0` or look for server logs: `"Splitting PDF ... into ... chunks"`.

**Q: What happens if one chunk fails?**  
A: Other chunks still return results. Check `metadata.failed_subjobs[]` for details. The job returns `status: "failed"` but includes partial results.

**Q: Does V2 work without splitting?**  
A: Yes! Just enable V2 without calling `.pdf_split_config()`. PDFs ≤ default chunk size behave identically to V1.
