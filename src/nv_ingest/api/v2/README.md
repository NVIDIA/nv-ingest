# NV-Ingest V2 API

## Overview

The V2 API introduces automatic PDF splitting at the REST layer to improve processing throughput. When a multi-page PDF is submitted, it's automatically split into configurable multi-page chunks (default 32 pages) before being sent to the Redis service that then communicates with our Ray processing backend.

## Key Changes from V1

1. **Automatic PDF Splitting**: PDFs over the configured `PDF_SPLIT_PAGE_COUNT` are automatically split into multi-page chunks
2. **Parent-Child Job Tracking**: Parent jobs maintain relationships with their subjobs via Redis
3. **Transparent Aggregation**: Results are automatically aggregated when fetching parent jobs
4. **Backward Compatible**: PDFs with page counts ≤ `PDF_SPLIT_PAGE_COUNT` behave identical to V1

## Tracing & Aggregated Metadata

- V2 endpoints open an OpenTelemetry span using the shared `traced_endpoint` decorator. The span name defaults to the function name, or can be overridden when applying the decorator.
- `submit_job_v2` records the parent span's `trace_id` into each subjob's `tracing_options`, enabling downstream Ray stages (e.g., the message broker sink) to attach chunk-level telemetry consistently.
- Response headers still return `x-trace-id` derived from the active span context, allowing clients to correlate downstream work.
- When `/v2/fetch_job/{parent_id}` aggregates completed chunks, it captures any `trace` / `annotations` dictionaries emitted by the sink for each subjob and includes them in the response payload (see "Aggregated response" below).

This behaviour matches the V1 tracing model and sets the foundation for adding W3C `traceparent` propagation in future changes.

## How It Works

1. **Submit**: When a PDF with pages exceeding `PDF_SPLIT_PAGE_COUNT` is submitted to `/v2/submit_job`:
   - The PDF is split into page chunks (size determined by `PDF_SPLIT_PAGE_COUNT`)
   - Each chunk becomes a subjob with deterministic IDs derived from the parent
   - Source IDs are modified to maintain association: `document.pdf#page_1`
   - Parent-child mapping is stored in Redis

2. **Processing**: Each subjob is processed independently by Ray, appearing as chunk-sized PDFs that honor the configured `PDF_SPLIT_PAGE_COUNT`

3. **Fetch**: When fetching the parent job via `/v2/fetch_job/{parent_id}`:
   - Subjob states and results are retrieved concurrently (bounded by the Redis connection pool)
   - If all complete, results are aggregated in original page order
   - Pending work returns 202 (processing)
   - Failed chunks are noted without failing the entire job; metadata records which chunks failed

### Aggregated response

The fetch endpoint returns a JSON body shaped like the following:

```json
{
  "data": [...],
  "status": "success",
  "trace": {
    "trace::entry::pdf_extractor": 1000,
    "trace::exit::pdf_extractor": 2150,
    "trace::resident_time::pdf_extractor": 250,
    "trace::entry::table_extractor": 1200,
    "trace::exit::table_extractor": 2300,
    "trace::resident_time::table_extractor": 300
    // ... parent-level aggregated traces only (clean, V1-compatible)
  },
  "annotations": {
    "annotation::uuid-1": {"task_id": "pdf_extractor", "task_result": "SUCCESS"},
    "annotation::uuid-2": {"task_id": "table_extractor", "task_result": "SUCCESS"}
    // ... all annotations from all chunks (annotations have unique UUIDs)
  },
  "metadata": {
    "parent_job_id": "<uuid>",
    "total_pages": 320,
    "pages_per_chunk": 32,
    "original_source_id": "document.pdf",
    "subjob_ids": ["...", "..."],
    "subjobs_failed": 0,
    "failed_subjobs": [],
    "chunks": [
      {
        "job_id": "...",
        "chunk_index": 1,
        "start_page": 1,
        "end_page": 32,
        "page_count": 32
      }
      // ... additional chunks ...
    ],
    "trace_segments": [
      {
        "job_id": "...",
        "chunk_index": 1,
        "start_page": 1,
        "end_page": 32,
        "trace": {"trace::entry::pdf_extractor": 1.7599e18, ...}
      }
      // ... per-chunk trace details
    ],
    "annotation_segments": [
      {
        "job_id": "...",
        "chunk_index": 1,
        "start_page": 1,
        "end_page": 32,
        "annotations": {"annotation::uuid": {...}, ...}
      }
      // ... per-chunk annotation details
    ]
  }
}
```

**Top-level trace and annotations** (V1 compatibility):
- `trace`: Contains **only parent-level aggregated traces** for clean V1 compatibility
  - `trace::entry::<stage>` - Earliest entry time across all chunks
  - `trace::exit::<stage>` - Latest exit time across all chunks
  - `trace::resident_time::<stage>` - Sum of all chunk durations (total compute time)
- `annotations`: Merged annotations from all chunks (annotations have unique UUIDs so merge safely)
- These fields match V1 structure, allowing existing client code to work without modification

**Note:** Chunk-level trace details are available in `metadata.trace_segments[]` for granular analysis

**Parent-Level Trace Aggregation:**

For split PDFs, parent-level metrics are automatically computed for each stage (including nested stages):

- `trace::entry::<stage>` - Earliest entry time across all chunks (when first chunk entered stage)
- `trace::exit::<stage>` - Latest exit time across all chunks (when last chunk exited stage)
- `trace::resident_time::<stage>` - Sum of all chunk durations (total compute time in stage)

**Supports arbitrary nesting depth:**
- Simple: `trace::entry::pdf_extractor`
- Nested: `trace::entry::pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0`

**Example:**
```json
{
  "trace": {
    "trace::entry::pdf_extractor": 1000,
    "trace::exit::pdf_extractor": 2150,
    "trace::resident_time::pdf_extractor": 250
    // ... only parent-level aggregations (clean, concise)
  },
  "metadata": {
    "trace_segments": [
      {
        "chunk_index": 1,
        "start_page": 1,
        "end_page": 32,
        "trace": {
          "trace::entry::pdf_extractor": 1000,
          "trace::exit::pdf_extractor": 1100
        }
      },
      {
        "chunk_index": 2,
        "trace": {
          "trace::entry::pdf_extractor": 2000,
          "trace::exit::pdf_extractor": 2150
        }
      }
    ]
  }
}
```

**Note:** `resident_time` represents total compute time (sum of chunk durations), while `exit - entry` shows wall-clock span.

**Detailed metadata** (V2-specific):
- `trace_segments`: **Chunk-level trace data** with page ranges for granular per-chunk analysis
- `annotation_segments`: Per-chunk annotation data with page ranges
- Clients can correlate chunk data by matching `job_id` or `chunk_index` across arrays
- Failed chunk entries remain in `failed_subjobs`; missing chunks indicate the sink did not emit telemetry
- **To access chunk traces:** Use `metadata.trace_segments[]` - each segment contains the full trace dict for that chunk

## Testing

Use the V2 test script with environment variable:
```bash
# Run with V2 endpoints
DATASET_DIR=/data/splits python scripts/tests/cases/dc20_v2_e2e.py
```

Or set the API version for any existing code:
```bash
export NV_INGEST_API_VERSION=v2
```
