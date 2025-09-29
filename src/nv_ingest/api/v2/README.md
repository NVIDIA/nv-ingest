# NV-Ingest V2 API

## Overview

The V2 API introduces automatic PDF splitting at the REST layer to improve processing throughput. When a multi-page PDF is submitted, it's automatically split into single-page subjobs before being sent to the Ray processing backend.

## Key Changes from V1

1. **Automatic PDF Splitting**: PDFs over the configured `PDF_SPLIT_PAGE_COUNT` are automatically split into multi-page chunks
2. **Parent-Child Job Tracking**: Parent jobs maintain relationships with their subjobs via Redis
3. **Transparent Aggregation**: Results are automatically aggregated when fetching parent jobs
4. **Backward Compatible**: PDFs with page counts ≤ `PDF_SPLIT_PAGE_COUNT` behave identical to V1

## How It Works

1. **Submit**: When a PDF with pages exceeding `PDF_SPLIT_PAGE_COUNT` is submitted to `/v2/submit_job`:
   - The PDF is split into page chunks (size determined by `PDF_SPLIT_PAGE_COUNT`)
   - Each chunk becomes a subjob with deterministic IDs derived from the parent
   - Source IDs are modified to maintain association: `document.pdf#page_1`
   - Parent-child mapping is stored in Redis

2. **Processing**: Each subjob is processed independently by Ray, appearing as individual single-page PDFs

3. **Fetch**: When fetching the parent job via `/v2/fetch_job/{parent_id}`:
   - Subjob states and results are retrieved concurrently (bounded by the Redis connection pool)
   - If all complete, results are aggregated in original page order
   - Pending work returns 202 (processing)
   - Failed chunks are noted without failing the entire job; metadata records which chunks failed

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

## Performance

Initial testing shows improved pages/second processing time due to parallel page processing in Ray.
