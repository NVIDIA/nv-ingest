# NV-Ingest V2 API

## Overview

The V2 API introduces automatic PDF splitting at the REST layer to improve processing throughput. When a multi-page PDF is submitted, it's automatically split into single-page subjobs before being sent to the Ray processing backend.

## Key Changes from V1

1. **Automatic PDF Splitting**: PDFs with more than 3 pages are automatically split into single-page subjobs
2. **Parent-Child Job Tracking**: Parent jobs maintain relationships with their subjobs via Redis
3. **Transparent Aggregation**: Results are automatically aggregated when fetching parent jobs
4. **Backward Compatible**: Single PDFs and PDFs with ≤3 pages behave identical to V1

## How It Works

1. **Submit**: When a PDF with >3 pages is submitted to `/v2/submit_job`:
   - The PDF is split into individual pages
   - Each page becomes a subjob with ID: `{parent_id}_page_{n}`
   - Source IDs are modified to maintain association: `document.pdf#page_1`
   - Parent-child mapping is stored in Redis

2. **Processing**: Each subjob is processed independently by Ray, appearing as individual single-page PDFs

3. **Fetch**: When fetching the parent job via `/v2/fetch_job/{parent_id}`:
   - All subjob states are checked
   - If all complete, results are aggregated in page order
   - If any are pending, returns 202 (processing)
   - Failed pages are noted but don't fail the entire job

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
