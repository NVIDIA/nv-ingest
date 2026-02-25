# Retriever Online Ingest – Docker

Build and run the **online ingest** Ray Serve API in Docker. The container serves `POST /ingest` for document ingestion and uses the same pipeline as the inprocess mode (PDF extraction, page elements, OCR, embedding, LanceDB).

The image is based on **NVIDIA CUDA 13** (`nvidia/cuda:13.0.0-runtime-ubuntu22.04`) with **Python 3.12** installed for GPU-accelerated inference.

## Build

Build from the **repository root** (parent of `retriever/`) so path dependencies `api`, `client`, and `src` are available:

```bash
# From nv-ingest repo root (CUDA 13 + Python 3.12)
docker build -f retriever/Dockerfile -t retriever-online:latest .
```

For a **CPU-only** image, use a custom build with an Ubuntu 22.04 base and the same Python 3.12 steps, or override `BASE_IMAGE` to an Ubuntu image that provides Python 3.12 (the default image uses Ubuntu for the deadsnakes PPA).

## Run

```bash
# Persist LanceDB under /data; optional model mount for Nemotron OCR
docker run --rm -p 7670:7670 \
  -v /path/to/lancedb:/data \
  -v /path/to/nemotron-ocr-v1:/workspace/models/nemotron-ocr-v1 \
  retriever-online:latest
```

- **LanceDB**: Default env `ONLINE_LANCEDB_URI=/data/lancedb`. Mount a host dir at `/data` to persist the vector DB.
- **Nemotron OCR**: Set `NEMOTRON_OCR_MODEL_DIR` or mount the model at `/workspace/models/nemotron-ocr-v1` (required if the pipeline uses table/chart/infographic extraction).
- **Embedding**: To use a remote NIM instead of a local model, set `ONLINE_EMBED_ENDPOINT` (e.g. `http://embedding:8000/v1`).

### Optional env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `ONLINE_LANCEDB_URI` | `/data/lancedb` | LanceDB directory |
| `ONLINE_LANCEDB_TABLE` | `nv-ingest` | Table name |
| `ONLINE_EMBED_ENDPOINT` | (none) | Embedding NIM URL (optional) |
| `NEMOTRON_OCR_MODEL_DIR` | `/workspace/models/nemotron-ocr-v1` | Nemotron OCR v1 model path |

## Use the API

```bash
# Health
curl http://localhost:7670/health

# Ingest a PDF
curl -X POST http://localhost:7670/ingest \
  -F "file=@document.pdf" \
  -H "X-Source-Path:/path/document.pdf"
```

Or use the CLI from the host:

```bash
retriever online submit document.pdf --base-url http://localhost:7670
```
