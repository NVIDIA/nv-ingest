# Retriever

RAG ingestion pipeline for PDFs: extract structure (text, tables, charts, infographics), embed, optionally upload to LanceDB, and run recall evaluation.

## Prerequisites

- **CUDA 13**
- **Python 3.12**
- **UV** (required) — [install UV](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation

Installation is done with **UV** from the **nv-ingest root**. UV manages the environment and dependencies; pip is not supported.

From the repo root:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./retriever
```

This installs the retriever in editable mode and its in-repo dependencies. Core dependencies (see `retriever/pyproject.toml`) include Ray, pypdfium2, pandas, LanceDB, PyYAML, torch, transformers, and the Nemotron packages (page-elements, graphic-elements, table-structure). The retriever also depends on the sibling packages `nv-ingest`, `nv-ingest-api`, and `nv-ingest-client` in this repo.

## Quick start

From the nv-ingest root, install with UV then run the batch pipeline with a directory of PDFs:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./retriever
uv run python retriever/src/retriever/examples/batch_pipeline.py /path/to/pdfs
```

Pass the directory that contains your PDFs as the first argument (`input-dir`). For recall evaluation, the pipeline uses `bo767_query_gt.csv` in the current directory by default; override with `--query-csv <path>`. Recall is skipped if the query CSV file does not exist. By default, per-query details (query, gold, hits) are printed; use `--no-recall-details` to print only the missed-gold summary and recall metrics. To use an existing Ray cluster, pass `--ray-address auto`.

For **HTML** or **text** ingestion, use `--input-type html` or `--input-type txt` with the same examples (e.g. `batch_pipeline.py <dir> --input-type html`). HTML files are converted to markdown via markitdown, then chunked with the same tokenizer as .txt. Staged CLI: `retriever html run --input-dir <dir>` writes `*.html_extraction.json`; then `retriever local stage5 run --input-dir <dir> --pattern "*.html_extraction.json"` and `retriever local stage6 run --input-dir <dir>`.

### Starting a Ray cluster

This project uses Ray Data. You can start a Ray cluster yourself to use the dashboard and control GPU usage.

**Cluster with Ray Dashboard**

Start a head node (dashboard is enabled by default on port 8265):

```bash
ray start --head
```

Open the dashboard at **http://127.0.0.1:8265**. Run your pipeline in the same machine; pass `--ray-address auto` to attach to this cluster (e.g. `uv run python retriever/src/retriever/examples/batch_pipeline.py /path/to/pdfs --ray-address auto` or the batch pipeline CLI with `--ray-address auto`).

**Single-GPU cluster (multi-GPU nodes)**

To use only one GPU on a node that has more (e.g. for testing), limit visible devices and start Ray with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```

Then run your pipeline as above (e.g. `--ray-address auto` for the CLI, or `ray_address="auto"` in Python).

### Running multiple NIM service instances on multi-GPU hosts

If you want more than one `page-elements` and `ocr` instance on the same machine, run separate Docker Compose projects and pin each project to a specific physical GPU.

From the `nv-ingest` repo root:

```bash
# GPU 0 stack
GPU_ID=0 \
PAGE_ELEMENTS_HTTP_PORT=8000 PAGE_ELEMENTS_GRPC_PORT=8001 PAGE_ELEMENTS_METRICS_PORT=8002 \
OCR_HTTP_PORT=8019 OCR_GRPC_PORT=8010 OCR_METRICS_PORT=8011 \
docker compose -p ingest-gpu0 up -d page-elements ocr

# GPU 1 stack
GPU_ID=1 \
PAGE_ELEMENTS_HTTP_PORT=8100 PAGE_ELEMENTS_GRPC_PORT=8101 PAGE_ELEMENTS_METRICS_PORT=8102 \
OCR_HTTP_PORT=8119 OCR_GRPC_PORT=8110 OCR_METRICS_PORT=8111 \
docker compose -p ingest-gpu1 up -d page-elements ocr
```

The `-p` values create isolated stacks, while `GPU_ID` pins each stack to a different physical GPU. Distinct host ports prevent collisions and keep both stacks externally accessible.

Useful checks:

```bash
docker compose -p ingest-gpu0 ps
docker compose -p ingest-gpu1 ps
```

To stop and remove both stacks:

```bash
docker compose -p ingest-gpu0 down
docker compose -p ingest-gpu1 down
```
