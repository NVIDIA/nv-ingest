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
uv pip install -e ./retriever
```

This installs the retriever in editable mode and its in-repo dependencies. Core dependencies (see `retriever/pyproject.toml`) include Ray, pypdfium2, pandas, LanceDB, PyYAML, torch, transformers, and the Nemotron packages (page-elements, graphic-elements, table-structure). The retriever also depends on the sibling packages `nv-ingest`, `nv-ingest-api`, and `nv-ingest-client` in this repo.

## Quick start

From the nv-ingest root, install with UV then run the batch pipeline with a directory of PDFs:

```bash
cd /path/to/nv-ingest
uv pip install -e ./retriever
uv run python retriever/src/retriever/examples/batch_pipeline.py /path/to/pdfs
```

Pass the directory that contains your PDFs as the first argument (`input-dir`). For recall evaluation, the pipeline uses `bo767_query_gt.csv` in the current directory by default; override with `--query-csv <path>`. Recall is skipped if the query CSV file does not exist. By default, per-query details (query, gold, hits) are printed; use `--no-recall-details` to print only the missed-gold summary and recall metrics. To use an existing Ray cluster, pass `--ray-address auto`.

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

## Ways to run

### 1. In-process (Python API)

Single process, no Ray. Good for local dev and small runs. Uses local models when endpoints are not set.

```python
from retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("/path/to/*.pdf")
    .extract(method="pdfium", extract_text=True, extract_tables=True, extract_charts=True, extract_infographics=True)
    .embed(model_name="nemo_retriever_v1")
    .vdb_upload(lancedb_uri="lancedb", table_name="nv-ingest", overwrite=False, create_index=True)
)
ingestor.ingest(show_progress=True)
```

Or run the inprocess pipeline example: `uv run python retriever/src/retriever/examples/inprocess_pipeline.py /path/to/pdfs` (optional: `--query-csv`, `--no-recall-details`). It does not use Ray.

### 2. Staged CLI (stage1 → stage7)

Run the pipeline as discrete file-based stages. Each stage reads/writes JSON/Parquet so you can inspect or re-run steps. Configs: `pdf_stage_config.yaml`, `infographic_stage_config.yaml`, `table_stage_config.yaml`, `chart_stage_config.yaml`, `embedding_stage_config.yaml`.

| Stage | Purpose |
|-------|---------|
| stage1 | PDF → primitives (`retriever pdf` / page-elements config) |
| stage2 | Infographic extraction |
| stage3 | Table extraction |
| stage4 | Chart extraction |
| stage5 | Text embeddings |
| stage6 | Upload to LanceDB |
| stage7 | Query + recall@k |

Example chain (replace paths and configs as needed):

```bash
retriever local stage1 page-elements --config retriever/pdf_stage_config.yaml
retriever local stage2 run --config retriever/infographic_stage_config.yaml --input <stage1_output_dir>
retriever local stage3 run --input <stage2_output_dir>
retriever local stage4 run --input <stage3_output_dir>
retriever local stage5 run --input-dir <dir> --endpoint-url http://localhost:8012/v1
retriever local stage6 run --input-dir <dir>
retriever local stage7 run --query-csv <query.csv> --embedding-endpoint http://localhost:8012/v1
```

### 3. Ray batch pipeline

For large-scale batch runs:

```bash
python retriever/src/retriever/ingest-batch-pipeline.py \
  --input-dir /path/to/pdfs \
  --method pdfium \
  --extract-text --extract-tables --extract-charts --extract-infographics \
  --text-depth page \
  --vdb-upload \
  --output-dir /path/to/output
```

Uses `retriever/ingest-config.yaml` when present.

## Other commands

- `retriever pdf` — PDF extraction (same as stage1).
- `retriever image render` — Visualize page-element (or other) detections on images.
- `retriever compare json` / `retriever compare results` — Compare JSON or result outputs.
- `retriever recall` — Query LanceDB and compute recall (e.g. `recall-with-main`).
