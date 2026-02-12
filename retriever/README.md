# Retriever

RAG ingestion pipeline for PDFs: extract structure (text, tables, charts, infographics), embed, optionally upload to LanceDB, and run recall evaluation.

## Install

From the **monorepo root** (nv-ingest), install the retriever and its in-repo deps:

```bash
cd /path/to/nv-ingest
uv pip install -e ./retriever
```

Or with pip:

```bash
pip install -e ./retriever
```

Core dependencies (see `retriever/pyproject.toml`) include Ray, pypdfium2, pandas, LanceDB, PyYAML, torch, transformers, and the Nemotron packages (page-elements, graphic-elements, table-structure). The retriever also depends on the sibling packages `nv-ingest`, `nv-ingest-api`, and `nv-ingest-client` in this repo.

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

See `retriever/src/retriever/examples/inprocess_pipeline_jp20.py` for a full example including recall evaluation.

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
