# Retriever

RAG ingestion pipeline for PDFs: extract structure (text, tables, charts, infographics), embed, optionally upload to LanceDB, and run recall evaluation.

## Prerequisites

- **CUDA 13** — required for **OCR** (Nemotron); text extraction and other stages may work without it.
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

### OCR and CUDA 13 runtime (conda)

The Nemotron OCR native extension requires **libcudart.so.13** (CUDA 13 runtime). If you see:

```text
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

your system CUDA Toolkit is missing or older than 13. Use the provided conda environment to get the CUDA 13 runtime and UV without changing the system CTK:

1. From the **nv-ingest root**:
   ```bash
   conda env create -f conda/environments/retriever_libcudart.yml
   ```
   (If the env already exists, use `conda env update -f conda/environments/retriever_libcudart.yml --name retriever_libcudart` to refresh it.)

2. Activate the environment:
   ```bash
   conda activate retriever_libcudart
   ```

3. Expose the conda env's CUDA runtime so the OCR extension can load it (needed because conda does not set `LD_LIBRARY_PATH` for this env by default):
   ```bash
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
   ```

4. Follow the **Installation** steps above unchanged: `uv venv .retriever`, `source .retriever/bin/activate`, `uv pip install -e ./retriever`. The conda env supplies the `uv` binary and the CUDA 13 runtime (the dynamic linker finds `libcudart.so.13` in the conda env's `lib`).

If your system already has a suitable CUDA 13 (or you are not using OCR), you can use the UV-only path above without the conda env.

## Quick start

From the nv-ingest root, install with UV then run the batch pipeline with a directory of PDFs:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./retriever
uv run python retriever/src/retriever/examples/batch_pipeline.py /path/to/pdfs
```

Pass the directory that contains your PDFs as the first argument (`input-dir`). For recall evaluation, the pipeline uses `bo767_query_gt.csv` in the current directory by default; override with `--query-csv <path>`. Recall is skipped if the query CSV file does not exist. By default, per-query details (query, gold, hits) are printed; use `--no-recall-details` to print only the missed-gold summary and recall metrics. To use an existing Ray cluster, pass `--ray-address auto`. If OCR fails with a missing `libcudart.so.13`, use the conda-based workflow under **OCR and CUDA 13 runtime (conda)** above.

For **HTML** or **text** ingestion, use `--input-type html` or `--input-type txt` with the same examples (e.g. `batch_pipeline.py <dir> --input-type html`). HTML files are converted to markdown via markitdown, then chunked with the same tokenizer as .txt. Staged CLI: `retriever html run --input-dir <dir>` writes `*.html_extraction.json`; then `retriever local stage5 run --input-dir <dir> --pattern "*.html_extraction.json"` and `retriever local stage6 run --input-dir <dir>`.

### Audio pipeline

Audio ingestion uses `.files("mp3/*.mp3").extract_audio(...).embed().vdb_upload().ingest()` in batch, inprocess, or fused mode. **ASR** (speech-to-text) can be:

- **Local**: When `audio_endpoints` are not set (e.g. `[null, null]` in `ingest-config.yaml` under `audio_asr`), the pipeline uses the local HuggingFace model **nvidia/parakeet-ctc-1.1b** (Transformers first, with NeMo fallback if needed). No NIM or gRPC endpoint required.
- **Remote**: When `audio_endpoints` is set (e.g. Parakeet NIM or self-deployed Riva gRPC), the pipeline uses the remote client. Set `AUDIO_GRPC_ENDPOINT`, `NGC_API_KEY`, and optionally `AUDIO_FUNCTION_ID` for NGC cloud ASR.

See `ingest-config.yaml` (`audio_chunk`, `audio_asr`) and the audio scripts under `retriever/scripts/` for examples.

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
