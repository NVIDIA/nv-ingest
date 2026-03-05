# NeMo Retriever Library

Nemo Retriever Library is a RAG ingestion pipeline for PDFs that extracts document structure (text, tables, charts, infographics), generates embeddings, optionally uploads them to LanceDB, and then runs recall evaluation.

## 1. Prerequisites

- The host must be running **CUDA 13.x** (so that `libcudart.so.13` is available).
- Ensure GPUs are visible and compatible with CUDA 13.x.

If OCR fails with a `libcudart.so.13` error, install the CUDA 13 runtime for your platform and set `LD_LIBRARY_PATH` to include the CUDA `lib64` directory before rerunning the pipeline.

---

2. Create and activate the NeMo Retriever environment

From anywhere:

```bash
uv venv .nemotron-ocr-test --python 3.12
source .nemotron-ocr-test/bin/activate
uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nemo-retriever
```
This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library.

## 3. Install NeMo Retriever Library (nightly) and dependencies

Install the current nightly NeMo Retriever Library–related packages.

```bash
uv pip install -i https://test.pypi.org/simple nemo-retriever==2026.3.3.dev20260303 nemo-retriever-api==2026.3.3.dev20260303 nemo-retriever-client==2026.3.3.dev20260303 --no-deps
uv pip install nemo-retriever nemo-retriever-api nemo-retriever-client
```
These packages provide the ingestion pipeline and APIs used by NeMo Retriever Library until everything is consolidated under the single `nemo-retriever` surface.

## 4. Install CUDA 13 builds of Torch and Torchvision

Use the CUDA 13.0 wheels from the dedicated index by running the following command.

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.9.1 torchvision -i https://download.pytorch.org/whl/cu130
```
This ensures the OCR and GPU‑accelerated components in NeMo Retriever Library run against the right CUDA runtime.

## 5. Set up the NeMo Retriever Library project environment

Run the following code from the NeMo Retriever Library repo root (NVIDIA/NeMo-Retriever).

```bash
cd /path/to/NeMo-Retriever
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
```
This creates a project-local environment and installs the `nemoretriever` Python package in editable mode for running the examples.

## 6. Run the batch pipeline on PDFs

Run the batch pipeline script and point it at the directory that contains your PDFs using the following command.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs
```

The first positional argument is the `input-dir`, the directory with the PDF files to ingest.

For recall evaluation, the pipeline uses bo767_query_gt.csv from the current working directory by default; you can override this by running the following command.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs \
  --query-csv /path/to/custom_query_gt.csv
```

If the specified query CSV does not exist, recall evaluation is skipped automatically and only the ingestion process runs.

By default, the pipeline prints per‑query details (query text, gold answers, and hits); use `--no-recall-details` to show only the missed‑gold summary and overall recall metrics.

To reuse an existing Ray cluster, append --ray-address using the following command.

```bash
--ray-address auto
```

By doing so the pipeline connects to the running Ray deployment instead of starting a new one.

7. Ingest HTML or plain text instead of PDFs
To run the same batch example on HTML or plain text:

bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py <dir> --input-type html
# or
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py <dir> --input-type txt
Use --input-type html for HTML files and --input-type txt for plain text.

HTML inputs are converted to markdown using markitdown, then chunked with the same tokenizer used for .txt ingestion.

Staged HTML CLI flow
For a more staged CLI workflow with HTML:

```bash
retriever html run --input-dir <dir>
retriever local stage5 run --input-dir <dir> --pattern "*.html_extraction.json"
retriever local stage6 run --input-dir <dir>
```
retriever html run writes *.html_extraction.json sidecar files into the input directory.

retriever local stage5 run performs downstream processing over those JSON files.

retriever local stage6 run completes the final ingestion stages (such as embedding and upload, depending on configuration).

8. Quick end‑to‑end test
After setup, run a small batch to confirm everything works:

bash
uv run python -m nemo_retriever.examples.batch_pipeline /datasets/nemo-retriever/bo20
This uses the module form of the NeMo Retriever batch pipeline example and points it at a sample dataset directory, verifying both ingestion and OCR under CUDA 13.

Benchmark harness (run, sweep, nightly)
The NeMo Retriever Library includes a lightweight benchmark harness for orchestration without Docker.

Config files:

nemo_retriever/harness/test_configs.yaml

nemo_retriever/harness/nightly_config.yaml

CLI entrypoint is nested under retriever harness.

First pass is LanceDB-only and enforces recall-required pass/fail by default.

Single-run artifact directories default to <dataset>_<timestamp>.

Single run
bash
# Dataset preset from test_configs.yaml (recall-required example)
retriever harness run --dataset jp20 --preset single_gpu

# Direct dataset path
retriever harness run --dataset /datasets/nemo-retriever/bo767 --preset single_gpu
Sweep runs
bash
retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
Nightly session
bash
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --dry-run
Harness artifacts
Each run writes a compact artifact set (no full stdout/stderr log persistence):

results.json (normalized metrics + pass/fail + config snapshot)

command.txt (exact invoked command)

runtime_metrics/ (Ray runtime summary + timeline files)

By default, detection totals are embedded into results.json under detection_summary; to emit a separate detection file set write_detection_file: true in nemo_retriever/harness/test_configs.yaml.

Sweep/nightly sessions additionally write:

session_summary.json (overall pass/fail rollup).

Runtime metrics interpretation
runtime_metrics/ includes:

run.runtime.summary.json: run totals (input files, pages, elapsed seconds)

run.ray.timeline.json: detailed Ray execution timeline

run.rd_dataset.stats.txt: Ray dataset stats dump

For routine benchmark comparison, use results.json; use runtime_metrics/ when investigating throughput regressions or stage-level behavior.

Artifact size profile
Current runs show LanceDB data dominates footprint:

bo20: ~9.0 MiB total, ~8.6 MiB LanceDB

jp20: ~36.8 MiB total, ~36.2 MiB LanceDB

Audio pipeline
Audio ingestion uses:

python
.files("mp3/*.mp3").extract_audio(...).embed().vdb_upload().ingest()
in batch, inprocess, or fused mode within NeMo Retriever.

ASR options:

Local: When audio_endpoints are not set, the pipeline uses local HuggingFace ASR (nvidia/parakeet-ctc-1.1b) via Transformers with NeMo fallback; no NIM or gRPC endpoint required.

Remote: When audio_endpoints is set (e.g., Parakeet NIM or self-deployed Riva gRPC), the pipeline uses the remote client; set AUDIO_GRPC_ENDPOINT, NGC_API_KEY, and optionally AUDIO_FUNCTION_ID.

See ingest-config.yaml (audio_chunk, audio_asr) and audio scripts under retriever/scripts/ for examples.

Starting a Ray cluster
NeMo Retriever uses Ray Data.

Cluster with Ray Dashboard

bash
ray start --head
Open http://127.0.0.1:8265 for the dashboard and run your pipeline on the same machine with --ray-address auto to attach.

Single-GPU cluster on multi-GPU nodes

bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
Then run your pipeline as above with --ray-address auto.

Running multiple NIM service instances on multi-GPU hosts
To run more than one page-elements and ocr instance on a host, run separate Docker Compose projects and pin each project to a specific GPU:

bash
# GPU 0 stack
GPU_ID=0 \
PAGE_ELEMENTS_HTTP_PORT=8000 PAGE_ELEMENTS_GRPC_PORT=8001 PAGE_ELEMENTS_METRICS_PORT=8002 \
OCR_HTTP_PORT=8019 OCR_GRPC_PORT=8010 OCR_METRICS_PORT=8011 \
docker compose -p retriever-gpu0 up -d page-elements ocr

# GPU 1 stack
GPU_ID=1 \
PAGE_ELEMENTS_HTTP_PORT=8100 PAGE_ELEMENTS_GRPC_PORT=8101 PAGE_ELEMENTS_METRICS_PORT=8102 \
OCR_HTTP_PORT=8119 OCR_GRPC_PORT=8110 OCR_METRICS_PORT=8111 \
docker compose -p retriever-gpu1 up -d page-elements ocr
The -p values create isolated stacks, while GPU_ID pins each stack to a different physical GPU and distinct host ports avoid collisions.

Useful checks:

bash
docker compose -p retriever-gpu0 ps
docker compose -p retriever-gpu1 ps
To stop and remove both stacks:

bash
docker compose -p retriever-gpu0 down
docker compose -p retriever-gpu1 down