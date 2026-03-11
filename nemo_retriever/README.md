# Quick Start for NeMo Retriever Library

NeMo Retriever Library is a retrieval-augmented generation (RAG) ingestion pipeline for documents that can parse text, tables, charts, and infographics. NeMo Retriever Library parses documents, creates embeddings, optionally stores embeddings in LanceDB, and performs recall evaluation.

This quick start guide shows how to run NeMo Retriever Library in library mode, directly from your application, without Docker. In library mode, NeMo Retriever Library supports two deployment options:
- Load Hugging Face models locally on your GPU.
- Use locally deployed NeMo Retriever NIM endpoints for embedding and OCR.

You’ll set up a CUDA 13–compatible environment, install the library and its dependencies, and run GPU‑accelerated ingestion pipelines that convert PDFs, HTML, plain text, and audio into vector embeddings stored in LanceDB, with optional Ray‑based scaling and built‑in recall benchmarking.

## Prerequisites

> **Warning:** The `online` and `fused` run modes are experimental and not fully supported. They may be incomplete, unstable, or subject to breaking changes. Use `batch` or `inprocess` modes for production workloads.

Before you start, make sure your system meets the following requirements:

- The host is running CUDA 13.x so that `libcudart.so.13` is available.
- Your GPUs are visible to the system and compatible with CUDA 13.x.
​
If optical character recognition (OCR) fails with a `libcudart.so.13` error, install the CUDA 13 runtime for your platform and update `LD_LIBRARY_PATH` to include the CUDA lib64 directory, then rerun the pipeline. 

For example, the following command can be used to update the `LD_LIBRARY_PATH` value.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

## Setup your environment

Complete the following steps to setup your environment. You will create and activate isolated Python and project virtual environments, install the NeMo Retriever Library and its CUDA 13–compatible GPU dependencies, and then run the ingestion, benchmarking, and audio pipelines to validate the full setup.

1. Create and activate the NeMo Retriever Library environment

Before installing NeMo Retriever Library, create an isolated Python environment so its dependencies do not conflict with other projects on your system. In this step, you set up a new virtual environment and activate it so that all subsequent installs are scoped to NeMo Retriever Library.

In your terminal, run the following commands from any location.

```bash
uv venv .nemotron-ocr-test --python 3.12
source .nemotron-ocr-test/bin/activate
uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nemo-retriever
```
This creates a dedicated Python environment and installs the `nemo-retriever` PyPI package, the canonical distribution for the NeMo Retriever Library.

2. Install NeMo Retriever Library and Dependencies

Install the latest nightly builds of the NeMo Retriever Library so you can test the most recent features and fixes before they are rolled into a stable release. 

In this step, you install the core library, its API layer, and the client package, ensuring the ingestion pipeline and related tooling all come from a consistent, up‑to‑date version set.

In your terminal, run the following commands from any location.


```bash
uv pip install -i https://test.pypi.org/simple nemo-retriever==2026.3.3.dev20260303 nemo-retriever-api==2026.3.3.dev20260303 nemo-retriever-client==2026.3.3.dev20260303 --no-deps
uv pip install nemo-retriever nemo-retriever-api nemo-retriever-client
```
These packages provide the ingestion pipeline and APIs used by NeMo Retriever Library until everything is consolidated under the single `nemo-retriever` surface.

3. Install CUDA 13 builds of Torch and Torchvision

To ensure NeMo Retriever Library’s OCR and GPU‑accelerated components run correctly on your system, you need PyTorch and TorchVision builds that are compiled for CUDA 13. In this step, you uninstall any existing Torch/TorchVision packages and reinstall them from a dedicated CUDA 13.0 wheel index so they link against the same CUDA runtime as the rest of your pipeline.

Use the CUDA 13.0 wheels from the dedicated index by running the following command.

```bash
uv pip uninstall torch torchvision
uv pip install torch==2.9.1 torchvision -i https://download.pytorch.org/whl/cu130
```
This ensures the OCR and GPU‑accelerated components in NeMo Retriever Library run against the right CUDA runtime.

4. Set up the NeMo Retriever Library project environment

For local development, you need a project-scoped environment tied directly to the NeMo Retriever Library source tree. 

In this step, you create a virtual environment in the repo itself and install the `nemo_retriever` package in editable mode so you can run examples, tweak the code, and pick up changes without reinstallation.

Run the following code from the NeMo Retriever Library repo root (NVIDIA/NeMo-Retriever).

```bash
cd /path/to/NeMo-Retriever
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
```
This creates a project-local environment and installs the `nemo_retriever` Python package in editable mode for running the examples.

5. Run the batch pipeline on PDFs

In this procedure, you run the end‑to‑end NeMo Retriever Library batch pipeline to ingest a collection of PDFs and generate embeddings for them. Pointing the script at a directory of PDF files lets the pipeline handle parsing, OCR, embedding, optional LanceDB upload, and (if configured) recall evaluation in a single command.

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

By doing this the pipeline connects to the running Ray deployment instead of starting a new one.

6. Ingest HTML or plain text instead of PDFs

If your documents aren't stored as PDFs, you can point the same NeMo Retriever Library batch pipeline to directories of HTML or plain text files instead. 

In this step, you either pass an input‑type flag to the batch example for a simple one‑shot run, or use a staged HTML CLI flow for more control over each phase of ingestion.

To run the batch example directly on HTML or plain text, use one of the following commands in your terminal.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py <dir> --input-type html
```
or

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py <dir> --input-type txt
```
Pass the directory that contains your PDFs as the first argument (`input-dir`). For recall evaluation, the pipeline uses `bo767_query_gt.csv` in the current directory by default; override with `--query-csv <path>`. For document-level recall, use `--recall-match-mode pdf_only` with `query,expected_pdf` data. Recall is skipped if the query file does not exist. By default, per-query details (query, gold, hits) are printed; use `--no-recall-details` to print only the missed-gold summary and recall metrics. To use an existing Ray cluster, pass `--ray-address auto`. If OCR fails with a missing `libcudart.so.13`, install the CUDA 13 runtime and set `LD_LIBRARY_PATH` as shown above.

Use `--input-type html` for HTML files and `--input-type txt` for plain text.  HTML inputs are converted to markdown using the same tokenizer and chunking strategy used for `.txt` ingestion.

For more step‑by‑step control with HTML, use the following staged HTML CLI flow commands instead.

```bash
retriever html run --input-dir <dir>
retriever local stage5 run --input-dir <dir> --pattern "*.html_extraction.json"
retriever local stage6 run --input-dir <dir>
```
`retriever html run` parses the HTML and writes `*.html_extraction.json` sidecar files into the input directory. `retriever local stage5 run` performs downstream processing over those JSON files, and `retriever local stage6 run` completes the final ingestion stages, such as embedding and optional upload, using the same core extraction pipeline.

- Config files:
  - `nemo_retriever/harness/test_configs.yaml`
  - `nemo_retriever/harness/nightly_config.yaml`
- CLI entrypoint is nested under `retriever harness`.
- First pass is LanceDB-only and enforces recall-required pass/fail by default.
- Single-run artifact directories default to `<dataset>_<timestamp>`.
- Dataset-specific recall adapters are supported via config:
  - `recall_adapter: none` (default passthrough)
  - `recall_adapter: page_plus_one` (convert zero-indexed `page` CSVs to `pdf_page`)
  - `recall_adapter: financebench_json` (convert FinanceBench JSON to `query,expected_pdf`)
  - `recall_match_mode: pdf_page|pdf_only` controls recall matching mode.
- Dataset presets configured under `/datasets/nv-ingest/...` will fall back to `/raid/$USER/...` when the dataset is not present in `/datasets`.
- Relative `query_csv` entries in harness YAML resolve from the config file directory first, then fall back to the repo root.
- The default `financebench` dataset preset now points at `data/financebench_train.json` and enables recall out of the box.

After you’ve finished installing and configuring NeMo Retriever Library, it's a good idea to validate the entire pipeline with a small, known dataset. In this step, you run the batch pipeline module against the sample `bo20` dataset to confirm that ingestion, OCR under CUDA 13, embedding, and any configured recall evaluation all run end‑to‑end without errors.

```bash
uv run python -m nemo_retriever.examples.batch_pipeline /datasets/nemo-retriever/bo20
```
This uses the module form of the NeMo Retriever Library batch pipeline example and points it at a sample dataset directory, verifying both ingestion and OCR under CUDA 13.

7. Ingest image files

NeMo Retriever Library can ingest standalone image files through the same detection, OCR, and embedding pipeline used for PDFs. Supported formats are PNG, JPEG, BMP, TIFF, and SVG. SVG support requires the optional `cairosvg` package. Each image is treated as a single page.

To run the batch pipeline on a directory of images, use `--input-type image` to match all supported formats at once.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/images \
  --input-type image
```

You can also pass a single-format shortcut to restrict which files are picked up.

```bash
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/images \
  --input-type png
```

Valid single-format values are `png`, `jpg`, `jpeg`, `bmp`, `tiff`, `tif`, and `svg`.

For in-process mode, build the ingestor chain with `extract_image_files` instead of `extract`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.params import ExtractParams, EmbedParams

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("images/*.png")
    .extract_image_files(
        ExtractParams(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
        )
    )
    .embed()
    .vdb_upload()
    .ingest()
)
```

All `ExtractParams` options (`extract_text`, `extract_tables`, `extract_charts`, `extract_infographics`) apply to image ingestion.

### Render one document as markdown

If you want a readable page-by-page markdown view of a single in-process result, pass the
single-document result from `results[0]` to `nemo_retriever.io.to_markdown`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.io import to_markdown

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("data/multimodal_test.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
)
results = ingestor.ingest()
print(to_markdown(results[0]))
```

Use `to_markdown_by_page(results[0])` when you want a `dict[int, str]` instead of one concatenated
markdown document.

## Benchmark harness

NeMo Retriever Library includes a lightweight benchmark harness that lets you run repeatable evaluations and sweeps without using Docker. [NeMo Retriever Library benchmarking documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/benchmarking/)

1. Configuration

The harness is configured using the following configuration files:

- `nemo_retriever/harness/test_configs.yaml`  
- `nemo_retriever/harness/nightly_config.yaml`  

The CLI entrypoint is nested under `retriever harness`. The first pass is LanceDB‑only and enforces recall‑required pass/fail by default, and single‑run artifact directories default to `<dataset>_<timestamp>`. [NeMo Retriever Library benchmarking documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/benchmarking/)

2. Single run

You can run a single benchmark either from a preset dataset name or a direct path.

Preset dataset name
```bash
# Dataset preset from test_configs.yaml (recall-required example)
retriever harness run --dataset jp20 --preset single_gpu
```
or

# Direct dataset path
retriever harness run --dataset /datasets/nv-ingest/bo767 --preset single_gpu

# Add repeatable run or session tags for later review
retriever harness run --dataset jp20 --preset single_gpu --tag nightly --tag candidate
```

3. Sweep runs

To sweep multiple runs defined in a config file use the following command.

```bash
retriever harness sweep --runs-config nemo_retriever/harness/nightly_config.yaml
```

4. Nightly sessions

To orchestrate a full nightly benchmark session use the following command.

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml
retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml --skip-slack
retriever harness nightly --dry-run
retriever harness nightly --replay nemo_retriever/artifacts/nightly_20260305_010203_UTC
```

`nemo_retriever/harness/nightly_config.yaml` supports a small top-level `preset:` and `slack:`
block alongside `runs:`. Keep the webhook secret out of YAML and source control; provide it only
through the `SLACK_WEBHOOK_URL` environment variable. If the variable is missing, nightly still
runs and writes artifacts but skips the Slack post. `--replay` lets you resend a previous session
directory, run directory, or `results.json` file after fixing webhook access.

For reusable box-local automation, the harness also includes shell entrypoints:

```bash
# One-shot nightly run using the repo-local .retriever env
bash nemo_retriever/harness/run_nightly.sh

# Forever loop that sleeps until the next UTC schedule window, then runs nightly
tmux new-session -d -s retriever-nightly \
  "cd /path/to/nv-ingest && export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...' && \
   bash nemo_retriever/harness/run_nightly_loop.sh"
```

`run_nightly_loop.sh` is intended as a pragmatic fallback for boxes where cron or timers are
unreliable. It does not require an interactive SSH session once launched inside `tmux`, but it is
still less robust than a real scheduler such as `systemd` or a cluster job scheduler.

The `--dry-run` option lets you verify the planned runs without executing them. [NeMo Retriever Library benchmarking documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/benchmarking/)

5. Harness artifacts

Each harness run writes a compact artifact set (no full stdout/stderr log persistence):

- `results.json` (normalized metrics + pass/fail + config snapshot + `run_metadata`)
- `command.txt` (exact invoked command)
- `runtime_metrics/` (Ray runtime summary + timeline files)

Recall metrics in `results.json` are normalized as `recall_1`, `recall_5`, and `recall_10`.
Nightly/sweep rollups intentionally focus on compact `summary_metrics`:

- `pages`
- `ingest_secs`
- `pages_per_sec_ingest`
- `recall_5`

By default, detection totals are embedded into `results.json` under `detection_summary`.
If you want a separate detection file for ad hoc inspection, set `write_detection_file: true` in
`nemo_retriever/harness/test_configs.yaml`.
When tags are supplied with `--tag`, they are persisted in `results.json` and in session rollups for sweep/nightly runs.

`results.json` also includes a nested `run_metadata` block for lightweight environment context:

- `host`
- `gpu_count`
- `cuda_driver`
- `ray_version`
- `python_version`

These fields use best-effort discovery and fall back to `null` or `"unknown"` rather than failing a run.

Sweep/nightly sessions additionally write:

The `runtime_metrics/` directory contains:

When Slack posting is enabled, the nightly summary is built from `session_summary.json` plus each
run's `results.json`, so the on-disk artifacts remain the source of truth even if you need to replay
or troubleshoot a failed post later.

### Runtime metrics interpretation

- **`run.runtime.summary.json`** - run totals (input files, pages, elapsed seconds)  
- **`run.ray.timeline.json`** - detailed Ray execution timeline  
- **`run.rd_dataset.stats.txt`** - Ray dataset stats dump  

Use `results.json` for routine benchmark comparison, and use the files under `runtime_metrics/` when investigating throughput regressions or stage‑level behavior. [NeMo Retriever Library benchmarking documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/benchmarking/)

6. Artifact size profile

Current benchmark runs show that the LanceDB data dominates the artifact footprint:

### Cron / timer setup

For a simple machine-local schedule, run the nightly command from `cron` or a `systemd` timer on the
GPU host that already has dataset access and the retriever environment installed.

Example cron entry:

```bash
0 2 * * * cd /path/to/nv-ingest && source .retriever/bin/activate && \
  export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..." && \
  retriever harness nightly --runs-config nemo_retriever/harness/nightly_config.yaml \
  >> nemo_retriever/artifacts/nightly_cron.log 2>&1
```

If you prefer `systemd`, keep the same command in an `ExecStart=` line and move
`SLACK_WEBHOOK_URL` into an environment file owned by the machine user so the secret stays out of
the repo.

### Artifact size profile

- **`bo20`** - ~9.0 MiB total, ~8.6 MiB LanceDB  
- **`jp20`** - ~36.8 MiB total, ~36.2 MiB LanceDB 

## Audio ingestion pipeline

NeMo Retriever Library also supports audio ingestion alongside documents. Audio pipelines typically follow a chained pattern such as the following.  

```python
.files("mp3/*.mp3").extract_audio(...).embed().vdb_upload().ingest()
```

This can be run in batch, in‑process, or fused mode within NeMo Retriever Library. [NeMo Retriever Library audio extraction documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/audio/)

### ASR options

For automatic speech recognition (ASR), you have the following two options:

- Local: When `audio_endpoints` are not set, the pipeline uses local HuggingFace ASR (`nvidia/parakeet-ctc-1.1b`) through Transformers with NeMo fallback; no NIM or gRPC endpoint is required. [Parakeet CTC 1.1B model on Hugging Face](https://huggingface.co/nvidia/parakeet-ctc-1.1b)
- Remote: When `audio_endpoints` is set (for example, Parakeet NIM or self‑deployed Riva gRPC), the pipeline uses the remote client; set `AUDIO_GRPC_ENDPOINT`, `NGC_API_KEY`, and optionally `AUDIO_FUNCTION_ID`. [NeMo Retriever Library audio extraction documentation (25.6.3)](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/audio/)

See `ingest-config.yaml` (sections `audio_chunk`, `audio_asr`) and audio scripts under `retriever/scripts/` for concrete configuration examples. [NeMo Retriever Library audio extraction documentation](https://docs.nvidia.com/nemo/retriever/latest/extraction/audio/)

## Ray cluster setup

NeMo Retriever Library uses Ray Data for distributed ingestion and benchmarking. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

### Local Ray cluster with dashboard

To start a Ray cluster with the dashboard on a single machine use the following command.

```bash
ray start --head
```

Open `http://127.0.0.1:8265` in your browser for the Ray Dashboard, and run your NeMo Retriever Library pipeline on the same machine with `--ray-address auto` to attach to this cluster. [Connecting to a remote Ray cluster on Kubernetes](https://discuss.ray.io/t/connecting-to-remote-ray-cluster-on-k8s/7460)

### Single‑GPU cluster on multi‑GPU nodes

To restrict Ray to a single GPU on a multi‑GPU node use the following command.

```bash
CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1
```
Then run your pipeline as before with `--ray-address auto` so it connects to this single‑GPU Ray cluster. [NeMo Ray run guide](https://docs.nvidia.com/nemo/run/latest/guides/ray.html)

## Running multiple NIM instances on multi‑GPU hosts

### Resource heuristics (batch mode)

By default, batch mode computes resources using this order:

1. Auto-detected resources (Ray cluster if connected, otherwise local machine)
2. Environment variables
3. Explicit function arguments (highest precedence)

This means defaults are deterministic but easy to override when you need fixed behavior.

### Default behavior

- `cpu_count` / `gpu_count` are detected from Ray (`cluster_resources`) or local host.
- Worker heuristics:
  - `page_elements_workers = gpu_count * page_elements_per_gpu`
  - `detect_workers = gpu_count * ocr_per_gpu`
  - `embed_workers = gpu_count * embed_per_gpu`
  - minimum of `1` per stage
- Stage GPU defaults:
  - If `gpu_count >= 2` and `concurrent_gpu_stage_count == 3`, uses high-overlap values for page-elements/OCR/embed.
  - Otherwise uses `min(max_gpu_per_stage, gpu_count / concurrent_gpu_stage_count)`.

### Override variables

| Variable | Where to set | Meaning |
|---|---|---|
| `override_cpu_count`, `override_gpu_count` | function args | Highest-priority CPU/GPU override |

### Running multiple NIM service instances on multi-GPU hosts

### Start two stacks on separate GPUs

```bash
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
```

The `-p` project names create isolated stacks, `GPU_ID` pins each stack to a specific physical GPU, and distinct host ports avoid collisions between the services.  

### Check and tear down stacks

To verify that both stacks are running use the following command.

```bash
docker compose -p retriever-gpu0 ps
docker compose -p retriever-gpu1 ps
```

To stop and remove both stacks use the following command.

```bash
docker compose -p retriever-gpu0 down
docker compose -p retriever-gpu1 down
```
