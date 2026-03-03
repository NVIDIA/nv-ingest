# Retriever

RAG ingestion pipeline for PDFs: extract structure (text, tables, charts, infographics), embed, optionally upload to LanceDB, and run recall evaluation.

## Prerequisites

- **CUDA 13** — required for **OCR** (Nemotron); text extraction and other stages may work without it.
- **Python 3.12**
- **UV** (required) — [install UV](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation

Installation is done with **UV** from the **nv-ingest root** using **uv pip install** (no lockfile/sync so optional extras stay independent). Pip is not supported.

From the repo root:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
```

This installs the retriever in editable mode and its in-repo dependencies. Core dependencies (see `nemo_retriever/pyproject.toml`) include Ray, pypdfium2, pandas, LanceDB, PyYAML, torch, transformers (4.x), vLLM 0.16, and the Nemotron packages (page-elements, graphic-elements, table-structure). The retriever also depends on the sibling packages `nv-ingest`, `nv-ingest-api`, and `nv-ingest-client` in this repo.

### Optional: ASR extra (local Parakeet)

For **local ASR** (nvidia/parakeet-ctc-1.1b with `audio_endpoints` unset), install the `[asr]` extra. This pulls in `transformers>=5`, `soundfile`, and `scipy` and is mutually exclusive with the default stack (vLLM 0.16 uses transformers&lt;5):

```bash
uv pip install -e "./nemo_retriever[asr]"
```

Docker: build with ASR support using `--build-arg INSTALL_ASR=1`.

### OCR and CUDA 13 runtime

The Nemotron OCR native extension requires **libcudart.so.13** (CUDA 13 runtime). If you see:

```text
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

your system CUDA Toolkit is missing or older than 13. Install CUDA 13 runtime support on your host, or run the retriever Docker image which includes the required runtime.

If CUDA libraries are installed in a non-standard path, expose them explicitly:
   ```bash
   export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Quick start

From the nv-ingest root, install with UV then run the batch pipeline with a directory of PDFs:

```bash
cd /path/to/nv-ingest
uv venv .retriever
source .retriever/bin/activate
uv pip install -e ./nemo_retriever
uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs
```

Pass the directory that contains your PDFs as the first argument (`input-dir`). For recall evaluation, the pipeline uses `bo767_query_gt.csv` in the current directory by default; override with `--query-csv <path>`. Recall is skipped if the query CSV file does not exist. By default, per-query details (query, gold, hits) are printed; use `--no-recall-details` to print only the missed-gold summary and recall metrics. To use an existing Ray cluster, pass `--ray-address auto`. If OCR fails with a missing `libcudart.so.13`, install the CUDA 13 runtime and set `LD_LIBRARY_PATH` as shown above.

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

Open the dashboard at **http://127.0.0.1:8265**. Run your pipeline in the same machine; pass `--ray-address auto` to attach to this cluster (e.g. `uv run python nemo_retriever/src/nemo_retriever/examples/batch_pipeline.py /path/to/pdfs --ray-address auto` or the batch pipeline CLI with `--ray-address auto`).

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

## Embedding backends

Embeddings can be served by a **remote HTTP endpoint** (NIM, vLLM, or any OpenAI-compatible server) or by a **local HuggingFace model** when no endpoint is configured.

- **Config**: Set `embedding_nim_endpoint` in `ingest-config.yaml` or stage config (e.g. `http://localhost:8000/v1`). Leave empty or null to use the local HF embedder.
- **CLI**: Use `--embed-invoke-url` (inprocess/batch pipelines) or `--embedding-endpoint` / `--embedding-http-endpoint` (recall CLI) to point at a remote server.

### Using vLLM for embeddings

You can serve an embedding model with [vLLM](https://docs.vllm.ai/) and point the retriever at it. vLLM exposes an OpenAI-compatible `/v1/embeddings` API. Set the embedding endpoint to the vLLM base URL (e.g. `http://localhost:8000/v1`).

**vLLM compatibility**: The default NIM-style client sends `input_type` and `truncate` in the request body; some vLLM versions or configs may not accept these. When using a **vLLM** server, enable the vLLM-compatible payload:

- **Ingest**: `--embed-use-vllm-compat` (inprocess pipeline) or set `embed_use_vllm_compat: true` in `EmbedParams`.
- **Recall**: `--embedding-use-vllm-compat` (recall CLI).

This sends only `model`, `input`, and `encoding_format` (minimal OpenAI-compatible payload).

### llama-nemotron-embed-1b-v2 with vLLM

For **nvidia/llama-nemotron-embed-1b-v2**, follow the model’s official vLLM instructions:

1. Use **vllm==0.11.0**.
2. Clone the [model repo](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) and **overwrite `config.json` with `config_vllm.json`** from that repo.
3. Start the server (replace `<path_to_the_cloned_repository>` and `<num_gpus_to_use>`):

   ```bash
   vllm serve \
       <path_to_the_cloned_repository> \
       --trust-remote-code \
       --runner pooling \
       --model-impl vllm \
       --override-pooler-config '{"pooling_type": "MEAN"}' \
       --data-parallel-size <num_gpus_to_use> \
       --dtype float32 \
       --port 8000
   ```

4. Set the retriever embedding endpoint to `http://localhost:8000/v1` and use `--embed-use-vllm-compat` / `--embedding-use-vllm-compat` as above.

See the [model README](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) for the canonical vLLM setup and client example.

### Using vLLM offline batched inference

You can run the same embedding model (e.g. llama-nemotron-embed-1b-v2) **without a vLLM server** by using vLLM’s Python API for batched inference. This loads the model in-process and runs `LLM.embed()` in batches.

- **When to use**: No server to run; same model and behavior as vLLM server; good for batch ingest or recall in a single process.
- **Install**: vLLM is an optional dependency. Install with `pip install -e ".[vllm]"` or `uv pip install -e ".[vllm]"` (requires vllm>=0.11.0 for llama-nemotron-embed-1b-v2).
- **Model path**: You can pass a HuggingFace model id (e.g. `nvidia/llama-nemotron-embed-1b-v2`) or a **local path**. For llama-nemotron-embed-1b-v2, a local clone with `config.json` replaced by `config_vllm.json` (from the model repo) may be required for vLLM to load it correctly.
- **Ingest**: Set `embed_use_vllm_offline: true` in `EmbedParams` or use `--embed-use-vllm-offline` in the inprocess pipeline. Optionally set `embed_model_path` (or `--embed-model-path`) to a local model path.
- **Recall**: Use `--embedding-use-vllm-offline` (recall CLI). Optionally `--embedding-vllm-model-path` to override the model path.
