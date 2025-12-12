# llama-3.2-nv-rerankqa-1b-v2 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2` reranker. Everything lives here to keep upstream source untouched.

## What nv-ingest expects
- HTTP endpoint on `${NIM_HTTP_API_PORT:-8000}` that accepts `POST /v1/ranking` (plus compatibility alias `/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking`).
- Request body: `{"model":"nvidia/llama-3.2-nv-rerankqa-1b-v2","query":{"text": "<query>"},"passages":[{"text":"..."},...],"truncate":"END"}`. `model` defaults to the server’s model when omitted. `truncate` supports `END` (default), `START`, or `NONE`.
- Response shape: `{"rankings":[{"index":0,"logit":6.8},...]}` sorted by descending `logit`.
- Health: `GET /v1/health/live` and `/v1/health/ready` plus Triton `/v2/health/{live,ready}`. Metadata: `GET /v1/models`, `/v1/metadata`, and Triton `/v2/models/*` (model config/metadata/ready).
- Batch limit discovered from `TRITON_MODEL_CONFIG_PATH` when present or `NIM_TRITON_MAX_BATCH_SIZE` (default `64`); requests exceeding the limit return 400/INVALID_ARGUMENT.
- Auth: Bearer token is required for HTTP and gRPC (tokens from `NGC_API_KEY`/`NVIDIA_API_KEY`/`NIM_NGC_API_KEY`, toggled by `NIM_REQUIRE_AUTH`).
- gRPC/Triton: exposes `ModelInfer` (query + `PASSAGES` bytes tensors, scores as FP32), model control (optional via `NIM_TRITON_ENABLE_MODEL_CONTROL`), statistics, and repository index.
- Metrics: Prometheus on `${NIM_TRITON_METRICS_PORT:-8002}` and `/metrics`; OTEL tracing is optional (`NIM_ENABLE_OTEL` + `NIM_OTEL_*`).

## Model source
- Uses the open model weights from https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2.
- Applies the official prompt template (`question:{query} \n \n passage:{passage}`) and returns raw logits per passage.

## Running locally
Prereqs: Python 3.10+, `git-lfs`, and an appropriate PyTorch build for your hardware (GPU recommended).

```bash
cd nim/nvidia/llama-3.2-nv-rerankqa-1b-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a torch build that matches your CUDA version (e.g., via https://pytorch.org/get-started/locally/)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124  # adjust as needed
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Quick smoke test:
```bash
curl -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/ranking \
  -H "content-type: application/json" \
  -d '{"model":"nvidia/llama-3.2-nv-rerankqa-1b-v2","query":{"text":"which way should i go?"},"passages":[{"text":"two roads diverged in a yellow wood..."},{"text":"then took the other, as just as fair..."}],"truncate":"END"}'
```

## Compose/K8s alignment
- Exposes HTTP `${NIM_HTTP_API_PORT:-8000}`, gRPC `${NIM_TRITON_GRPC_PORT:-8001}`, and metrics `${NIM_TRITON_METRICS_PORT:-8002}`. Compatible with the `docker-compose.yaml` `reranker` service and Helm values (`llama_3_2_nv_rerankqa_1b_v2`).
- Mirrors key envs: `NIM_HTTP_API_PORT`, `NIM_TRITON_GRPC_PORT`, `NIM_TRITON_METRICS_PORT`, `NIM_TRITON_MAX_BATCH_SIZE`, `NIM_TRITON_RATE_LIMIT`, `NIM_TRITON_LOG_VERBOSE`, `NIM_TRITON_ENABLE_MODEL_CONTROL`, `NIM_REQUIRE_AUTH`, `MODEL_VERSION`, `LOG_LEVEL`, `MAX_SEQUENCE_LENGTH`, optional `NIM_ENABLE_OTEL` and `NIM_OTEL_*`.

## Known deltas vs. closed NIM
- Only the reranking API surface (`/v1/ranking` and the compatibility path) is implemented; no async job polling endpoints are present.
