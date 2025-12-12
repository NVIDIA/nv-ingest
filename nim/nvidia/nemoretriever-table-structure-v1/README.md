# nemoretriever-table-structure-v1 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/nemoretriever-table-structure-v1` NIM. Everything lives here to keep upstream source untouched.

## What nv-ingest expects
- HTTP endpoint on `${NIM_HTTP_API_PORT:-8000}` with `POST /v1/infer` body `{"input":[{"type":"image_url","url":"data:image/png;base64,..."}]}` → `{"data":[{"bounding_boxes":{"border":[],"cell":[],"row":[],"column":[],"header":[]}}]}` where boxes are `{x_min,y_min,x_max,y_max,confidence}` in normalized [0,1].
- gRPC/Triton endpoint on `${NIM_GRPC_API_PORT:-8001}` using model name `${TRITON_MODEL_NAME:-yolox_ensemble}`. Inputs: `INPUT_IMAGES` (BYTES, shape `[N]`, base64 data URLs) and `THRESHOLDS` (FP32, `[N,2]`, optional). Output: `OUTPUT` (BYTES, `[N]`) with JSON per image (`{label:[[x0,y0,x1,y1,score],...]}`).
- Health: `GET /v1/health/live`, `/v1/health/ready`, gRPC `ServerLive/ServerReady/ModelReady`. Metadata: `GET /v1/metadata`, gRPC `ModelMetadata`/`ModelConfig`/`RepositoryIndex`. Models list: `GET /v1/models`.
- Batch limit discovered from Triton model config (`TRITON_MODEL_CONFIG_PATH` if provided) or `NIM_TRITON_MAX_BATCH_SIZE` (default `32`); requests beyond that are rejected on HTTP and gRPC.
- Metrics: Prometheus text on `${NIM_METRICS_API_PORT:-8002}/metrics` and the HTTP `/metrics` path.
- Auth: Bearer/NGC key required on inference/metadata/model-control surfaces (HTTP `Authorization`/`X-API-Key`, gRPC `authorization` metadata). Set `NIM_REQUIRE_AUTH=false` to disable locally.
- Default version tag via `YOLOX_TABLE_STRUCTURE_TAG` (defaults to `1.6.0`). Use `LOG_LEVEL` to adjust verbosity.

## Model source
- Uses the open weights and helpers from https://huggingface.co/nvidia/nemotron-table-structure-v1 (`define_model("table_structure_v1")` + `postprocess_preds_table_structure`).
- Labels: `border`, `cell`, `row`, `column`, `header` (the latter two are passed through for completeness even if downstream only consumes `cell`/`row`/`column`).

## Running locally
Prereqs: Python 3.10+, `git-lfs`, and a PyTorch build that matches your GPU/CPU.

```bash
cd nim/nvidia/nemoretriever-table-structure-v1
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a torch build suited to your hardware (e.g., via https://pytorch.org/get-started/locally/)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124  # adjust as needed
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Quick smoke test:
```bash
curl -s http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/health/ready
curl -s -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/infer \
  -H "content-type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<your_base64_png>"}]}'
```

## Compose/K8s alignment
- Ports: `8000` (HTTP), `8001` (gRPC/Triton), `8002` (metrics). Map `YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT` to `http://nemoretriever-table-structure-v1:8000/v1/infer` and `YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT` to `nemoretriever-table-structure-v1:8001`.
- Env mirrors compose/Helm: `NIM_TRITON_MAX_BATCH_SIZE`, `NIM_TRITON_RATE_LIMIT`, `NIM_TRITON_LOG_VERBOSE`, `NIM_REQUIRE_AUTH`, `NGC_API_KEY`/`NVIDIA_API_KEY`, `NIM_ENABLE_OTEL`, `NIM_OTEL_*`, `TRITON_OTEL_*`, `OMP_NUM_THREADS`, `NIM_TRITON_CUDA_MEMORY_POOL_MB`.
- Ready probe path: `/v1/health/ready`; liveness: `/v1/health/live`; metadata: `/v1/metadata`; metrics: `/metrics`.

## Known deltas vs. closed NIM
- No async job/queue management; per-request inference only.
- Triton control APIs are scoped to model load/unload/index/metadata/config; other administrative endpoints are stubbed out.
