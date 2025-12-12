# nemoretriever-page-elements-v3 (open replacement)

Reverse‑engineered drop-in for the private `nim/nvidia/nemoretriever-page-elements-v3` NIM (formerly hosted at `nvcr.io/nim/nvidia/nemoretriever-page-elements-v3`). The goal is upstream alignment without touching core source: everything lives in this directory (`nims/nvidia/nemoretriever-page-elements-v3`).

## What nv-ingest expects
- HTTP: `http://nemoretriever-page-elements-v3:8000/v1/infer` returning bounding boxes in the same shape as the closed NIM. Batch size is discovered via `NIM_TRITON_MAX_BATCH_SIZE`.
- gRPC/Triton: port `8001` exposes the Triton gRPC API with `ModelReady/ModelMetadata/ModelConfig/RepositoryIndex` plus `ModelInfer` using inputs `INPUT_IMAGES` (BYTES, shape `[batch]` of base64 PNG) and `THRESHOLDS` (FP32, shape `[batch,2]`) and output `OUTPUT` (BYTES JSON per image). HTTP-style metadata/config are mirrored under `/v2/models/...`.
- Health: `GET /v1/health/live` and `GET /v1/health/ready` (aliases under `/v2/health/...`) return 200 for readiness checks.
- Metadata: `GET /v1/metadata` exposes `modelInfo[0].shortName` so `get_model_name()` resolves `nemoretriever-page-elements-v3`; `/v2/models/{name}` and `/v2/models/{name}/config` mirror the Triton replies.
- Metrics: Prometheus metrics are exported at `/metrics` (port 8000) and on a dedicated listener if `NIM_METRICS_PORT` is set (default 8002). Triton clients can also scrape via `/v2/metrics`.
- Auth: All inference/metadata calls (HTTP and gRPC) require `Authorization: Bearer $NGC_API_KEY` (or `NIM_NGC_API_KEY`). Health remains unauthenticated for probes.
- Inference request body: `{"input": [{"type": "image_url", "url": "data:image/png;base64,..."}]}` with batches up to `NIM_TRITON_MAX_BATCH_SIZE`.
- Inference response shape: `{"data": [{"bounding_boxes": {"table": [...], "chart": [...], "title": [...], "infographic": [...], "paragraph": [...], "header_footer": [...]}}]}` where each box is `{x_min,y_min,x_max,y_max,confidence}` in normalized [0,1] coords. The private NIM uses `paragraph` (not `text`), so this server remaps the open model’s `text` label accordingly. gRPC responses return the normalized array payloads the client post-processes.

Key envs mirrored from `docker-compose.yaml` / `helm/values.yaml`:
- `NIM_HTTP_API_PORT` (default 8000)
- `NIM_TRITON_MAX_BATCH_SIZE` (default 8 here; set to 32 to match upstream config)
- `YOLOX_TAG` (metadata tag, default `1.7.0`)
- `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, OTEL vars are respected by the container runtime; the app itself is CPU/GPU agnostic via PyTorch device detection.

## Model source
- Uses the open model weights and helpers from https://huggingface.co/nvidia/nemotron-page-elements-v3.
- Official usage example is embedded into the server (`define_model("page_element_v3")` + `postprocess_preds_page_element`).

## Running locally (HTTP only)
Prereqs: Python 3.10+, `git-lfs`, and a CUDA-capable PyTorch if you want GPU.

```bash
cd nims/nvidia/nemoretriever-page-elements-v3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # installs the HF package; install torch that matches your GPU if needed
uvicorn app:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Quick smoke test:
```bash
curl -s http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/health/ready
curl -s -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/infer \
  -H "content-type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<your_base64_png>"}]}'
```

For gRPC parity (after installing `tritonclient`), point nv-ingest at `page-elements:8001` with protocol `grpc`; the server reports max batch from `NIM_TRITON_MAX_BATCH_SIZE` and supports model control/index endpoints used by the client.

## Compose/K8s alignment
- Expose ports `8000` (HTTP) and optionally `8001`/`8002` if you front this with Triton later; the server itself only binds HTTP.
- Map env vars from existing charts/compose to keep `YOLOX_HTTP_ENDPOINT` working (`http://nemoretriever-page-elements-v3:8000/v1/infer`).
- Ready probe path: `/v1/health/ready`; metadata path: `/v1/metadata`.

## Parity notes
- Triton gRPC is implemented (metadata/config/ready/repository index, model control, infer, statistics).
- Per-request `THRESHOLDS` overrides are honored in gRPC as `[conf_thresh, iou_thresh]` per image; batch limit stays at `NIM_TRITON_MAX_BATCH_SIZE`.
- System and CUDA shared memory register/status/unregister are implemented; inputs/outputs can target registered system regions.
- Model control load/unload flips readiness and removes/creates the model instance accordingly.
- Output includes all six v3 labels, with `text` remapped to `paragraph` to keep downstream post-processing intact.
