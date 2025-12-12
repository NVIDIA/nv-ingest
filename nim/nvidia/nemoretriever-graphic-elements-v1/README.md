Nemotron Graphic Elements v1 (open)
====================================

Reverse-engineered replacement for `nvcr.io/nim/nvidia/nemoretriever-graphic-elements-v1` using the open-source `nvidia/nemotron-graphic-elements-v1` model.

What the upstream NIM exposes
-----------------------------
- Endpoints used by nv-ingest (see `docker-compose.yaml`, `helm/values.yaml`, and `src/nv_ingest/pipeline/default_pipeline_impl.py`):
  - gRPC: `YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT` (defaults to `graphic-elements:8001`)
  - HTTP: `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT` (defaults to `http://graphic-elements:8000/v1/infer`)
  - Protocol toggle: `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL` (defaults to `grpc`; libmode defaults to `http`).
- Model contract (from `chart_extractor.py` and `yolox.py`):
  - Model name: `yolox_ensemble`
  - gRPC inputs: `INPUT_IMAGES` (BYTES, shape `[batch]`, base64 PNG) and `THRESHOLDS` (FP32, shape `[batch,2]`, values `[0.01,0.25]`).
  - gRPC output: one tensor of bytes/JSON per image containing a dict keyed by class label with normalized `[x_min,y_min,x_max,y_max,score]`.
  - HTTP request payload: `{"input": [{"type": "image_url", "url": "data:image/png;base64,<...>"} , ...]}`.
  - HTTP response payload: `{"data": [{"bounding_boxes": {"chart_title": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}], ...}}]}`.
  - Class labels: `["chart_title","x_title","y_title","xlabel","ylabel","other","legend_label","legend_title","mark_label","value_label"]`.
  - Max batch size is probed from Triton; nv-ingest requests up to 8 for chart extraction. Postprocessing expects **normalized** coords; nv-ingest scales them to pixel space later.

Open implementation in this directory
-------------------------------------
- `server.py`: FastAPI + Triton-compatible gRPC service that wraps the HF model. It:
  - Accepts `input` items with `image_url` (data URLs or remote/local paths).
  - Runs the HF `define_model("graphic_element_v1")` inference, applies the official `postprocess_preds_graphic_element`, and returns bounding boxes per class with normalized coords and confidences.
  - Provides HTTP parity endpoints (`/v1/infer`, `/v1/health/live`, `/v1/health/ready`, `/v1/metadata`), Triton metadata/config/ready HTTP shims, and `/metrics`.
  - Provides a Triton gRPC surface (ready/live/metadata/config/repository + `ModelInfer` with `INPUT_IMAGES`/`THRESHOLDS` → JSON bytes tensor).
  - Enforces Bearer/NGC auth (`Authorization: Bearer <token>` or `ngc-api-key: <token>`) across HTTP, gRPC, and metrics. Health is open.
  - Exposes OTEL (via `NIM_ENABLE_OTEL`, `NIM_OTEL_EXPORTER_OTLP_ENDPOINT`, `NIM_OTEL_SERVICE_NAME`) and Prometheus metrics on `NIM_METRICS_PORT` (default 8002).
  - Respects `NIM_TRITON_MAX_BATCH_SIZE` (default 8), optional `NIM_TRITON_RATE_LIMIT`, and model control toggles (`NIM_TRITON_ENABLE_MODEL_CONTROL`).
- `requirements.txt`: Minimal pinned runtime deps. Install the HF package separately (weights live in the HF repo, not here).
- Notes:
  - HTTP `/v1/infer` behavior is unchanged from the previous drop; gRPC now matches the NIM contract so nv-ingest can stay on `grpc` protocol.
  - Thresholds default to the upstream values (`conf_thresh=0.01`, `iou_thresh=0.25`, final score cutoff `0.1`). Override via `THRESHOLD` env if you need parity tweaks; gRPC `THRESHOLDS` input overrides per-request.

Setup
-----
1) Grab the model repo with weights:
```
git lfs install
git clone https://huggingface.co/nvidia/nemotron-graphic-elements-v1
pip install -e ./nemotron-graphic-elements-v1
```

2) Install server deps (uv recommended):
```
cd /Users/fran/Source/open-nv-ingest
uv pip install -r nim/nvidia/nemoretriever-graphic-elements-v1/requirements.txt
```

3) Run the server (CPU or GPU depending on `CUDA_VISIBLE_DEVICES`; override device/threshold via env):
```
NIM_HTTP_API_PORT=8000 \
NIM_GRPC_API_PORT=8001 \
NIM_METRICS_PORT=8002 \
NIM_TRITON_MAX_BATCH_SIZE=8 \
NGC_API_KEY=your_token_here \
python nim/nvidia/nemoretriever-graphic-elements-v1/server.py
# or: uvicorn --app-dir nim/nvidia/nemoretriever-graphic-elements-v1 server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

4) Wire nv-ingest:
- gRPC (parity): set `YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT=<host>:8001`, `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=http://<host>:8000/v1/infer`, `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=grpc`, and provide `NGC_API_KEY` (or `NIM_NGC_API_KEY`/`NVIDIA_API_KEY`).
- HTTP-only (legacy): set `YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=http://<host>:8000/v1/infer`, leave gRPC endpoint empty, and set `YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=http`.

Container build (optional)
--------------------------
`docker-compose.yaml` now builds `nemoretriever-graphic-elements-v1-open` from this directory by default. To build/push manually:
```
docker build -t nemoretriever-graphic-elements-v1-open:latest nim/nvidia/nemoretriever-graphic-elements-v1
# optional: docker tag nemoretriever-graphic-elements-v1-open:latest <your-registry>/nemoretriever-graphic-elements-v1-open:latest
# optional: docker push <your-registry>/nemoretriever-graphic-elements-v1-open:latest
```

Request/response example
------------------------
```
curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<base64_png>"}]}'
```
Response:
```
{
  "data": [
    {
      "bounding_boxes": {
        "chart_title": [{"x_min":0.02,"y_min":0.05,"x_max":0.94,"y_max":0.12,"confidence":0.91}],
        "x_title": [],
        "y_title": [],
        "xlabel": [...],
        "ylabel": [...],
        "other": [...]
      }
    }
  ]
}
```

Open questions / future parity work
-----------------------------------
- If gRPC drop-in parity is required, the server would need to expose a Triton-compatible gRPC surface that accepts `INPUT_IMAGES`/`THRESHOLDS` and returns the JSON bytes payloads described above. The current implementation targets the HTTP code path used in libmode and is sufficient once the env vars point nv-ingest to HTTP.
- Upstream NIM enables Triton model control and max-batch discovery; if you need identical batching behaviour, expose `max_batch_size` metadata (or keep HTTP and rely on nv-ingest batching in the client).
