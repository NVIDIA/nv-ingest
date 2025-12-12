nemoretriever-ocr-v1 (open replacement)
=======================================

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/nemoretriever-ocr-v1` NIM using the open-source `nvidia/nemotron-ocr-v1` model.

What nv-ingest expects
----------------------
- Endpoints (see `docker-compose.yaml`, `helm/values.yaml`, and `default_pipeline_impl.py`):
  - HTTP: `OCR_HTTP_ENDPOINT` → defaults to `http://ocr:8000/v1/infer`
  - gRPC: `OCR_GRPC_ENDPOINT` → defaults to `ocr:8001` (can be left empty to force HTTP)
  - Protocol toggle: `OCR_INFER_PROTOCOL` → defaults to `grpc`; set to `http` for this server.
- Request payload (HTTP): `{"input":[{"type":"image_url","url":"data:image/png;base64,<...>"}], "merge_levels":["paragraph", ...]}` where `merge_levels` is optional (defaults to `paragraph`).
- Response payload (HTTP): `{"data":[{"text_detections":[{"bounding_box":{"points":[{"x":..,"y":..},...]},"text_prediction":{"text":"...", "confidence": <float>}}, ...]}]}`.
  - Bounding boxes are normalized coordinates (0–1) with four points per detection.
- Health: `/v1/health/live` and `/v1/health/ready` must return HTTP 200 for readiness checks.
- Metadata: `/v1/metadata` should expose `modelInfo[0].shortName` so discovery works if queried (we return `nemoretriever-ocr-v1`).

Open implementation in this directory
-------------------------------------
- `server.py`: FastAPI app that wraps the official `NemotronOCR` pipeline from the HF repo.
  - Accepts batches of `image_url` inputs (data URLs, http/https, or local paths) and optional `merge_levels` (`word|sentence|paragraph`).
  - Runs the HF model one image at a time (the reference pipeline is single-image) and emits `text_detections` with normalized quadrilateral boxes and confidences matching nv-ingest parsing logic.
  - Exposes `/v1/infer`, `/v1/health/live`, `/v1/health/ready`, `/v1/metadata`, and Triton-compatible HTTP (`/v2/*`) + gRPC (port `NIM_TRITON_GRPC_PORT`) surfaces including model control and statistics. Metrics are on `NIM_TRITON_METRICS_PORT`.
  - Honors env vars: `NIM_HTTP_API_PORT` (port), `MERGE_LEVEL` (default merge level, default `paragraph`), `NIM_TRITON_MAX_BATCH_SIZE` (how many inputs to accept per request; processed sequentially), `NIM_TRITON_ENABLE_MODEL_CONTROL` (enable load/unload), `MODEL_DIR` (path to HF checkpoints), and bearer tokens via `NGC_API_KEY`/`NVIDIA_API_KEY`.
- `requirements.txt`: Runtime deps for the server; install the HF package separately (`nemotron-ocr` from the repo).

Setup
-----
1) Fetch the model repo + weights (requires git-lfs, CUDA-capable system):
```
git lfs install
git clone https://huggingface.co/nvidia/nemotron-ocr-v1
cd nemotron-ocr-v1/nemotron-ocr
pip install -v .
```

2) Install the server deps (uv or pip):
```
cd /Users/fran/Source/open-nv-ingest
uv pip install -r nim/nvidia/nemoretriever-ocr-v1/requirements.txt
```

3) Run the server:
```
NIM_HTTP_API_PORT=8000 \
MERGE_LEVEL=paragraph \
NIM_TRITON_MAX_BATCH_SIZE=8 \
MODEL_DIR=/Users/fran/Source/open-nv-ingest/nemotron-ocr-v1/checkpoints \
python nim/nvidia/nemoretriever-ocr-v1/server.py
# or uvicorn --app-dir nim/nvidia/nemoretriever-ocr-v1 server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

4) Wire nv-ingest to it:
- Set `OCR_INFER_PROTOCOL=http`.
- Point `OCR_HTTP_ENDPOINT=http://<host>:8000/v1/infer`.
- Clear or ignore `OCR_GRPC_ENDPOINT` to avoid accidental gRPC selection.
- If using Helm, override `nimOperator.nemoretriever_ocr_v1.expose.service.port` to 8000 and set envs above.

Request/response example
------------------------
```
curl -X POST http://localhost:8000/v1/infer \
  -H "content-type: application/json" \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64,<base64_png>"}],"merge_levels":["word"]}'
```
Response:
```
{
  "data": [
    {
      "text_detections": [
        {
          "bounding_box": {
            "points": [
              {"x":0.12,"y":0.18},
              {"x":0.42,"y":0.18},
              {"x":0.42,"y":0.24},
              {"x":0.12,"y":0.24}
            ],
            "type": "quadrilateral"
          },
          "text_prediction": {"text":"Example", "confidence":0.97}
        }
      ]
    }
  ]
}
```

Known gaps vs. closed NIM
-------------------------
- The HF pipeline runs per-image; batches are processed sequentially up to `NIM_TRITON_MAX_BATCH_SIZE`.
