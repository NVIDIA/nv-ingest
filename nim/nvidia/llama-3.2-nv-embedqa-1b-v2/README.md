# llama-3.2-nv-embedqa-1b-v2 (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2` text embedding NIM. Lives under `nim/nvidia/llama-3.2-nv-embedqa-1b-v2` to keep upstream sources untouched.

## What nv-ingest expects
- HTTP endpoint on `${NIM_HTTP_API_PORT:-8000}` with `POST /v1/embeddings`. Payload: `{"model":"nvidia/llama-3.2-nv-embedqa-1b-v2","input":["..."],"encoding_format":"float","input_type":"passage","truncate":"END","dimensions":2048}`. `model` defaults to the server model; `input` may be a single string or list. `input_type` controls prefixing (`query:` vs `passage:`); `truncate` supports `END`, `START`, or `NONE`. `dimensions` is optional Matryoshka down-projection.
- Response shape: `{"object":"list","model":"...","data":[{"object":"embedding","index":0,"embedding":[...float...]}],"usage":{"prompt_tokens":N,"total_tokens":N}}`. `encoding_format="base64"` returns base64-encoded float32 bytes in `embedding`.
- Health: `GET /v1/health/live` and `/v1/health/ready` return 200. Metadata: `GET /v1/models` lists the model id; `GET /v1/metadata` exposes `modelInfo[0].shortName` for `get_model_name()` compatibility.
- Batch limit enforced by `NIM_TRITON_MAX_BATCH_SIZE` (default `30`, Helm defaults to `3` for ONNX profile). Requests exceeding the limit are rejected with `400`.

## Model source
- Uses the open weights from https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2.
- Applies the official prefixing (`query:` / `passage:`) and mean-pooling with L2 normalization. `dimensions` slices the leading components and renormalizes (Matryoshka embeddings).

## Running locally
Prereqs: Python 3.10+, `git-lfs`, and a PyTorch build for your hardware (GPU recommended).

```bash
cd nim/nvidia/llama-3.2-nv-embedqa-1b-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a torch build matching your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124  # adjust as needed
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

Smoke test:
```bash
curl -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/embeddings \
  -H "content-type: application/json" \
  -d '{"model":"nvidia/llama-3.2-nv-embedqa-1b-v2","input":["example passage text"],"input_type":"passage","truncate":"END"}'
```

## Compose/K8s alignment
- Exposes port `8000` (no gRPC). Compatible with the `docker-compose.yaml` `embedding` service and Helm values (`embedqa` section).
- Mirrors key envs: `NIM_HTTP_API_PORT`, `NIM_TRITON_MAX_BATCH_SIZE`, `MODEL_VERSION` (for `modelInfo.shortName`), `LOG_LEVEL`, optional `MAX_SEQUENCE_LENGTH` to override tokenizer max length.

## Known deltas vs. closed NIM
- gRPC/Triton endpoints are not implemented; HTTP parity is preserved for nv-ingest usage.
- Authorization headers are accepted but not enforced.
- Only the embeddings surface (`/v1/embeddings`) plus health/metadata is implemented; no async job APIs.
