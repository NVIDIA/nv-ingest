# parakeet-1-1b-ctc-en-us (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` Riva/Parakeet ASR NIM. Everything lives under `nim/nvidia/parakeet-1-1b-ctc-en-us` to keep upstream sources untouched.

## What nv-ingest expects
- gRPC endpoint on `${GRPC_PORT:-50051}` implementing the Riva `RivaSpeechRecognition/Recognize` RPC. nv-ingest’s `ParakeetClient` calls `offline_recognize()` with WAV bytes (base64-decoded) that have already been converted to mono 44.1kHz; the server resamples to 16kHz for the model.
- `RecognitionConfig.enable_word_time_offsets` is honored; word timings are derived from the open model’s logits and returned in `SpeechRecognitionAlternative.words` so `process_transcription_response()` can build segments.
- Streaming: incremental decode caches logits between emissions (interim responses at `STREAMING_EMIT_INTERVAL_MS`) with VAD-driven endpointing to mirror Riva’s incremental stability.
- Metadata/config: `GetRivaSpeechRecognitionConfig` returns the model name, batch/concurrency limits, and punctuation/diarization defaults so discovery calls do not fail.
- Health/metadata HTTP on `${HTTP_PORT:-9000}` (set to `0` to disable HTTP). Auth is enforced if enabled; liveness/ready can stay public via `ALLOW_UNAUTH_HEALTH`:
  - `GET /v1/health/live` → `{ "status": "SERVING" }`
  - `GET /v1/health/ready` → `{ "status": "READY", "uptime_seconds": <int>, "model": "<name>", "version": "<version>" }`
  - `GET /v1/metadata` → exposes `modelInfo.shortName` so `get_model_name()` continues to work (`parakeet-1-1b-ctc-en-us:${MODEL_VERSION}`).
  - `GET /metrics` → Prometheus metrics (protected when auth is enabled).

## Riva control-plane parity additions
- Auth: optional Bearer/NGC token checks across gRPC and HTTP (`AUTH_TOKENS`, `NGC_API_KEYS`, `REQUIRE_AUTH`, `ALLOW_UNAUTH_HEALTH`).
- Config: honors `language_code`, punctuation toggle, diarization flag, and exposes batch/concurrency caps in `GetRivaSpeechRecognitionConfig` and `/v1/metadata`.
- Streaming: incremental decode caches logits between emissions to avoid full re-decodes; interim responses are emitted at `STREAMING_EMIT_INTERVAL_MS` with VAD-backed endpointing/pipeline states to drive incremental stability.
- Punctuation/diarization: punctuation uses a dedicated model (`PUNCTUATION_MODEL_ID`), and diarization assigns speaker tags via lightweight spectral clustering on voiced segments when enabled.
- Observability: `/metrics` (Prometheus), optional OTLP tracing (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`).
- Health/metadata: readiness includes uptime/model info; metadata includes limit/features surfaces for control-plane discovery.
- Triton control-plane: `/v2/health/{live,ready}`, `/v2/models/{name}`, `/v2/models/{name}/config`, `/v2/models/{name}/ready`, and repository index are mirrored over HTTP and gRPC (`GRPCInferenceService`) for control-plane parity.

Key envs mirrored from `docker-compose.yaml` / Helm values:
- `GRPC_PORT` (default `50051`)
- `HTTP_PORT` (default `9000`, set to `0` to disable the HTTP shim)
- `MODEL_ID` (default `nvidia/parakeet-ctc-1.1b`)
- `MODEL_VERSION` (default `1.1.0`)
- `MAX_WORKERS` (gRPC thread pool, default `4`)
- `MAX_CONCURRENT_REQUESTS` (gRPC request concurrency guard, default `16`)
- `MAX_STREAMING_SESSIONS` (concurrent streaming sessions, default `8`)
- `MAX_BATCH_SIZE` (surfaced via metadata/config for control plane compatibility, default `4`)
- `AUTH_TOKENS` / `NGC_API_KEYS` (comma-separated tokens for Bearer/NGC auth; set `REQUIRE_AUTH=true` to enforce)
- `ALLOW_UNAUTH_HEALTH` (default `true`, set `false` to protect health/metadata/metrics)
- `STREAMING_EMIT_INTERVAL_MS` (cadence for interim streaming results, default `400`)
- `ENDPOINT_START_HISTORY_MS` / `ENDPOINT_START_THRESHOLD` / `ENDPOINT_STOP_HISTORY_MS` / `ENDPOINT_STOP_THRESHOLD` / `ENDPOINT_EOU_HISTORY_MS` / `ENDPOINT_EOU_THRESHOLD` / `VAD_FRAME_MS` (default `240/0.6/1200/0.35/1600/0.2/30`; controls VAD/endpointer behavior, timing, and streaming finalization)
- `PUNCTUATION_MODEL_ID` (defaults to `kredor/punctuate-all`, used when punctuation is enabled)
- `MAX_SPEAKER_COUNT` (max speakers to surface when diarization is enabled; default `2`)
- `METRICS_NAMESPACE` (Prometheus metric prefix, default `parakeet`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_SERVICE_NAME` (optional OpenTelemetry traces)
- `LOG_LEVEL`
- `CUDA_VISIBLE_DEVICES` is respected by PyTorch for GPU selection.

## Model source
- Uses the open Hugging Face weights https://huggingface.co/nvidia/parakeet-ctc-1.1b and follows the official CTC decoding example (`processor.decode(..., output_word_offsets=True)` with `time_offset = inputs_to_logits_ratio / sampling_rate`).
- Input is resampled to the model’s 16kHz expectation and normalized to float32.

## Running locally
Prereqs: Python 3.10+, git-lfs, and a compatible PyTorch build (install a CUDA wheel if you want GPU).

```bash
cd nim/nvidia/parakeet-1-1b-ctc-en-us
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a torch wheel that matches your system before the rest if you need GPU acceleration
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu124  # adjust CUDA version as needed
python -m pip install -r requirements.txt
python server.py  # starts gRPC on $GRPC_PORT and HTTP health on $HTTP_PORT
```

Smoke test (requires `grpcurl`):
```bash
BASE64_WAV=$(python - <<'PY'
import base64, sys
print(base64.b64encode(open("sample.wav","rb").read()).decode())
PY
)
grpcurl -plaintext -d "{\"config\":{\"enable_word_time_offsets\":true},\"audio\":\"$BASE64_WAV\"}" localhost:${GRPC_PORT:-50051} nvidia.riva.asr.RivaSpeechRecognition/Recognize
```

## Compose/K8s alignment
- Swap `AUDIO_IMAGE` to point at this folder (for example by setting `AUDIO_IMAGE=parakeet-1-1b-ctc-en-us-open` and adding a build stanza mirroring other open NIMs).
- Exposes ports `50051` (gRPC) and `9000` (HTTP health/metadata). The nv-ingest runtime uses only the gRPC endpoint (`AUDIO_GRPC_ENDPOINT=audio:50051`).
- Auth is optional; set `AUTH_TOKENS` / `NGC_API_KEYS` and `REQUIRE_AUTH=true` to enforce across gRPC/HTTP (health can stay public via `ALLOW_UNAUTH_HEALTH`).

## Known deltas vs. closed NIM
- Uses the open Hugging Face checkpoint; accuracy and word offsets may differ slightly from the closed Parakeet/Riva build.
- Diarization uses lightweight spectral clustering on voiced windows; overlapped speech or highly reverberant audio may differ from the closed pipeline.
