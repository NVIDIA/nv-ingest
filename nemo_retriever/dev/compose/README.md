# NeMo Retriever Development Compose Helpers

These files provide a development-only, standalone service deployment. Hosted
inference remains the zero-profile default. Production deployments and split
gateway/realtime/batch topology use the [Helm chart](../../helm/README.md).

Run commands from the repository root. Docker Compose 2.23.1 or newer is
required for optional dependencies and inline configs.

Building the service image pulls its Ubuntu base image from NVCR. If the build
fails with `403 Forbidden` while pulling `nvcr.io/nvidia/base/ubuntu`,
authenticate to the registry with an NGC API key:

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

This registry login is separate from the `NVIDIA_API_KEY` used for hosted
inference at runtime.

## Hosted inference (default)

The default starts the retriever and LanceDB only. Document and query content
is sent to NVIDIA-hosted endpoints, so confirm that this is appropriate for
your data.

```bash
export NVIDIA_API_KEY=nvapi-...
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
curl -fsSL http://localhost:7670/v1/health
```

The same default stack exposes collection and document lifecycle APIs. Optional
runtime tokens enable a public bearer credential and a separate credential for
the private Retriever-to-VectorDB hop:

```bash
export NRL_API_TOKEN="<development-api-token>"
export NRL_INTERNAL_VDB_TOKEN="<internal-service-token>"
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

Leave both values unset to preserve the existing unauthenticated development
experience. In this default Compose configuration, `NRL_API_TOKEN` is bound to
the `default` scope, so clients use `X-NRL-Scope: default` (or SDK
`scope="default"`). Multi-scope deployments use the service's Secret-backed
token mapping. Tokens are runtime inputs and must not be committed.

Endpoints, models, ports, worker counts, and the service image can all be
overridden explicitly. The most commonly tuned variables are
`NIM_PAGE_ELEMENTS_URL`, `NIM_TABLE_STRUCTURE_URL`, `NIM_OCR_URL`,
`NIM_EMBED_URL`, `NIM_EMBED_MODEL`, `RETRIEVER_HTTP_PORT`,
`PIPELINE_REALTIME_WORKERS`, `PIPELINE_REALTIME_QUEUE_SIZE`,
`PIPELINE_BATCH_WORKERS`, and `PIPELINE_BATCH_QUEUE_SIZE`. Explicit shell
variables override preset values, enabling mixed hosted/self-hosted stacks.

## Self-hosted NIM profiles

Use the NVCR authentication described above before pulling a self-hosted NIM.
Keep `NGC_API_KEY` exported so it is also available to the NIM containers. The
hosted-only stack does not require `NGC_API_KEY` at runtime.

Routing is selected when Compose renders the configuration: self-hosted mode
requires both the `nims-core` preset and profile, and it does not fail over to
hosted endpoints if a local NIM becomes unavailable.

Start the four core extraction/retrieval NIM services with their checked-in internal
endpoint wiring:

```bash
docker compose --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --profile nims-core \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

Optional wiring presets are `nim-caption.env`, `nim-answer.env`, and
`nim-audio.env`. Select their matching Compose profiles explicitly. For
example, layer answer generation over the core NIMs with:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --profile nims-core --profile nim-answer \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

To start all nine NIM services, layer the four wiring presets and select every NIM
profile:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/nim-caption.env \
  --env-file nemo_retriever/dev/compose/presets/nim-answer.env \
  --env-file nemo_retriever/dev/compose/presets/nim-audio.env \
  --profile nims-core --profile nim-reranker --profile nim-parse \
  --profile nim-caption --profile nim-answer --profile nim-audio \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

The `nim-caption` profile defaults to `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant`. Set `NIM_CAPTION_IMAGE` or `NIM_CAPTION_TAG` only when you need a different registry or image tag.

Reranker and Parse need only `--profile nim-reranker` or
`--profile nim-parse`. They are lifecycle/API-only and intentionally are not
injected into retriever service configuration, matching Helm.

Every NIM has a persistent model or cache volume, a configurable GPU assignment, and a
configurable host port. Variables follow the service prefix, for example
`NIM_OCR_GPU_ID`, `NIM_OCR_HOST_PORT`, `NIM_OCR_CACHE_VOLUME`,
`NIM_OCR_CACHE_PATH`, `NIM_OCR_IMAGE`, and `NIM_OCR_TAG`. The defaults form a
collision-free assignment for the combined-profile example: core NIMs use GPUs
0, 1, 2, and 3 (page-elements, table-structure, OCR, and embedding), reranker uses 4,
parse uses 5, caption uses 6, answer uses 7, and audio uses 9. Change the
defaults to match the active profiles and host before startup. The one-GPU
answer NIM uses `NIM_ANSWER_GPU_ID_0`; for example, an answer-only run can set
`NIM_ANSWER_GPU_ID_0=0`. Compose lifecycle support
means image pull, startup, readiness, persistent model or cache data, restart, logs, and
teardown; NIM Operator reconciliation, NIMCache CRDs, and model-profile
selection remain Kubernetes-only.

Answer behavior is configurable through `ANSWER_LLM_ENABLED`,
`ANSWER_LLM_MODEL`, `ANSWER_LLM_API_BASE_YAML`, `ANSWER_LLM_TEMPERATURE`,
`ANSWER_LLM_TOP_P`, `ANSWER_LLM_MAX_TOKENS`, `ANSWER_LLM_EXTRA_PARAMS`,
`ANSWER_LLM_NUM_RETRIES`, `ANSWER_LLM_TIMEOUT`, and
`ANSWER_LLM_REASONING_ENABLED`. Caption uses `NIM_CAPTION_URL` and
`NIM_CAPTION_MODEL_YAML`; audio uses `NIM_AUDIO_GRPC_ENDPOINT` and its `_YAML`
counterpart.

## Local Hugging Face GPU models

Local mode builds the `service-gpu` target for both retriever and VectorDB,
enables local extraction, embedding, and ASR, and persists Hugging Face model
downloads. It reserves separate GPUs by default (`0` for retriever and `1` for
query embedding); override `LOCAL_MODELS_GPU_ID` and `LOCAL_VECTORDB_GPU_ID`
as needed.

```bash
docker compose --env-file nemo_retriever/dev/compose/presets/local-models.env \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml \
  -f nemo_retriever/dev/compose/service-mode.local-models.compose.yaml \
  up --build -d
```

Local mode is mutually exclusive with the `nims-core` profile. Optional answer,
caption, and audio NIMs may be layered onto it by adding their wiring preset
and matching `--profile`; their remote endpoints take precedence for those
stages. Tune warmup, process-pool cap, embedding memory fraction, and cache
name with the `LOCAL_MODELS_*`, `LOCAL_EMBED_GPU_MEMORY_UTILIZATION`, and
`HF_CACHE_VOLUME` variables.

## Observability

The observability preset starts OpenTelemetry Collector and Zipkin and enables
retriever/NIM OTLP export. The collector batches telemetry, retains its debug
exporter, exposes Prometheus-format metrics at `http://localhost:8889/metrics`,
and forwards traces to Zipkin. Open `http://localhost:9411` to inspect traces.

```bash
docker compose --env-file nemo_retriever/dev/compose/presets/observability.env \
  --profile observability \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

To combine it with core NIMs, supply both presets and profiles:

```bash
docker compose \
  --env-file nemo_retriever/dev/compose/presets/nims-core.env \
  --env-file nemo_retriever/dev/compose/presets/observability.env \
  --profile nims-core --profile observability \
  -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

Host ports are configurable with `OTEL_GRPC_HOST_PORT`,
`OTEL_HTTP_HOST_PORT`, `OTEL_PROMETHEUS_HOST_PORT`, and
`ZIPKIN_HOST_PORT`. Telemetry is disabled in the default stack; Compose
starts the retriever once the optional collector process starts and relies on
OTLP exporter retries while its listeners become ready. Collector failure does
not block the retriever service. Compose intentionally does not add Prometheus,
Grafana, ServiceMonitor, or autoscaling.

## Hardware-gated smoke checks

These checks pull large images/models and require suitable NVIDIA GPUs. Use
`docker compose config` for configuration-only validation without launching
the stack.

1. Core extraction/retrieval: start `nims-core.env`, wait for the four NIMs plus
   Retriever and VectorDB to report healthy, then ingest a representative PDF with the service CLI:

   ```bash
   retriever ingest service /path/to/document.pdf \
     --service-url http://localhost:7670
   ```

   Query the service directly through `/v1/query`:

   ```bash
   curl -fsSL -X POST http://localhost:7670/v1/query \
     -H 'Content-Type: application/json' \
     --data '{"query":"What is in this document?","top_k":5}'
   ```

   The equivalent CLI invocation is:

   ```bash
   retriever query service "What is in this document?" \
     --service-url http://localhost:7670
   ```

   Confirm the results include extracted text from the ingested document.
2. Answer: layer `nim-answer`, ingest/query a small collection, call
   `/v1/answer`, and confirm an answer plus retrieved context is returned.
3. Caption/audio: layer each preset independently; ingest an image with the
   caption stage and an audio file with audio extraction, then confirm caption
   text and transcript in job results.
4. Lifecycle-only APIs: start reranker and Parse and verify readiness plus API
   reachability at `http://localhost:8005` and `http://localhost:8006`; do not
   expect retriever auto-wiring.
5. Local models: start the local override, ingest a PDF and audio sample, query
   the collection, and confirm both service logs show local model loading while
   VectorDB performs local query embedding.
6. Tracing: add observability, perform ingestion/query, open Zipkin on port
   `9411`, and confirm retriever spans (and NIM spans when supported by the
   selected NIM profile) are visible.

Use `docker compose ... logs -f`, `ps`, and `down` for normal lifecycle work.
Named data/model caches survive `down`; use `down -v` only when a clean cache
and database are intentionally required.

## Other helpers

The judge and Neo4j helpers remain independent of service-mode profiles.

### Local judge

`judge.compose.yaml` starts a local OpenAI-compatible Nemotron NIM for
`retriever skill-eval` runs that use a local judge endpoint. Set `NGC_API_KEY`
or `NIM_NGC_API_KEY` before starting it, then authenticate to NGC and launch
the helper:

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge
```

Set `judge.model` to `openai/nvidia/nemotron-3.5-lightning-30b-a3b` and
`judge.api_base` to `http://localhost:8000/v1` in the skill-eval configuration.
Override the host port with `JUDGE_HTTP_PORT` when needed.

### Neo4j

`neo4j.compose.yaml` starts the graph development database. Set
`NEO4J_PASSWORD` in the environment or a repository-root `.env` file before
starting it; `NEO4J_USERNAME` defaults to `neo4j`.

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j
```
