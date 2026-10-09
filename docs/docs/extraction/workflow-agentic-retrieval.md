# Workflow: Agentic retrieval

Use this workflow after you have ingested documents into a LanceDB table. Agentic retrieval does not ingest files. It queries the same table, embedding model, and storage flags as one-pass `retriever query`.

**Agentic retrieval** runs a large language model (LLM) Reason and Act (ReAct)
loop that issues several retrieval sub-queries. In the default `select` mode,
the workflow fuses candidates with reciprocal rank fusion and selects a final
document ranking. In `answer` mode, the ReAct loop instead returns an integrated
answer with validated citations to documents retrieved during that run.
**One-pass retrieval** sends a single dense or hybrid query and returns
text-enriched chunk hits. For the concept distinction, refer to
[Agentic retrieval (concept)](agentic-retrieval-concept.md).

## Query with the CLI { #query-with-the-cli }

`retriever query --agentic` is the one-shot CLI path. It searches the LanceDB table built by `retriever ingest`. Reuse the same `--lancedb-uri`, `--table-name`, and embedding model that you used at ingest. When `--embed-model-name` is omitted, agentic retrieval uses the selected table's model.

### Local in-process vLLM { #local-in-process-vllm }

The CLI and NRB agentic benchmark paths default to an in-process local vLLM agent
LLM. If you omit `--agentic-llm-model` and `--agentic-invoke-url`, the library
loads `nemotron-8b` (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) on the local CUDA
host. This requires a Linux CUDA GPU and the `[local]` extra.

GPU placement follows process-level vLLM behavior. Set `CUDA_VISIBLE_DEVICES` before you start the command.

```bash
CUDA_VISIBLE_DEVICES=0 retriever query "find documents about parser behavior" --agentic
```

The larger `super-49b` profile is also supported. Pass `--agentic-local-tensor-parallel-size 2` with two visible GPUs for that profile.

Custom in-process LLMs are not supported. The agent loop depends on OpenAI-style tool-call messages. Use an OpenAI-compatible endpoint for custom models.

### Remote OpenAI-compatible NIM or hosted endpoint { #remote-openai-compatible-endpoint }

Providing `--agentic-invoke-url` routes the agent to that remote chat-completions endpoint. `--agentic-llm-model` is required on the remote path and is sent as the remote model ID. The LLM client defaults to `callable`, which calls the endpoint over the shared chat-completions HTTP client and needs no extra LLM SDK.

Self-hosted NIM or a local OpenAI-compatible server:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/nemotron-3.5-lightning-30b-a3b \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions
```

NVIDIA-hosted Build endpoint (requires `NVIDIA_API_KEY`; `NGC_API_KEY` is the fallback):

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/nemotron-3.5-lightning-30b-a3b \
  --agentic-invoke-url https://integrate.api.nvidia.com/v1/chat/completions
```

Add `--agentic-mode answer` to either local or remote commands when you want an
integrated answer rather than a ranked document list:

```bash
retriever query "explain the parser behavior" \
  --agentic \
  --agentic-mode answer \
  --include-usage
```

`--agentic-local-tensor-parallel-size` is ignored when `--agentic-invoke-url` is set. For hosted model IDs, refer to [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints). For key setup, refer to [Authentication and API keys](api-keys.md).

This self-hosted NIM configuration gap does not apply to NVIDIA-hosted Build endpoints. A Helm-deployed Nemotron 3.5 Lightning NIM rejects tool-call requests until you add the passthrough arguments. Refer to [Self-hosted Helm Nemotron 3.5 Lightning](#self-hosted-helm-lightning).

### CLI options { #cli-options }

The following options apply only with `--agentic`. For the full flag list, refer to [Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#agentic-retrieval) in the CLI reference.

| Option | Default | Notes |
|---|---|---|
| `--agentic-mode` | `select` | `select` returns ranked documents; `answer` returns an integrated answer, validated citation IDs, and hydrated citation hits. |
| `--agentic-llm-model` | `nemotron-8b` when no invoke URL is set | Local profile alias (`nemotron-8b` or `super-49b`) or remote model ID when `--agentic-invoke-url` is set. |
| `--agentic-invoke-url` | unset (local vLLM) | OpenAI-compatible `/v1/chat/completions` endpoint. Required together with `--agentic-llm-model` for remote runs. |
| `--agentic-local-tensor-parallel-size` | `1` | vLLM `tensor_parallel_size` for the in-process agent LLM. Set to `2` for local `super-49b`. Ignored when `--agentic-invoke-url` is set. |
| `--agentic-react-max-steps` | `50` | Maximum ReAct loop iterations. |
| `--agentic-reasoning-effort` | `high` | Forwarded on OpenAI-compatible agent LLM calls. Ignored by the local adapter. |
| `--include-usage` | off | In select mode, print an object with `hits` and provider-reported LLM `usage`. In answer mode, add `usage` to the answer response object. |

Embedding credentials use `NVIDIA_API_KEY` or `NGC_API_KEY` when you call a remote embedding endpoint. The CLI also reuses `--embed-invoke-url`, `--top-k`, `--lancedb-uri`, and `--table-name` from standard retrieval.

## Self-hosted Helm Nemotron 3.5 Lightning { #self-hosted-helm-lightning }

Use this path when the agent LLM is the Helm-deployed Nemotron 3.5 Lightning NIM rather than local in-process vLLM or an NVIDIA-hosted Build endpoint.

`nimOperator.answer_llm.enabled=true` deploys Nemotron 3.5 Lightning and auto-wires it only
to `serviceConfig.llm` for the default `POST /v1/answer` classic path. That path
sends a plain text-generation request and does not require tool calling.
`POST /v1/answer` with `"mode": "agentic"` uses `serviceConfig.agentic`
instead. That block stays empty unless you set it, and its LLM must support tool
calling.

The chart starts that NIM with `NIM_PASSTHROUGH_ARGS=--reasoning-parser nemotron_v3`. The agentic ReAct loop sends OpenAI-style tool-call messages with `tool_choice=auto`. A self-hosted vLLM-backed Nemotron 3.5 Lightning NIM rejects those requests with HTTP 400 unless you also pass `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder`.

You can reuse the same Nemotron 3.5 Lightning NIM for agentic retrieval after you add those arguments. `POST /v1/answer` continues to work.

If you set `nimOperator.answer_llm.env` in a values file, include the full list. Change only the `NIM_PASSTHROUGH_ARGS` value.

```yaml
nimOperator:
  answer_llm:
    enabled: true
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_MODEL_NAME
        value: "nvidia/nemotron-3.5-lightning-30b-a3b"
      - name: NIM_SERVED_MODEL_NAME
        value: "nvidia/nemotron-3.5-lightning-30b-a3b"
      - name: NIM_TENSOR_PARALLEL_SIZE
        value: "1"
      - name: NIM_PASSTHROUGH_ARGS
        value: "--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
```

Equivalent `--set` override when you do not use a values file. Helm `--set` replaces the `env` list, so include every Nemotron 3.5 Lightning environment entry and change only the `NIM_PASSTHROUGH_ARGS` value:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true \
  --set nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT \
  --set-string nimOperator.answer_llm.env[0].value=8000 \
  --set nimOperator.answer_llm.env[1].name=NIM_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[1].value=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set nimOperator.answer_llm.env[2].name=NIM_SERVED_MODEL_NAME \
  --set-string nimOperator.answer_llm.env[2].value=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set nimOperator.answer_llm.env[3].name=NIM_TENSOR_PARALLEL_SIZE \
  --set-string nimOperator.answer_llm.env[3].value=1 \
  --set nimOperator.answer_llm.env[4].name=NIM_PASSTHROUGH_ARGS \
  --set-string nimOperator.answer_llm.env[4].value="--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
```

After the NIM is Ready, confirm the passthrough arguments:

```bash
kubectl exec -n <namespace> deploy/answer-llm -- printenv NIM_PASSTHROUGH_ARGS
```

The value must include `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder`.

Forward the answer LLM for CLI use:

```bash
kubectl port-forward -n <namespace> service/answer-llm 9000:8000
```

Then run the remote command in [Remote OpenAI-compatible NIM or hosted endpoint](#remote-openai-compatible-endpoint). Point `--agentic-invoke-url` at `http://localhost:9000/v1/chat/completions` and set `--agentic-llm-model` to `nvidia/nemotron-3.5-lightning-30b-a3b`. Reuse the same embedding invoke URL and model name that you used at ingest.

For service-mode `POST /v1/query` with `agentic=true` and the MCP `agentic_query` tool, also set `serviceConfig.agentic`. The chart does not copy `answer_llm` into this block.

```yaml
serviceConfig:
  agentic:
    enabled: true
    llmModel: nvidia/nemotron-3.5-lightning-30b-a3b
    invokeUrl: http://answer-llm:8000/v1/chat/completions
```

`invokeUrl` uses the in-cluster Nemotron 3.5 Lightning service. Change the hostname if you override `nimOperator.answer_llm.nimServiceName`. `llmModel` is the model ID advertised by the NIM, not the LiteLLM `openai/` prefix used by `serviceConfig.llm.model`.

If you register MCP retrieval tools, also set `serviceConfig.mcp.enabled=true` and set `serviceConfig.mcp.queryMethods` to `agentic` or `all`. Helm leaves MCP disabled by default. Agentic MCP tools are omitted unless `serviceConfig.agentic.enabled` is true. Refer to [Enable MCP on Helm](#enable-mcp-on-helm).

For other self-hosted OpenAI-compatible NIMs, enable automatic tool choice and the parser that model requires. Nemotron 3.5 Lightning uses `qwen3_coder` for tool calls and `nemotron_v3` for reasoning.

For chart keys, refer to [Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm) in the Helm chart README.

## Enable agentic retrieval in the service { #enable-agentic-retrieval-in-the-service }

Retriever Service exposes agentic retrieval on `POST /v1/query` and
`POST /v1/answer` when `agentic.enabled` is true. Service mode requires remote
OpenAI-compatible LLM and embedding endpoints. Local in-process vLLM remains
available on the one-shot CLI path.

Enable agentic retrieval in `retriever-service.yaml`:

```yaml
agentic:
  enabled: true
  llm_model: nvidia/nemotron-3.5-lightning-30b-a3b
  invoke_url: https://your-llm.example/v1/chat/completions
  reasoning_effort: high
  backend_top_k: 20
  react_max_steps: 50
  request_timeout_s: 1800
```

`agentic.invoke_url` and `agentic.llm_model` are required when `agentic.enabled` is true. The VectorDB process owns the LanceDB volume and executes the agentic workflow. Start it with matching `--agentic`, `--agentic-llm-model`, and `--agentic-invoke-url` options. LLM and embedding credentials are resolved from the service process environment (`NVIDIA_API_KEY`, then `NGC_API_KEY`).

Agentic service requests use the configured remote embedding endpoint for retrieval. The result-selection graph does not require a local embedding model or Hugging Face cache.

On Kubernetes, the Helm chart maps the same knobs under `serviceConfig.agentic`. Enabling `nimOperator.answer_llm` does not populate this block. Refer to [Self-hosted Helm Nemotron 3.5 Lightning](#self-hosted-helm-lightning) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm).

The VectorDB service runs up to four non-agentic queries concurrently by default.
Set `--max-concurrent-queries` when starting `nemo_retriever.service.vectordb_app`
to use a different positive limit.

REST clients select document-ranking mode on `/v1/query`:

```bash
curl -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "find documents about parser behavior", "top_k": 5, "agentic": true}'
```

Set `agentic_mode` to `answer` on `/v1/query`, or set `mode` to `agentic` on
the gateway `/v1/answer` endpoint, to return an integrated answer:

```bash
curl -X POST http://localhost:7670/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"query": "explain the parser behavior", "top_k": 5, "mode": "agentic"}'
```

The Python service client exposes the same path through
`RetrieverServiceClient.agentic_answer()` and `aagentic_answer()`. Their request
timeout defaults to 1,800 seconds to match long-running agentic service
requests, and can be overridden with `request_timeout_s`.

When service auth is enabled, send `Authorization: Bearer <token>` (`NEMO_RETRIEVER_API_TOKEN`). Requests with `agentic: true` return HTTP `400` when agentic retrieval is not configured on the service.

`top_k` cannot exceed the configured `agentic.backend_top_k` (default 20). Agentic queries are capped at 4,096 characters.

## Query with MCP { #query-with-mcp }

`retriever service start` mounts a FastMCP HTTP endpoint when `mcp.enabled` is true. The default mount path is `/mcp`. Set `mcp.path`, or Helm `serviceConfig.mcp.path`, to use a different path. The bundled non-Helm `retriever-service.yaml` sets `mcp.enabled` to `true` by default. Model Context Protocol (MCP) agents can use that endpoint to call the running service for health checks, pipeline introspection, document ingestion, job status, VectorDB query, agentic retrieval, and answer generation. If service auth is enabled, the MCP endpoint uses the same bearer-token middleware as the REST API.

The Helm chart does not enable that mount. `serviceConfig.mcp.enabled` is `false`, so a chart-rendered service returns HTTP `404` at the MCP path until you opt in. Refer to [Enable MCP on Helm](#enable-mcp-on-helm).

Plain and agentic selection share `POST /v1/query` and the same hits response
envelope. Integrated agentic answering has a separate response contract. MCP
exposes distinct tools so agents can choose explicitly:

- `query` calls `POST /v1/query` with `agentic=false` for one-pass dense or hybrid retrieval.
- `agentic_query` calls `POST /v1/query` with `agentic=true` and runs the ReAct retrieval workflow. It is added to MCP when `agentic.enabled` is true.
- `agentic_answer` calls `POST /v1/answer` with `mode=agentic` and returns an
  integrated answer, validated citations, hydrated citation hits, status, error,
  and usage fields. It is added when `agentic.enabled` is true.

Use `--query-methods classic` (default), `agentic`, or `all` to choose which retrieval tools the MCP server registers. The mounted MCP endpoint uses the same knob through `mcp.query_methods` in the service config. Agentic tools are omitted unless `agentic.enabled` is also true.

For local stdio-based agents, run the MCP server as a shim that points at an existing retriever service:

```bash
retriever service mcp-stdio \
  --service-url http://localhost:7670 \
  --query-methods agentic \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```
For remote agents, expose the retriever service URL and configure the agent to connect to the MCP mount path. The default is:

```text
https://<retriever-service-host>/mcp
```

If you set `mcp.path` or Helm `serviceConfig.mcp.path`, use that configured path instead of `/mcp`. Helm does not mount the path until you set `serviceConfig.mcp.enabled=true`. Refer to [Enable MCP on Helm](#enable-mcp-on-helm).

The `ingest_documents` MCP tool accepts either paths visible to the MCP server process or inline `content_base64` document bytes. Use inline base64 for remote agents whose local files are not present on the service host.

### Enable MCP on Helm { #enable-mcp-on-helm }

The supported Helm chart renders `mcp.enabled: false`. Remote agents that connect to the MCP path receive HTTP `404` unless you enable the mount. The default path is `/mcp`. If you set `serviceConfig.mcp.path`, agents must use that path.

Add the following flag to your chart install:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set serviceConfig.mcp.enabled=true
```

`serviceConfig.mcp.queryMethods` selects which retrieval tools FastMCP registers: `classic` (default, `query` only), `agentic` (`agentic_query` only), or `all` (both). The `agentic_query` tool is omitted unless `serviceConfig.agentic.enabled` is also `true`. Enable both when remote agents must call agentic retrieval:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set serviceConfig.mcp.enabled=true \
  --set serviceConfig.mcp.queryMethods=all \
  --set serviceConfig.agentic.enabled=true \
  --set serviceConfig.agentic.llmModel=nvidia/nemotron-3.5-lightning-30b-a3b \
  --set serviceConfig.agentic.invokeUrl=http://answer-llm:8000/v1/chat/completions
```

The agentic `--set` values still require a remote chat-completions endpoint. For self-hosted Nemotron 3.5 Lightning, also add the tool-call passthrough arguments in [Self-hosted Helm Nemotron 3.5 Lightning](#self-hosted-helm-lightning).

Confirm the rendered ConfigMap before you rely on the endpoint:

```bash
helm template retriever ./nemo_retriever/helm \
  --set serviceConfig.vectordb.enabled=false \
  --set serviceConfig.mcp.enabled=true \
  | sed -n '/^    mcp:/,/^    llm:/p'
```

The block must show `enabled: true`. For chart keys, refer to [Service configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#service-configuration-rendered-into-retriever-serviceyaml) in the Helm chart README.

## Result contract { #result-contract }

Every agentic Retriever run writes a lightweight Agent Trajectory Interchange
Format (ATIF) JSON trajectory under `./agentic-traces` by default. The
trajectory bounds observation content to keep the file lightweight. These
traces are not added to HTTP responses. If a trace cannot be persisted,
retrieval continues and emits a warning.

One-pass retrieval returns text-enriched chunk hits. Agentic select mode ranks
documents. Each selected document is rehydrated from the retrieval hop that
returned it. Agentic answer mode returns an integrated answer and only permits
citation IDs that a retrieval hop returned during that run. CLI and service
output then use different JSON shapes.

CLI `retriever query` without `--agentic` projects each hit to five fields: `modality`, `page_number`, `score`, `source`, and `text`. CLI `retriever query --agentic` does not use that projection. It prints the internal hit dictionary plus these ranking annotations:

- `doc_id` — the document identifier the agent selected.
- `rank` — the position in the final ranking.
- `result_source` — `final_results`, `rrf`, or `selection_agent`, depending on which stage produced the ranked ID.

`modality` and `score` exist only on the dense CLI path. Agentic CLI objects can include internal fields such as `content_type`, `_distance`, `metadata`, `path`, `pdf_basename`, `pdf_page`, and `source_id` when the retrieval hop returned them.

When the agent names a document that no retrieval hop returned, the CLI object contains only `doc_id`, `rank`, and `result_source`. Classic hit keys are absent, not present with null values.

CLI `retriever query --agentic` keeps its default output as a JSON list of those
hits. Add `--include-usage` to return a JSON object that contains `hits` and
provider-reported LLM `usage`:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --include-usage
```

```json
{
  "hits": [
    {
      "doc_id": "parser-guide",
      "rank": 1,
      "result_source": "final_results"
    }
  ],
  "usage": {
    "input_tokens": 1250,
    "cache_tokens": 400,
    "output_tokens": 184,
    "total_tokens": 1434
  }
}
```

The `usage` object reports the exact token counts returned by the LLM provider.
It contains `input_tokens`, observed cache-read `cache_tokens`, `output_tokens`,
and `total_tokens`. `cache_tokens` is `null` when no stage reports cache usage.
When available, `stages` preserves the provider-reported breakdown for the
ReAct and final-selection calls, including cache-creation counters. When a
provider reports uncached, cache-creation, and cache-read input separately,
`input_tokens` includes all three counters. Cache tokens are therefore not
added again when calculating `total_tokens`. If the provider does not report
usage, the response sets `usage` to `null`.
`--include-usage` applies only to agentic queries. Classic `retriever query`
output is unchanged.

Service `POST /v1/query` with `agentic=true` maps those ranked hits onto the classic hits envelope and can include the same optional `usage` object at the response root. Successful responses set `query_mode` to `"agentic"`. Classic dense or hybrid `/v1/query` (including `format=evidence`) sets `query_mode` to `"classic"` and does not add usage metadata. For backward compatibility with the previous agentic service contract, service and MCP hits also copy `rank` and `result_source` under `metadata`; the top-level fields are authoritative and carry the same values.

When no retrieval hop captured the document, the service envelope fills these classic fields with null: `text`, `source_id`, `path`, `page_number`, `pdf_basename`, and `pdf_page`. `source` falls back to `doc_id`. That null-key behavior applies to service and MCP hits only, not to CLI `--agentic` output.

CLI `--agentic-mode answer`, `/v1/query` with
`{"agentic": true, "agentic_mode": "answer"}`, `/v1/answer` with
`{"mode": "agentic"}`, the Python client's `agentic_answer` methods, and the MCP
`agentic_answer` tool return the answer contract:

- `answer` — the integrated answer, or null when no answer was produced.
- `citations` — deduplicated IDs of documents retrieved during the run.
- `citation_hits` — those cited documents rehydrated in citation order.
- `succeeded`, `message`, and `error` — termination and diagnostic state.
- `query_mode` — `"agentic_answer"` on service responses.
- `usage` — normalized provider-reported token usage when requested or available.

Answer mode terminates inside the ReAct loop and intentionally bypasses
reciprocal rank fusion and the selection agent, because those stages produce a
document ranking rather than an answer.

## Failure and retry behavior { #failure-and-retry-behavior }

Operational failures from the agent LLM or retrieval tool, including embedding, vector database, and reranker endpoint failures, terminate the query with an error instead of returning a successful empty result.

An HTTP `400` from the chat-completions NIM with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` means the self-hosted endpoint is not tool-call ready. Refer to [Self-hosted Helm Nemotron 3.5 Lightning](#self-hosted-helm-lightning). The CLI then exits with `Agentic retrieval failed (llm_call_failed)`.

On the service:

- HTTP `400` when `agentic.enabled` is false.
- HTTP `422` when the query exceeds 4,096 characters, `format` is not `hits`, or `rerank` is combined with `agentic`.
- HTTP `501` when agentic service queries have no remote embedding endpoint.
- HTTP `503` with a `Retry-After: 30` header when every dedicated agentic worker is busy. The service sheds load instead of queueing behind a multi-minute run.
- HTTP `502` when the gateway cannot reach the VectorDB process.

Agentic runs use a dedicated worker pool in the VectorDB process so they cannot exhaust the capacity used by plain queries. A ReAct run cannot be interrupted once started, so a worker stays occupied until it finishes even if the caller times out or disconnects.

## Limitations and resource requirements { #limitations-and-resource-requirements }

- Local in-process agent LLMs are limited to the tested `nemotron-8b` and `super-49b` profiles. Custom in-process models require an OpenAI-compatible endpoint instead.
- Local CLI runs need a CUDA GPU host and the `[local]` extra. `super-49b` needs two visible GPUs and `--agentic-local-tensor-parallel-size 2`.
- Retriever Service agentic queries require a remote chat-completions URL, a remote embedding endpoint, and matching credentials in the process environment.
- The default Helm `answer_llm` Nemotron 3.5 Lightning NIM is limited to `POST /v1/answer` until you add the tool-call passthrough arguments. Enabling `nimOperator.answer_llm` does not configure `serviceConfig.agentic`.
- Helm leaves `serviceConfig.mcp.enabled` at `false`. Remote MCP agents require `--set serviceConfig.mcp.enabled=true` and must use the configured mount path, which defaults to `/mcp`. Refer to [Enable MCP on Helm](#enable-mcp-on-helm).
- Agentic ranking is document-level. Rehydrated hits include chunk `text` when a retrieval hop returned the document. Otherwise load the source document by `doc_id`.
- Service agentic queries accept a single query string, `format=hits` only, and cannot combine `rerank=true` on the same `/v1/query` request. On the CLI, `--rerank` applies to each agent retrieve hop.

## Related Topics { #related-topics }

- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Metadata and filtering](vdbs.md#metadata-and-filtering)
- [Evaluate on your data](evaluate-on-your-data.md)
- [Authentication and API keys](api-keys.md)
- [CLI reference: Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#agentic-retrieval)
- [Helm chart README: Agentic retrieval (self-hosted Nemotron 3.5 Lightning)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm)
- [Helm chart README: Service configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#service-configuration-rendered-into-retriever-serviceyaml)
- [Release notes](releasenotes.md#retrieval-and-rag)
