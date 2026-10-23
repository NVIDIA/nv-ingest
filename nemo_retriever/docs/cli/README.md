# Retriever CLI

This page describes the public `retriever` command-line workflow for document
ingest and retrieval.

For product-facing examples, prefer these commands:

- `retriever ingest` - ingest supported documents and media into a Retriever index.
- `retriever query` - query a local LanceDB table written by local or batch ingest.
- `retriever query service` - query a Retriever service deployment.
- `retriever service` - operate a Retriever service deployment.

Format names and internal stages are not root commands. Use `retriever ingest`
for PDF, HTML, TXT, image, Office, audio, and video inputs; it owns extraction,
embedding, and index creation as one workflow.

## Public ingest shape

`retriever ingest` defaults to local, in-process ingest:

```bash
retriever ingest DOCUMENTS...
```

Explicit modes are also available:

```bash
retriever ingest local DOCUMENTS...
retriever ingest batch DOCUMENTS...
retriever ingest service DOCUMENTS...
```

The root ingest CLI uses subcommands instead of a `--run-mode` flag. Choose
the command that matches where ingest runs and where results are stored.

| Command | What It Does | Writes To | Use When |
|---|---|---|---|
| `retriever ingest ...` | Local in-process ingest | local LanceDB | Default local ingest and CI/small corpus runs. |
| `retriever ingest local ...` | Local in-process ingest | local LanceDB | Same as the default, but explicit. |
| `retriever ingest batch ...` | Ray-backed batch ingest | local LanceDB | Larger or batch-tuned runs. |
| `retriever ingest service ...` | Sends documents to a Retriever service | service-configured storage | Remote service ingest. |

This separation keeps invalid flag combinations out of the parser. For example,
service ingest does not expose LanceDB target flags, Ray tuning, local endpoint
configuration, local embed backend selection, or local media controls.

<!-- --8<-- [start:quickstart] -->

> Use `retriever ingest` and `retriever query` for product-facing workflows.

## Quick start

Local ingest embeds on an available local GPU when `--embed-invoke-url` is unset. Install
the `[local]` extra before you run the default `retriever ingest` or
`retriever ingest batch` commands:

```bash
pip install "nemo-retriever[local]"
```

On this development branch, omitting `--embed-model-name` selects Nemotron 3.5 Embed 1B for local inference. These changes are not part of the documented 26.08.1 release. The default local backend is vLLM. For in-process ingest, NeMo Retriever Library selects `nvidia/Nemotron-3.5-Embed-1B-NVFP4` when every visible CUDA device supports native NVFP4. Otherwise, it selects `nvidia/Nemotron-3.5-Embed-1B-BF16`. With `--local-ingest-embed-backend hf`, it selects BF16 because the compressed-tensors NVFP4 checkpoint requires vLLM.

For local GPU embedding, `retriever ingest batch` automatically selects BF16 because Ray cluster GPU counts do not establish worker GPU architectures. Explicit Nemotron 3.5 precision checkpoint IDs remain unchanged. The CLI treats explicit Nemotron 3 precision IDs as the logical Nemotron 3 model and selects BF16 in batch mode. Select Nemotron 3.5 NVFP4 explicitly only when every eligible worker supports it; the driver's GPU alone does not establish that compatibility.

The Nemotron 3.5 integration remains experimental while checkpoint access is private. Its model-specific vLLM adapter preserves bidirectional text attention without patching vendor model classes. Validate retrieval quality before production use.

The default modality remains text. Nemotron 3.5 also supports image and combined text-and-image embeddings when you select the matching modality. Its prerelease repositories require access: set `HF_TOKEN` to a token from an authorized Hugging Face account. Do not add query or document prefixes yourself; the library applies the checkpoint's retrieval chat template.

Remote embedding and Helm retain `nvidia/nemotron-3-embed-1b` as the text-only default. Explicitly selected Nemotron 3 checkpoints retain their existing local behavior.

When graph execution selects the CPU embedding actor without an explicit URL, it uses `https://integrate.api.nvidia.com/v1/embeddings`. This hosted fallback retains the Nemotron 3 default, including the model identity recorded on the index. Set `NVIDIA_API_KEY` or `NGC_API_KEY` before ingest.

If you installed the base package for Remote NIM with no local GPU, keep that
install and pass `--embed-invoke-url` instead. Refer to
[Route ingest to hosted or self-hosted NIM endpoints](#route-ingest-to-hosted-or-self-hosted-nim-endpoints).

### Ingest a PDF locally

From a clone of this repository, `./data/multimodal_test.pdf` is a valid
first-run input. If you installed from PyPI, pass a PDF file that you supply.

```bash
retriever ingest ./data/multimodal_test.pdf
```

Then query the default LanceDB table:

```bash
retriever query "What is in this document?"
```

By default, local ingest auto-detects supported input formats and writes to
`lancedb/nemo-retriever`; `retriever query` reads from the same table. Use
explicit high-level options when a task needs behavior beyond the current ingest
defaults.

Python `.vdb_upload()` and default `Retriever()` use the same table.

The plain `retriever query` examples below apply to local and batch ingest output
written to LanceDB. Use `retriever query service` to query a Retriever service.

### Ingest a larger corpus with batch mode

Replace `/path/to/your/pdfs` with a directory of PDF files that you supply. The
repository and the PyPI package do not include a `pdf_corpus` dataset.

```bash
retriever ingest batch /path/to/your/pdfs \
  --profile fast-text \
  --pdf-extract-workers 4 \
  --embed-workers 2
```

Batch mode exposes Ray runtime and batch tuning flags such as `--ray-address`,
`--pdf-extract-workers`, `--ocr-workers`, and `--embed-workers`.

### Ingest through a Retriever service

Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest service /path/to/your/pdfs \
  --service-url http://localhost:7670 \
  --service-concurrency 8
```

Use `--service-api-token` or `NEMO_RETRIEVER_API_TOKEN` when the service requires
a bearer token. Service ingest does not expose `--lancedb-uri`; the service
configures its vector database. Query the service with:

```bash
retriever query service "What is in this corpus?" \
  --service-url http://localhost:7670
```


### Start a local service with VectorDB

Use `retriever service start --launch-vectordb` to run a local service with a supervised VectorDB child on `127.0.0.1:7671`. The child uses `nim_endpoints.embed_invoke_url` when configured. Otherwise, it uses local Hugging Face embedding when `local_models.enabled` and `local_models.embed.enabled` are both `true`. The command waits for VectorDB readiness and terminates the child when the service exits. If the child does not exit promptly, the service forcefully stops it. You can set the same behavior in YAML with `vectordb.launch_on_start: true` and a loopback `vectordb.vectordb_url`.

The child inherits credentials from the service environment. Set `NVIDIA_API_KEY` or `NGC_API_KEY` for remote embedding, and set `NRL_INTERNAL_VDB_TOKEN` or `NRL_INTERNAL_VDB_TOKEN_FILE` for the VectorDB internal credential. Do not place credentials in the service YAML.

For a fully local deployment, use a CUDA-capable host and install the service and local extras. Install the `multimedia` extra when you ingest audio or video.

```bash
pip install "nemo-retriever[service,local]"
scripts/launch_local_service_with_vectordb.sh \
  nemo_retriever/examples/retriever-service.local.yaml
```

The example configuration leaves NIM endpoints unset and uses local Hugging Face models. The launcher validates `/v1/health` on the service and VectorDB, then keeps both processes running until you stop it with `Ctrl+C`.

```bash
retriever service start --config my-retriever-service.yaml --launch-vectordb
```

Use the command above when `nim_endpoints.embed_invoke_url` is configured. Omit the flag to use an existing VectorDB. Helm continues to deploy VectorDB as a separate pod.

If VectorDB exits during startup or does not become ready, inspect the VectorDB output in the terminal that started the service. Verify the VectorDB configuration, embedding model setup and credentials, writable LanceDB directory, and that port `7671` is available.

### Route ingest to hosted or self-hosted NIM endpoints

```bash
export NVIDIA_API_KEY=nvapi-...

retriever ingest ./data/multimodal_test.pdf \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/nemotron-3-embed-1b
```

`NVIDIA_API_KEY` is required only when those URLs point at hosted
build.nvidia.com endpoints. `NGC_API_KEY` is used separately when pulling or
running self-hosted NIM containers.

To rerank local query results with the hosted vision-language reranker, pass the
NVIDIA-hosted `/reranking` endpoint and model. Use the same `NVIDIA_API_KEY` that
authorizes the hosted embedding URL:

```bash
retriever query "What is in this document?" \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/nemotron-3-embed-1b \
  --reranker-invoke-url https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking \
  --reranker-model-name nvidia/llama-nemotron-rerank-vl-1b-v2 \
  --reranker-api-key-env NVIDIA_API_KEY
```

Passing `--rerank` without `--reranker-invoke-url` uses the local GPU reranker,
not this hosted endpoint.

A Cohere-style `/v1/rerank` URL is a gateway route, not an NVIDIA-hosted NIM
endpoint. Pass the full URL that your gateway exposes, for example
`https://<your-gateway>/v1/rerank`, and the model name that gateway expects.
Set `--reranker-api-key-env` to an environment variable that holds a credential
issued by that gateway. A gateway that expects a LiteLLM virtual key that starts
with `sk-` rejects NVIDIA `nvapi-` keys and NGC keys.

### Query result controls

Both `retriever query` and `retriever query service` return compact JSON hits
with `source`, `page_number`, and `text`. Use `--candidate-k`, `--page-dedup`,
and `--content-types` to control how results are selected after vector
retrieval:

```bash
retriever query "annual revenue by region" \
  --top-k 5 \
  --candidate-k 40 \
  --content-types table
```

`--top-k` is the final number of results to return after filtering and
deduplication. `--candidate-k` is the number of raw results to retrieve from
LanceDB or the Retriever service before filtering, page deduplication, and
final truncation. If omitted, the candidate pool is the same size as
`--top-k`. Set `--candidate-k` larger than `--top-k` when page deduplication
or content-type filtering might remove too many of the nearest retrieved rows.
It must always be greater than or equal to `--top-k`.

Page deduplication and content-type filtering are applied after vector
retrieval, preserving retriever ranking order and truncating the final output to
`--top-k`. Local and batch ingest record the canonical embedding model on the
LanceDB table, and non-service query uses that model automatically. Use
`--embed-model-name` only as an explicit override. A dense or hybrid query
rejects the request if that override differs from the model recorded on the
table. It also rejects a legacy or third-party table without model metadata,
because the query cannot verify its embedding space. Sparse queries are exempt
because they do not create dense query vectors. Endpoint URLs and provider
prefixes remain runtime configuration, so continue to pass
`--embed-invoke-url` and `--embed-model-provider-prefix` when the selected model
must be routed remotely.
For example, a table can store the canonical model
`nvidia/nemotron-3-embed-1b` while a LiteLLM-routed request uses
`nvidia/nemotron-3-embed-1b`. The endpoint and routing prefix
are intentionally not persisted on the table.

The local checkpoint default changes to Nemotron 3.5 Embed 1B on this development branch. This does not change an existing table's recorded model. Before
you migrate a persistent table, back up its LanceDB directory and retain the
original corpus. To adopt the new default for a CLI-managed table, re-ingest the
complete corpus with the same URI and table name and pass `--overwrite`
explicitly. Do not use `--append` to mix embeddings from the two models. You can
continue to query a tagged old table with its recorded model, but an untagged
dense or hybrid table must be rebuilt. For service deployments, rebuild persisted
legacy and collection tables before access or configure the service with the old
index model. Refer to [Keep the embedding model aligned](https://docs.nvidia.com/nemo/retriever/latest/extraction/vdbs/#lancedb-embedding-model-compatibility).

`--content-types` accepts comma-separated content types such as `text`, `table`,
`chart`, `image`, and `infographic`. `images` is accepted as an alias for
captioned image rows emitted by ingest. This option filters by content-type
metadata only; it does not filter by source, page, or other metadata
predicates. Hits with missing or unknown content-type metadata are excluded
while `--content-types` is active. In service mode, results must include
content-type metadata to match this filter. Default display values in the JSON
output are not used for content-type matching.

### Agentic retrieval

`--agentic` swaps the single dense pass for an LLM-driven ReAct loop: the agent
issues several retrieval sub-queries, fuses the candidates, and selects a final
ranking. It searches the same LanceDB table built by `retriever ingest`. You can
reuse the same table, embedding flags, and `--top-k` as standard retrieval.
The JSON hit shape is not a drop-in replacement for dense `retriever query`
output.

By default, agentic retrieval runs the agent LLM in process with local vLLM and
`nemotron-8b` (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`). This requires a CUDA GPU
host and the local extras installed. Provide `--agentic-invoke-url` when you want
a custom model or a separately hosted OpenAI-compatible endpoint.

```bash
# default local vLLM agent LLM: nemotron-8b
retriever query "how does the ingestion pipeline handle tables?" \
  --agentic

# custom/self-hosted model through an OpenAI-compatible endpoint
retriever query "summarize the deployment options" \
  --agentic \
  --agentic-llm-model custom-remote-model \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions \
  --embed-invoke-url http://localhost:8000/v1 \
  --agentic-react-max-steps 5

# integrated answer plus validated citation IDs and hydrated citation hits
retriever query "what changed in the latest report?" \
  --agentic \
  --agentic-mode answer \
  --include-usage
```

Agentic select mode returns the agent's ranked documents as JSON. The dense path
projects each hit to five fields: `modality`, `page_number`, `score`,
`source`, and `text`. Agentic mode does not use that projection. It prints
the internal hit dictionary plus `doc_id`, `rank`, and `result_source`.
`result_source` is `final_results`, `rrf`, or `selection_agent`, depending
on which stage produced the ranking.
`modality` and `score` exist only on the dense path. Fields such as
`content_type`, `_distance`, `metadata`, `path`, `pdf_basename`,
`pdf_page`, and `source_id` appear on the agentic path when the retrieval
hop returned them.

Hit fields are rehydrated at the end of the loop from the retrieval hop
that returned the document. When the agent names a document that no
retrieval hop returned, the object contains only `doc_id`, `rank`, and
`result_source`. Classic hit keys such as `text` and `source` are
absent. They are not present with null values.

Agentic retrieval reuses the same `--top-k`, `--lancedb-uri`, `--table-name`,
`--embed-invoke-url`, and `--embed-model-name` options as standard retrieval.
Agentic retrieval uses the selected table's model automatically when
`--embed-model-name` is omitted.

The default `retriever query --agentic` select-mode output remains a JSON hits
list. Add `--include-usage` to print a JSON object with `hits` and exact
provider-reported LLM usage:

```bash
retriever query "how does the ingestion pipeline handle tables?" \
  --agentic \
  --include-usage
```

```json
{
  "hits": [
    {
      "doc_id": "ingestion-guide",
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

The `usage` object reports observed cache reads as `cache_tokens` and can also
include `stages`, which preserves the provider-reported breakdown for the ReAct
and final-selection calls, including cache creation. When a provider reports
uncached, cache-creation, and cache-read input separately, `input_tokens`
includes all three counters; cache is not added again to `total_tokens`.
`cache_tokens` is `null` when no stage reports cache usage. The output sets
`usage` to `null` when the LLM provider does not report it. This flag applies
only with `--agentic`; classic query behavior and output are unchanged.

**How it works.** Each agentic query runs `Query -> ReActAgentOperator -> (RRF
fusion) -> SelectionAgentOperator -> ranked results`:

- `ReActAgentOperator` runs the per-query ReAct loop; every `retrieve` tool call
  delegates to the standard `Retriever`, so the agent searches the same vector
  DB and embedding config as dense retrieval.
- `RRFAggregatorOperator` fuses candidates from the loop's multiple searches with
  reciprocal rank fusion.
- `SelectionAgentOperator` runs a final LLM selection pass over the fused set and
  emits ranked document IDs. Those IDs are then rehydrated from the retrieval-hop
  hit dictionary.

`--agentic-mode answer` uses the same iterative retriever but ends inside the
ReAct loop with an integrated answer. It intentionally bypasses RRF and the
selection agent because those stages produce a document ranking, not an answer.
The output contains `answer`, `citations`, `citation_hits`, `succeeded`, `error`,
and optional `usage`. Citation IDs must have been returned by a retrieval hop;
`citation_hits` preserves citation order and contains the corresponding full
retrieval metadata. In answer mode, `--include-usage` adds `usage` to that answer
object; it does not return the select-mode `{ "hits": ..., "usage": ... }`
envelope.

Agentic-only knobs (apply only with `--agentic`):

- `--agentic-mode` (default `select`) — return ranked documents with `select`, or
  an integrated answer and citations with `answer`.
- `--agentic-llm-model` — local profile alias/model ID when no invoke URL is
  provided (`nemotron-8b` by default; `super-49b` also supported), or the remote
  model ID when `--agentic-invoke-url` is provided.
- `--agentic-local-tensor-parallel-size` (default `1`) — vLLM
  `tensor_parallel_size` for the in-process agent LLM. Use `2+` with matching
  `CUDA_VISIBLE_DEVICES` for multi-GPU local profiles (for example
  `super-49b`). Ignored when `--agentic-invoke-url` is set. When the first
  `tensor_parallel_size` CUDA-visible GPUs are not NVLink-connected (typical
  dual-GPU PCIe workstations), tensor-parallel startup automatically sets
  `NCCL_NVLS_ENABLE=0` and `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`, because NVLink
  multicast collectives abort vLLM startup there; set either variable yourself
  to override. Detection is scoped to that TP device group, not the whole host
  or extra visible GPUs outside the shard.
- `--agentic-invoke-url` — OpenAI-compatible chat-completions endpoint for the
  agent LLM. Providing it routes agent LLM calls to that remote endpoint; omit it
  to run the in-process local model.
- `--agentic-llm-client` (optional) — LLM client that builds the agent LLM.
  Defaults to `callable`. It drives the in-process
  adapter when `--agentic-invoke-url` is omitted, and the shared chat-completions
  HTTP client when it is set.
- `--agentic-reasoning-effort` (default `high`) — `reasoning_effort` forwarded on
  OpenAI-compatible agentic LLM calls; ignored by the local adapter.
- `--agentic-react-max-steps` (default `50`) — maximum ReAct loop iterations.
- `--agentic-text-truncation` (default `0`) — max characters of each candidate
  shown to the agent; `0` disables truncation.
- `--agentic-temperature` (default: unset) — sampling temperature for agent LLM
  calls; omit to use the endpoint/model default (`0.0` = greedy). Local and
  non-NVIDIA OpenAI-compatible endpoints allow up to `2.0`; NVIDIA-hosted
  endpoints allow up to `1.0`.
- `--include-usage` (default: off) — in select mode, replace the default hits
  list with an object containing `hits` and provider-reported LLM `usage`; in
  answer mode, add `usage` to the answer, citations, status, and error object.

<!-- --8<-- [end:quickstart] -->

## Common ingest options

### Local and batch ingest

These options apply to `retriever ingest`, `retriever ingest local`, and
`retriever ingest batch` unless otherwise noted.

| Option | Default | Notes |
|---|---|---|
| `DOCUMENTS...` | required | Files, directories, or shell globs. Supported file families are detected automatically. |
| `--profile` | `auto` | `auto` uses manifest-routed ingest and selects `pdfium_hybrid` for PDFs. `fast-text` selects `pdfium` and disables Page Elements, image, table, and chart extraction for text-only PDFs. |
| `--lancedb-uri` | `lancedb` | LanceDB database URI. |
| `--table-name` | `nemo-retriever` | LanceDB table name. Must match query-time storage flags. Python `.vdb_upload()` and default `Retriever()` use the same default. |
| `--overwrite/--append` | overwrite | Overwrite the table by default; use `--append` to add rows. |
| `--index-mode` | `auto` | Recommended: leave this unset. `auto` creates a hybrid vector + BM25/FTS configuration for new tables and preserves an existing table on append. Use `dense`, `hybrid`, or `sparse` only for explicit experiments or specialized deployments. |
| `--embed-model-name` | unset | Local GPU inference selects Nemotron 3.5 Embed 1B: NVFP4 with native-capable GPUs and vLLM, or BF16 otherwise. Ray batch GPU embedding automatically selects BF16. Explicit Nemotron 3.5 precision IDs remain unchanged. The CLI treats explicit Nemotron 3 precision IDs as the logical Nemotron 3 model and selects BF16 for batch GPU embedding. Remote inference, including implicit hosted CPU fallback, retains `nvidia/nemotron-3-embed-1b`. |
| `--local-ingest-embed-backend` | `vllm` | Local embedding backend. With no model override, explicit `hf` use selects the Nemotron 3.5 BF16 checkpoint. |
| `--method` | profile default | PDF extraction method: `pdfium`, `pdfium_hybrid`, `ocr`, or `nemotron_parse`. The `auto` profile selects `pdfium_hybrid`; `fast-text` selects `pdfium`. An explicit value overrides the profile-selected method. |
| `--extract-text`, `--extract-tables`, `--extract-charts` | planner default | Enable or disable extraction families. |
| `--ocr-version` | planner default | OCR engine version for local extraction. |
| `--ocr-lang` | planner default | OCR v2 language selector for local extraction. |
| `--caption` | off | Add a captioning stage. |
| `--caption-model-name` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | Local vLLM caption model. The default has approximately 62 GiB of BF16 weights. On a dedicated 80 GB GPU, its local profile reserves `0.95` of GPU memory for vLLM model and KV-cache use. Nano models retain the `0.5` profile default and remain available as explicit overrides. For remote endpoints, pass the endpoint API model ID. |
| `--caption-gpu-memory-utilization` | model profile | Fraction of a local caption GPU that vLLM can reserve. The Omni BF16 profile defaults to `0.95`; other local caption profiles default to `0.5`. Use this option only with `--caption` and local vLLM captioning. |
| `--dedup/--no-dedup` | automatic with captioning | Captioning a non-image document automatically enables content-hash and bounding-box image deduplication with an intersection over union threshold of `0.45`. Use `--dedup` to enable the default behavior explicitly, or use `--no-dedup` to disable both mechanisms and preserve all extracted image crops. Image-only inputs are exempt from automatic deduplication. |
| `--text-chunk` | off | Enable token chunking during extraction. |
| `--store-images-uri` | unset | Store extracted images at a local path or fsspec-compatible URI. |
| `--dry-run` | off | Print the resolved ingest plan without creating an ingestor. |
| `--quiet/--no-quiet` | quiet | Suppress verbose progress output by default. |

Batch-only options include `--ray-address`, `--ray-log-to-driver`,
`--pdf-split-batch-size`, `--pdf-extract-workers`, `--ocr-workers`,
`--table-structure-workers`, `--nemotron-parse-workers`, `--embed-workers`, and
related batch-size / CPU / GPU tuning flags.

### Service ingest

`retriever ingest service` exposes only service-supported request controls.
It does not expose LanceDB target flags, Ray tuning, local endpoint URLs/API
keys, local embed backend selection, `--ocr-lang`, or local audio/video media
controls.

| Option | Default | Notes |
|---|---|---|
| `DOCUMENTS...` | required | Files, directories, or shell globs sent to the service client. |
| `--service-url` | `http://localhost:7670` | Retriever service base URL. |
| `--service-concurrency` | `8` | Maximum concurrent document uploads. |
| `--service-api-token` | env fallback | Bearer token; also reads `NEMO_RETRIEVER_API_TOKEN`. |
| `--profile` | `auto` | Same profile names as local and batch ingest where supported. |
| `--caption`, `--text-chunk` | off | Service-supported ingest controls. |
| `--dedup/--no-dedup` | automatic with captioning | Explicitly enable or disable image deduplication for captioned non-image documents. Image-only inputs are exempt from automatic deduplication. |
| `--store-images-uri` | unset | Service-accessible image storage URI. |
| `--dry-run` | off | Print the resolved service ingest request. Tokens are redacted. |

## Examples

### Custom LanceDB location

```bash
retriever ingest ./data/multimodal_test.pdf \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

```bash
retriever query "What is in this document?" \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

### Fast text-only PDF fallback

Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest /path/to/your/pdfs \
  --profile fast-text
```

### Dense Nemotron embedding checkpoints

`--embed-model-name` accepts a Hugging Face repository ID or an on-disk
checkpoint compatible with a supported dense Nemotron text or vision-language
embedding profile:

```bash
retriever ingest /path/to/your/pdfs \
  --embed-model-name acme/my-finetuned-nemotron-embed
```

Supported official checkpoint profiles include the following:

- `nvidia/llama-3.2-nv-embedqa-1b-v2`
- `nvidia/llama-nemotron-embed-1b-v2`
- `nvidia/llama-nemotron-embed-vl-1b-v2`
- `nvidia/llama-nemotron-embed-vl-1b-v2-fp8`
- `nvidia/llama-nv-embed-reasoning-3b`
- `nvidia/llama-embed-nemotron-8b`
- `nvidia/Nemotron-3-Embed-1B-BF16`
- `nvidia/Nemotron-3-Embed-8B-BF16`
- `nvidia/Nemotron-3-Embed-1B-NVFP4`
- `nvidia/Nemotron-3.5-Embed-1B-BF16`
- `nvidia/Nemotron-3.5-Embed-1B-NVFP4`

Equivalent local checkpoints and weight-only fine-tunes are supported. A
compatible checkpoint must be complete and loadable, declare average pooling,
and expose a positive output width. Supported architectures include
`LlamaBidirectionalModel`, `LlamaNemotronVLModel`, `Mistral3Model` embedding checkpoints, and compatible Ministral 3
dense encoders. A compatible Ministral 3 dense checkpoint must declare
`model_type: "ministral3"`, `architectures: ["Ministral3Model"]`,
`is_causal: false`, `pooling: "avg"`, and a positive `hidden_size`. LanceDB
infers the schema from the produced vectors; the tested official checkpoints
use 2048, 3072, and 4096 dimensions. Query and document prompts are read from
`config_sentence_transformers.json` when the checkpoint supplies it.
Fine-tunes that require prefixes other than `query: ` and `passage: ` must
retain that prompt metadata.

Nemotron 3.5 uses `model_type: "mistral3"`, `architectures: ["Mistral3Model"]`, and a bidirectional text encoder with masked mean pooling. Its output width comes from `text_config.hidden_size`. The processor supplies its retrieval chat template and image preprocessing. Generic Mistral 3 generation checkpoints are not supported by this embedding profile.

For local Nemotron 3.5 paired text-and-image ingestion, the configured passage limit includes the template, processor-expanded image tokens, and text. Oversized paired text is split losslessly, with the unchanged full image repeated for each child. If the image and template leave no text budget, the library raises an error rather than truncating the image. Refer to [Embedding input limits](../../../docs/docs/extraction/embedding.md#text-input-overflow) for the policy and split metadata.

General Mistral 3 Base, Instruct, and vision-language model (VLM) generation
checkpoints remain unsupported because they do not match the dense encoder
contract. NeMo AutoModel `ministral3_bidirec` checkpoints also remain
unsupported. Other unsupported Nemotron RAG models include rerankers, ColEmbed
late-interaction models, Omni Embed, OCR, and parsing models. These models
require different dependencies, outputs, modalities, or operator contracts.

Unregistered Hub repositories are resolved to an immutable commit and loaded
with `trust_remote_code=True`; only use repositories you trust. The resolved
model name and revision are recorded on the LanceDB table and reused by local
query.

For a compatible ModelOpt or compressed-tensors checkpoint, including FP8 or NVFP4 variants, select
vLLM for ingest. Local query detects the quantization configuration and selects
vLLM automatically:

```bash
retriever ingest /path/to/your/pdfs \
  --embed-model-name /models/my-finetuned-nemotron-embed-fp8 \
  --local-ingest-embed-backend vllm

retriever query "What is in this corpus?" \
  --table-name nemo-retriever
```

Hugging Face remains the local query backend for checkpoints that do not require vLLM quantization support.
Local directories must contain `config.json`, and their absolute path must be
available to every Ray worker or service replica that loads the model.

### PDF extraction method

Use `--method` to select how the CLI extracts text from PDF pages. The default
`auto` profile selects `pdfium_hybrid` so that scanned pages can use OCR without
bypassing Page Elements. An explicit `--method` value overrides the method
selected by the profile.

- `pdfium` extracts native PDF text. It does not use OCR as a fallback for
  scanned-page text. Page Elements still supports enabled table and chart
  extraction, but it does not recover page text in this method.
- `pdfium_hybrid` extracts native text from text-bearing pages. For pages
  classified as scanned, it uses Page Elements and OCR to recover page text.
- `ocr` uses Page Elements and OCR for PDF page text on every page.
- `nemotron_parse` uses the Nemotron Parse visual extraction path instead of
  the Page Elements and OCR path.

The `fast-text` profile is the explicit text-only exception. It selects
`pdfium` and disables Page Elements, page rendering, image extraction, table
extraction, and chart extraction.

For example, select hybrid extraction for a PDF that contains scanned pages:

```bash
retriever ingest ./data/scanned.pdf \
  --method pdfium_hybrid
```

`--ocr-version` and `--ocr-lang` configure the local OCR engine when an enabled
stage uses OCR. These options do not select a PDF extraction method.

### OCR language mode

```bash
retriever ingest ./data/multimodal_test.pdf \
  --ocr-version v2 \
  --ocr-lang english
```

For mixed-script documents, use `--ocr-lang multi` where supported by the local
OCR engine.

### Text chunking

```bash
retriever ingest ./data/test.pdf \
  --text-chunk \
  --text-chunk-max-tokens 512 \
  --text-chunk-overlap-tokens 64
```

### Captioning and image storage

Captioning a non-image document automatically enables content-hash and bounding-box image deduplication before captioning. Bounding-box deduplication uses an intersection over union threshold of `0.45` and retains structured table, chart, or infographic regions instead of overlapping raw image crops. Image-only documents do not enable deduplication automatically.

```bash
retriever ingest ./data/test.pdf \
  --caption \
  --caption-invoke-url https://integrate.api.nvidia.com/v1/chat/completions \
  --api-key "${NVIDIA_API_KEY}" \
  --store-images-uri ./processed_docs/images
```

Use `--dedup` to enable the default behavior explicitly. Use `--no-dedup` when you need to preserve and caption every extracted image crop:

```bash
retriever ingest ./data/test.pdf \
  --caption \
  --no-dedup
```

`--dedup-iou-threshold` requires an explicit `--dedup`. You cannot combine the threshold option with `--no-dedup`.

Disabling deduplication can increase the number of caption requests, processing latency, and model cost. Use `--dry-run` to inspect the effective deduplication configuration, including automatic caption deduplication and an explicit `--no-dedup` override.

For mixed service inputs, dry-run output also includes `dedup_scope`, which lists enabled non-image families and exempt image families.

For local Hugging Face Omni BF16 captioning, use a dedicated GPU. The default
profile reserves `0.95` of GPU memory so that vLLM can allocate both the model
and its KV cache. Override that reservation when your deployment requires it:

```bash
retriever ingest ./data/test.png \
  --caption \
  --caption-gpu-memory-utilization 0.95
```

An 80 GB requirement for a self-hosted Omni NIM does not by itself establish
that direct local Hugging Face vLLM inference has sufficient KV-cache capacity.

## Results and diagnostics

Local and batch ingest report the number of input files and LanceDB rows written:

```text
Ingested 20 file(s) -> 1884 row(s) in LanceDB lancedb/nemo-retriever.
```

Service ingest reports the row count returned by the service result when
available:

```text
Ingested 20 file(s) -> 1940 row(s) through retriever service http://localhost:7670.
```

Use `--dry-run` on any ingest mode to inspect the resolved request without
creating an ingestor or contacting the service.
