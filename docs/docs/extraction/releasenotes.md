# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.08.2 Helm Chart Patch { #release-26082 }

The 26.08.2 Helm chart no longer sets `NIM_SERVER_MODE`,
`NIM_SERVER_MAX_WAIT_MS`, or `NIM_PIPELINE_MAX_BATCH_SIZE` for the default
Page Elements, Table Structure, and OCR NIMs. Those NIMs use their image
defaults for these settings. If you need to override one of them, set the
complete `nimOperator.<key>.env` list because Helm replaces environment-variable
lists rather than merging them.

## 26.08.1 Release Notes (26.8.1) { #release-26081 }

NVIDIA® NeMo Retriever Library version 26.08.1 includes a shared text-generation task API, configurable large language model (LLM) settings, grounded answer-generation model paths, agentic retrieval, and updated Helm NIM defaults. It builds on the graph ingest, multimodal extraction, and Helm-first deployment foundation.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md).

The following sections summarize user-visible changes included in 26.08.1 and foundational capabilities that remain current.

### Upgrade notes { #upgrade-notes }

- Nemotron OCR v2 is now the default OCR engine for local Hugging Face and hosted CPU actors. For Helm NIM deployments, Nemotron OCR v2 is the default. The previous release kept Helm on OCR v1. Refer to [Default Helm NIMs](prerequisites-support-matrix.md#default-helm-nims) for the chart image repository and tag.
- Helm replaces separate page-elements and table-structure NIMs with the combined `nemotron-object-detection:2.0.1` image. Development Compose uses the same combined object-detection image and OCR v2, but still defaults to `2.0.0` tags unless you override `NIM_*_TAG`.
- Helm changes its default embedding NIM from Llama Nemotron Embed VL 1B v2 to `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2`. The new default is text-only. The optional VL reranker remains on `2.3.0`. Development Compose uses the same Nemotron 3 Embed 1B image and logical model ID. Its optional VL reranker remains on the existing image unless you override `NIM_RERANK_TAG`.
- Before upgrading a persistent LanceDB deployment to the new embedding default, back up the table and re-ingest the complete corpus. For CLI-managed tables, pass `--overwrite` explicitly. For a service deployment, rebuild the persisted table before startup or explicitly configure `serviceConfig.vectordb.embedModel` with the model used to create a tagged existing table. Untagged dense or hybrid tables must be rebuilt; local sparse queries are exempt. Refer to [Keep the embedding model aligned](vdbs.md#lancedb-embedding-model-compatibility).
- Default VLM image captioning is Nemotron 3 Nano Omni for local and hosted paths. Chart-classified PDF regions remain on the layout and OCR path.
- Hosted Nemotron Parse and self-hosted Nemotron Parse use distinct HTTP contracts. Select the matching client path for your endpoint.
- macOS Intel (x86_64) is no longer supported for package installs. Use Apple Silicon (arm64) macOS, Windows x64, or Linux. Refer to [Packaging and platform](#packaging-and-platform).
- Legacy `nv-ingest` and compatibility pipeline CLI code paths are removed. Use `retriever ingest` and the graph stage registry.
- Self-hosted Parakeet on Helm requires both `nimOperator.audio.enabled=true` and `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051`. Enabling the audio NIM alone does not wire the service ASR endpoint.
- Changing a Helm NIM image repository or tag on an existing release cannot patch `NIMCache` `spec.source.ngc.modelPuller`. Delete the `NIMCache` and its PVC, then upgrade. The affected NIM is unavailable while the operator re-caches weights. Refer to [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).
- A document whose VectorDB write is not acknowledged now fails instead of reporting `completed` with a positive row count. Earlier builds failed only collection-managed writes and logged a legacy fixed-table failure as a warning. The worker acknowledgement timeout is configurable through `serviceConfig.vectordb.writeTimeoutSeconds` (rendered as `vectordb.write_timeout_s`) and defaults to 300 seconds. Refer to [Ingest fails with a VectorDB write error](troubleshoot.md#vectordb-write-not-acknowledged).
- Retriever Service OpenAPI `info.version` no longer reports a stale package-version value. The service reports the package version, and Helm sets `RETRIEVER_SERVICE_VERSION` from the running service image tag so `/openapi.json` matches the deployed release.

### Text generation and LLM configuration { #text-generation-and-llm-configuration }

- 26.08.1 includes a shared one-request-per-row text-generation abstraction: `TextGenerationTask` plus `TextGenerationOperator`. `GenericGenerationOperator` accepts a validated custom prompt. Refer to [One-shot text generation](nemo-retriever-api-reference.md#one-shot-text-generation).
- `SummarizeTask` inherits from `TextGenerationTask`. `SummarizationOperator` provides the built-in summarization behavior with the default prompt or a custom prompt.
- Configure those operators with `TextGenerationParams.from_kwargs(...)`. Supported fields include `model`, `api_base`, `api_key`, `temperature`, `top_p`, `max_tokens`, `extra_params`, `num_retries`, `timeout`, `prompt`, `system_prompt`, `rag_system_prompt`, `rag_system_prompt_prefix`, `reasoning_enabled`, and `max_workers`. Refer to [TextGenerationParams configuration](nemo-retriever-api-reference.md#textgenerationparams-configuration).

### Answer generation { #answer-generation }

- `Retriever.answer()` and optional `POST /v1/answer` remain the grounded answer-generation path. The default LLM is `nvidia/nemotron-3.5-lightning-30b-a3b` (Helm `nimOperator.answer_llm` image `nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant`). The generic slot also accepts another OpenAI-compatible LLM or vision-language model (VLM), including `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.
- Enabling the Omni caption Helm key does not enable `/v1/answer`. Use Omni as the answer backend by overriding the generic `answer_llm` slot or by pointing `serviceConfig.llm` at an Omni chat-completions endpoint. Refer to [Answer generation](prerequisites-support-matrix.md#answer-generation) and [Answer generation (operator-managed LLM)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#answer-generation-llm).

### Agentic retrieval { #agentic-retrieval }

- Agentic retrieval is available in 26.08.1. An LLM agent issues multiple searches, fuses candidates, and returns a document-level ranking. The CLI, Python query workflow, REST, and MCP surfaces share this path. Refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- `retriever query --agentic` runs that ReAct loop over the same LanceDB table as one-pass retrieval. Local CLI and harness runs default to in-process vLLM (`nemotron-8b`). Remote OpenAI-compatible NIM or NVIDIA-hosted endpoints use `--agentic-invoke-url`.
- Retriever Service exposes agentic retrieval on `POST /v1/query` with `agentic=true` and an `agentic_query` MCP tool when `agentic.enabled` is true. Service mode requires a remote OpenAI-compatible LLM endpoint. Agentic remains opt-in through `serviceConfig.agentic.enabled`.
- The Helm `answer_llm` Nemotron 3.5 Lightning NIM auto-wires `/v1/answer` only and enables the `nemotron_v3` reasoning parser. Self-hosted agentic retrieval against that NIM requires `--enable-auto-tool-choice --tool-call-parser qwen3_coder` on `NIM_PASSTHROUGH_ARGS` and explicit `serviceConfig.agentic` wiring. Refer to [Self-hosted Helm Nemotron 3.5 Lightning](workflow-agentic-retrieval.md#self-hosted-helm-lightning).
- Configurable auto-retrieval is available on the service query path, with evidence and coverage output formats on `/v1/query`. MCP query-method selection and rerank tools are available.

### Models, OCR, and NIM artifacts { #models-ocr-and-captioning }

- Nemotron OCR v2 is unified across library, hosted, and Helm defaults. Hosted OCR uses its own language behavior. Refer to [Default Helm NIMs](prerequisites-support-matrix.md#default-helm-nims) and [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#ocr-nim-configuration) for the chart image.
- Local OCR crop batching runs across page rows for throughput. The Helm embed NIM sets `NIM_ENGINE_COUNT=1` by default. `NIM_PERFORMANCE_MODE=1` is available as an optional commented setting for supported deployments.
- 26.08.1 Helm default and optional NIM images that affect mirroring, allowlisting, and troubleshooting include the following:
    - Combined object detection for page elements and table structure: `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1`
    - Image OCR: `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1`
    - Text embedding: `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2`
    - VL reranking (optional): `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:2.3.0`
    - Optional Omni caption and configurable answer VLM: `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant`
    - Optional answer-generation LLM: `nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant`
- Nemotron 3 Embed 1B is the default text embedder. The logical model ID is `nvidia/nemotron-3-embed-1b`.
    - Helm NIM: `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2`. The NIM selects a supported NVFP4 or BF16 engine.
    - Local vLLM: `nvidia/Nemotron-3-Embed-1B-NVFP4` when every visible CUDA device has compute capability 10.0 or later; `nvidia/Nemotron-3-Embed-1B-BF16` otherwise.
    - Local Hugging Face: `nvidia/Nemotron-3-Embed-1B-BF16`.
- The new default is text-only. Explicitly select `nvidia/llama-nemotron-embed-vl-1b-v2` when you need image or text-and-image embeddings. Refer to [Embeddings](embedding.md) and [Dense Nemotron embedding checkpoints](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#dense-nemotron-embedding-checkpoints).
- The `page_elements` and `table_structure` services share the combined object-detection image and select distinct models. Pull that image once for air-gapped or allowlisted deployments.
- Nemotron 3 Nano Omni is the canonical caption model. It is opt-in on Helm and has a larger GPU footprint than Nano caption profiles.
- Nemotron Parse endpoint wiring is available in service extraction workers, with documented hosted versus self-hosted contract selection.
- An embedder model router supports additional Llama Nemotron and Nemotron-3 embedding checkpoints, including 1B, 3B, 8B, local, fine-tuned, and ModelOpt.

### Pipeline and ingestion { #pipeline-and-ingestion }

- `retriever ingest` and the graph stage registry are the canonical ingestion paths. The compatibility pipeline CLI is retired.
- Documented Markdown, JSON, and shell text inputs, plus inline text ingestion support.
- Service-mode TXT and HTML chunking.
- `return_failures` is supported across in-process and batch ingest modes.
- Tabular ingestion and embedding improvements, including table-type handling.
- PDF render parameters are forwarded through ingestion graphs.

### CLI { #cli }

- Root CLI adds first-class `retriever ingest` and `retriever query` commands with NIM URL flags, batch tuning, and LanceDB overwrite/append controls
- `retriever query --agentic` runs an LLM-driven ReAct retrieval loop over the same LanceDB table as one-pass retrieval. Local CLI and NRB benchmark runs default to in-process vLLM (`nemotron-8b`). Remote OpenAI-compatible NIM or NVIDIA-hosted endpoints use `--agentic-invoke-url`. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- `retriever ingest` and `retriever query` replace the retired compatibility pipeline command. Other top-level subcommands—including `eval`, `benchmark`, and `skill-eval`—are development and experimental.

### Retriever Service and deployment { #retriever-service-and-deployment }

- Helm maps `serviceConfig.nimEndpoints.rerankInvokeUrl` / `rerankModelName` into `nim_endpoints.rerank_invoke_url` / `rerank_model_name`, and auto-wires those fields when `nimOperator.rerankqa.enabled=true`, so `/v1/query` with `rerank=true` works in split topology.
- Split topology renders an internal gateway startup Service so realtime and batch init containers can reach `GET /v1/live` before the gateway passes its deep `/v1/health` readiness check. This removes the clean-install deadlock that required manually patching `publishNotReadyAddresses` on the gateway Service.
- Retriever Service exposes agentic retrieval on `POST /v1/query` when `agentic.enabled` is true. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- Gateway worker pull scheduling replaces push routing.
- Development Docker Compose deployment is available for local service stacks.
- Zipkin tracing parity is available alongside OpenTelemetry.
- Helm maximum upload size configuration and OpenShift deployment follow-ups.
- Secret-backed Helm authentication for public and internal service tokens. Inline tokens are gated for insecure development only.

### Multimodal extraction { #multimodal-extraction }

- Fixed an issue where SDK batch ingestion with `extraction_mode="auto"` initialized Page Elements for PDF and image inputs when `use_page_elements=False` or no enabled extraction stage required its output. Page-level image embedding continues to render one full-page image for each PDF page without loading a Page Elements model.
- Fixed an issue that could cause local Hugging Face batch audio extraction to hang in interactive terminals when FFmpeg inherited the parent process's standard input.

### Retrieval and RAG { #retrieval-and-rag }

- `Retriever.answer()` supports the Nemotron 3.5 Lightning default LLM path and the Omni VLM-capable path documented under [Answer generation](#answer-generation).
- Service-mode `Retriever.answer` support and FastMCP integration are available for local and remote agents.
- Agentic retrieval is available on the CLI, Python query workflow, REST, and MCP surfaces. Refer to [Agentic retrieval](#agentic-retrieval).

### Vector database and retrieval { #vector-database-and-retrieval }

- True LanceDB hybrid retrieval.
- LanceDB retrieval-mode autodetection and persisted embedding identity for automatic local queries.
- Local dense and hybrid queries reject an explicit embedding model that differs from the model recorded on the LanceDB table. They also reject tables without embedding-model metadata. The dedicated VectorDB service checks its configured legacy table at startup and collection-scoped tables when they are accessed. Errors direct users to use the index model or rebuild the table; local sparse queries are exempt.
- Dense image-only VDB records are retained where applicable.
- Scope-isolated collection and document catalog APIs (`/v1/collections`) create, list, get, update, and delete collections and committed documents without exposing LanceDB table names. Ingest and replace use `POST /v1/ingest/job` with `collection_name`. Retrieval uses `POST /v1/query` with `collection_name`. Refer to [Collection management API](../reference/collection-management-api.md).
- Fixed an issue where concurrent ingests into one VectorDB pod could report success minutes before the rows were durable or queryable. Each write now commits its rows independently, and index maintenance runs in a separate serialized phase where concurrent writers share one coalesced rebuild. An index-readiness wait that expires logs a warning and leaves the committed rows queryable instead of failing the write. Refer to [LanceDB index creation fails during concurrent Helm ingestion](troubleshoot.md#lancedb-concurrent-index-creation).

### Packaging and platform { #packaging-and-platform }

- Public nightlies are published to PyPI while local install extras remain stable.
- Ray is raised to `>=2.56.1` for CVE remediation. The previous release used `>=2.49.0`. Ray no longer publishes wheels for macOS Intel (x86_64), so `pip` and `uv` installs fail on Intel Macs, including in-process library mode. Apple Silicon (arm64) macOS remains supported for slim remote or NIM-only installs, alongside Windows x64.

### Helm chart { #helm-chart }

- The Helm chart under `nemo_retriever/helm/` defaults to OCR v2, the combined object-detection NIM, and Nemotron 3 Embed 1B `2.2.2`. Optional NIMs include VL rerank `2.3.0`, Omni `2.0.4-variant`, Nemotron Parse, and the Nemotron 3.5 Lightning `answer_llm` slot. Refer to [Default Helm NIMs](prerequisites-support-matrix.md#default-helm-nims) and [Models, OCR, and NIM artifacts](#models-ocr-and-captioning).

### Documentation { #documentation }

- Published [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md) for CLI, service, REST, and MCP usage.
- Published [One-shot text generation](nemo-retriever-api-reference.md#one-shot-text-generation) for `TextGenerationTask`, `GenericGenerationOperator`, `SummarizationOperator`, and `TextGenerationParams`.
- Clarified Nemotron 3.5 Lightning and Omni answer-generation paths on this page and in [Answer generation](prerequisites-support-matrix.md#answer-generation). For Helm enablement and slot overrides, refer to [Answer generation (operator-managed LLM)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#answer-generation-llm).

### Current foundational capabilities { #current-foundational-capabilities }

The following foundational capabilities remain current. They are not new 26.08.1 highlights.

- Text splitting for graph and library ingest uses `.extract(split_config=...)` instead of standalone `.split()` on the graph ingest path. The service ingestor API can still expose `.split()` separately.
- Direct `Retriever(...)` construction uses `vdb_kwargs`, `embed_kwargs`, and `rerank` instead of flat `lancedb_uri`, `lancedb_table`, `embedder`, `embedding_endpoint`, `local_query_embed_backend`, and `reranker` arguments.
- For Helm audio and video extraction, set `service.installFfmpeg: true` in `values.yaml` (or pass `--set service.installFfmpeg=true`) when images no longer bundle `ffmpeg` and `ffprobe` by default.
- `nemo_retriever` requires Python 3.12.
- Manifest-based ingest routing replaces input-type routing. `retriever ingest` is input-aware for PDF, image, audio, video, text, HTML, DOCX, PPTX, SVG, and related types.
- `allow_no_gpu` skips the GPU requirement during ingest for CPU-only experimentation.
- Root CLI includes `retriever ingest` and `retriever query` with NIM URL flags, batch tuning, and LanceDB overwrite or append controls.
- Retriever Service v2 provides a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration.
- OpenTelemetry provides basic pipeline and service observability.
- Air-gapped deployment guidance is in [deployment options](deployment-options.md) and the Helm chart README.
- Nemotron Parse is an alternate PDF extraction method (v1.2 HTTP interface, optional Helm NIM, and local inference through vLLM where configured).
- VLM image captioning through vLLM, including Omni caption model profiles, is available.
- vLLM-backed text and vision-language embedders, a multimodal VL reranker, and torch 2.11 are available for local GPU installs.
- The video retrieval pipeline includes frame extraction, OCR, audio-visual fusion, and text deduplication.
- Long-audio Parakeet chunking provides time-aligned segments, punctuation-based audio segmenting, and ASR batch and streaming improvements.
- The live RAG SDK includes `Retriever.retrieve()`, reference answer generation through `Retriever.answer()`, and optional batch operator graphs through LiteLLM (`[llm]` extra).
- Vector database operators are integrated in the pipeline, with custom metadata support and updated LanceDB hybrid search guidance.
- LanceDB is the first-party vector path for new deployments. Milvus and MinIO guidance is removed from the primary extraction doc set.
- Evaluation includes a BEIR-centric overhaul and the experimental `retriever skill-eval` benchmark CLI.
- Text-to-SQL agent graph and tabular tooling support structured data retrieval, including tabular data ingestion.
- Optional install extras include `[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and slim remote or NIM-only installs on Mac and Windows.
- Documentation is aligned to a Helm-first supported path and consolidates extraction concepts, ingest workflow, embeddings, audio and video guides, prerequisites and support matrix, and UDF or custom stages in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph).

## Release Notes for Previous Versions { #previous-versions }

- [26.05](https://docs.nvidia.com/nemo/retriever/26.5.0/extraction/releasenotes-nv-ingest/)
- [26.03](https://docs.nvidia.com/nemo/retriever/26.3.0/extraction/releasenotes-nv-ingest/)
- [26.1.2](https://archive.docs.nvidia.com/nemo/retriever/26.1.2/extraction/releasenotes-nv-ingest/)
- [26.1.1](https://archive.docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes-nv-ingest/)
- [25.9.0](https://archive.docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes-nv-ingest/)
- [25.6.3](https://archive.docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes-nv-ingest/)
- [25.6.2](https://archive.docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes-nv-ingest/)
- [25.4.2](https://archive.docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes-nv-ingest/)
- [25.3.0](https://archive.docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/)

Release notes for 24.12.1 and 24.12.0 are on the [25.3.0 archived release notes](https://archive.docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/).

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Answer generation](prerequisites-support-matrix.md#answer-generation)
- [One-shot text generation](nemo-retriever-api-reference.md#one-shot-text-generation)
- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md)
- [Deployment options](deployment-options.md)
- [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md)
