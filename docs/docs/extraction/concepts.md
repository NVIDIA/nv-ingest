# Concepts

These terms appear throughout NeMo Retriever Library documentation.

## Job { #job }

An **ingestion job** is a unit of work you run on input content (documents, audio, video, and other supported types). Submit jobs through any of these supported entry points:

- **Python API** — `Ingestor` task chains such as `.extract(...)`. Library and batch modes run ingest in-process. Against a deployed Retriever service (`run_mode="service"`), the client wraps the REST contract below. Refer to the [Python API guide](nemo-retriever-api-reference.md).
- **`retriever ingest` CLI** — including `retriever ingest service` for a running service. Refer to the [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).
- **Retriever service REST API** — the public two-step ingest workflow:
  1. Create and configure the job aggregate with `POST /v1/ingest/job` and an `application/json` `JobCreateRequest` body. The JSON sets job-level fields such as `expected_documents`; it does not embed document bytes.
  2. Upload document content separately with multipart requests to job-scoped endpoints such as `POST /v1/ingest/job/{job_id}/document`.

Creating the JSON job aggregate does not complete ingestion. For the live OpenAPI schema, open `/docs` or `/openapi.json` on a running service. For how to run a service, refer to [Deployment options](deployment-options.md).

Default tasks target strong recall; customize behavior with task keyword arguments (including chunking and splitting on `.extract()`) or custom UDF-style operations. For UDFs and other extension paths, refer to [Customize & extend](customize-extend.md). Results are structured metadata and annotations (Ray Dataset, pandas `DataFrame`, or similar).

## Collection { #collection }

A **collection** is a scoped logical container for ingested documents on the Retriever service. The public catalog contract is REST `/v1/collections` on the published gateway. Python applications can use `RetrieverServiceClient`, which wraps those endpoints. Ingest and retrieval use `/v1/ingest/job` and `/v1/query` with `collection_name`. Callers do not supply LanceDB table names. Refer to [Collection management API](../reference/collection-management-api.md).

## Pipeline and tasks { #pipeline-and-tasks }

NeMo Retriever Library does **not** run one static pipeline on every document. You configure **tasks** such as parsing, chunking, embedding, storage, and filtering per job. For UDFs, custom graph stages, and other extension paths, refer to [Customize & extend](customize-extend.md).

## Extraction metadata { #extraction-metadata }

Output is a **Ray Dataset** (Ray Data) or **pandas** `DataFrame` listing extracted objects (text regions, tables, images, and so on), processing notes, and timing or trace data. Field-level detail is in the [metadata reference](content-metadata.md).

## Embeddings and retrieval { #embeddings-and-retrieval }

Optionally, the library can compute **embeddings** for extracted content and store vectors in [LanceDB](https://lancedb.com/) for downstream semantic search in your application. For upload and retrieval APIs, refer to [Vector databases](vdbs.md). For text and multimodal embedding options, refer to [Embeddings](embedding.md). For iterative, tool-driven retrieval over that index, refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

## Chunking { #chunking }

Chunking is built into the `.extract()` task and depends on **content type**:

- **PDF, DOCX, and PPTX** — Text is grouped using built-in **page** boundaries (one chunk per page where the format has pages).
- **Plain text (`.txt`) and HTML** — Formats without natural page breaks are split into segments of **1024 tokens** by default, using the revision-pinned [Nemotron 3 Embed 1B BF16 tokenizer](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16). Published service images bundle this tokenizer artifact without model weights, so default text chunking does not require Hugging Face access at runtime. The embedding stage separately checks inputs with the selected model's tokenizer and formatting. Refer to [Token-based splitting](#token-based-splitting) and [Environment variables](environment-config.md) for overrides and other runtimes.
- **Audio and video** — Media is split into **segments** for decoding and ASR using ffmpeg-based rules (configurable **size**, **time**, or **frame** split modes in the media chunking stage). With the Parakeet ASR path, you can optionally emit **sentence-like segments** by passing `asr_params=ASRParams(segment_audio=True)` to `.extract_audio(...)`. Refer to [Speech and audio extraction](audio-video.md#speech-and-audio-extraction) for the import and a runnable example.

For PDF parallelism before Ray processing (large files), refer to [PDF pre-splitting for parallel ingest](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest).

### Token-based splitting { #token-based-splitting }

Token-based splitting uses the revision-pinned `nvidia/Nemotron-3-Embed-1B-BF16` tokenizer with configurable `max_tokens` and `overlap_tokens`. For graph and library ingest (`create_ingestor(run_mode="inprocess")` or `create_ingestor(run_mode="batch")`), set those values on `.extract(split_config={"text": {"max_tokens": ..., "overlap_tokens": ...}})`, or omit `split_config` to use default text segmentation for unstructured text. Do not call `.split()` on `GraphIngestor`; that method exists only on `ServiceIngestor` (`run_mode="service"`). Published service images and the documented source builds include the tokenizer locally; source builds enable this with `--build-arg DOWNLOAD_DEFAULT_TOKENIZER=True`. The `service` image disables runtime Hub access, while `service-gpu` remains online for its other Hugging Face models. The base library install includes the tokenizer Python dependencies; pre-populate the Hugging Face cache before offline use. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

## Deployment modes { #deployment-modes }

- **Library mode** — Run without the full container stack where appropriate; refer to [Deployment options](deployment-options.md).
- **Kubernetes / Helm (self-hosted)** — Refer to [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md) and [deployment options](deployment-options.md) for running the full microservices pipeline on your infrastructure.
- **Notebooks** — [Jupyter examples](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md) for experimentation and RAG demos.

For a concise comparison, refer to [Deployment options](deployment-options.md).
