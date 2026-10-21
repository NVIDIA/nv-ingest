# NeMo Retriever API Reference

This page is the public Python SDK contract for NeMo Retriever Library. The generated signatures document `create_ingestor()` and the concrete objects it returns. They also document `Retriever` query and answer helpers, generation operators, and parameter models.

Import the factory and `GraphIngestionError` from `nemo_retriever`. Import generation operators from `nemo_retriever.operators.generation`. Import LLM client, task, and result types from `nemo_retriever.models.llm`.

## Error and failure contract { #error-and-failure-contract }

The Python API does not define a separate set of numeric NeMo Retriever
extraction error codes. Depending on the run mode and failing stage, callers
observe one or more of the following:

- Python configuration or dependency exceptions, such as `ValueError`,
  `ImportError`, or `RuntimeError`.
- `GraphIngestionError` for row-level failures from explicitly configured
  remote NIM stages in `run_mode="inprocess"` or `"batch"` when
  `error_policy="raise"` (the default).
- HTTP status codes or gRPC errors returned by a remote NIM or by the
  Retriever service. These are transport or upstream-service statuses, not
  NeMo Retriever-specific error codes.
- Per-document failures in `ServiceIngestResult.failures` when
  `run_mode="service"`.

The generated API signatures and parameter models below are the API contract.
Exception text and upstream response bodies can change between releases; do
not parse them as stable codes. The stable text-generation codes documented in
[One-shot text generation](#one-shot-text-generation) apply to generation
operator output columns, not to document extraction.

### Configure at least one input source

Before you call `.ingest()`, `.ingest_stream()`, or `.aingest_stream()`,
configure at least one input source by calling `.files()`, `.urls()`,
`.texts()`, or `.buffers()` with a nonempty value. Omitting input configuration
or passing an empty collection raises `ValueError` before pipeline execution.

A configured source can legitimately produce blank text or an empty result.
For example, OCR can find no text on an image-only page. This outcome does not
raise the missing-input error.

A nonempty optional glob passed to `.files()` also counts as a configured
source. If it matches no files, `.ingest()` can return an empty result, and the
streaming methods can yield no results.

In the `inprocess` and `batch` run modes, `.files()` and
`create_ingestor(documents=...)` also accept directories, which expand
recursively to their supported files. A directory that contains no supported
files raises `FileNotFoundError`.

### Fetch content from URLs { #fetch-content-from-urls }

Call `.urls()` with one absolute HTTP or HTTPS URL, or a sequence of URLs. URL
fetching is lazy and starts when you call an ingest method. NeMo Retriever
Library performs the HTTP GET requests in the Python SDK process. In service
mode, the client fetches the content and sends it to the Retriever service.

The response must match an existing [supported input
format](multimodal-extraction.md#supported-file-types-and-formats). NeMo
Retriever Library uses a recognized response content type and filename hints
to select the extraction path. It checks `Content-Disposition`, the final URL
after redirects, and then the submitted URL for filename hints. HTML responses
use the same MarkItDown conversion as local `.html` files.

Results retain the submitted URL as their base `path` and `source_id`,
including when the request follows a redirect. Extractors can add their normal
page suffix to a page-specific identity. NeMo Retriever Library preserves that
suffix while replacing the internal transport name with the submitted URL
before post-extraction stages such as embedding and `.vdb_upload()`.

The following example fetches and extracts a PDF.

```python
from nemo_retriever import create_ingestor

pdf_url = "https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf"

result, failures = (
    create_ingestor(run_mode="batch")
    .urls([pdf_url])
    .extract()
    .ingest(return_failures=True)
)
```

Pass `UrlFetchParams` or equivalent keyword arguments to `.urls()` when you
need to configure the HTTP requests.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import UrlFetchParams

fetch_params = UrlFetchParams(
    headers={"Authorization": "Bearer <token>"},
    request_timeout_s=60,
    follow_redirects=True,
    max_response_bytes=20_000_000,
    max_concurrency=4,
)

ingestor = create_ingestor(run_mode="inprocess").urls(
    ["https://example.com/private-document.html"],
    params=fetch_params,
)
```

The following settings apply to `inprocess`, `batch`, and `service` run modes.

| Field | Default | Behavior |
| --- | --- | --- |
| `headers` | `{}` | Adds the same request headers to every configured URL. |
| `request_timeout_s` | `30.0` | Sets the timeout for each HTTP connect, read, write, and connection-pool operation. It is not an end-to-end ingest deadline. |
| `follow_redirects` | `True` | Follows HTTP redirects. Redirect targets can provide filename hints, but they do not replace the submitted source URL. |
| `max_response_bytes` | `10_000_000` | Rejects an individual response after it exceeds this many downloaded bytes. |
| `max_concurrency` | `8` | Limits the number of concurrent URL fetches. |

Repeated `.urls()` calls append sources. A call without `params` or keyword
settings retains the existing `UrlFetchParams`. A later call with explicit
settings replaces the shared fetch configuration for all URLs on that
ingestor.

NeMo Retriever Library streams each accepted response into managed temporary
storage and removes that storage after ingestion. Service mode also streams
the spooled content during upload. This keeps memory use bounded by active I/O
instead of the total size of all fetched content.

Invalid URLs and unsupported URL schemes raise a configuration error. HTTP
error responses, timeouts, network failures, oversized responses, and
unsupported response formats are per-URL failures. Pass
`return_failures=True` to keep successful results and receive failures as
`(url, error_message)` tuples. Without that option, graph run modes raise
`GraphIngestionError` for URL fetch failures. With that option, service mode
returns `(ServiceIngestResult, failures)`. In service mode, URL fetch, upload,
and server-side document failures are also available in
`ServiceIngestResult.failures`. URL fetch and upload failures identify the
submitted URL instead of an internal transport filename.

### Select a supported extraction method

`ExtractParams` validates `method` when you construct the model. For PDF
extraction, use `pdfium`, `pdfium_hybrid`, `ocr`, or `nemotron_parse`. The
default, `pdfium_hybrid`, uses native PDF text and OCR for pages that it detects
as scanned. It also runs page-element detection and OCR for text-only
configurations, and it OCRs image inputs. To use native PDF text only, set
`method="pdfium"`.

The `audio` value remains available for the legacy params-driven audio path. For
new audio pipelines, use [`GraphIngestor.extract_audio()`](#graph-ingestor)
instead.

Any other value raises a Pydantic `ValidationError` before pipeline setup. The
error lists the supported values, so spelling and configuration errors do not
silently select another extraction path.

For local `nemotron_parse` extraction in NeMo Retriever Library 26.08, omit
`nemotron_parse_model` to use the default model, or set it to
`nvidia/NVIDIA-Nemotron-Parse-v1.2`. Other local model values, including
`nvidia/NVIDIA-Nemotron-Parse-2.0`, raise a Pydantic `ValidationError` before pipeline
execution. To use a remote Nemotron Parse endpoint, configure
`nemotron_parse_invoke_url` or `invoke_url` and select the model that matches
the endpoint contract.

### Choose raise or collect behavior

For graph run modes, `error_policy="raise"` raises `GraphIngestionError` when
an explicitly configured remote NIM stage reports a row-level error. The
exception retains the underlying records in `exc.records`. When available, its
message identifies the stage, invoke URL, and HTTP status in a form similar to
`[stage=OCR NIM url=https://... http=503]`, followed by a troubleshooting hint.

Use `error_policy="collect"` when partial results are useful and your
application inspects the error fields in every returned row. Alternatively,
pass `return_failures=True` to `.ingest()` to receive a `(result, failures)`
tuple. When no remote invoke URL is configured, `return_failures=True` scans
all output columns for row-level error fields so local failures are still
visible. In service mode, failures are also available from
`ServiceIngestResult.failures`.

### What the raise error policy covers

The strict policy applies only to stages where you explicitly configure a
remote NIM invoke URL. It does not raise for local-only PDFium parsing,
caption, audio or video, or ASR failures, even when those stages populate
row-level error fields.

| Configured invoke URL | DataFrame column scanned | Stage label in messages |
| --- | --- | --- |
| `page_elements_invoke_url` | `output_column` (default `page_elements_v3`) | Page Elements NIM |
| `ocr_invoke_url` | `ocr` | OCR NIM |
| `table_structure_invoke_url` | `table_structure_ocr_v1` | Table Structure NIM |
| `nemotron_parse_invoke_url` or `invoke_url` | `nemotron_parse_v1_2` | Nemotron Parse NIM |
| `embed_invoke_url` or `embedding_endpoint` | `output_column` (default `text_embeddings_1b_v2`) | Embedding NIM |

Caption and ASR use remote endpoints but are outside this raise path today.
Remote caption failures can abort the whole ingest instead of returning a
partial DataFrame. ASR failures can omit affected rows while logging a
warning, which can look like an empty transcript unless you inspect logs.

### Row-level error payloads

Most extraction stages write errors into the result row instead of raising
immediately. The common nested shape is:

```json
{
  "error": {
    "stage": "ocr_page_elements",
    "type": "HTTPError",
    "message": "HTTP 503 from https://example/v1/infer: ...",
    "traceback": "..."
  }
}
```

The `stage` string is a semi-stable operator identifier (for example
`remote_inference`, `nemotron_parse_pages`, or `split_pdf`). It is not a
product-wide error-code enum. HTTP status codes usually appear inside
`message` text rather than as a separate `status_code` field; when a
structured status is present, `GraphIngestionError` can include it in the
rendered exception.

```python
import os

from nemo_retriever import GraphIngestionError, create_ingestor
from nemo_retriever.common.params import ExtractParams

pipeline = (
    create_ingestor(run_mode="inprocess", error_policy="raise")
    .files(["document.pdf"])
    .extract(
        ExtractParams(
            method="ocr",
            ocr_invoke_url=os.environ["OCR_INVOKE_URL"],
        )
    )
)

try:
    result = pipeline.ingest()
except GraphIngestionError as exc:
    # Records can contain source paths, endpoint details, and upstream
    # response text. Extract only known-safe diagnostic fields before
    # logging or sending them to your support workflow.
    for record in exc.records:
        payload = record.get("error") if isinstance(record, dict) else record
        if isinstance(payload, dict):
            print(
                {
                    "column": record.get("column"),
                    "stage": payload.get("stage"),
                    "type": payload.get("type"),
                    "message": payload.get("message"),
                }
            )
        else:
            print(
                {
                    "column": record.get("column") if isinstance(record, dict) else None,
                    "message": str(payload),
                }
            )
```

For a support-oriented mapping of extraction paths, error signals, corrective
actions, and escalation criteria, refer to
[Python API error triage](troubleshoot.md#python-api-error-triage).

!!! note "Version-specific behavior"

    This reference describes the current NeMo Retriever Library. Older
    NV-Ingest releases, including `25.4.2`, can use different exception text
    and result shapes and might not include enriched `GraphIngestionError`
    diagnostics. When troubleshooting an older deployment, use the package and
    container versions from that deployment and include them in the support
    case.

## Write graph results to Parquet { #graph-parquet-output }

Call `.save_to_disk(output_directory)` before `.ingest()` to write graph rows
to a local Parquet dataset. The default `output_schema="native"` preserves
native graph columns. Select `output_schema="interleaved"` for ordered
multimodal rows projected from tagged Parse pages. This method configures the writer and returns
the ingestor. Extraction and writing start when you call `.ingest()`. It works
with `run_mode="inprocess"` and `run_mode="batch"`.

The following example uses a [served Nemotron 3.5 Parse
endpoint](multimodal-extraction.md#served-nemotron-35-parse) and returns counts
without retaining all extracted rows in the driver.

```python
from nemo_retriever import create_ingestor

summary = (
    create_ingestor(run_mode="batch", error_policy="collect")
    .files(["report.pdf"])
    .extract(
        method="nemotron_parse",
        nemotron_parse_invoke_url="http://localhost:8000/v1/chat/completions",
        nemotron_parse_model="nvidia/Nemotron-3.5-Parse",
        extract_text=True,
        extract_tables=True,
    )
    .save_to_disk("./parse-rows", rows_per_file=1024)
    .ingest(return_results=False)
)

print(summary.to_dict(orient="records"))
```

The destination must not already exist. Choose a new destination for each run;
the writer does not append to or replace an existing dataset.

The writer accepts the following settings.

| Argument | Default | Behavior |
| --- | --- | --- |
| `output_directory` | Required | Local directory to publish after the stream completes. |
| `compression` | `"snappy"` | PyArrow-supported Parquet compression codec, or `None` for no compression. |
| `rows_per_file` | `1024` | Positive maximum number of output rows in each Parquet part. This is not the model inference batch size. |
| `output_schema` | `"native"` | Preserve native graph rows, or use `"interleaved"` to project tagged Parse page results. |
| `passthrough_columns` | `()` | Input columns to copy onto interleaved page elements and error rows. Only valid with `output_schema="interleaved"`. |

With `return_results=False`, the returned DataFrame contains one summary row.
`input_rows` counts graph rows consumed by the writer. `written_rows` counts
rows in the published dataset. These counts match for native output;
interleaved output can contain several elements and document metadata rows per
input. Neither field counts successful pages or model requests.
`parquet_files` counts the written parts, including the readable empty part
created when the stream contains no rows.

The default `return_results=True` also returns the native extracted DataFrame
and retains those records in memory, regardless of the selected disk schema. Use that mode with `return_failures=True`
when you need a returned failure list. `return_failures=True` cannot be
combined with the Parquet summary mode; inspect the saved error fields instead.

The default native dataset preserves graph columns, including binary content
and nested extraction errors. A failed page remains a failed row; saving it does
not make its extraction successful. Empty dictionaries become null because
Parquet cannot represent an empty struct. Nullable fields introduced by later
batches are aligned across parts before publication. Incompatible values can
still cause a schema error.

The existing extraction error policy still applies. With the default
`error_policy="raise"`, configured remote-stage errors raise
`GraphIngestionError` after all rows have been saved and the dataset has been
published. In summary mode, the exception retains the first failing batch's
diagnostics; the dataset retains all failed rows. Use `error_policy="collect"`
to receive the summary and review the saved errors yourself. The existence of
a published dataset does not imply that `.ingest()` returned successfully.

The writer consumes the stream into a temporary sibling directory, writes
`_common_metadata` and a `_SUCCESS` record, then publishes the destination by
renaming the directory. If consumption or writing fails, it removes that
attempt's temporary directory. `_SUCCESS` records completed dataset writing,
not error-free page extraction.

You can verify the published dataset without loading all rows into memory.

```python
import pyarrow.dataset as ds

saved = ds.dataset("./parse-rows", format="parquet")
print(saved.count_rows())
print(saved.schema)

for batch in saved.to_batches(columns=["nemotron_parse_v1_2"], batch_size=128):
    for row in batch.to_pylist():
        metadata = row.get("nemotron_parse_v1_2") or {}
        if metadata.get("error"):
            print(metadata["error"])
```

Batch mode consumes Ray Data output in batches. In-process mode reuses its
operators across bounded document batches and closes them after the stream.
A single PDF can still produce many page rows in one document batch.
In-process streaming rejects operators that require the complete corpus as one
global batch because splitting that input would change their behavior.

`ServiceIngestor.save_to_disk()` is a separate service-mode API. It writes
per-document JSON or compressed JSON, not this Parquet dataset.

### Export tagged Parse pages as interleaved rows { #interleaved-parquet-output }

Set `output_schema="interleaved"` when a downstream reader needs ordered text,
table, and image elements. This export uses PyArrow and the library's Parse
postprocessor; it does not require an installed NeMo Curator package.

The following configuration keeps each page's native Parse diagnostics as an
additional column in the saved elements and error rows.

```python
summary = (
    create_ingestor(run_mode="batch", error_policy="collect")
    .files(["report.pdf"])
    .extract(
        method="nemotron_parse",
        nemotron_parse_invoke_url="http://localhost:8000/v1/chat/completions",
        nemotron_parse_model="nvidia/Nemotron-3.5-Parse",
        extract_text=True,
        extract_tables=True,
    )
    .save_to_disk(
        "./parse-interleaved",
        output_schema="interleaved",
        passthrough_columns=("nemotron_parse_v1_2",),
    )
    .ingest(return_results=False)
)
```

The projection requires tagged Parse output in `nemotron_parse_v1_2`, a
one-based input `page_number`, and a source path or source identity. It does
not convert arbitrary graph rows or untagged Markdown into a validated page.
Missing page information becomes an error row when a source identity exists;
missing source identity or duplicate document/page identities abort publication.

The eight reserved columns have the following meanings.

| Column | Meaning |
| --- | --- |
| `sample_id` | Document grouping key: the input `source_id`, or a hash of the source path when it is absent. |
| `position` | Element order within the document. The document metadata row has position `-1`; page elements and error rows start at `0`. |
| `modality` | `text`, `table`, `image`, or `metadata`. `Picture` elements become images; `Table` elements become tables; other classes become text. |
| `content_type` | `image/png` for picture crops, `text/markdown` for text and tables, or `application/json` for metadata and error rows. |
| `text_content` | Element text or document metadata. Table bodies retain their native markup, including merged-cell spans. |
| `binary_content` | Decoded PNG picture-crop bytes when the element is an image. |
| `source_ref` | JSON containing the source path. |
| `materialize_error` | JSON error details for a failed page, otherwise null. |

Additional identity and geometry columns are `page_number`, `source_path`,
`element_class`, `bbox_xyxy_norm`, and `bbox_coordinate_space`. The exported
`page_number` is zero-based. Bounding boxes refer to the coordinate space
recorded in `bbox_coordinate_space`, not directly to source PDF coordinates.

Cropping `Picture` elements in `normalized_page_content` coordinates requires
the optional model-team `nemotron-parse` package. This applies to tagged native
Parse 3.5 `/v1/parse` output as well as served 3.5 output. Follow the
[pinned package installation and client plugin guidance](multimodal-extraction.md#served-nemotron-35-parse).
If the package cannot be imported, the writer emits one `materialize_error`
row for the affected page and continues exporting other pages. Native
`/v1/parse` requests, native-schema output, and interleaved pages without those
picture crops do not require this package.

`passthrough_columns` copies explicitly selected input columns onto page
output rows. Missing selected values become null. Names must not collide with
the reserved or identity/geometry columns. Native output already keeps every
column, so it rejects `passthrough_columns` rather than silently ignoring it.

Each page must pass whole-page validation before its elements are published.
The complete response must parse into balanced tagged elements, and every
box must contain finite, ordered coordinates between `0` and `1`. Every
`Picture` must produce a decodable crop at least 10 pixels in each dimension.
An upstream error, incomplete response, invalid box, or invalid picture crop
produces one error row for that page. The writer does not publish the valid
prefix of a rejected page.

An empty Parse response is also a failed page. It is not classified as a
correctly blank page because this export does not establish whether the source
is blank. Passing structural validation does not establish extraction accuracy.
Projection-only failures are recorded in `materialize_error`; they do not
independently trigger `GraphIngestionError`. Inspect that column even when the
ingest call succeeds.

A temporary SQLite spool groups documents and orders their observed pages on
disk before writing the Parquet parts. Reserve space for that spool, the
Parquet output, and temporary part rewrites. This avoids concatenating the
complete corpus in memory, but it does not eliminate temporary disk use or
the memory needed to process an individual page.

Each document also receives one metadata row with `observed_pages`,
`failed_pages`, and an `extraction_status` of `success`, `partial`, or `failed`.
These fields describe only the pages received by the writer. They do not prove
that every page of the original PDF was ingested. Reconcile source-page counts
and identities separately when you require complete-document accounting.

Read interleaved Parquet with an Arrow dataset reader and filter
`materialize_error` to inspect page failures. `return_results=True` still
returns native graph rows, so read the published dataset to inspect its
interleaved projection.


## PDF pre-splitting for parallel ingest { #pdf-pre-splitting-for-parallel-ingest }

Large PDFs are split into page batches before Ray processing so extraction can run in parallel. This happens on the default ingest path; you do not need extra configuration for typical workloads.

To tune splitter throughput from the CLI, use `--pdf-split-batch-size` (Ray actor batch size for the splitter stage). Refer to [Local and batch ingest](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli#local-and-batch-ingest) in the CLI reference.

**Python client (`pdf_split_config`):** Only [`ServiceIngestor.pdf_split_config()`](#service-ingestor) records page-chunking settings in the request pipeline spec for the remote gateway. Obtain that object with `create_ingestor(run_mode="service")`. Local graph ingest (`run_mode="inprocess"` or `"batch"`) does not implement this method. PDFs are split automatically on the default graph ingest path without client-side configuration.

## One-shot text generation { #one-shot-text-generation }

`TextGenerationOperator` is the reusable base for synchronous, one-request-per-row text generation. It is a provisional text-only API: it does not support tool calls, agent loops, streaming, multiple choices, or structured domain results.

Concrete operators construct an immutable `TextGenerationTask` and provide reconstructible constructor state. Runtime task and client objects must not be included in graph constructor arguments. A custom completion client must be safe for concurrent calls or report that it does not support concurrent calls so the operator serializes access.

Embedding and captioning remain separate operator families because they use modality grouping, native batching, and specialized CPU/GPU lifecycles.

### Generic generation and summarization

Both operators consume a pandas DataFrame and add text, latency, model, and error columns without changing the input rows:

```python
import pandas as pd

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.operators.generation import GenericGenerationOperator, SummarizationOperator

summary_params = TextGenerationParams.from_kwargs(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
    temperature=0.0,
    max_tokens=512,
)
summaries = SummarizationOperator(summary_params).run(
    pd.DataFrame({"text": ["A long document to summarize."]})
)

prompt_params = TextGenerationParams.from_kwargs(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
    prompt="Write a {tone} title for: {text}",
)
titles = GenericGenerationOperator(
    prompt_params,
    input_columns={"tone": "style", "text": "document"},
    output_column="title",
).run(pd.DataFrame({"style": ["concise"], "document": ["Quarterly results"]}))
```

`SummarizationOperator` defaults to `text`, `summary`, `summary_latency_s`, `summary_model`, and `summary_error`. `GenericGenerationOperator` maps each named prompt placeholder to a physical DataFrame column and derives the metadata column names from `output_column`. Prompt contracts are validated when the operator is constructed, before any provider request runs. `SummarizeTask` inherits from `TextGenerationTask` and supplies the built-in summarization prompt unless you override `prompt` on `TextGenerationParams`.

### TextGenerationParams configuration { #textgenerationparams-configuration }

Construct `TextGenerationParams` with `TextGenerationParams.from_kwargs(...)`. The following fields are supported.

| Field | Purpose |
|-------|---------|
| `model` | Required provider model identifier. |
| `api_base` | Optional OpenAI-compatible API base URL. |
| `api_key` | Optional credential or `os.environ/<NAME>` reference. Literal keys are not written to persisted graph JSON. |
| `temperature` | Optional sampling override. Valid values are 0.0 through 2.0. |
| `top_p` | Optional sampling override. Valid values are 0.0 through 1.0. |
| `max_tokens` | Optional positive token-limit override. Omit the field to inherit the task default. |
| `extra_params` | Optional provider-specific request keys. Dedicated fields such as `model`, `messages`, sampling, and credentials must not be duplicated here. |
| `num_retries` | Transport retry count. The default is 3. The value must be 0 or greater. |
| `timeout` | Transport timeout in seconds. The default is 120.0. The value must be greater than 0. |
| `prompt` | Optional user prompt or prompt template. `GenericGenerationOperator` requires this field. |
| `system_prompt` | Optional system prompt. |
| `rag_system_prompt` | Optional retrieval-augmented generation (RAG) system prompt. |
| `rag_system_prompt_prefix` | Optional prefix applied to the RAG system prompt. |
| `reasoning_enabled` | Optional reasoning toggle. When unset, transport reasoning defaults to true. |
| `max_workers` | Concurrent row workers. The default is 8. The value must be 1 or greater. |

Sampling fields that you omit inherit the task defaults. `temperature`, `top_p`, and `max_tokens` are applied only when you pass them explicitly.

To define another one-request/one-text-result task, subclass `TextGenerationTask`, declare `required_inputs`, and implement `build_request()`. Then construct it from a `TextGenerationOperator` subclass with explicit logical-input-to-DataFrame-column mappings. This abstraction is intentionally text-only; use a separate operator family for embeddings, captioning, tools, streaming, or structured domain results.

Generation failures are collected per row using stable error codes: `empty_input`, `request_error`, `transport_error`, `unsupported_response`, `parse_error`, `empty_output`, and the RAG-specific `thinking_truncated`. Raw provider exceptions and credentials are not written to DataFrame outputs.

Generated signatures for `TextGenerationOperator`, `GenericGenerationOperator`, and `SummarizationOperator` appear in [Generation operators](#generation-operators). Generated signatures for `TextGenerationTask`, `LiteLLMClient`, `LLMClient`, `AnswerJudge`, `AnswerResult`, and related types appear in [LLM clients, tasks, and results](#llm-clients-tasks-and-results). `Retriever.answer()` requires an `LLMClient` and returns `AnswerResult`.

## Persisted graphs are trusted configuration { #persisted-graphs-are-trusted-configuration }

Graph loading imports operator classes and invokes their constructors. Load graph JSON only from trusted sources; do not expose graph payloads, callable references, or class names as model- or user-controlled agent tools.

Version 2 graph files preserve shared-node DAG identity and reject cycles. Constructor state must consist of supported JSON-native values, typed Pydantic models, paths, sets and tuples, or importable type/callable references. Runtime data such as DataFrames and opaque client objects is not persistable.

API keys are never written into graph JSON. Use an explicit environment reference in persisted configuration:

```python
from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

QAGenerationOperator(
    model="openai/gpt-4o-mini",
    api_key="os.environ/OPENAI_API_KEY",
)
```

For LiteLLM `nvidia_nim/...` models, including the default `LiteLLMClient` and `LLMJudge` models, use `os.environ/NVIDIA_API_KEY`. Refer to [Authentication and API keys](api-keys.md#credential-references-in-persisted-graphs).

Serializing a graph containing a literal API key fails with a contextual error instead of guessing which provider credential should be used on a worker.

## Generated Python API { #generated-python-api }

The signatures below are the supported public ingest, retrieve, generation, and parameter surfaces. Private helpers, module loggers, and duplicate re-exports are omitted.

Use the following public import paths:

- Import `create_ingestor` and `GraphIngestionError` from `nemo_retriever`.
- Import `GraphIngestor` from `nemo_retriever.ingestor.graph_ingestor`.
- Import `ServiceIngestor` from `nemo_retriever.service.service_ingestor`.
- Import generation operators from `nemo_retriever.operators.generation`.
- Import LLM client, task, and result types from `nemo_retriever.models.llm`.
- Import parameter models from `nemo_retriever.common.params`.
- Import `RetrieveVdbOperator` from `nemo_retriever.operators.vdb`.
- Import `NemotronRerankActor` from `nemo_retriever.operators.rerank`.

### Public ingestion factory { #public-ingestion-factory }

`create_ingestor()` returns a concrete ingestion client from `run_mode`. The supported values are `inprocess`, `batch`, and `service`.

| `run_mode` | Runtime type | Execution |
| --- | --- | --- |
| `inprocess` | `GraphIngestor` | Local in-process graph. This is the default. |
| `batch` | `GraphIngestor` | Ray Data graph. |
| `service` | `ServiceIngestor` | Remote Retriever service. |

The function is annotated as returning the shared `Ingestor` interface. At runtime it returns `GraphIngestor` or `ServiceIngestor`. Use the generated class entries below for run-mode-specific methods.

Factory keyword arguments merge into `IngestorCreateParams`. Common fields include `documents`, `base_url`, `api_key`, `error_policy`, `ray_address`, and `max_concurrency`. Refer to `IngestorCreateParams` in [Parameter models](#generated-parameter-models).

```python
from nemo_retriever import GraphIngestionError, create_ingestor

graph = create_ingestor(run_mode="inprocess")
batch = create_ingestor(run_mode="batch")
service = create_ingestor(run_mode="service", base_url="http://localhost:7670")
```

::: nemo_retriever.ingestor.core.create_ingestor
    options:
      heading_level: 4
      show_docstring_description: false

### Graph ingest { #graph-ingestor }

`create_ingestor(run_mode="inprocess")` and `create_ingestor(run_mode="batch")` return `GraphIngestor`. Import the class from `nemo_retriever.ingestor.graph_ingestor` when you need the type explicitly. Import `GraphIngestionError` from `nemo_retriever`.

`GraphIngestor` methods include `extract_html()`, `extract_audio()`, `extract_video()`, `save_to_disk()`, `get_error_rows()`, and `get_dataset()`. `get_error_rows()` filters rows that contain stage error payloads from a pandas DataFrame or Ray Dataset. If you omit `dataset`, it uses the dataset retained from the last `ingest()` call. `get_dataset()` returns that retained dataset.

With `.save_to_disk()`, `.ingest(return_results=False)` returns a Parquet
writing summary in either graph run mode. Refer to [Write graph results to
Parquet](#graph-parquet-output) for its counts and error handling.

Without `.save_to_disk()`, this flag requires batch mode, a final streaming VDB
upload stage, and the default raise error policy. Refer to [Return a batch
ingest summary](vdbs.md#batch-ingest-summary) for that path's counts and
requirements. You can also pass `IngestExecuteParams(return_results=False)`
through `params`. In batch mode, `get_dataset()` then returns the summary.

::: nemo_retriever.ingestor.graph_ingestor.GraphIngestor
    options:
      heading_level: 4
      docstring_style: numpy
      show_docstring_description: false
      filters:
        - "!^_"
        - "!^pdf_split_config$"

::: nemo_retriever.ingestor.graph_ingestor.GraphIngestionError
    options:
      heading_level: 4
      show_docstring_description: false

### Service ingest { #service-ingestor }

`create_ingestor(run_mode="service")` returns `ServiceIngestor`. Import the class from `nemo_retriever.service.service_ingestor`. `ingest()` returns `ServiceIngestResult`.

Service-only methods include `split()`, `pdf_split_config()`, `ingest_stream()`, `aingest_stream()`, and `cancel()`. Its `save_to_disk()` method writes service results as JSON and has a different signature from the graph Parquet writer. `cancel()` is part of the public class. It currently raises `NotImplementedError` because the service does not expose a cancel endpoint.

#### Choose a service result schema { #service-result-schema }

`ServiceIngestor.ingest()` returns result rows in the `legacy` schema by
default. You can pass `result_schema="compact"` to use the compact schema.
The schemas return text and embeddings as follows.

| Schema | Text | Embeddings |
| --- | --- | --- |
| `legacy` | Ordinary string values are not truncated. | Pass `return_embeddings=True` to preserve embedding payloads in their legacy columns and nested fields. |
| `compact` | The top-level `text` field preserves the complete extracted text. | Pass `return_embeddings=True` to add a top-level `embedding` field when the source row contains an embedding. |

For legacy rows, the service preserves extracted text and string-valued
metadata in full, including returned strings nested in table, chart,
infographic, and image results. Raw images and embeddings remain opt-in.
Arrays, binary values, and oversized non-text collections remain summarized.

The default `return_embeddings=False` omits the top-level `embedding` field
from compact rows. This default keeps the compact response shape and payload
size unchanged. When you configure `EmbedParams.output_column`, compact rows
read the embedding from that column and normalize it to the top-level
`embedding` field. Raw image payloads remain available only in legacy rows
when you pass `return_images=True`.

The following example requests compact rows with their embeddings.

```python
from nemo_retriever import create_ingestor

result = (
    create_ingestor(run_mode="service", base_url="http://localhost:7670")
    .texts(["Text to embed and return."])
    .embed()
    .ingest(result_schema="compact", return_embeddings=True)
)

for row in result.dataframe.to_dict(orient="records"):
    print(row["text"], row.get("embedding"))
```

::: nemo_retriever.service.service_ingestor.ServiceIngestor
    options:
      heading_level: 4
      docstring_style: numpy
      show_docstring_description: false

::: nemo_retriever.service.service_ingestor.ServiceIngestResult
    options:
      heading_level: 4
      docstring_style: numpy
      show_docstring_description: false

### Retrieve { #generated-retrieve-api }

`Retriever` embeds queries, retrieves from a vector database, and can rerank results. Pass `embed_kwargs` that match `EmbedParams`, `vdb_kwargs` for `RetrieveVdbOperator`, and `rerank_kwargs` for `NemotronRerankActor`.

With the default query graph, `Retriever.queries()` returns one hit list per
input query, preserving input order and repeated identical inputs. Overlength
queries are split losslessly, searched, and fused before optional reranking
with the original full query text. `top_k` limits the final hits per original
query. For the fusion policy, refer to
[Text inputs that exceed the model limit](embedding.md#text-input-overflow).

`Retriever.answer()` requires an `LLMClient` and returns `AnswerResult`. Pass an `AnswerJudge` only when you also pass `reference`. Those types are documented in [LLM clients, tasks, and results](#llm-clients-tasks-and-results).

`Retriever.pipeline()` returns `RetrieverPipelineBuilder`. `generate()` accepts a `LiteLLMClient` instance or `model=` keyword arguments. `judge()` accepts an `LLMJudge` instance or `model=` keyword arguments. When you include both `.judge()` and `.score()`, call `.judge()` first. `.score()` derives `failure_mode` from the `judge_score` present at that step. A later `.judge()` does not recompute `failure_mode`.

The package exports `nemo_retriever.retriever` as a default `Retriever()` instance. Construct `Retriever` directly when you need a configured object.

::: nemo_retriever.graph.retriever.Retriever
    options:
      heading_level: 4
      show_docstring_description: false

::: nemo_retriever.graph.retriever.RetrieverPipelineBuilder
    options:
      heading_level: 4
      show_docstring_description: false

### Generation operators { #generation-operators }

Import these classes from `nemo_retriever.operators.generation`. Usage notes and parameter-field tables appear in [One-shot text generation](#one-shot-text-generation).

::: nemo_retriever.operators.generation.TextGenerationOperator
    options:
      heading_level: 4
      show_docstring_description: false

::: nemo_retriever.operators.generation.GenericGenerationOperator
    options:
      heading_level: 4
      show_docstring_description: false

::: nemo_retriever.operators.generation.SummarizationOperator
    options:
      heading_level: 4
      show_docstring_description: false

### LLM clients, tasks, and results { #llm-clients-tasks-and-results }

Import these names from `nemo_retriever.models.llm`. That package path is the supported public import. `LiteLLMClient` and `LLMJudge` load from implementation submodules. Import those classes from `nemo_retriever.models.llm`. Use `from_kwargs()` for a flat constructor.

::: nemo_retriever.models.llm
    options:
      heading_level: 4
      show_docstring_description: false
      members:
        - AnswerJudge
        - AnswerRequest
        - AnswerResult
        - GeneratedTextResult
        - GenerationRequest
        - GenerationResult
        - GenerationTaskError
        - GenericPromptTask
        - JudgeResult
        - LLMClient
        - RagAnswerTask
        - RetrievalResult
        - RetrieverStrategy
        - SummarizeTask
        - TextCompletionClient
        - TextGenerationTask

::: nemo_retriever.models.llm.clients.litellm.LiteLLMClient
    options:
      heading_level: 4
      show_docstring_description: false

::: nemo_retriever.models.llm.clients.judge.LLMJudge
    options:
      heading_level: 4
      show_docstring_description: false

### Parameter models { #generated-parameter-models }

::: nemo_retriever.common.params
    options:
      heading_level: 4
      show_submodules: false
      filters:
        - "!^_"
        - "!^__all__$"
