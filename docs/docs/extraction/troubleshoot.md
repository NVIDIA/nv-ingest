# Troubleshoot NeMo Retriever Library

Use this documentation to troubleshoot issues that arise when you use [NeMo Retriever Library](overview.md).

## Python API error triage { #python-api-error-triage }

NeMo Retriever Library does not assign product-specific numeric error codes to
each extraction method. The Python API surfaces Python exception types,
row-level failure records, and HTTP or gRPC statuses from the service or
upstream NIM. Treat the exception type, failing stage, upstream status, and
response detail together as the error identifier.

The current graph API enriches `GraphIngestionError` for explicitly configured
Page Elements, OCR, Table Structure, Nemotron Parse, and embedding endpoints.
When the upstream payload contains an HTTP status, the message includes the
stage, configured URL, and status. For example:

```text
Graph ingestion detected row-level errors from an explicitly configured
remote NIM endpoint. row 0, column ocr
[stage=OCR NIM url=https://example.invalid/v1/infer http=503], path error: ...
Troubleshooting: OCR NIM ... returned a 5xx server error ...
```

This enrichment is not a new error-code namespace. The `503` in this example
is the upstream HTTP status. Exception wording and upstream response bodies are
not stable API fields and should not be parsed programmatically.

In many row-level payloads, the HTTP status appears only inside
`error.message` (for example `HTTP 503 from ...`), not as a separate
`status_code` field. Inspect `exc.records` or the failing DataFrame column
when troubleshooting. The `error` value can be a nested object or a string.

### Coverage limits of the raise error policy

`error_policy="raise"` scans only remote NIM stages with an explicitly
configured invoke URL: Page Elements, OCR, Table Structure, Nemotron Parse,
and embedding. It does not automatically raise for:

- Local-only pipelines (`pdfium` without remote URLs), even when rows contain
  `metadata.error` or column-level error payloads.
- Caption or remote VLM stages. Missing credentials fail at actor setup;
  inference failures can abort the entire ingest.
- Audio or video ASR over gRPC or HTTP. Failures can drop individual rows and
  log warnings instead of raising `GraphIngestionError`.

For those paths, use `error_policy="collect"`, pass `return_failures=True`, or
inspect row columns and service logs directly.

### Error signals and first response

| Signal | Typical meaning | L1/L2 response |
| --- | --- | --- |
| `ValueError` or Pydantic validation error before execution | An unsupported run mode, parameter value, protocol, or parameter combination | Compare the call with the current [Python API reference](nemo-retriever-api-reference.md). Remove unknown parameters and reproduce with the smallest valid pipeline. |
| `ImportError`, `ModuleNotFoundError`, or a missing-dependency `RuntimeError` | The selected local extraction path requires a package or executable that is not installed | Install the documented package extra or system dependency. Confirm that the Python environment running the worker, not only the client shell, contains it. |
| `GraphIngestionError` with no HTTP status | The named remote stage returned a row-level error, its error payload omitted a status, or the endpoint was unreachable | Check DNS, routing, TLS, the endpoint URL, and the NIM readiness endpoint from the worker or service pod. Inspect `exc.records` after removing secrets and document content. |
| HTTP `401` or `403` from a NIM | Missing, expired, or unauthorized credentials | Verify `NVIDIA_API_KEY`, `NGC_API_KEY`, or the stage-specific credential in the environment that makes the request. Do not attach API keys to a support case. |
| HTTP `403` from the Retriever service | Authentication failure or a deployment policy that disallows the requested endpoint, stage, sink, or override | Read the response `detail`. Verify the service token and compare the requested pipeline with `/v1/ingest/pipeline-config`. |
| HTTP `404` or `410` while opening a service ingest job | The Python SDK and Retriever service can be on incompatible API versions | A current client raises `RetrieverServiceCompatibilityError`. Align the Python package, service image, and Helm chart versions. |
| Other HTTP `4xx` | The upstream service rejected the request | Check file type, rendered page or image size, model name, endpoint path, and request schema. For `413` or `422`, reduce the payload or image size and verify the endpoint's input limits. |
| HTTP `429` | The remote service is rate-limiting requests | Reduce concurrency or batch size and retry with backoff. Escalate only if throttling persists within the service quota. |
| HTTP `5xx`, including `503` | The upstream NIM is unavailable, overloaded, not ready, or failed during inference | Check readiness, pod restarts, GPU memory, server logs, and request volume. Retry a minimal input after the NIM is healthy. |
| Timeout, connection reset, DNS, TLS, or gRPC transport error | The client could not complete transport to the service or NIM | Test connectivity from the process or pod that runs the stage. Verify protocol, port, certificate trust, proxy, and network policy. Preserve the gRPC status and details when present. |
| A per-document entry in `ServiceIngestResult.failures` | Upload or pipeline processing failed after a service job was created | Correlate the document ID with the job ID and service logs. Other documents in the same result can still have succeeded. |
| Successful ingest with fewer rows than inputs (caption or ASR enabled) | Caption inference failed before row collection, or ASR dropped failed rows and logged warnings | Re-run with logging enabled. For caption, verify endpoint credentials and payload limits. For ASR, verify gRPC endpoint, `function_id`, and `NVIDIA_API_KEY`. |
| OOM, worker exit, or pod restart | Host or GPU resources were exhausted, or an orchestrator terminated the worker | Reduce batch size or concurrency, use smaller document groups, and inspect host, Ray, Kubernetes, and NIM resource telemetry. |
| `Infeasible Ray CPU/GPU plan` | Explicit worker counts or node overrides, including required Ray Data source capacity for filesystem inputs, exceed resources currently available to Ray. | Reduce `*_workers` or per-node concurrency, or wait for shared-cluster capacity. Refer to the [performance guide](performance_guide.md). |

The service can retry some transient transport, `429`, and `5xx` failures.
Report the final status returned after retries, not an intermediate warning.

### Representative extraction paths

Use the failing stage—not only the top-level `method` value—to select the
troubleshooting path. A single document can pass through several stages.

| API path | Components that can fail | Representative signals |
| --- | --- | --- |
| `ExtractParams(method="pdfium")` | File loading, PDF splitting, PDFium parsing, page rendering; optionally Page Elements and Table Structure when enabled | Malformed or encrypted input, `pypdfium2` import failure, local Python exception, or remote-stage `GraphIngestionError` when an invoke URL is explicitly configured |
| `ExtractParams(method="pdfium_hybrid")` | PDFium plus Page Elements, OCR, and optionally Table Structure | The local PDF signals above, or a row-level/HTTP failure attributed to Page Elements, OCR, or Table Structure |
| `ExtractParams(method="ocr")` | Page rendering, Page Elements, and the local or remote OCR backend | Missing local model dependencies, invalid image payload, authentication/transport status, or OCR row-level failure |
| `ExtractParams(method="nemotron_parse")` | PDF rendering and local Nemotron Parse model or configured Nemotron Parse NIM | Missing `open_clip`, missing local model configuration, unsupported image input, or Nemotron Parse row-level/HTTP failure |
| `.caption(...)` | Local caption model or remote VLM endpoint | `ValueError` at setup when credentials or endpoint/protocol are invalid; remote inference failures can abort the whole ingest rather than populate a row error column |
| `.embed(...)` | Local embedding model or configured embedding NIM | Model/dependency error, input-size or schema rejection, authentication/transport status, or embedding row-level failure; `GraphIngestionError` when a remote embed URL is configured |
| Audio or video extraction | `ffmpeg`/`ffprobe`, media decoding, frame/chunk creation, and local or remote ASR | Missing executable, malformed media, codec failure, gRPC status, or credential error; ASR failures may omit rows and log warnings instead of raising, so verify logs when output is unexpectedly empty |

`pdfium` itself is primarily a local parser, so a Page Elements, Table
Structure, OCR, caption, or embedding HTTP status comes from an enabled
downstream stage rather than from PDFium.

### Collect diagnostics safely

Before escalating, collect the following:

1. Package, image or Helm versions, and `run_mode`.
2. Exception class and sanitized message. For `GraphIngestionError`, include
   sanitized `exc.records`. For row-level failures, include `stage`, `type`,
   and `message` when present.
3. Extraction method, enabled stages, and endpoint hostnames with credentials
   and signed query parameters removed.
4. HTTP or gRPC status, response detail, job ID, document ID, trace ID, and
   timestamp when available.
5. Whether the endpoint readiness check succeeds from the worker or service
   pod.
6. A minimal non-confidential reproducing input, or characteristics such as
   format, page count, dimensions, and size.
7. Relevant client, service, Ray, NIM, and Kubernetes logs for the same
   timestamp.

Never include API keys, bearer tokens, document contents, or unredacted signed
URLs in logs or support cases.

Escalate to NVIDIA L3 when the failure is reproducible on a supported,
version-aligned configuration after L1 and L2 support has verified input
validity, credentials, endpoint readiness, connectivity, and resource
availability. Escalate immediately for repeatable crashes, incorrect
successful output, or a `5xx` from a healthy NVIDIA-owned NIM with a minimal
valid input. Keep configuration, dependency, customer network, quota, and
malformed-input issues with L1 and L2 support unless the documented behavior
is incorrect.

!!! note "Older NV-Ingest releases"

    Error text and result shapes differ by release. NV-Ingest `25.4.2`
    predates some current enriched diagnostics. Do not assume that a field
    shown in current NeMo Retriever Library output exists in `25.4.2`; include
    the exact old exception and logs when escalating.

## Can't process long, non-language text strings { #cant-process-long-non-language-text-strings }

NeMo Retriever Library is designed to process language and language-length strings.
If you submit a document that contains extremely long, or non-language text strings,
such as a DNA sequence, errors or unexpected results occur.

## Can't process malformed input files { #cant-process-malformed-input-files }

Malformed PDFs and images usually produce a row-level error instead of
aborting the entire job. Inspect `metadata.error` on the returned row, pass
`return_failures=True`, or use `error_policy="collect"`.

- For a malformed PDF, `metadata.error` is an object with `stage`, `type`,
  `message`, and `traceback` fields. For example:
  `{"stage": "page_processing", "type": "PdfiumError", "message": "...", "traceback": "..."}`.
- For a malformed standalone image, `metadata.error` is a string. For example:
  `UnidentifiedImageError: cannot identify image file`.

Verify or repair the input format and resubmit the job. Do not rely on legacy
messages such as `Failed to extract image` or `File may be malformed`; current
NeMo Retriever Library extraction operators do not emit those strings. For
safe handling examples, refer to
[Row-level error payloads](nemo-retriever-api-reference.md#row-level-error-payloads).

## Audio or video extraction reports missing media dependencies { #audio-or-video-extraction-reports-missing-media-dependencies }

When you run audio or video extraction, you might see an error similar to one
of the following:

```text
Audio extraction requires media dependencies; missing: ffmpeg.
VideoFrameActor requires media dependencies; missing: ffprobe.
```

The `ffmpeg-python` wrapper and `nemo-retriever[multimedia]` do not install the
`ffmpeg` or `ffprobe` binaries the pipeline executes.

For air-gapped or locked-down clusters, refer to [Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment).

**Connected environments:**

On Debian or Ubuntu hosts:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg
```

For the bundled service container at runtime:

```bash
docker run -e INSTALL_FFMPEG=true nemo-retriever-service
```

For Helm, when package-repo egress and the image security policy allow startup install:

```yaml
service:
  installFfmpeg: true
```

This path fails with `allowPrivilegeEscalation: false` or `readOnlyRootFilesystem: true`.

## Can't start new thread error { #cant-start-new-thread-error }

In rare cases, when you run a job you might an see an error similar to `can't start new thread`.
This error occurs when the maximum number of processes available to a single user is too low.
To resolve the issue, set or raise the maximum number of processes (`-u`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-u` setting, consider the following:

- Apply the `-u` setting directly to the user (or the environment of the pod or process) that runs your ingest service.
- For `-u` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -u 10000
```



## Out-of-Memory (OOM) Error when Processing Large Datasets { #out-of-memory-oom-error-when-processing-large-datasets }

When you process a very large dataset with thousands of documents, you might encounter an Out-of-Memory (OOM) error.
This happens because NeMo Retriever Library materializes extraction results in system memory (RAM) while the job runs.
If the total size of the results exceeds the available memory, the process fails.

To reduce memory pressure, try one or more of the following:

- Process documents in smaller batches instead of submitting the entire corpus in one job.
- Route outputs to a sink (for example, `.vdb_upload(...)`, `.webhook(...)`, or `.store(...)`) so results are written out instead of held in memory until the job finishes.
- In `run_mode="service"`, pass `return_results=False` to `.ingest(...)` when you do not need the full result payload returned to the client. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).
- Increase available host or pod memory for the ingest workload.



## Embedding service fails to start with an unsupported batch size error { #embedding-service-fails-unsupported-batch-size }

On some GPUs, for example RTX 6000, a self-hosted embedding NIM can fail
during startup with an error similar to the following:

```text
ValueError: Configured max_batch_size (30) is larger than the model's supported max_batch_size (3).
```

This error comes from the embedding NIM process. NeMo Retriever Library does
not read `EMBEDDER_BATCH_SIZE`. Setting that variable in the SDK, CLI, or
Helm process environment does not change NIM startup.

Configure the embedding NIM container instead. Use the supported maximum
from the error message. In this example, that value is `3`.

**Helm:** The default chart deploys `nemotron-3-embed-1b:2.2.2`
as `nimOperator.vlm_embed`. For that image, set `NIM_PIPELINE_MAX_BATCH_SIZE`
on `nimOperator.vlm_embed.env`. That list replaces the chart default, so
keep the default entries. The following example keeps those defaults, leaves
the optional performance mode disabled, and adds the batch-size variable.
Uncomment `NIM_PERFORMANCE_MODE` only if your NIM build supports it for this
SKU:

```yaml
nimOperator:
  vlm_embed:
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_TRITON_LOG_VERBOSE
        value: "1"
      - name: OMP_NUM_THREADS
        value: "1"
      - name: NIM_ENGINE_COUNT
        value: "1"
      # Optional: enable if your NIM build supports throughput mode for this SKU.
      # - name: NIM_PERFORMANCE_MODE
      #   value: "1"
      - name: NIM_PIPELINE_MAX_BATCH_SIZE
        value: "3"
```

**Development Compose:** The default `nim-embedding` image is also `nemotron-3-embed-1b:2.2.2`. Add `NIM_PIPELINE_MAX_BATCH_SIZE` to its existing environment mapping in `nemo_retriever/dev/compose/service-mode.compose.yaml`:

```yaml
NIM_PIPELINE_MAX_BATCH_SIZE: "3"
```

**Library or CLI with a remote NIM:** Set the image-specific batch-size
variable on the NIM container or NIMService that serves your embed URL.
`--embed-batch-size` and `.embed(inference_batch_size=...)` batch requests
after the NIM is running. They do not start the NIM.

**NVIDIA-hosted Build endpoints:** NVIDIA operates the NIM. This startup
error does not apply.

For image-specific variables, refer to
[Troubleshoot NVIDIA NeMo Retriever Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/troubleshoot.html)
and
[Environment Variables for NVIDIA NeMo Retriever Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/environment-variables.html).
For the Helm env list contract, refer to
[NIM Operator sub-stack](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#nim-operator-sub-stack).



## ModuleNotFoundError: No module named open_clip when using nemotron_parse { #modulenotfounderror-no-module-named-open-clip-when-using-nemotron-parse }

When you run PDF extraction with `method="nemotron_parse"`, you might see an error similar to the following:

```text
ModuleNotFoundError: No module named 'open_clip'
```

The Nemotron Parse NIM client requires the `open_clip` Python module, provided by `open-clip-torch`. That package is not part of the default `nemo-retriever` install or the `[local]` extra.

Install the dedicated PyPI extra before running Nemotron Parse extraction:

```bash
pip install "nemo-retriever[nemotron-parse]"
```

For local GPU inference with Nemotron Parse, combine extras:

```bash
pip install "nemo-retriever[local,nemotron-parse]"
```

Also refer to [NeMo Retriever Library Overview](overview.md) and [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#software-requirements).

## Extract method nemotron-parse doesn't support image files { #extract-method-nemotron-parse-doesnt-support-image-files }

Currently, extraction with Nemotron parse doesn't support image files, only scanned PDFs.
To work around this issue, convert image files to PDFs before you use `method="nemotron_parse"`.

## Nemotron Parse model and endpoint mismatch { #nemotron-parse-model-endpoint-mismatch }

When you run PDF extraction with `method="nemotron_parse"`, a mismatched model and endpoint can fail with an error similar to the following:

```text
HTTP 400: Content cannot be a plain string. The model does not support text input.
```

This can occur when you send a versioned self-hosted model (for example `nvidia/nemotron-parse-v1.2`) to the NVIDIA-hosted Build endpoint, which expects the image-only `nvidia/nemotron-parse` contract. The library may replace the raw HTTP error with a targeted model/contract mismatch hint.

To use hosted Build, omit `nemotron_parse_model` so the library selects `nvidia/nemotron-parse` automatically, or set `nemotron_parse_model="nvidia/nemotron-parse"` explicitly. Send `nvidia/nemotron-parse-v1.2` only to a compatible self-hosted chat endpoint.

Do not combine hosted Build and self-hosted endpoints in one `nemotron_parse_invoke_url` list. The library rejects this configuration because one workflow cannot send different model IDs and request contracts to individual endpoints. Setting `nemotron_parse_model` does not override this restriction. Use a homogeneous endpoint list, or configure separate ingestors or extraction workflows for hosted Build and self-hosted capacity. For more information, refer to [Nemotron Parse: hosted Build and self-hosted NIM contracts](prerequisites-support-matrix.md#nemotron-parse-hosted-vs-self-hosted).

## Hosted Page Elements NIM image size limits { #hosted-page-elements-nim-image-size-limits }

[NVIDIA-hosted Page Elements NIM](https://build.nvidia.com/nvidia/nemotron-page-elements-v3) endpoints on `ai.api.nvidia.com` accept only **inline** PNG or JPEG payloads. The matching build.nvidia.com model experience uses the same contract. The same `/v1/infer` request shape applies to hosted **Table Structure** and **Graphic Elements** object-detection NIMs.

The [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html) documents image URLs as `data:image/<format>;base64,<data>`. Hosted Page Elements inference does not accept NVCF Asset API identifiers in that `url` field.

The following table summarizes inline payload limits by deployment:

| Deployment | Inline base64 limit | If the image exceeds the limit |
|------------|---------------------|--------------------------------|
| Hosted (`build.nvidia.com`, `ai.api.nvidia.com`) | About **180,000 characters** on the base64 portion of the data URL (roughly 180 KB; build.nvidia.com validates `len(image_b64) < 180_000`) | Resize or re-encode the image so the inline payload fits. Self-host the NIM if you need larger images. Do not use the NVCF Asset API as a hosted fallback. |
| Self-hosted NIM container | Higher; the NeMo Retriever client downscales HTTP payloads above **512,000 characters** before calling the NIM | Resize or re-encode the source image, or rely on the client downscaling |

That API reference states only that "very large images may cause processing issues." For hosted integrations, treat **180,000 characters** as the inline cap unless NVIDIA publishes a different limit for your endpoint.

!!! important

    The build.nvidia.com playground can tell you to use the NVCF Asset API when an image exceeds the inline cap. Creating and uploading an asset can succeed. Hosted Page Elements inference still rejects `data:image/<format>;asset_id,<asset-id>` with HTTP 422. That rejection occurs with or without the `NVCF-INPUT-ASSET-REFERENCES` header. There is no Asset API recovery path for this hosted endpoint.

### NeMo Retriever Library pipeline users

When you route extraction to hosted Page Elements NIM URLs (for example `page_elements_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"`), the library:

- Renders PDF pages with default `render_mode="fit_to_model"` (targets about 1024 px on the long edge instead of full raster DPI).
- Downscales base64 page images before remote object-detection NIM HTTP calls when payloads exceed the client limit (512,000 characters for Page Elements and Table Structure).

!!! important

    The library downscales payloads to **512,000** characters before HTTP calls to object-detection NIMs. Hosted endpoints still reject inline base64 above **180,000** characters. Treat the lower hosted cap as the effective limit when `page_elements_invoke_url` points at `ai.api.nvidia.com`.

If you still receive **422** responses mentioning invalid image URLs on hosted endpoints, lower `dpi` in `ExtractParams` and keep `render_mode="fit_to_model"`. For very large standalone image inputs, preprocess the files before ingest. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

### Direct Page Elements NIM API calls

When you call Page Elements NIM **directly**, send only inline base64. Direct calls include the build playground, curl, and custom integrations that do not go through the NeMo Retriever pipeline. The following example shows the required `input` payload:

```json
{
  "input": [
    {
      "type": "image_url",
      "url": "data:image/png;base64,<BASE64_ENCODED_IMAGE>"
    }
  ]
}
```

Use this form only when `len(base64_image) < 180_000`. If the encoded image is larger, resize or re-encode it to PNG or JPEG until the base64 string is under 180,000 characters. You can also self-host Page Elements NIM.

Do not send `"url": "data:image/png;asset_id,<assetId>"`. Hosted Page Elements rejects that encoding with HTTP 422.

Hosted calls require the same [`NVIDIA_API_KEY`](api-keys.md#nvidia-api-key) you use for other build.nvidia.com NIM endpoints. For the request schema, refer to the [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html).

Supported formats remain **PNG** and **JPEG**, encoded as `data:image/<format>;base64,<data>`. OpenAPI specs for Page Elements v2 and v3 are linked from the [Object Detection NIM API reference](https://docs.nvidia.com/nim/ingestion/object-detection/latest/api-reference.html#openapi-reference-for-page-elements).

## Too many open files error { #too-many-open-files-error }

In rare cases, when you run a job you might an see an error similar to `too many open files` or `max open file descriptor`.
This error occurs when the open file descriptor limit for your service user account is too low.
To resolve the issue, set or raise the maximum number of open file descriptors (`-n`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-n` setting, consider the following:

- Apply the `-n` setting directly to the user (or the environment of the pod or process) that runs your ingest service.
- For `-n` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -n 10000
```



## Triton server INFO messages incorrectly logged as errors { #triton-server-info-messages-incorrectly-logged-as-errors }

Self-hosted NIM containers can wrap Triton server INFO lines as ERROR in
the container log. The logger can show a filename such as `nimutils.py`.
That logger is inside the NIM image. It is not part of the NeMo Retriever
Library source.

You can ignore messages whose Triton payload starts with `I` (INFO).
Treat them as informational. They do not mean ingest failed.

```text
ERROR 2025-04-24 22:49:44.268 nimutils.py:68] I0424 22:49:44.265292 98 cache_manager.cc:480] "Create CacheManager with cache_dir: '/opt/tritonserver/caches'"
ERROR 2025-04-24 22:49:44.431 nimutils.py:68] I0424 22:49:44.431796 98 pinned_memory_manager.cc:277] "Pinned memory pool is created at '0x7f8e4a000000' with size 268435456"
```

If a real failure occurs, inspect the NIM pod or Compose container logs for
the same timestamp. Do not treat these startup INFO lines as the root cause.



## LanceDB index creation fails during concurrent Helm ingestion { #lancedb-concurrent-index-creation }

Concurrent ingest workers can finish extraction at nearly the same time and
send overlapping writes to the VectorDB service. In affected releases, those
writes can rebuild the LanceDB vector or full-text-search index concurrently.
LanceDB rejects one of the competing commits with an error similar to the
following:

```text
Retryable commit conflict for version <version>:
This CreateIndex transaction was preempted by concurrent transaction CreateIndex.
Please retry.
```

Upgrade to a NeMo Retriever Library release that separates row commits from
index maintenance. The VectorDB service admits concurrent
`/internal/vectordb/write` requests that reach the same pod. Each request
commits its rows under a short-lived lock, so those rows are durable as soon
as the commit lands. LanceDB search also scans rows that no index covers yet,
so a committed row is queryable before the next rebuild includes it.

Only index maintenance is serialized, because LanceDB rejects competing index
commits. Concurrent writers share one coalesced rebuild: a rebuild that starts
after a batch was committed also indexes that batch. A write therefore never
waits behind another writer's index build. There is no Helm value to configure
this behavior.

An index-readiness wait that expires no longer fails the write. The VectorDB
pod logs a warning similar to the following, and the committed rows stay
queryable until the next rebuild covers them:

```text
LanceDB index on column 'vector' did not report coverage of 512 row(s) within
0:01:00. Queries still scan unindexed rows and the next rebuild will cover them.
```

Keep the VectorDB deployment at one replica. Row and index serialization is
local to a single VectorDB process. It does not coordinate writes across
multiple pods or independently deployed processes that share a LanceDB
directory.

If a request failed before the upgrade, inspect the table and the ingest job
before you resubmit it. A write can append rows before a later index rebuild
fails. The legacy append path does not deduplicate rows, so rerunning the same
input can create duplicate rows.

## Ingest fails with a VectorDB write error { #vectordb-write-not-acknowledged }

A worker posts extracted records to the VectorDB service before it reports a
document as complete. When the VectorDB service rejects that write or does not
acknowledge it within the configured timeout, the document fails with an error
similar to the following:

```text
RuntimeError: VectorDB write failed for report.pdf: <error>. The extracted rows
are not confirmed durable, so the document is not queryable.
```

A write into a managed collection reports
`Collection write failed for report.pdf: <error>` instead.

The worker pod logs the underlying failure at error level:

```text
Failed to POST 128 records to vectordb for report.pdf: <error>
```

Earlier releases logged this failure as a warning for writes to the legacy
fixed table and still reported the document as `completed` with a positive row
count. Queries then returned no rows for a document that the ingest job called
successful. A failed document now means the rows are not confirmed durable.

Complete the following checks:

1. Run `kubectl get pods --namespace <namespace>` and confirm the VectorDB pod is `Running` and ready. A pod that is restarting or unschedulable does not accept writes.
2. Read the VectorDB pod logs for the same records. A rejected write reports a request or backend error. A write that the worker abandoned on timeout can still be in progress on the VectorDB pod.
3. If writes time out under load or on slow storage, raise the acknowledgement timeout. `serviceConfig.vectordb.writeTimeoutSeconds` defaults to `300` seconds and covers the row commit plus the index maintenance that follows it.
4. Resubmit the failed documents after the write path is healthy. A write that timed out on the worker can still have committed its rows, and the legacy append path does not deduplicate rows, so check the table row count first.

To raise the timeout on an existing release, run the following command:

```bash
helm upgrade retriever ./nemo_retriever/helm \
  --reuse-values \
  --set serviceConfig.vectordb.writeTimeoutSeconds=900
```

For the rendered service key and sibling VectorDB values, refer to [Service configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#service-configuration-rendered-into-retriever-serviceyaml).


## Helm install succeeds but PersistentVolumeClaims stay Pending { #helm-pending-pvcs }

`helm install` can report `STATUS: deployed` while every default PersistentVolumeClaim stays `Pending`. That status means Helm rendered the release. It does not mean the retriever service, VectorDB, or core NIM workloads can schedule.

A representative claim event looks like the following:

```text
Type    Reason         From                          Message
Normal  FailedBinding  persistentvolume-controller   no persistent volumes available for this claim and no storage class is set
```

This event means the claim omitted `storageClassName` and the cluster has neither a default StorageClass nor a compatible classless persistent volume.

Complete the following checks:

1. Run `kubectl get storageclass` and `kubectl get pv`. Confirm a default StorageClass, a named class you set on every default claim, or compatible `Available` persistent volumes.
2. Run `kubectl get pvc --namespace <namespace>`. A default install creates seven claims. All seven must reach `Bound` before the functional workloads can start.
3. If you intended a named StorageClass, set the three chart-managed paths and the four per-NIM `nimOperator.<key>.storage.pvc.storageClass` paths. Do not set only `nimOperator.nimCache.pvc.storageClass`. That chart-level value is not applied to the core NIMCache resources.
4. After you add a default StorageClass or compatible volumes, confirm the claims become `Bound`. If they remain `Pending`, uninstall and reinstall after the storage strategy is in place.

For the default claim list, Helm value paths, and preflight commands, refer to [Kubernetes Helm Storage Requirements](prerequisites-support-matrix.md#kubernetes-helm-storage-requirements) and [Persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#persistent-storage-prerequisite).

## Core NIM pods stay Pending for GPU { #helm-pending-gpus }

`helm install` can report `STATUS: deployed` while one or more core NIM pods stay `Pending`. The default chart creates four NIMService workloads. Each requests `nvidia.com/gpu: 1`. On a conventional cluster without MIG or time-slicing, the scheduler needs four allocatable GPU slots across eligible nodes.

A representative pod event looks like the following:

```text
Warning  FailedScheduling  default-scheduler  0/1 nodes are available:
  1 Insufficient nvidia.com/gpu.
```

Complete the following checks:

1. Run `kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu` and sum `GPU` across eligible nodes. A default core install needs four slots across the cluster. Four one-GPU nodes are enough. A single node needs four slots only when you pack all four core NIMs onto one physical GPU with sharing and placement constraints.
2. Run `kubectl get pods --namespace <namespace>` and `kubectl describe pod <nim-pod>`. Confirm the Pending pods are the core NIMServices (`nemotron-page-elements-v3`, `nemotron-table-structure-v1`, `nemotron-ocr-v2`, and `nemotron-3-embed-1b`).
3. Either add GPU capacity so four slots are allocatable across the cluster, or configure GPU Operator time-slicing with at least four replicas before you reinstall. Time-slicing creates logical slots. MIG is an advanced GPU Operator configuration outside this chart. For one-GPU placement, cluster-wide oversubscription, and MIG constraints, refer to [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite).
4. After sharing or extra GPUs are in place, confirm the four core NIM pods reach `Running`.

For VRAM versus scheduling, the time-slicing ConfigMap, and ClusterPolicy patch, refer to [Kubernetes Helm GPU scheduling](prerequisites-support-matrix.md#kubernetes-helm-gpu-scheduling) and [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite).

## Split topology Helm install or upgrade times out with Deployments not ready { #helm-split-topology-startup-deadlock }

`helm upgrade --install --wait` can time out in split topology
(`topology.mode: split`) with errors similar to the following:

```text
Release "<release>" failed:
resource Deployment/<release>-nemo-retriever-gateway not ready
resource Deployment/<release>-nemo-retriever-realtime not ready
resource Deployment/<release>-nemo-retriever-batch not ready
context deadline exceeded
```

This deadlock occurs when the gateway readiness probe uses deep
`GET /v1/health`, which returns HTTP `503` until realtime and batch workers are
healthy, while worker Pods cannot start until their `wait-for-gateway` init
container reaches the gateway's shallow `GET /v1/live` endpoint. The externally
exposed gateway Service does not publish endpoints for an unready Pod, so neither
side can become ready on a clean install.

The chart renders an internal gateway startup Service named
`<release>-nemo-retriever-gateway-startup` with `publishNotReadyAddresses: true`.
Worker init containers poll `/v1/live` through that Service so startup completes
without manual intervention. Client traffic continues to use the
readiness-gated gateway Service.

If you run an older chart that does not render the startup Service, you can
unblock the install with a one-time patch on the gateway Service:

```bash
kubectl patch service <release>-nemo-retriever-gateway --type=merge \
  -p '{"spec":{"publishNotReadyAddresses":true}}'
```

Upgrade to a chart version that includes the startup Service so you do not need
to repeat that patch after every reinstall. Refer to [Health probes](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#health-probes) and [Service networking](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#service-networking) in the Helm chart README.

## Helm upgrade fails when changing a NIM image repository or tag { #helm-nimcache-modelpuller-immutable }

`helm upgrade` can fail when you change `nimOperator.<key>.image.repository` or `nimOperator.<key>.image.tag` on an existing release. The chart reuses the `NIMCache` name, for example `nemotron-page-elements-v3`, while `spec.source.ngc.modelPuller` changes to the new `repository:tag` value.

The NIM Operator `NIMCache` CRD marks `modelPuller` immutable. Kubernetes rejects the update with a message similar to the following:

```text
modelPuller is an immutable field. Please create a new NIMCache resource instead when you want to change this container.
```

Helm can apply other release resources before that rejection. The `NIMCache` then remains on the old image while the rest of the release has moved.

Do not retry `helm upgrade` until you delete the existing `NIMCache`. Complete the following steps:

1. Drain ingest traffic that depends on the affected NIM.
2. Run `kubectl get nimcache <name> --namespace <namespace>` and confirm the live `modelPuller` value differs from the new `repository:tag`.
3. Delete the `NIMCache`. Helm `keep` annotations do not block `kubectl delete`.
4. If the operator-created PVC remains, delete it so the new image re-pulls weights. Default claim names use a `-pvc` suffix, for example `nemotron-page-elements-v3-pvc`.
5. Re-run `helm upgrade` with the new repository or tag. Helm creates a new `NIMCache`.
6. Wait until the new cache is ready before you send traffic.

The affected NIM is unavailable during re-cache. Repeat the sequence for every NIM whose image changes.

Changing `service.image.repository` or `service.image.tag` does not use `NIMCache` and is not subject to this rule.

For default cache names, PVC cleanup, and the full upgrade sequence, refer to [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).

## NIMCache or NIMService still uses ngc-secret after a global Secret rename { #helm-nim-secret-names }

Retriever Pods inherit `ngcImagePullSecret.name` and `ngcApiSecret.name`. Empty per-NIM `image.pullSecrets` and `authSecret` inherit the same names.

If NIMCache or NIMService still shows `ngc-secret` or `ngc-api` after a rename, you still have a non-empty per-NIM override. The retriever Deployment can become Ready while NIM model-download Jobs and NIM Pods fail because they reference Secrets that do not exist.

Complete the following checks:

1. Render the chart with the NIM Operator CRDs enabled. Inspect `pullSecret` on every `NIMCache` and `pullSecrets` on every `NIMService`, plus `authSecret` on both.
2. Confirm those fields match Secrets that exist in the release namespace.
3. If a NIM still lists `ngc-secret` or `ngc-api` after you renamed the global Secret names, clear `nimOperator.<key>.image.pullSecrets` and `nimOperator.<key>.authSecret`, or set them to the new names. Empty values inherit the global names.
4. Top-level `imagePullSecrets` applies only to Retriever Pods. It does not update NIM Operator custom resources.

For value paths and a rename example, refer to [Use externally managed Secrets](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#use-externally-managed-secrets) and [Secrets](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#secrets) in the Helm chart README.

## Agentic retrieval fails with auto tool choice HTTP 400 { #agentic-auto-tool-choice }

`retriever query --agentic`, `POST /v1/query` with `agentic=true`, or the MCP `agentic_query` tool can fail on the first LLM call with HTTP 400 from a self-hosted chat-completions NIM:

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

The CLI then exits with `Agentic retrieval failed (llm_call_failed)`.

For the Helm `answer_llm` Nemotron 3.5 Lightning NIM, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` to `NIM_PASSTHROUGH_ARGS`, retaining `--reasoning-parser nemotron_v3`, and set `serviceConfig.agentic` for service mode. `POST /v1/answer` is a separate path and does not require tool calling.

For the copy-paste Helm values and CLI command, refer to [Self-hosted Helm Nemotron 3.5 Lightning](workflow-agentic-retrieval.md#self-hosted-helm-lightning).

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Kubernetes Helm Storage Requirements](prerequisites-support-matrix.md#kubernetes-helm-storage-requirements)
- [Kubernetes Helm GPU scheduling](prerequisites-support-matrix.md#kubernetes-helm-gpu-scheduling)
- [Deployment options](deployment-options.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md)
- [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag)
- [Use externally managed Secrets](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#use-externally-managed-secrets)
- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md#self-hosted-helm-lightning)
- [About getting started](getting-started-about.md) (prerequisites and deployment)
