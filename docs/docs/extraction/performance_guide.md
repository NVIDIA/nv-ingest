# Performance Guide

This page is a starting point for NeMo Retriever Library performance tuning guidance.

## Scope

Use this guide to document practical recommendations for:

- Extraction throughput and latency tuning
- Task-level settings (for example `extract`, `caption`, and `embed`)
- Deployment-specific tuning for library mode and Kubernetes/Helm
- NIM endpoint sizing and concurrency settings
- Benchmarking methodology and repeatable test setups

## Batch resource sizing

In batch mode, NeMo Retriever Library sizes unspecified Ray actor pools from Ray CPU and GPU resources. The library uses the resources that Ray reports as available immediately before it submits the pipeline. This prevents default extraction, OCR, and embedding pools from reserving more resources than the cluster can schedule.

For filesystem inputs, the library reserves CPU capacity for each Ray Data `ReadBinary` source task before it sizes actor pools. This reservation lets the input stage start instead of being blocked by persistent extraction actors. Inputs that are already Ray datasets, such as inline text rows, do not require this reservation.

When ingestion combines multiple extraction datasets, preflight also reserves 1 CPU for schema-normalization work. This keeps actor pools from occupying all CPU capacity before the datasets have compatible schemas.

Known text inputs use a direct `TxtSplitActor` graph that targets up to 8 actors, limited by the available CPUs. Shared preflight can reduce this default when the same batch job requires other actor pools or task reservations.

Explicit `BatchTuningParams` worker counts and integer `node_overrides` concurrency values must fit the available Ray CPU and GPU budget, including required task reservations. For an explicit Ray actor-pool tuple, validation uses `minimum` for `(minimum, maximum)` or `initial` for `(minimum, maximum, initial)`. The library preserves the tuple's `maximum` so Ray can grow the pool as capacity becomes available.

Only automatically derived concurrency can be reduced. A node override that sets `batch_size` or other settings without `concurrency` leaves automatic concurrency sizing enabled. The library raises an error before submitting work if the plan is infeasible. Reduce `*_workers`, per-node concurrency, or a tuple's starting size, or wait for shared-cluster capacity before retrying.

### Override worker counts

The library does not read environment variables to set worker counts or CPU and GPU totals. `CUDA_VISIBLE_DEVICES` still controls which GPUs Ray can see. To limit GPU count, start a Ray cluster with a restricted GPU set, for example `CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1`.

Set explicit worker counts with batch-mode CLI flags or with `BatchTuningParams` on `.extract()` and `.embed()`.

The following CLI example sets worker counts for a batch ingest. Replace
`/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs \
  --pdf-extract-workers 4 \
  --page-elements-workers 3 \
  --ocr-workers 3 \
  --embed-workers 2
```

The following Python example passes the same worker counts through `BatchTuningParams`:

```python
from pathlib import Path

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

documents = [str(Path("data/multimodal_test.pdf"))]

chunks = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract(
        batch_tuning=BatchTuningParams(
            pdf_extract_workers=4,
            page_elements_workers=3,
            ocr_workers=3,
            ocr_inference_batch_size=8,
        )
    )
    .embed(
        batch_tuning=BatchTuningParams(
            embed_workers=2,
        )
    )
    .ingest()
)
```

Related batch-size, CPU, and GPU-per-actor flags are documented in the [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md).

Use the Ray dashboard to verify the available-resource snapshot and the planned worker allocation when you tune throughput.

## Profile batch ingestion with Nsight Systems

Batch mode emits NVIDIA Tools Extension (NVTX) ranges from the driver and Ray
actor processes. Use NVIDIA Nsight Systems to correlate these semantic stages
with CUDA and operating system activity. Start with a representative, bounded
input because profiler tracing can increase runtime and report size.

Run a batch ingest under `nsys profile` as follows:

```bash
nsys profile \
  --trace=nvtx,cuda,osrt \
  --sample=none \
  --wait=all \
  --output=nrl-batch-profile \
  -- retriever ingest batch /path/to/your/pdfs
```

The command writes `nrl-batch-profile.nsys-rep`. Open the report in the Nsight
Systems graphical interface, or run `nsys-ui nrl-batch-profile.nsys-rep` on a
workstation with the interface installed. Search the timeline for
`nrl.batch::` to find the following ranges.

| Range | Interpretation |
| --- | --- |
| `nrl.batch::page_elements.startup` | One Page Elements actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::page_elements.batch` | One Page Elements actor processing call. |
| `nrl.batch::ocr.startup` | One OCR actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::ocr.batch` | One OCR actor processing call. |
| `nrl.batch::embedding.startup` | One embedding actor constructor. A backend that loads lazily can perform more startup work in the first batch. |
| `nrl.batch::embedding.batch` | One embedding actor processing call. |
| `nrl.batch::ray.materialize` | Materialization of a terminal Ray dataset into a pandas DataFrame. This range can include lazy upstream execution. |
| `nrl.batch::pipeline.terminal_stream` | Consumption of the lazy upstream pipeline and streaming of its records to the vector database backend. This range is not pure vector database time. |
| `nrl.batch::result.concat` | Concatenation of retained terminal frames when the caller requests full results. This range is absent or trivial on paths that do not retain full results. |

Ranges from concurrent Ray actors can overlap. Their durations include CPU
preparation, waits, and calls into inference backends. They do not represent
CUDA kernel time by themselves. In the timeline, compare range boundaries with
the CUDA activity rows to identify the stage that remains active near pipeline
completion and to find idle gaps between batches.

You can also print an aggregate NVTX summary:

```bash
nsys stats --report nvtx_sum nrl-batch-profile.nsys-rep
```

The summary adds durations from all matching ranges. With concurrent actors,
the summed duration can exceed wall-clock time, so use the timeline to reason
about overlap and the critical path.

## Tune remote OCR request batching

Remote OCR batches cropped regions across the page rows supplied to one OCR actor call. This behavior applies to in-process, batch, and service ingestion with a remote OCR NIM. It preserves page and region output order.

For in-process and batch ingestion, set `BatchTuningParams.ocr_inference_batch_size` through `.extract(batch_tuning=...)` to limit the cropped regions in each OCR HTTP request. This value takes precedence over `ExtractParams.inference_batch_size` for OCR. When you do not set it, OCR uses `ExtractParams.inference_batch_size`, which defaults to `8`.

Set `ExtractParams.remote_retry.remote_max_pool_workers` through `.extract(remote_retry=...)` to cap concurrent remote requests per actor. Each submitted group contains at most `ocr_inference_batch_size * remote_max_pool_workers` cropped regions, using the effective OCR batch size. Requests for different pages can overlap when a group contains multiple requests.

The NIM HTTP client applies the configured retry policy. If a group spans multiple pages and all concurrent failures are HTTP `400`, `413`, or `422`, OCR retries each page separately to isolate input-specific failures. Any authentication failure, exhausted rate-limit retry, transport failure, server error, or invalid response count prevents page-level retries for that group. If a remote request or response processing fails a page, OCR discards that page's results, including results from earlier groups, and reports the page error.

The execution mode determines how page rows reach the OCR actor and how many actors can submit requests.

| Mode | OCR batching and concurrency |
| --- | --- |
| `inprocess` | OCR batches across the page rows in the current graph stage, including pages of one PDF. It uses the remote request pool without Ray. |
| `batch` | In the dedicated PDF graph, Ray supplies page-row batches to OCR actors. `ocr_inference_batch_size` also sets this Ray row-batch size, while `ocr_workers` controls the number of actors. Remote request limits apply separately within each actor. |
| `service` | Each service worker runs an in-process graph. The whole-document route splits the PDF inside that graph, so OCR can batch across its pages. Service worker counts control concurrent work items. |

In service mode, use `.extract(inference_batch_size=...)` to set the request batch size. The default service policy does not accept client overrides for `batch_tuning` or `remote_retry`. The server controls NIM endpoints and credentials. Across Ray actors or service workers, total request concurrency can exceed the per-actor limit.

A smaller OCR inference batch size creates more, smaller HTTP requests. In Ray batch mode, it also reduces the number of page rows available to each actor call. A larger value creates fewer, larger requests. Benchmark both latency and throughput with representative documents, and monitor NIM GPU memory, HTTP `429` responses, errors, and output counts. Overlapping HTTP requests does not establish concurrent GPU execution; the NIM controls backend queueing and execution.

## Shared preflight for custom Ray Data graphs

`GraphIngestor` reserves source capacity automatically. For custom graphs, declare source capacity before calling `preflight_executors(...)`. Set `source_cpu_reservation=1` on each `RayDataExecutor` that will receive a filesystem path or glob. `source_cpu_reservation` must be a finite, non-negative CPU value. An executor that only receives an existing Ray dataset can omit the reservation.

```python
file_executor = RayDataExecutor(graph, source_cpu_reservation=1)
inline_executor = RayDataExecutor(graph)
preflight_executors([file_executor, inline_executor], cluster_resources)
```

The shared preflight records these reservations. NeMo Retriever Library rejects a later filesystem input when its executor lacks the required reservation. It rejects the input before it starts Ray work. Construct a new executor with `source_cpu_reservation=1`, and include it in a new shared preflight instead.
