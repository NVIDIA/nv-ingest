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

### Use an elastic embedding worker range

Use an elastic embedding actor pool to let Ray Data adjust the embedding actor count within explicit bounds. This option is disabled by default. It uses Ray's native actor-pool scheduling, not an adaptive policy that NeMo Retriever Library enables automatically. Continue to set `embed_workers` when you need a fixed embedding pool.

Set all three elastic worker fields together. Do not combine them with `embed_workers`. Each value must be a positive integer, and the values must satisfy `embed_workers_min <= embed_workers_initial <= embed_workers_max`.

The following Python example requests four initial embedding actors, sets a minimum of four actors, and permits growth to eight actors. These values illustrate the controls, not a recommended allocation for every workload:

```python
chunks = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract()
    .embed(
        batch_tuning=BatchTuningParams(
            embed_workers_min=4,
            embed_workers_initial=4,
            embed_workers_max=8,
            embed_cpus_per_actor=0.5,
            gpu_embed=0.5,
        )
    )
    .ingest()
)
```

The fields control the Ray Data actor pool as follows:

| Field | Behavior |
| --- | --- |
| `embed_workers_min` | Sets the lower bound for the active actor pool. |
| `embed_workers_initial` | Sets the actor count that Ray requests when the pool starts and that resource preflight includes in the initial plan. |
| `embed_workers_max` | Sets the upper bound that Ray can grow toward while embedding work is queued and resources are available. |

Preflight budgets the initial actors alongside other pipeline stages and required task reservations. It does not reserve capacity for the maximum actor count or guarantee that the pool reaches that count. Each additional actor must obtain its requested logical CPU and GPU resources at runtime.

Retriever harness runfiles expose the same range under `ingest.embed.batch`. The following `set` object applies the preceding example:

```json
{
  "set": {
    "ingest.embed.batch.embed_workers_min": 4,
    "ingest.embed.batch.embed_workers_initial": 4,
    "ingest.embed.batch.embed_workers_max": 8,
    "ingest.embed.batch.embed_cpus_per_actor": 0.5,
    "ingest.embed.batch.embed_gpus_per_actor": 0.5
  }
}
```

The root CLI currently exposes only the fixed `--embed-workers` option. Use `BatchTuningParams` or a harness runfile to configure an elastic range.

Ray CPU and GPU resource values are logical scheduling units. For example, `gpu_embed=0.5` permits two embedding actors to share one logical GPU, but it does not limit each actor to half of the GPU memory. NeMo Retriever Library does not validate model memory, batch memory, or the safe number of colocated actors. Profile GPU memory before increasing the maximum actor count.

An elastic range does not pin actors to devices, wait for named upstream stages to finish, or guarantee when upstream actors release resources. Ray controls pool growth and shrinkage. Additional actors incur startup costs, including model loading for local inference, so growth does not guarantee an end-to-end speedup. Verify the actor count, device placement, queue backlog, and GPU memory in a representative run before using the range for larger workloads.

To isolate the effect of embedding concurrency, compare fixed and elastic pools using the same frozen extracted chunks, stable chunk identifiers, modalities, and embedding inputs. Keep model revisions, precision, embedding settings, and index and retrieval settings unchanged. Changing extraction outputs between runs prevents attributing a quality difference to the worker range alone.

Check retrieval quality with the same queries and relevance labels. Compare per-query results and metrics, not only aggregate scores or output row counts. A worker-range setting alone does not establish quality invariance. Measure end-to-end latency, including actor startup, under controlled cache and startup conditions before choosing a range for your deployment.

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
