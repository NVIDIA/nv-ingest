# HTTP vs ES Hadoop Elastic search index creation

This benchmark is an experimental workflow for the PR 2766 checkout. It is not a released Retriever CLI workflow.

The benchmark compares the PR's NeMo Retriever Library (NRL) Python Elasticsearch writer with the Elasticsearch Hadoop connector's Spark writer. Both writers send Elasticsearch bulk requests over HTTP. The connector changes the client execution and serialization path; it does not replace Elasticsearch's HTTP protocol.

## Completed 50-million-document comparison

Both runs indexed the same 50,000,000 documents and existing 768-dimensional embeddings on a node with eight NVIDIA H100 NVL GPUs. Eight Elasticsearch nodes each used one GPU, with eight primary shards, zero replicas, cosine similarity, and matching `int8_hnsw` settings: `m=64` and `ef_construction=2000`.

The HTTP run completed first, followed by the Hadoop connector run. The runner waited for background merges to drain between runs. The recorded comparison contains the following measurements.

| Measurement | NRL HTTP writer | ES Hadoop connector |
| --- | ---: | ---: |
| Documents | 50,000,000 | 50,000,000 |
| Total wall time (seconds) | 6,316.388837 | 3,660.648710 |
| Throughput (documents/second) | 7,915.915 | 13,658.781 |
| Dataset preflight (seconds) | 0.163307 | 0.194536 |
| Runtime startup (seconds) | 5.157455 | 4.983130 |
| Read, prepare, write, and cuVS build (seconds) | 6,306.898618 | 3,652.326492 |
| Refresh and validation (seconds) | 2.296216 | 1.322828 |

The Hadoop path completed in 42.05% less wall time, corresponding to 1.725 times the throughput. This is one sequential comparison of the complete Ray/Python and Spark/JVM ingestion paths. It does not isolate transport costs or establish a repeatable performance difference.

The total covers accepted documents, refreshed search visibility, flush, document counts, and sample vector and source validation. It excludes runtime shutdown, background merge drain, and post-run recovery. Small differences between the sum of named phases and the total reflect work outside those phase timers. Concurrent worker and shard durations overlap and cannot be added to these wall-clock intervals.

The original Hadoop segment-level vector check included 5,653 deleted documents from retry replacements. The corrected live-vector check counts searchable documents with a `vector` field. Both indexes contained exactly 50,000,000 live vector documents. Post-run recovery and this correction occurred outside the recorded timings; the table reports the original ingestion measurements.

## Keep the indexing workload comparable

Use the same input files, document selection, vector dimensions, document IDs, payload, index mapping, and Elasticsearch cluster for both runs. Create a separate empty index for each run. Run the writers sequentially so they do not compete for Elasticsearch, GPU, CPU, memory, or disk resources.

The input at `/localhome/local-jperez/data/fineweb/part-000` contains existing embeddings. The benchmark converts the vectors to float32 for ingestion. It does not extract documents or generate new embeddings. The input contains 50,000,000 documents in 200 Parquet files, with 768-dimensional vectors.

The Python run retains the PR's Ray Data ingestion and prepared-batch Elasticsearch path. The connector run uses Spark and the Elasticsearch Hadoop connector. This comparison therefore measures the complete ingestion paths, including their different readers and preparation work.

Use matching `int8_hnsw` mappings, with eight primary shards, zero replicas, `m=64`, and `ef_construction=2000`. These are benchmark settings, not recommendations for every deployment.

## Prepare the cluster

Use an Elasticsearch build with the PR's cuVS dependencies. The Docker build recipe is in [`../docker/elasticsearch-cuvs/Dockerfile`](../docker/elasticsearch-cuvs/Dockerfile).

For the eight-GPU node, assign one GPU to each Elasticsearch node and distribute the primary shards across the eight nodes. GPU indexing happens in Elasticsearch. Ray and Spark preparation do not re-embed the input on GPUs.

Verify the Elasticsearch license, GPU indexing capability, cluster health, and shard placement before timing ingestion. A GPU visible to a container does not establish that Elasticsearch used it to build the index. Preserve GPU capability results, sampled utilization, and server logs with the benchmark artifacts.

Provide enough approved storage for the indexes, translogs, merges, logs, and client temporary files. The benchmark does not authorize formatting or repurposing a block device.

## Run the comparison

Run the script from this checkout with a Python environment that imports its NRL source and includes Ray, PyArrow, and the Elasticsearch Python client. For the connector run, also provide Spark 3.5.7, a compatible Java runtime, the `elasticsearch-spark-30_2.12` 9.5.2 connector JAR, and a compiled JAR for [`VectorBase64.java`](VectorBase64.java). The JVM user-defined function encodes vectors as big-endian float32 base64, matching the NRL writer.

Check the available options before running the benchmark.

```bash
python nemo_retriever/scripts/benchmark_elasticsearch_writers.py --help
```

The following commands ingest every Parquet file in `part-000`. They use the default endpoints `http://127.0.0.1:9200` through `http://127.0.0.1:9207`. Replace the artifact directories and JAR paths with prepared local paths. Use `--urls` to specify another cluster.

```bash
python nemo_retriever/scripts/benchmark_elasticsearch_writers.py \
  --writer http \
  --input /localhome/local-jperez/data/fineweb/part-000 \
  --index-name fineweb_pr2766_http \
  --output-dir /path/to/benchmark-artifacts/http \
  --cpus 128 --workers 128 --read-workers 32 --prepare-workers 64 \
  --batch-size 500 --ray-object-store-gib 256

python nemo_retriever/scripts/benchmark_elasticsearch_writers.py \
  --writer hadoop \
  --input /localhome/local-jperez/data/fineweb/part-000 \
  --index-name fineweb_pr2766_hadoop \
  --output-dir /path/to/benchmark-artifacts/hadoop \
  --cpus 128 --workers 128 --spark-memory-gib 128 --batch-size 500 \
  --jars /path/to/elasticsearch-spark-30_2.12-9.5.2.jar,/path/to/nrl-vector-base64.jar
```

Each index name and output directory must be new. The script rejects an existing index or output directory. It preserves a partial index after an ingestion failure. To rerun, use new names or remove only the failed benchmark artifacts after confirming that you no longer need them.

The settings above allocate substantial CPU and memory resources. Confirm that the node can provide those resources alongside all Elasticsearch nodes. `--workers` controls Python bulk concurrency in the HTTP run and the maximum coalesced Spark partition count in the Hadoop run. Identical values do not guarantee identical active request concurrency.

For a smoke test, point `--input` at the same single Parquet file in both runs. The script accepts a file or a directory; it does not implement a row-limit flag.

Each run produces the following artifacts.

| Artifact | Content |
| --- | --- |
| `config.json` | Writer arguments and runtime settings. |
| `mapping.json` | Expected index definition. |
| `result.json` | Status, wall time, named phase timings, document count, server statistics, and GPU capability results. |
| `resources.jsonl` | Sampled Elasticsearch node statistics and `nvidia-smi` output every 10 seconds. |
| `ray_stats.txt` | Ray dataset statistics for the HTTP run. |
| `spark-events/` | Spark event logs for the Hadoop run. |
| `stage_metrics.json` | Ray operator metrics or Spark stage and cumulative task metrics, generated by the comparison runner. |

Capture standard output and standard error separately to retain client logs and phase messages. `total_seconds` ends before runtime shutdown; report that boundary when comparing completion times.

The sequential runner starts the HTTP run, waits for background merges to drain, starts the Hadoop run, and writes `comparison.json` and `comparison.csv`. It preserves `state.json` and each writer's log. On this node, the prepared connector and UDF JARs are in `/raid/nrl-es-benchmark/tools`.

```bash
PYTHONPATH="$PWD/nemo_retriever/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  SPARK_LOCAL_IP=127.0.0.1 .venv-bench/bin/python \
  nemo_retriever/scripts/run_elasticsearch_comparison.py \
  --root /raid/nrl-es-benchmark/full \
  --input /localhome/local-jperez/data/fineweb/part-000 \
  --jars /raid/nrl-es-benchmark/tools/elasticsearch-spark-30_2.12-9.5.2.jar,/raid/nrl-es-benchmark/tools/vector-base64.jar
```

The runner assigns the client to the 128 logical CPUs reserved for Ray or Spark on this node. The Elasticsearch containers use the other 128 logical CPUs. These CPU sets are specific to this node's topology.

Compile the vector encoder against the installed Spark dependencies before running the connector path.

```bash
mkdir -p /raid/nrl-es-benchmark/tools/classes
javac -cp '.venv-bench/lib/python3.12/site-packages/pyspark/jars/*' \
  -d /raid/nrl-es-benchmark/tools/classes nemo_retriever/scripts/VectorBase64.java
jar cf /raid/nrl-es-benchmark/tools/vector-base64.jar \
  -C /raid/nrl-es-benchmark/tools/classes .
```

## Interpret the measurements

Compare total wall time and successful document throughput first. Confirm that both indexes contain the expected document count and matching vector mappings before comparing timings.

The top-level startup, dataset preflight, streaming write, and refresh and validation intervals describe the client workflow. Distributed reads, vector preparation, serialization, network transfer, Elasticsearch indexing, and cuVS work can overlap inside the streaming interval.

Treat Ray and Spark task durations and Elasticsearch indexing and merge counters as supporting diagnostics. They are cumulative worker or shard measurements, not independent wall-clock stages. Do not sum them to produce an end-to-end time.

A returned bulk response establishes document acceptance. Refresh establishes search visibility. Background merges can continue after refresh. State the completion condition alongside the timing results and use the same condition for both writers.

Retain the source revision, dependency versions, selected input files, document selection, writer settings, index mapping, cluster configuration, resource samples, logs, and final validation results. File-system caches, warm processes, retries, and retained indexes can affect later runs. Report the run order and repeat the comparison when assessing a stable performance difference.
