#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compare PR 2766's unchanged Ray/NRL sink with Elasticsearch's Spark connector.

Input is FineWeb Parquet with existing 768-dimensional `embedding` arrays.
Both writers use identical payloads and NRL's index definition. No re-embedding.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def normalize_batch(table):
    import pyarrow as pa
    import pyarrow.compute as pc

    vector = pc.cast(table["embedding"], pa.list_(pa.float32(), 768))
    ids = pc.cast(table["id"], pa.string())
    if ids.null_count or vector.null_count:
        raise ValueError("IDs and vectors must be non-null")
    fields = {"id": ids, "document_id": ids, "source_id": ids,
              "text": pc.fill_null(pc.cast(table["text"], pa.string()), ""), "vector": vector}
    for name in ("url", "file_path", "dump"):
        fields[name] = pc.fill_null(pc.cast(table[name], pa.string()), "")
    return pa.table(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer", choices=["http", "hadoop"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urls", nargs="+", default=[f"http://127.0.0.1:{p}" for p in range(9200, 9208)])
    parser.add_argument("--cpus", type=int, default=128)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--read-workers", type=int, default=32)
    parser.add_argument("--prepare-workers", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--ray-object-store-gib", type=int, default=256)
    parser.add_argument("--spark-memory-gib", type=int, default=128)
    parser.add_argument("--jars", help="Comma-separated connector and VectorBase64 jars (Hadoop mode)")
    args = parser.parse_args()
    if args.writer == "hadoop" and not args.jars:
        parser.error("--jars is required for Hadoop mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"config": {**vars(args), "output_dir": str(args.output_dir)},
              "phases_seconds": {}, "status": "running"}
    started = time.perf_counter()
    (args.output_dir / "config.json").write_text(json.dumps(result["config"], indent=2))

    @contextmanager
    def phase(name):
        before = time.perf_counter()
        print(f"PHASE_START {name}", flush=True)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - before
            result["phases_seconds"][name] = elapsed
            print(f"PHASE_END {name} {elapsed:.6f}", flush=True)

    from nemo_retriever.common.vdb.elasticsearch import Elasticsearch
    from elasticsearch import Elasticsearch as Client
    import pyarrow.parquet as pq
    import pyarrow as pa

    client = Client(args.urls, request_timeout=600, max_retries=3, retry_on_timeout=True)
    if client.indices.exists(index=args.index_name):
        raise ValueError("Benchmark index must be new; existing indexes are never overwritten")
    sink_kwargs = dict(url=args.urls, index_name=args.index_name, vector_dim=768,
                       index_type="int8_hnsw", metric="cosine", hnsw_m=64,
                       hnsw_ef_construction=2000, number_of_shards=8, number_of_replicas=0,
                       input_format="embedding_parquet", batch_size=args.batch_size,
                       bulk_workers=args.workers, bulk_queue_size=args.workers,
                       refresh_interval="-1", request_timeout=600, encode_vectors_base64=True,
                       delete_index_on_failure=False, overwrite=False)
    backend = Elasticsearch(**sink_kwargs)
    with phase("dataset_preflight"):
        path = Path(args.input)
        files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError("No input Parquet files")
        expected_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
        sample = normalize_batch(pa.Table.from_batches([next(pq.ParquetFile(files[0]).iter_batches(batch_size=8))]))
        result["expected_rows"] = expected_rows
        result["input_files"] = [str(p) for p in files]
        sample_ids = sample["id"].to_pylist()
        result["gpu_usage_before"] = backend._assert_gpu_indexing()
        (args.output_dir / "mapping.json").write_text(json.dumps(backend._index_definition(), indent=2))

    stop = threading.Event()
    def monitor():
        with (args.output_dir / "resources.jsonl").open("w") as output:
            while not stop.is_set():
                entry = {"elapsed_seconds": time.perf_counter() - started}
                try:
                    entry["nodes"] = client.nodes.stats(metric=["indices", "jvm", "os", "process", "thread_pool"]).body
                    entry["gpu"] = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], text=True).strip()
                except Exception as exc:
                    entry["monitor_error"] = str(exc)
                output.write(json.dumps(entry) + "\n")
                output.flush()
                stop.wait(10)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    spark = None
    ray = None
    try:
        if args.writer == "http":
            with phase("runtime_startup"):
                import ray as ray_module
                from nemo_retriever.common.ray_runtime import build_local_ray_runtime_env
                ray_module.init(num_cpus=args.cpus, num_gpus=0,
                                object_store_memory=args.ray_object_store_gib << 30,
                                _temp_dir=str(args.output_dir / "ray"), log_to_driver=True,
                                runtime_env=build_local_ray_runtime_env())
                ray = ray_module
                from nemo_retriever.graph.executor import RayDataExecutor
                from nemo_retriever.graph.pipeline_graph import Graph
                from nemo_retriever.operators.vdb import IngestVdbOperator
                context = ray_module.data.DataContext.get_current()
                context.target_max_block_size = 128 << 20
                context.parquet_chunker_target_chunk_size = 128 << 20
                context.override_object_store_memory_limit_fraction = 1.0
                dataset = ray_module.data.read_parquet(
                    [str(p) for p in files], columns=["id", "text", "url", "file_path", "dump", "embedding"],
                    concurrency=args.read_workers, override_num_blocks=max(len(files) * 8, args.cpus * 4),
                ).map_batches(normalize_batch, batch_format="pyarrow", concurrency=args.prepare_workers,
                              batch_size=4096, num_cpus=1)
                sink = IngestVdbOperator(vdb_op="elasticsearch", vdb_kwargs=sink_kwargs)
                executor = RayDataExecutor(
                    Graph() >> sink, retain_stream_ingest_output=False,
                    stream_ingest_prepare_concurrency=args.prepare_workers,
                    stream_ingest_prefetch_batches=8, stream_ingest_queue_size=64,
                    source_cpu_reservation=1,
                )
            with phase("read_prepare_write_and_cuvs_build"):
                executor.ingest(dataset)
            result["stream_ingest_stats"] = getattr(executor, "_last_stream_ingest_stats", {})
            (args.output_dir / "ray_stats.txt").write_text(dataset.stats())
        else:
            with phase("runtime_startup"):
                os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-memory {args.spark_memory_gib}g pyspark-shell"
                os.environ["SPARK_LOCAL_DIRS"] = str(args.output_dir / "spark-tmp")
                from pyspark.sql import SparkSession, functions as f, types as t
                events = args.output_dir / "spark-events"
                events.mkdir()
                spark = (SparkSession.builder.master(f"local[{args.cpus}]")
                         .appName("NRL Elasticsearch Hadoop benchmark")
                         .config("spark.jars", args.jars)
                         .config("spark.eventLog.enabled", "true")
                         .config("spark.eventLog.dir", str(events))
                         .config("spark.sql.files.maxPartitionBytes", 64 << 20)
                         .config("spark.sql.shuffle.partitions", args.workers)
                         .config("spark.sql.adaptive.enabled", "false")
                         .getOrCreate())
                spark.udf.registerJavaFunction("nrl_vector_b64", "nrl.benchmark.VectorBase64", t.StringType())
                raw = spark.read.parquet(*[str(p) for p in files])
                docid = f.col("id").cast("string")
                source = [docid.alias("document_id"), docid.alias("source_id")]
                source += [f.coalesce(f.col(name), f.lit("")).alias(name) for name in ("url", "file_path", "dump")]
                source.append(f.when(f.length(f.col("url")) > 0, f.col("url")).alias("source_name"))
                rows = raw.select(
                    docid.alias("id"), f.coalesce(f.col("text"), f.lit("")).alias("text"),
                    f.struct(docid.alias("id"), f.lit("text").alias("type"), docid.alias("document_id")).alias("metadata"),
                    f.struct(*source).alias("source"), f.expr("nrl_vector_b64(embedding)").alias("vector"),
                )
                # Native file splitting avoids a full-data repartition shuffle.
                rows = rows.coalesce(args.workers)
                backend.create_index()
            with phase("read_prepare_write_and_cuvs_build"):
                (rows.write.format("org.elasticsearch.spark.sql")
                 .option("es.nodes", ",".join(url.removeprefix("http://") for url in args.urls))
                 .option("es.resource", args.index_name).option("es.mapping.id", "id")
                 .option("es.index.auto.create", "false").option("es.write.operation", "index")
                 .option("es.batch.size.entries", args.batch_size).option("es.batch.size.bytes", "100mb")
                 .option("es.batch.write.refresh", "false").option("es.batch.write.retry.count", "3")
                 .option("es.batch.write.retry.wait", "2s").option("es.http.timeout", "10m")
                 .option("es.spark.dataframe.write.null", "false").mode("append").save())
                client.indices.refresh(index=args.index_name)
        with phase("refresh_and_validation"):
            client.indices.refresh(index=args.index_name)
            client.indices.flush(index=args.index_name, wait_if_ongoing=True)
            client.cluster.health(index=args.index_name, wait_for_status="green", timeout="10m")
            actual = client.count(index=args.index_name)["count"]
            if actual != expected_rows:
                raise RuntimeError(f"Indexed {actual} unique IDs but expected {expected_rows} input rows")
            response = client.mget(index=args.index_name, ids=sample_ids)
            if not all(doc.get("found") for doc in response["docs"]):
                raise RuntimeError("Sample source IDs missing from index")
            for doc, original in zip(response["docs"], sample.to_pylist()):
                if doc["_source"]["text"] != original["text"]:
                    raise RuntimeError("Indexed text differs from input")
            import numpy as np
            hits = client.search(index=args.index_name, query={"ids": {"values": sample_ids}},
                                 source={"exclude_vectors": False}, size=len(sample_ids))["hits"]["hits"]
            actual_sources = {doc["_id"]: doc["_source"] for doc in hits}
            prepared = backend.prepare_embedding_parquet_batch(sample)
            for action, original in zip(backend._prepared_actions([prepared]), sample.to_pylist()):
                stored = dict(actual_sources[action["_id"]])
                np.testing.assert_allclose(stored.pop("vector"), original["vector"], rtol=0, atol=1e-7)
                expected = dict(action["_source"])
                expected.pop("vector")
                if stored != expected:
                    raise RuntimeError("Indexed source metadata differs from input")
            result["indexed_rows"] = actual
            result["index_stats"] = client.indices.stats(index=args.index_name).body
            # Segment-level vector statistics also include retry-replaced documents
            # until merges remove them. Validate live documents instead.
            result["live_vector_documents"] = client.count(
                index=args.index_name, query={"exists": {"field": "vector"}}
            )["count"]
            if result["live_vector_documents"] != expected_rows:
                raise RuntimeError("Live vector document count differs from input row count")
            result["gpu_usage_after"] = backend._assert_gpu_indexing()
            result["cluster_health"] = client.cluster.health(index=args.index_name).body
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        result["total_seconds"] = time.perf_counter() - started
        stop.set()
        thread.join(timeout=15)
        if spark is not None:
            spark.stop()
        if ray is not None:
            ray.shutdown()
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps({"status": result["status"], "total_seconds": result["total_seconds"],
                          "phases_seconds": result["phases_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
