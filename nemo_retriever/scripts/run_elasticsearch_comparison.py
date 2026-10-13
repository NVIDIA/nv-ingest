#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run both full benchmarks sequentially and preserve a comparison report."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request


def diagnostics(directory):
    stages = {"measurement_note": "Distributed task and shard times overlap; they are not additive wall-clock phases."}
    metrics = []
    for path in (directory / "ray" / "session_latest" / "logs" / "ray-data").glob("ray-data-dataset*.log"):
        content = path.read_text(errors="replace")
        for name, data in re.findall(r"Operator (.*?) completed\. Operator Metrics:\n(\{[^\n]+\})", content):
            metrics.append({"operator": name, "log": str(path), **ast.literal_eval(data)})
    if metrics:
        stages["ray_operators"] = metrics
    events = []
    for path in (directory / "spark-events").glob("*"):
        for line in path.open():
            event = json.loads(line)
            if event["Event"] == "SparkListenerStageCompleted":
                info = event["Stage Info"]
                events.append({"stage_id": info["Stage ID"], "name": info["Stage Name"],
                               "tasks": info["Number of Tasks"],
                               "wall_seconds": (info["Completion Time"] - info["Submission Time"]) / 1000})
            elif event["Event"] == "SparkListenerTaskEnd":
                # Task metrics include cumulative executor, serialization, read and GC work.
                task = event.get("Task Metrics", {})
                aggregate = stages.setdefault("spark_task_totals", {})
                for key in ["Executor Run Time", "Executor CPU Time", "Executor Deserialize Time",
                            "Result Serialization Time", "JVM GC Time"]:
                    aggregate[key] = aggregate.get(key, 0) + task.get(key, 0)
                for key in ["Input Metrics", "Output Metrics", "Shuffle Read Metrics", "Shuffle Write Metrics"]:
                    values = aggregate.setdefault(key, {})
                    for metric, value in task.get(key, {}).items():
                        if isinstance(value, (int, float)):
                            values[metric] = values.get(metric, 0) + value
    if events:
        stages["spark_stages"] = events
    (directory / "stage_metrics.json").write_text(json.dumps(stages, indent=2))
    return stages


def compare(root, names):
    results = [json.loads((root / name / "result.json").read_text()) for name in names]
    for name in names:
        diagnostics(root / name)
    rows = []
    for result in results:
        seconds = result["total_seconds"]
        stats = result.get("index_stats", {}).get("_all", {}).get("total", {})
        rows.append({"writer": result["config"]["writer"], "status": result["status"],
                     "documents": result.get("indexed_rows", 0), "total_seconds": seconds,
                     "documents_per_second": result.get("indexed_rows", 0) / seconds,
                     **result["phases_seconds"],
                     "server_indexing_shard_seconds": stats.get("indexing", {}).get("index_time_in_millis", 0) / 1000,
                     "server_refresh_shard_seconds": stats.get("refresh", {}).get("total_time_in_millis", 0) / 1000,
                     "server_merge_shard_seconds": stats.get("merges", {}).get("total_time_in_millis", 0) / 1000})
    report = {"runs": rows, "run_order": names,
              "hadoop_speedup": rows[0]["total_seconds"] / rows[1]["total_seconds"],
              "completion_condition": "Accepted documents, refreshed search visibility, flush, count and sample vector/source validation. Total excludes runtime shutdown; background merge drain between runs is recorded separately.",
              "caveat": "Full ingestion paths differ (Ray/Python versus Spark/JVM); one sequential comparison does not isolate transport costs or establish stable performance."}
    (root / "comparison.json").write_text(json.dumps(report, indent=2))
    with (root / "comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--jars", required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--names", nargs=2, default=["http-full1", "hadoop-full1"])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        compare(args.root, args.names)
        return
    state = {"pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "input": args.input, "runs": args.names, "status": "running"}
    def save():
        tmp = args.root / "state.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.root / "state.json")
    try:
        for writer, name in zip(["http", "hadoop"], args.names):
            state["active_writer"] = writer
            save()
            command = ["taskset", "-c", "32-63,96-127,160-191,224-255", sys.executable,
                       str(Path(__file__).with_name("benchmark_elasticsearch_writers.py")),
                       "--writer", writer, "--input", args.input, "--index-name", "fineweb-" + name,
                       "--output-dir", str(args.root / name)]
            if writer == "hadoop":
                command += ["--jars", args.jars]
            with (args.root / (name + ".log")).open("w") as output:
                subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
            # Prevent the first writer's background merge work from overlapping the next.
            state["active_writer"] = "draining_merges_after_" + writer
            save()
            started = time.monotonic()
            idle = 0
            while idle < 3:
                data = json.load(urllib.request.urlopen("http://127.0.0.1:9200/_nodes/stats/indices?filter_path=nodes.*.indices.merges.current", timeout=30))
                idle = idle + 1 if not any(node["indices"]["merges"]["current"] for node in data["nodes"].values()) else 0
                if time.monotonic() - started > 3600:
                    raise TimeoutError("Cluster merges did not drain within one hour")
                time.sleep(10)
            state.setdefault("merge_drain_seconds", {})[writer] = time.monotonic() - started
            save()
        compare(args.root, args.names)
        state["status"] = "complete"
        state["active_writer"] = None
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
