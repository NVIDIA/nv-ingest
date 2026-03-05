# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI commands for ``retrieval-bench evaluate dense-retrieval`` and
``retrieval-bench evaluate agentic-retrieval``.
"""

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Set

import typer

from retrieval_bench.pipeline_evaluation import (
    BasePipeline,
    aggregate_results,
    evaluate_retrieval,
    load_vidore_dataset,
    print_dataset_info,
)
from retrieval_bench.pipeline_evaluation.tracing import dataset_trace_dir, default_trace_run_name
from retrieval_bench.pipelines.backends import VALID_BACKENDS
from vidore_benchmark.utils.logging_utils import setup_logging

# Import pipeline utility commands (report-results, compare-results, etc.)
from retrieval_bench.cli.pipeline_evaluation import app as _pipeline_utils_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Evaluate retrieval pipelines on ViDoRe v3 / BRIGHT datasets.",
    no_args_is_help=True,
)

app.add_typer(
    _pipeline_utils_app,
    name="utils",
    help="Reporting & trace utilities (list-datasets, report-results, compare-results, ...)",
)


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


# ---------------------------------------------------------------------------
# Query-id selector helpers (duplicated minimally from pipeline_evaluation)
# ---------------------------------------------------------------------------


def _parse_query_ids_selector(selector: str) -> Set[str]:
    selector = (selector or "").strip()
    if not selector:
        raise ValueError("Empty --query-ids selector.")
    ids: Set[str] = set()
    for tok in [t.strip() for t in selector.split(",") if t.strip()]:
        if "-" in tok:
            parts = [p.strip() for p in tok.split("-", 1)]
            start, end = int(parts[0]), int(parts[1])
            for i in range(start, end + 1):
                ids.add(str(i))
        else:
            ids.add(str(int(tok)))
    return ids


def _parse_index_selector(selector: str) -> Set[int]:
    selector = (selector or "").strip()
    if not selector:
        raise ValueError("Empty --query-ids selector.")
    ids: Set[int] = set()
    for tok in [t.strip() for t in selector.split(",") if t.strip()]:
        if "-" in tok:
            parts = [p.strip() for p in tok.split("-", 1)]
            start, end = int(parts[0]), int(parts[1])
            for i in range(start, end + 1):
                ids.add(i)
        else:
            ids.add(int(tok))
    return ids


def _query_ids_are_numeric(query_ids: list[str]) -> bool:
    return all(str(qid).isdigit() for qid in query_ids)


def _filter_queries_by_ids(query_ids, queries, qrels, query_languages, requested_ids):
    fq_ids, fq = [], []
    for qid, q in zip(query_ids, queries):
        if qid in requested_ids:
            fq_ids.append(qid)
            fq.append(q)
    if not fq_ids:
        raise ValueError("After applying --query-ids, zero queries remain.")
    fqrels = {qid: qrels[qid] for qid in fq_ids if qid in qrels}
    fql = {qid: query_languages.get(qid, "unknown") for qid in fq_ids}
    return fq_ids, fq, fqrels, fql


def _filter_queries_by_positions(query_ids, queries, qrels, query_languages, requested_positions):
    fq_ids, fq = [], []
    for idx, (qid, q) in enumerate(zip(query_ids, queries)):
        if idx in requested_positions:
            fq_ids.append(qid)
            fq.append(q)
    if not fq_ids:
        raise ValueError("After applying --query-ids, zero queries remain.")
    fqrels = {qid: qrels[qid] for qid in fq_ids if qid in qrels}
    fql = {qid: query_languages.get(qid, "unknown") for qid in fq_ids}
    return fq_ids, fq, fqrels, fql


# ---------------------------------------------------------------------------
# Shared evaluation runner
# ---------------------------------------------------------------------------

_METRICS = [
    "ndcg_cut_1",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "ndcg_cut_100",
    "recall_1",
    "recall_5",
    "recall_10",
    "recall_20",
    "recall_50",
    "recall_100",
    "P_1",
    "P_5",
    "P_10",
    "P_20",
    "map",
    "map_cut_1",
    "map_cut_10",
    "map_cut_100",
    "recip_rank",
]


def _run_evaluation(
    *,
    pipeline: BasePipeline,
    dataset_name: str,
    split: str,
    language: Optional[str],
    query_ids_selector: Optional[str],
    trace_run_name: Optional[str],
    traces_dir: str,
    output_file: Optional[str],
    show_dataset_info: bool,
    pipeline_label: str,
    pipeline_args_for_output: Dict[str, Any],
    cache_only: bool = False,
) -> None:
    """Shared dataset-loading / evaluation / result-display logic."""

    # Load dataset
    try:
        query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels, query_languages, excluded_ids_by_query = (
            load_vidore_dataset(dataset_name=dataset_name, split=split, language=language)
        )
    except Exception as e:
        print(f"\nError loading dataset: {e}\n")
        raise typer.Exit(code=1)

    # Apply query-id filter
    if query_ids_selector:
        try:
            if _query_ids_are_numeric(query_ids):
                requested = _parse_query_ids_selector(query_ids_selector)
                query_ids, queries, qrels, query_languages = _filter_queries_by_ids(
                    query_ids,
                    queries,
                    qrels,
                    query_languages,
                    requested,
                )
            else:
                requested_pos = _parse_index_selector(query_ids_selector)
                query_ids, queries, qrels, query_languages = _filter_queries_by_positions(
                    query_ids,
                    queries,
                    qrels,
                    query_languages,
                    requested_pos,
                )
        except ValueError as e:
            print(f"\nError parsing/applying --query-ids: {e}\n")
            raise typer.Exit(code=1)

    if show_dataset_info:
        print_dataset_info(dataset_name, query_ids, queries, corpus_ids, corpus_images, corpus_texts, qrels)

    # Cache-only mode: build corpus embeddings and exit without running queries.
    if cache_only:
        pipeline.dataset_name = dataset_name
        pipeline.index(corpus_ids=corpus_ids, corpus_images=corpus_images, corpus_texts=corpus_texts)
        print("Corpus embeddings cached. Exiting (--cache-only).")
        return

    # Evaluate
    print("\nRunning evaluation...")
    try:
        pipeline.dataset_name = dataset_name
        trace_run_name_eff = trace_run_name or default_trace_run_name(pipeline)
        results = evaluate_retrieval(
            pipeline=pipeline,
            query_ids=query_ids,
            queries=queries,
            corpus_ids=corpus_ids,
            corpus_images=corpus_images,
            corpus_texts=corpus_texts,
            qrels=qrels,
            traces_dir=traces_dir,
            trace_run_name=trace_run_name_eff,
            dataset_name=dataset_name,
            split=split,
            language=language,
            query_ids_selector=query_ids_selector,
            excluded_ids_by_query=excluded_ids_by_query,
            metrics=_METRICS,
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}\n")
        raise typer.Exit(code=1)

    timing_info = results.get("_timing", {}) if isinstance(results, dict) else {}
    aggregated = aggregate_results(results, query_languages)

    # Display results
    _display_results(aggregated, timing_info)

    # Save results
    if output_file is None:
        pipeline_name = trace_run_name_eff
        dataset_short = dataset_trace_dir(dataset_name)
        os.makedirs(f"results/{pipeline_name}", exist_ok=True)
        output_file = f"results/{pipeline_name}/{dataset_short}.json"

    output_path = Path(output_file)
    wall_time_per_query_ms = None
    if (
        isinstance(timing_info, dict)
        and timing_info.get("total_wall_time_milliseconds") is not None
        and len(query_ids) > 0
    ):
        wall_time_per_query_ms = timing_info["total_wall_time_milliseconds"] / len(query_ids)

    output_data = {
        "dataset": dataset_name,
        "split": split,
        "language": language,
        "query_ids_selector": query_ids_selector,
        "traces_dir": traces_dir,
        "trace_run_name": trace_run_name_eff,
        "pipeline_label": pipeline_label,
        "pipeline_args": pipeline_args_for_output,
        "aggregated_metrics": aggregated,
        "wall_time_per_query_milliseconds": wall_time_per_query_ms,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}\n")

    if isinstance(timing_info, dict):
        missing_num = timing_info.get("missing_num_queries", 0)
        expected_num = timing_info.get("expected_num_queries", len(query_ids))
        if isinstance(missing_num, int) and missing_num > 0:
            typer.secho(
                f"WARNING: {missing_num}/{expected_num} expected queries were not evaluated.",
                fg=typer.colors.RED,
                bold=True,
            )


def _display_results(aggregated: Dict[str, Any], timing_info: Any) -> None:
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    key_metrics = ["ndcg_cut_10", "ndcg_cut_5", "recall_10", "recall_5", "map", "recip_rank"]

    if "overall" in aggregated and "by_language" in aggregated:
        overall_metrics = aggregated["overall"]
        timing_info = aggregated.get("timing", {})

        print("\n--- Overall Results ---")
        for metric in key_metrics:
            if metric in overall_metrics:
                print(f"  {metric:25s}: {overall_metrics[metric]:.4f}")

        print("\n--- Results by Language ---")
        for lang, lang_metrics in aggregated["by_language"].items():
            num_queries = lang_metrics.get("num_queries", 0)
            print(f"\n{lang.capitalize()} ({num_queries} queries):")
            for metric in key_metrics:
                if metric in lang_metrics:
                    print(f"  {metric:25s}: {lang_metrics[metric]:.4f}")

        other_count = len(overall_metrics) - len([m for m in key_metrics if m in overall_metrics])
        if other_count > 0:
            print(f"\n({other_count} additional metrics saved to file)")

        if timing_info:
            print("\n--- Timing Metrics ---")
            for metric, value in timing_info.items():
                if "milliseconds" in metric:
                    print(f"  {metric:40s}: {value:.2f}ms")
                else:
                    print(f"  {metric:40s}: {value:.2f}")
    else:
        timing_metrics = {
            k: v for k, v in aggregated.items() if k.startswith(("total_", "average_", "queries_", "num_"))
        }
        retrieval_metrics = {k: v for k, v in aggregated.items() if k not in timing_metrics}

        if retrieval_metrics:
            print("\nKey Retrieval Metrics:")
            for metric in key_metrics:
                if metric in retrieval_metrics:
                    print(f"  {metric:25s}: {retrieval_metrics[metric]:.4f}")
            other_count = len(retrieval_metrics) - len([m for m in key_metrics if m in retrieval_metrics])
            if other_count > 0:
                print(f"\n  ({other_count} additional metrics saved to file)")

        if timing_metrics:
            print("\nTiming Metrics:")
            for metric, value in timing_metrics.items():
                if "milliseconds" in metric:
                    print(f"  {metric:40s}: {value:.2f}ms")
                else:
                    print(f"  {metric:40s}: {value:.2f}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_backend_help = "Dense retriever backend. One of: " + ", ".join(sorted(VALID_BACKENDS))


@app.command("dense-retrieval")
def dense_retrieval(
    dataset_name: Annotated[str, typer.Option(help="Dataset name (e.g. 'bright/biology', 'vidore/vidore_v3_hr')")],
    backend: Annotated[str, typer.Option(help=_backend_help)],
    top_k: Annotated[int, typer.Option(help="Number of results per query")] = 100,
    language: Annotated[Optional[str], typer.Option(help="Language filter (e.g. 'english')")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    query_ids_selector: Annotated[
        Optional[str],
        typer.Option("--query-ids", help="Query-id selector, e.g. '0-99,120'"),
    ] = None,
    output_file: Annotated[Optional[str], typer.Option(help="Path to save results JSON")] = None,
    trace_run_name: Annotated[Optional[str], typer.Option(help="Trace subdirectory name")] = None,
    traces_dir: Annotated[str, typer.Option(help="Traces root directory")] = "traces",
    show_dataset_info: Annotated[bool, typer.Option(help="Show dataset info")] = True,
    pipeline_args: Annotated[
        Optional[str],
        typer.Option(help="JSON string of additional backend overrides"),
    ] = None,
    cache_only: Annotated[
        bool, typer.Option("--cache-only", help="Only build corpus embedding cache, skip evaluation")
    ] = False,
):
    """
    Evaluate a dense retrieval backend on a dataset.

    Examples:

        retrieval-bench evaluate dense-retrieval \\
            --dataset-name bright/biology \\
            --backend llama-nv-embed-reasoning-3b

        retrieval-bench evaluate dense-retrieval \\
            --dataset-name vidore/vidore_v3_hr \\
            --backend llama-nemotron-embed-vl-1b-v2 \\
            --language english
    """
    if backend not in VALID_BACKENDS:
        print(f"\nUnknown backend: {backend!r}")
        print(f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
        raise typer.Exit(code=1)

    overrides: Dict[str, Any] = {}
    if pipeline_args:
        try:
            overrides = json.loads(pipeline_args)
        except json.JSONDecodeError as e:
            print(f"\nError parsing --pipeline-args: {e}\n")
            raise typer.Exit(code=1)

    from retrieval_bench.pipelines.dense import DenseRetrievalPipeline

    pipeline = DenseRetrievalPipeline(backend=backend, top_k=top_k, **overrides)

    _run_evaluation(
        pipeline=pipeline,
        dataset_name=dataset_name,
        split=split,
        language=language,
        query_ids_selector=query_ids_selector,
        trace_run_name=trace_run_name,
        traces_dir=traces_dir,
        output_file=output_file,
        show_dataset_info=show_dataset_info,
        pipeline_label=f"dense-retrieval/{backend}",
        pipeline_args_for_output={"backend": backend, "top_k": top_k, **overrides},
        cache_only=cache_only,
    )


@app.command("agentic-retrieval")
def agentic_retrieval(
    dataset_name: Annotated[str, typer.Option(help="Dataset name (e.g. 'bright/biology', 'vidore/vidore_v3_hr')")],
    backend: Annotated[str, typer.Option(help=_backend_help)],
    llm_model: Annotated[str, typer.Option(help="LLM model identifier (e.g. 'gpt-4o', 'openai/my-model')")],
    num_concurrent: Annotated[int, typer.Option(help="Number of concurrent agent queries")] = 1,
    reasoning_effort: Annotated[str, typer.Option(help="Reasoning effort level")] = "high",
    target_top_k: Annotated[int, typer.Option(help="Target number of final results per query")] = 10,
    retriever_top_k: Annotated[int, typer.Option(help="Retriever top-k (overrides default 500)")] = 500,
    language: Annotated[Optional[str], typer.Option(help="Language filter (e.g. 'english')")] = None,
    split: Annotated[str, typer.Option(help="Dataset split")] = "test",
    query_ids_selector: Annotated[
        Optional[str],
        typer.Option("--query-ids", help="Query-id selector, e.g. '0-99,120'"),
    ] = None,
    output_file: Annotated[Optional[str], typer.Option(help="Path to save results JSON")] = None,
    trace_run_name: Annotated[Optional[str], typer.Option(help="Trace subdirectory name")] = None,
    traces_dir: Annotated[str, typer.Option(help="Traces root directory")] = "traces",
    show_dataset_info: Annotated[bool, typer.Option(help="Show dataset info")] = True,
    pipeline_args: Annotated[
        Optional[str],
        typer.Option(help="JSON string of additional overrides (backend and agent)"),
    ] = None,
):
    """
    Evaluate an agentic retrieval pipeline (dense retrieval + LLM agent).

    Examples:

        retrieval-bench evaluate agentic-retrieval \\
            --dataset-name bright/biology \\
            --backend llama-nv-embed-reasoning-3b \\
            --llm-model your-llm-model \\
            --num-concurrent 10

        retrieval-bench evaluate agentic-retrieval \\
            --dataset-name vidore/vidore_v3_hr \\
            --backend llama-nemotron-embed-vl-1b-v2 \\
            --llm-model your-llm-model
    """
    if backend not in VALID_BACKENDS:
        print(f"\nUnknown backend: {backend!r}")
        print(f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
        raise typer.Exit(code=1)

    overrides: Dict[str, Any] = {}
    if pipeline_args:
        try:
            overrides = json.loads(pipeline_args)
        except json.JSONDecodeError as e:
            print(f"\nError parsing --pipeline-args: {e}\n")
            raise typer.Exit(code=1)

    from retrieval_bench.pipelines.agentic import AgenticRetrievalPipeline

    pipeline = AgenticRetrievalPipeline(
        backend=backend,
        retriever_top_k=retriever_top_k,
        num_concurrent=num_concurrent,
        llm_model=llm_model,
        reasoning_effort=reasoning_effort,
        target_top_k=target_top_k,
        **overrides,
    )

    _run_evaluation(
        pipeline=pipeline,
        dataset_name=dataset_name,
        split=split,
        language=language,
        query_ids_selector=query_ids_selector,
        trace_run_name=trace_run_name,
        traces_dir=traces_dir,
        output_file=output_file,
        show_dataset_info=show_dataset_info,
        pipeline_label=f"agentic-retrieval/{backend}",
        pipeline_args_for_output={
            "backend": backend,
            "llm_model": llm_model,
            "num_concurrent": num_concurrent,
            "reasoning_effort": reasoning_effort,
            "target_top_k": target_top_k,
            "retriever_top_k": retriever_top_k,
            **overrides,
        },
    )
