# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core evaluation orchestration using pytrec_eval.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytrec_eval
from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from retrieval_bench.pipeline_evaluation.tracing import (
    dataset_trace_dir,
    default_trace_run_name,
    extract_run_and_time_ms,
    load_trace_file,
    trace_path,
    write_trace_file,
)


def evaluate_retrieval(
    pipeline: BasePipeline,
    query_ids: List[str],
    queries: List[str],
    corpus_ids: List[str],
    corpus_images: List[Any],
    corpus_texts: List[str],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str] = None,
    track_time: bool = True,
    traces_dir: str = "traces",
    trace_run_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: str = "test",
    language: Optional[str] = None,
    query_ids_selector: Optional[str] = None,
    excluded_ids_by_query: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a pipeline using pytrec_eval.

    Args:
        pipeline: Instance of BasePipeline with user's pipeline logic
        query_ids: List of query identifiers
        queries: List of query texts
        corpus_ids: List of corpus item identifiers
        corpus_images: List of corpus images (PIL.Image objects)
        corpus_texts: List of corpus texts (markdown strings)
        qrels: Ground truth relevance judgments in pytrec_eval format
               {query_id: {doc_id: relevance_score}}
        metrics: List of metrics to calculate (default: ['ndcg_cut_10'])
        track_time: Whether to track retrieval time (default: True)
        traces_dir: Directory for per-query trace files (default: 'traces')
        trace_run_name: Name for this trace run. If None, auto-generated from pipeline class and model ID.
        dataset_name: Dataset identifier used for trace directory layout.
        split: Data split (default: 'test')
        language: Optional language filter passed through to traces.
        query_ids_selector: Optional selector string recorded in traces for provenance.
        excluded_ids_by_query: Per-query doc IDs to exclude from scored results (BRIGHT semantics).

    Returns:
        Dictionary of evaluation results per query:
        {
            'q1': {'ndcg_cut_10': 0.85, ...},
            'q2': {'ndcg_cut_10': 0.72, ...},
            ...
        }
        If track_time=True, also includes timing information in a special '_timing' key.
    """
    if metrics is None:
        metrics = ["ndcg_cut_10"]

    wall_start = time.perf_counter() if track_time else None

    def _filtered_run_for_query(qid: str, run_q: Any) -> Any:
        if not isinstance(run_q, dict):
            return run_q
        if not isinstance(excluded_ids_by_query, dict):
            return run_q
        excluded = excluded_ids_by_query.get(str(qid), None)
        if not isinstance(excluded, list):
            return run_q
        out = dict(run_q)
        for did in set(str(x) for x in excluded):
            if did != "N/A":
                out.pop(did, None)
        return out

    # Dataset context (for trace directory layout).
    dataset_name_eff = dataset_name or getattr(pipeline, "dataset_name", None) or "unknown_dataset"

    # Trace run name: always enabled; default is <PipelineClass>__<model_short>.
    trace_run_name_eff = trace_run_name or default_trace_run_name(pipeline)
    dataset_dir = dataset_trace_dir(dataset_name_eff, split=split, language=language)

    # Build a quick lookup for query text by id.
    query_by_id: Dict[str, str] = {qid: q for qid, q in zip(query_ids, queries)}

    # Load cached per-query runs/timing if present; otherwise schedule query for execution.
    cached_run: Dict[str, Dict[str, float]] = {}
    per_query_time_ms: Dict[str, float] = {}
    to_run_ids: List[str] = []
    to_run_queries: List[str] = []

    for qid in query_ids:
        p = trace_path(traces_dir, trace_run_name_eff, dataset_dir, qid)
        trace_obj = load_trace_file(p)
        extracted = extract_run_and_time_ms(trace_obj) if trace_obj is not None else None
        if extracted is None:
            to_run_ids.append(qid)
            to_run_queries.append(query_by_id.get(qid, ""))
        else:
            run_q, t_ms = extracted
            cached_run[qid] = run_q
            per_query_time_ms[qid] = t_ms

    # Indexing step: always call index() so the pipeline can set up embeddings/indices.
    start_time_indexing = time.time()
    pipeline.index(corpus_ids=corpus_ids, corpus_images=corpus_images, corpus_texts=corpus_texts)
    indexing_time = time.time() - start_time_indexing

    if indexing_time < 1e-5:
        indexing_time = 0.0

    # Execute only the missing queries (if any) and always write traces for executed queries.
    executed_run: Dict[str, Dict[str, float]] = {}
    pipeline_infos: Optional[Dict[str, Any]] = None
    pipeline_infos_public: Optional[Dict[str, Any]] = None
    executed_call_wall_ms: Optional[float] = None

    if to_run_ids:
        # Provide tracing context to pipelines that want to write per-query traces incrementally.
        setattr(
            pipeline,
            "tracing_context",
            {
                "traces_dir": traces_dir,
                "trace_run_name": trace_run_name_eff,
                "dataset": dataset_name_eff,
                "dataset_dir": dataset_dir,
                "split": split,
                "language": language,
                "query_ids_selector": query_ids_selector,
                "pipeline_class": pipeline.__class__.__name__,
                "model_id": getattr(pipeline, "model_id", None),
                "llm_model": getattr(pipeline, "llm_model", None),
            },
        )
        setattr(pipeline, "excluded_ids_by_query", excluded_ids_by_query)

        call_start = time.perf_counter() if track_time else None
        result = pipeline.search(query_ids=to_run_ids, queries=to_run_queries)
        if isinstance(result, tuple):
            executed_run, pipeline_infos = result
        else:
            executed_run, pipeline_infos = result, None
        if track_time and call_start is not None:
            executed_call_wall_ms = (time.perf_counter() - call_start) * 1000.0

        if not isinstance(executed_run, dict):
            raise ValueError(f"Pipeline must return a dict, got {type(executed_run)}")

        # Pull per-query timing from pipeline infos if provided, else fall back to equal-split wall time.
        provided_times: Dict[str, Any] = {}
        if isinstance(pipeline_infos, dict):
            provided_times = pipeline_infos.get("per_query_retrieval_time_milliseconds", {}) or {}

        fallback_ms = None
        if track_time and executed_call_wall_ms is not None and to_run_ids:
            fallback_ms = executed_call_wall_ms / len(to_run_ids)

        for qid in to_run_ids:
            t_ms = provided_times.get(qid, None) if isinstance(provided_times, dict) else None
            if isinstance(t_ms, (int, float)):
                per_query_time_ms[qid] = float(t_ms)
            elif fallback_ms is not None:
                per_query_time_ms[qid] = float(fallback_ms)

        per_query_trace: Dict[str, Any] = {}
        if isinstance(pipeline_infos, dict):
            pqt = pipeline_infos.get("per_query_trace", None)
            if isinstance(pqt, dict):
                per_query_trace = pqt

        if isinstance(pipeline_infos, dict):
            pipeline_infos_public = dict(pipeline_infos)
            pipeline_infos_public.pop("per_query_trace", None)

            llm_error_query_ids: List[str] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            any_usage = False
            _agg_trajectory_steps: List[int] = []
            _agg_llm_turns: List[int] = []
            _agg_retrieval_calls: List[int] = []
            for qid in to_run_ids:
                extra = per_query_trace.get(qid, None)
                if isinstance(extra, dict) and isinstance(extra.get("llm_error", None), str):
                    llm_error_query_ids.append(qid)
                usage = extra.get("llm_usage", None) if isinstance(extra, dict) else None
                if isinstance(usage, dict):
                    pt = usage.get("prompt_tokens", None)
                    ct = usage.get("completion_tokens", None)
                    tt = usage.get("total_tokens", None)
                    if isinstance(pt, int):
                        total_prompt_tokens += pt
                        any_usage = True
                    if isinstance(ct, int):
                        total_completion_tokens += ct
                        any_usage = True
                    if isinstance(tt, int):
                        total_tokens += tt
                        any_usage = True
                if isinstance(extra, dict):
                    for _key, _lst in [
                        ("trajectory_steps", _agg_trajectory_steps),
                        ("llm_turns", _agg_llm_turns),
                        ("retrieval_calls", _agg_retrieval_calls),
                    ]:
                        v = extra.get(_key, None)
                        if isinstance(v, int):
                            _lst.append(v)
            pipeline_infos_public["llm_error_query_ids"] = llm_error_query_ids
            if any_usage:
                pipeline_infos_public["llm_total_prompt_tokens"] = total_prompt_tokens
                pipeline_infos_public["llm_total_completion_tokens"] = total_completion_tokens
                pipeline_infos_public["llm_total_tokens"] = total_tokens
            if _agg_trajectory_steps:
                pipeline_infos_public["avg_trajectory_steps"] = sum(_agg_trajectory_steps) / len(_agg_trajectory_steps)
            if _agg_llm_turns:
                pipeline_infos_public["avg_llm_turns"] = sum(_agg_llm_turns) / len(_agg_llm_turns)
            if _agg_retrieval_calls:
                pipeline_infos_public["avg_retrieval_calls"] = sum(_agg_retrieval_calls) / len(_agg_retrieval_calls)

        for qid in to_run_ids:
            p = trace_path(traces_dir, trace_run_name_eff, dataset_dir, qid)
            existing = load_trace_file(p)
            if isinstance(existing, dict) and extract_run_and_time_ms(existing) is not None:
                continue

            run_q = executed_run.get(qid, None)
            run_q = _filtered_run_for_query(qid, run_q)
            t_ms = per_query_time_ms.get(qid, None)
            payload: Dict[str, Any] = {
                "query_id": qid,
                "dataset": dataset_name_eff,
                "dataset_dir": dataset_dir,
                "split": split,
                "language": language,
                "query_ids_selector": query_ids_selector,
                "trace_run_name": trace_run_name_eff,
                "pipeline_class": pipeline.__class__.__name__,
                "model_id": getattr(pipeline, "model_id", None),
                "retrieval_time_milliseconds": t_ms,
                "run": run_q,
            }
            extra = per_query_trace.get(qid, None)
            if isinstance(extra, dict):
                payload["pipeline_trace"] = extra
            write_trace_file(p, payload)

    # Combined run (cached + executed). Executed wins on collisions.
    run: Dict[str, Dict[str, float]] = {}
    run.update(cached_run)
    run.update(executed_run)

    expected_query_ids = list(query_ids)
    returned_set = set(run.keys())

    evaluated_query_ids = [qid for qid in expected_query_ids if qid in returned_set]
    missing_query_ids = [qid for qid in expected_query_ids if qid not in returned_set]

    if not evaluated_query_ids:
        raise ValueError(
            "Pipeline returned no results for any expected query_ids. " "Refusing to compute metrics on an empty set."
        )

    # Restrict evaluation to only queries we have both run + qrels for.
    run_eval = {qid: _filtered_run_for_query(qid, run[qid]) for qid in evaluated_query_ids}
    qrels_eval = {qid: qrels[qid] for qid in evaluated_query_ids if qid in qrels}

    # Create pytrec_eval evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_eval, set(metrics))

    # Evaluate
    results = evaluator.evaluate(run_eval)

    # Persist per-query evaluation metrics into trace files and aggregate pipeline-level
    # summaries from traces (e.g. LLM token usage), so fully-cached runs still get totals.
    llm_error_query_ids_from_traces: List[str] = []
    llm_total_prompt_tokens = 0
    llm_total_completion_tokens = 0
    llm_total_tokens = 0
    llm_any_usage = False
    _trace_trajectory_steps: List[int] = []
    _trace_llm_turns: List[int] = []
    _trace_retrieval_calls: List[int] = []
    for qid in evaluated_query_ids:
        p = trace_path(traces_dir, trace_run_name_eff, dataset_dir, qid)
        trace_obj = load_trace_file(p) or {}
        if not isinstance(trace_obj, dict):
            trace_obj = {}

        pipeline_trace = trace_obj.get("pipeline_trace", None)
        if isinstance(pipeline_trace, dict):
            if isinstance(pipeline_trace.get("llm_error", None), str):
                llm_error_query_ids_from_traces.append(qid)
            usage = pipeline_trace.get("llm_usage", None)
            if isinstance(usage, dict):
                pt = usage.get("prompt_tokens", None)
                ct = usage.get("completion_tokens", None)
                tt = usage.get("total_tokens", None)
                if isinstance(pt, int):
                    llm_total_prompt_tokens += pt
                    llm_any_usage = True
                if isinstance(ct, int):
                    llm_total_completion_tokens += ct
                    llm_any_usage = True
                if isinstance(tt, int):
                    llm_total_tokens += tt
                    llm_any_usage = True
            for _keys, _lst in [
                (("trajectory_steps", "agent_steps"), _trace_trajectory_steps),
                (("llm_turns", "num_agent_steps"), _trace_llm_turns),
                (("retrieval_calls", "num_retrieval_steps"), _trace_retrieval_calls),
            ]:
                for _key in _keys:
                    v = pipeline_trace.get(_key, None)
                    if isinstance(v, int):
                        _lst.append(v)
                        break

        trace_obj.update(
            {
                "query_id": qid,
                "dataset": dataset_name_eff,
                "dataset_dir": dataset_dir,
                "split": split,
                "language": language,
                "query_ids_selector": query_ids_selector,
                "trace_run_name": trace_run_name_eff,
                "pipeline_class": pipeline.__class__.__name__,
                "model_id": getattr(pipeline, "model_id", None),
                "per_query_metrics": results.get(qid, None),
            }
        )
        write_trace_file(p, trace_obj)

    # Add timing information if tracking
    if track_time:
        num_queries = len(query_ids)
        num_corpus = len(corpus_ids)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0 if wall_start is not None else None

        total_retrieval_ms = 0.0
        for qid in evaluated_query_ids:
            t_ms = per_query_time_ms.get(qid, None)
            if isinstance(t_ms, (int, float)):
                total_retrieval_ms += float(t_ms)

        avg_ms = (total_retrieval_ms / len(evaluated_query_ids)) if evaluated_query_ids else 0.0
        qps = (len(evaluated_query_ids) / (total_retrieval_ms / 1000.0)) if total_retrieval_ms > 0 else 0.0

        num_loaded = len([qid for qid in evaluated_query_ids if qid in cached_run])
        num_executed = len([qid for qid in evaluated_query_ids if qid in executed_run])

        executed_retrieval_ms = sum(
            float(per_query_time_ms[qid])
            for qid in evaluated_query_ids
            if qid in executed_run and isinstance(per_query_time_ms.get(qid, None), (int, float))
        )

        results["_timing"] = {
            "total_retrieval_time_milliseconds": total_retrieval_ms,
            "total_retrieval_time_milliseconds_executed": executed_retrieval_ms,
            "indexing_time_milliseconds": indexing_time * 1000,
            "average_time_per_query_milliseconds": avg_ms,
            "total_wall_time_milliseconds": wall_ms,
            "expected_num_queries": num_queries,
            "num_queries": len(evaluated_query_ids),
            "num_corpus": num_corpus,
            "missing_num_queries": len(missing_query_ids),
            "queries_per_second": qps,
            "num_queries_loaded_from_trace": num_loaded,
            "num_queries_executed": num_executed,
        }

    # Attach evaluation infos (and keep any pipeline-provided infos nested).
    eval_infos: Dict[str, Any] = {
        "tracing": {
            "traces_dir": str(Path(traces_dir)),
            "trace_run_name": trace_run_name_eff,
            "dataset_dir": dataset_dir,
        },
    }

    pipeline_infos_summary: Dict[str, Any] = {}
    if pipeline_infos_public is not None:
        pipeline_infos_summary.update(pipeline_infos_public)
    elif isinstance(pipeline_infos, dict):
        pipeline_infos_summary.update(pipeline_infos)
        pipeline_infos_summary.pop("per_query_trace", None)

    pipeline_infos_summary["llm_error_query_ids"] = sorted(
        llm_error_query_ids_from_traces, key=lambda x: int(x) if x.isdigit() else x
    )
    if llm_any_usage:
        pipeline_infos_summary["llm_total_prompt_tokens"] = llm_total_prompt_tokens
        pipeline_infos_summary["llm_total_completion_tokens"] = llm_total_completion_tokens
        pipeline_infos_summary["llm_total_tokens"] = llm_total_tokens
    if _trace_trajectory_steps:
        pipeline_infos_summary["avg_trajectory_steps"] = sum(_trace_trajectory_steps) / len(_trace_trajectory_steps)
    if _trace_llm_turns:
        pipeline_infos_summary["avg_llm_turns"] = sum(_trace_llm_turns) / len(_trace_llm_turns)
    if _trace_retrieval_calls:
        pipeline_infos_summary["avg_retrieval_calls"] = sum(_trace_retrieval_calls) / len(_trace_retrieval_calls)

    if pipeline_infos_summary:
        eval_infos["pipeline_infos"] = pipeline_infos_summary
    results["_infos"] = eval_infos

    return results


def aggregate_results(
    results: Dict[str, Dict[str, float]], query_languages: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Calculate aggregate statistics across all queries.

    If query_languages is provided, also computes per-language aggregates.

    Args:
        results: Per-query evaluation results from evaluate_retrieval()
        query_languages: Optional mapping of query_id to language

    Returns:
        Dictionary of aggregated metrics. If query_languages is provided:
        {
            'overall': {'ndcg_cut_10': 0.85, ...},
            'by_language': {
                'english': {'ndcg_cut_10': 0.87, ...},
                'french': {'ndcg_cut_10': 0.82, ...},
            },
            'timing': {...}  # if timing info present
            'infos': {...}  # if pipeline infos present
        }
        Otherwise, just returns flat aggregated metrics.
    """
    if not results:
        return {}

    # Extract meta information if present (without mutating input)
    timing_info = results.get("_timing", None)
    infos = results.get("_infos", None)

    # Filter to actual per-query metrics (ignore meta keys like _timing/_infos)
    query_results = {qid: qres for qid, qres in results.items() if not str(qid).startswith("_")}
    if not query_results:
        final = {"timing": timing_info} if timing_info else {}
        if infos is not None:
            final["infos"] = infos
        return final

    # Get all metric names from first query
    metric_names = list(next(iter(query_results.values())).keys())

    # If no language splitting requested, return simple aggregation
    if query_languages is None:
        aggregated = {}
        for metric in metric_names:
            scores = [query_results[qid][metric] for qid in query_results]
            aggregated[metric] = sum(scores) / len(scores)

        if timing_info:
            aggregated.update(timing_info)
        if infos is not None:
            aggregated["infos"] = infos

        return aggregated

    # Split results by language
    results_by_language = defaultdict(dict)
    for query_id, per_query_results in query_results.items():
        lang = query_languages.get(query_id, "unknown")
        results_by_language[lang][query_id] = per_query_results

    # Compute overall aggregates
    overall_aggregated = {}
    for metric in metric_names:
        scores = [query_results[qid][metric] for qid in query_results]
        overall_aggregated[metric] = sum(scores) / len(scores)

    # Compute per-language aggregates
    by_language_aggregated = {}
    for lang, lang_results in results_by_language.items():
        lang_aggregated = {}
        for metric in metric_names:
            scores = [lang_results[qid][metric] for qid in lang_results]
            lang_aggregated[metric] = sum(scores) / len(scores)
        lang_aggregated["num_queries"] = len(lang_results)
        by_language_aggregated[lang] = lang_aggregated

    # Build final result structure
    final_result = {
        "overall": overall_aggregated,
        "by_language": by_language_aggregated,
    }

    if timing_info:
        final_result["timing"] = timing_info
    if infos is not None:
        final_result["infos"] = infos

    return final_result
