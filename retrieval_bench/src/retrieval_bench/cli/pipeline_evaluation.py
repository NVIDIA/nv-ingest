# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline utility CLI commands (report-results, compare-results, etc.).

The evaluation commands (dense-retrieval, agentic-retrieval) live in
``retrieval_bench.cli.evaluate``.  This module retains the utility /
reporting commands that operate on saved results and traces.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Optional, Set, Tuple

import typer

from retrieval_bench.pipeline_evaluation import (
    BasePipeline,
    get_available_datasets,
)
from retrieval_bench.pipeline_evaluation.tracing import dataset_trace_dir
from vidore_benchmark.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="""
    CLI for evaluating abstract pipelines on ViDoRe v3 datasets.

    Evaluate custom retrieval pipelines that inherit from BasePipeline.
    Supports built-in pipelines (random, file-based) and custom Python implementations.
    """,
    no_args_is_help=True,
)


@app.callback()
def main(log_level: Annotated[str, typer.Option("--log", help="Logging level")] = "warning"):
    """Initialize logging configuration."""
    setup_logging(log_level)
    logger.info("Logging level set to `%s`", log_level)


def _parse_query_ids_selector(selector: str) -> Set[str]:
    """
    Parse a comma-separated list of integer IDs and inclusive ranges.

    Examples:
      - "0-99,120,200-210" -> {"0","1",...,"99","120","200",...,"210"}
      - "5" -> {"5"}
    """
    selector = (selector or "").strip()
    if not selector:
        raise ValueError("Empty --query-ids selector.")

    ids: Set[str] = set()
    tokens = [t.strip() for t in selector.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Empty --query-ids selector.")

    for tok in tokens:
        if "-" in tok:
            parts = [p.strip() for p in tok.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range token '{tok}'. Expected format like '0-10'.")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid range token '{tok}'. Range bounds must be integers.")
            if start < 0 or end < 0:
                raise ValueError(f"Invalid range token '{tok}'. Query IDs must be >= 0.")
            if start > end:
                raise ValueError(f"Invalid range token '{tok}'. Range start must be <= end.")
            for i in range(start, end + 1):
                ids.add(str(i))
        else:
            try:
                val = int(tok)
            except ValueError:
                raise ValueError(f"Invalid id token '{tok}'. Expected an integer or a range like '0-10'.")
            if val < 0:
                raise ValueError(f"Invalid id token '{tok}'. Query IDs must be >= 0.")
            ids.add(str(val))

    return ids


def _parse_index_selector(selector: str) -> Set[int]:
    """
    Parse a comma-separated list of integer indices and inclusive ranges.

    Examples:
      - "0-99,120,200-210" -> {0,1,...,99,120,200,...,210}
      - "5" -> {5}
    """
    selector = (selector or "").strip()
    if not selector:
        raise ValueError("Empty --query-ids selector.")

    ids: Set[int] = set()
    tokens = [t.strip() for t in selector.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Empty --query-ids selector.")

    for tok in tokens:
        if "-" in tok:
            parts = [p.strip() for p in tok.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range token '{tok}'. Expected format like '0-10'.")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid range token '{tok}'. Range bounds must be integers.")
            if start < 0 or end < 0:
                raise ValueError(f"Invalid range token '{tok}'. Indices must be >= 0.")
            if start > end:
                raise ValueError(f"Invalid range token '{tok}'. Range start must be <= end.")
            for i in range(start, end + 1):
                ids.add(int(i))
        else:
            try:
                val = int(tok)
            except ValueError:
                raise ValueError(f"Invalid id token '{tok}'. Expected an integer or a range like '0-10'.")
            if val < 0:
                raise ValueError(f"Invalid id token '{tok}'. Indices must be >= 0.")
            ids.add(int(val))

    return ids


def _query_ids_are_numeric(query_ids: list[str]) -> bool:
    return all(str(qid).isdigit() for qid in query_ids)


def _filter_queries_by_ids(
    query_ids: list[str],
    queries: list[str],
    qrels: dict,
    query_languages: dict,
    requested_ids: Set[str],
):
    """Filter query_ids/queries/qrels/query_languages in sync, preserving original order."""
    available = set(query_ids)
    missing = sorted(requested_ids - available, key=lambda x: int(x) if x.isdigit() else x)
    if missing:
        raise ValueError(
            "Some requested query IDs are not available after filtering (e.g. language).\n"
            f"Missing IDs: {', '.join(missing[:50])}" + (" ..." if len(missing) > 50 else "")
        )

    filtered_query_ids: list[str] = []
    filtered_queries: list[str] = []
    for qid, q in zip(query_ids, queries):
        if qid in requested_ids:
            filtered_query_ids.append(qid)
            filtered_queries.append(q)

    if not filtered_query_ids:
        raise ValueError("After applying --query-ids, zero queries remain to evaluate.")

    filtered_qrels = {qid: qrels[qid] for qid in filtered_query_ids if qid in qrels}
    filtered_query_languages = {qid: query_languages.get(qid, "unknown") for qid in filtered_query_ids}

    return filtered_query_ids, filtered_queries, filtered_qrels, filtered_query_languages


def _filter_queries_by_positions(
    query_ids: list[str],
    queries: list[str],
    qrels: dict,
    query_languages: dict,
    requested_positions: Set[int],
):
    """
    Filter query_ids/queries/qrels/query_languages in sync by *position* (0-based index).

    This is used for datasets whose query IDs are non-numeric (e.g. BRIGHT aops),
    so we can still use compact numeric selectors like '0-99,120,200-210'.
    """
    if not requested_positions:
        raise ValueError("Empty --query-ids selector.")

    n = len(query_ids)
    bad = sorted([i for i in requested_positions if i < 0 or i >= n])
    if bad:
        raise ValueError(
            "Some requested query indices are out of range.\n"
            f"Valid range: 0..{max(0, n-1)}\n"
            f"Bad indices (first 50): {bad[:50]}" + (" ..." if len(bad) > 50 else "")
        )

    filtered_query_ids: list[str] = []
    filtered_queries: list[str] = []
    for idx, (qid, q) in enumerate(zip(query_ids, queries)):
        if idx in requested_positions:
            filtered_query_ids.append(qid)
            filtered_queries.append(q)

    if not filtered_query_ids:
        raise ValueError("After applying --query-ids, zero queries remain to evaluate.")

    filtered_qrels = {qid: qrels[qid] for qid in filtered_query_ids if qid in qrels}
    filtered_query_languages = {qid: query_languages.get(qid, "unknown") for qid in filtered_query_ids}
    return filtered_query_ids, filtered_queries, filtered_qrels, filtered_query_languages


def _extract_aggregated_metric(obj: Any, metric: str) -> Optional[float]:
    """
    Extract a metric from a pipeline results JSON object.

    Supports both formats:
    - language-split: aggregated_metrics.overall.<metric>
    - flat: aggregated_metrics.<metric>
    """
    if not isinstance(obj, dict):
        return None
    aggregated = obj.get("aggregated_metrics", None)
    if not isinstance(aggregated, dict):
        return None

    if "overall" in aggregated and isinstance(aggregated.get("overall"), dict):
        val = aggregated["overall"].get(metric, None)
    else:
        val = aggregated.get(metric, None)

    if isinstance(val, (int, float)):
        return float(val)
    return None


def _extract_pipeline_infos(obj: Any) -> Optional[dict]:
    """
    Extract the pipeline_infos object from a results JSON payload (if present).

    Expected location:
      aggregated_metrics._infos.pipeline_infos
    """
    if not isinstance(obj, dict):
        return None
    aggregated = obj.get("aggregated_metrics", None)
    if not isinstance(aggregated, dict):
        return None
    infos = aggregated.get("infos", None) or aggregated.get("_infos", None)
    if not isinstance(infos, dict):
        return None
    pipeline_infos = infos.get("pipeline_infos", None)
    if not isinstance(pipeline_infos, dict):
        return None
    return pipeline_infos


def _try_extract_llm_summary_from_results(obj: Any) -> Optional[Tuple[int, int, int, int, Optional[float]]]:
    """
    Return (llm_error_count, prompt_tokens, completion_tokens, total_tokens, avg_trajectory_steps)
    if present and valid. Otherwise return None.
    """
    pi = _extract_pipeline_infos(obj)
    if not isinstance(pi, dict):
        return None

    err_ids = pi.get("llm_error_query_ids", None)
    pt = pi.get("llm_total_prompt_tokens", None)
    ct = pi.get("llm_total_completion_tokens", None)
    tt = pi.get("llm_total_tokens", None)

    if not isinstance(err_ids, list) or not all(isinstance(x, str) for x in err_ids):
        return None
    if not isinstance(pt, int) or not isinstance(ct, int) or not isinstance(tt, int):
        return None

    avg_traj = pi.get("avg_trajectory_steps", None)
    avg_traj_f = float(avg_traj) if isinstance(avg_traj, (int, float)) else None

    return (len(err_ids), int(pt), int(ct), int(tt), avg_traj_f)


def _extract_wall_time_and_nq(obj: Any) -> Tuple[Optional[float], Optional[int]]:
    """
    Return (total_wall_time_milliseconds, num_queries) from a results JSON object.

    Handles both flat (BRIGHT) and language-split (vidore v3) aggregated_metrics layouts.
    Returns (None, None) when the fields are unavailable.
    """
    if not isinstance(obj, dict):
        return None, None
    aggregated = obj.get("aggregated_metrics", None)
    if not isinstance(aggregated, dict):
        return None, None

    if "overall" in aggregated and isinstance(aggregated.get("overall"), dict):
        timing = aggregated.get("timing", {})
    else:
        timing = aggregated

    if not isinstance(timing, dict):
        return None, None

    wall_ms = timing.get("total_wall_time_milliseconds", None)
    nq = timing.get("num_queries", None)
    if not isinstance(wall_ms, (int, float)) or not isinstance(nq, (int, float)):
        return None, None
    return float(wall_ms), int(nq)


def _compute_llm_summary_from_traces(
    traces_dir: str, trace_run_name: str, dataset_short: str
) -> Optional[Tuple[int, int, int, int, Optional[float], int]]:
    """
    Return (llm_error_count, prompt_tokens, completion_tokens, total_tokens, avg_trajectory_steps, num_traces)
    by scanning per-query trace files.
    Returns None if the trace directory does not exist.
    """
    trace_root = Path(traces_dir) / trace_run_name / dataset_short
    if not trace_root.exists() or not trace_root.is_dir():
        return None

    llm_error_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    trajectory_steps_vals: list[int] = []
    num_traces = 0

    for p in sorted(trace_root.glob("*.json")):
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        num_traces += 1

        pipeline_trace = obj.get("pipeline_trace", None)
        if not isinstance(pipeline_trace, dict):
            continue

        if isinstance(pipeline_trace.get("llm_error", None), str):
            llm_error_count += 1

        usage = pipeline_trace.get("llm_usage", None)
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens", None)
            ct = usage.get("completion_tokens", None)
            tt = usage.get("total_tokens", None)
            if isinstance(pt, int):
                prompt_tokens += pt
            if isinstance(ct, int):
                completion_tokens += ct
            if isinstance(tt, int):
                total_tokens += tt

        ts = pipeline_trace.get("trajectory_steps", None)
        if isinstance(ts, int):
            trajectory_steps_vals.append(ts)

    avg_traj = (sum(trajectory_steps_vals) / len(trajectory_steps_vals)) if trajectory_steps_vals else None
    return (llm_error_count, prompt_tokens, completion_tokens, total_tokens, avg_traj, num_traces)


def _compute_llm_errors_from_traces(
    traces_dir: str, trace_run_name: str, dataset_short: str, *, max_error_len: int = 500
) -> Optional[list[tuple[str, str]]]:
    """
    Return a list of (query_id, llm_error) by scanning per-query trace files.
    Returns None if the trace directory does not exist.
    """
    trace_root = Path(traces_dir) / trace_run_name / dataset_short
    if not trace_root.exists() or not trace_root.is_dir():
        return None

    out: list[tuple[str, str]] = []
    for p in sorted(trace_root.glob("*.json")):
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        pipeline_trace = obj.get("pipeline_trace", None)
        if not isinstance(pipeline_trace, dict):
            continue

        err = pipeline_trace.get("llm_error", None)
        if not isinstance(err, str):
            continue

        qid = obj.get("query_id", None)
        qid_s = str(qid) if qid is not None else p.stem
        if max_error_len <= 0:
            err_s = err
        else:
            err_s = (
                err
                if len(err) <= max_error_len
                else (err[:max_error_len] + f"... [truncated, original_len={len(err)}]")
            )
        out.append((qid_s, err_s))

    # Sort numerically when possible (query ids are usually numeric strings).
    out.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return out


def _load_pipeline_from_module(module_path: str, class_name: str, **kwargs) -> BasePipeline:
    """
    Dynamically load a pipeline class from a Python file.

    Args:
        module_path: Path to the Python file containing the pipeline class
        class_name: Name of the pipeline class to instantiate
        **kwargs: Arguments to pass to the pipeline constructor

    Returns:
        Instantiated pipeline object
    """
    module_path = Path(module_path).resolve()

    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("custom_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_pipeline"] = module
    spec.loader.exec_module(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_path}")

    pipeline_class = getattr(module, class_name)

    # Verify it's a BasePipeline subclass
    if not issubclass(pipeline_class, BasePipeline):
        raise TypeError(f"Class '{class_name}' must inherit from BasePipeline")

    # Instantiate the pipeline
    return pipeline_class(**kwargs)


@app.command()
def list_datasets():
    """
    List all available ViDoRe v3 datasets.

    Example:
        retrieval-bench pipeline list-datasets
    """
    datasets = get_available_datasets()

    print("\n" + "=" * 70)
    print("Available ViDoRe v3 Datasets")
    print("=" * 70)
    for i, dataset_name in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset_name}")
    print("=" * 70)
    print(f"\nTotal: {len(datasets)} datasets\n")


@app.command()
def list_backends():
    """
    List available dense retrieval backends.

    Example:
        retrieval-bench evaluate utils list-backends
    """
    from retrieval_bench.pipelines.backends import VALID_BACKENDS as _backends

    print("\n" + "=" * 70)
    print("Available Dense Retrieval Backends")
    print("=" * 70)
    for i, b in enumerate(sorted(_backends), 1):
        print(f"  {i}. {b}")
    print("=" * 70)
    print("\nUsage:")
    print("  retrieval-bench evaluate dense-retrieval --backend <backend> --dataset-name <dataset>")
    print("  retrieval-bench evaluate agentic-retrieval --backend <backend> --dataset-name <dataset>\n")


@app.command()
def report_results(
    results_dir: Annotated[
        str,
        typer.Option(
            "--results-dir",
            help="Directory containing per-dataset result JSON files (e.g. results/ColEmbedPipeline).",
        ),
    ],
):
    """
    Report NDCG@10 and query-coverage per dataset for a results directory.

    This command expects the JSON structure produced by `retrieval-bench pipeline evaluate`
    and `retrieval-bench pipeline evaluate-all`, i.e. each file contains:
      - aggregated_metrics (including ndcg_cut_10)
      - timing info (including expected_num_queries / num_queries / missing_num_queries) when available

    We intentionally do not require per-id lists (evaluated/missing ids); coverage is computed
    from counts stored in the result file.
    """
    root = Path(results_dir)
    if not root.exists() or not root.is_dir():
        raise typer.BadParameter(f"--results-dir must be an existing directory: {root}")

    datasets = get_available_datasets()

    print("\n" + "=" * 70)
    print(f"Results report: {root}")
    print("=" * 70)
    print(f"{'dataset':28s}  {'ndcg@10':>8s}  {'coverage':>12s}  {'status':>8s}")
    print("-" * 70)

    for dataset_name in datasets:
        ds_short = dataset_trace_dir(dataset_name)
        path = root / f"{ds_short}.json"
        if not path.exists():
            print(f"{ds_short:28s}  {'-':>8s}  {'-':>12s}  {'MISSING':>8s}")
            continue

        try:
            with open(path, "r") as f:
                obj = json.load(f)
        except Exception:
            print(f"{ds_short:28s}  {'-':>8s}  {'-':>12s}  {'ERROR':>8s}")
            continue

        aggregated: Any = obj.get("aggregated_metrics", {}) if isinstance(obj, dict) else {}
        if not isinstance(aggregated, dict):
            aggregated = {}

        # Extract NDCG@10 (handle both nested language-split format and flat format).
        ndcg = None
        timing: Any = {}
        if "overall" in aggregated and isinstance(aggregated.get("overall"), dict):
            ndcg = aggregated["overall"].get("ndcg_cut_10", None)
            timing = aggregated.get("timing", {}) if isinstance(aggregated.get("timing", {}), dict) else {}
        else:
            ndcg = aggregated.get("ndcg_cut_10", None)
            # In flat mode, timing fields are merged into aggregated_metrics.
            timing = aggregated

        ndcg_str = f"{float(ndcg):.4f}" if isinstance(ndcg, (int, float)) else "-"

        expected = timing.get("expected_num_queries", None) if isinstance(timing, dict) else None
        num = timing.get("num_queries", None) if isinstance(timing, dict) else None
        missing = timing.get("missing_num_queries", None) if isinstance(timing, dict) else None

        coverage = "-"
        status = "UNKNOWN"
        if isinstance(expected, (int, float)) and isinstance(num, (int, float)):
            expected_i = int(expected)
            num_i = int(num)
            coverage = f"{num_i}/{expected_i}"
            status = "FULL" if expected_i == num_i else "PARTIAL"
        elif isinstance(expected, (int, float)) and isinstance(missing, (int, float)):
            expected_i = int(expected)
            missing_i = int(missing)
            num_i = max(0, expected_i - missing_i)
            coverage = f"{num_i}/{expected_i}"
            status = "FULL" if missing_i == 0 else "PARTIAL"

        print(f"{ds_short:28s}  {ndcg_str:>8s}  {coverage:>12s}  {status:>8s}")

    print("-" * 70 + "\n")


@app.command()
def compare_results(
    results_dirs: Annotated[
        list[str],
        typer.Option(
            "--results-dirs",
            help="One or more pipeline results directories (e.g. results/ColEmbedPipeline__...).",
        ),
    ],
    all_datasets: Annotated[
        bool,
        typer.Option(
            "--all-datasets",
            "--all",
            help="Report across all available ViDoRe v3 datasets (overrides --datasets).",
        ),
    ] = False,
    datasets: Annotated[
        Optional[str],
        typer.Option(
            "--datasets",
            help="Optional comma-separated dataset filter (accepts either full names like 'vidore/vidore_v3_hr' "
            "or shorts like 'vidore_v3_hr'). Defaults to all datasets.",
        ),
    ] = None,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Aggregated metric key to report (default: ndcg_cut_10)."),
    ] = "ndcg_cut_10",
    csv: Annotated[
        bool,
        typer.Option(
            "--csv",
            help="Print only a comma-separated header + one row per dataset (easy to paste into Google Docs/Sheets).",
        ),
    ] = False,
):
    """
    Compare aggregated metrics (default: NDCG@10) across multiple pipeline results directories.
    """
    if not results_dirs:
        raise typer.BadParameter("--results-dirs must include at least one directory.")

    # Canonical dataset list.
    all_ds = get_available_datasets()
    short_by_full = {ds: dataset_trace_dir(ds) for ds in all_ds}
    full_by_short = {v: k for k, v in short_by_full.items()}

    # Optional dataset filter.
    selected = list(all_ds)
    if (not all_datasets) and datasets:
        toks = [t.strip() for t in datasets.split(",") if t.strip()]
        if not toks:
            raise typer.BadParameter("--datasets was provided but empty.")
        selected = []
        for tok in toks:
            # Allow specifying either the full dataset name (e.g. 'bright/biology')
            # or the dataset file key used on disk (e.g. 'bright__biology').
            if tok in short_by_full:
                selected.append(tok)
            elif tok in full_by_short:
                selected.append(full_by_short[tok])
            else:
                raise typer.BadParameter(
                    f"Unknown dataset '{tok}'. Expected one of: {', '.join(short_by_full.values())}"
                )

    # Pipeline labels for columns.
    dirs: list[Path] = [Path(d) for d in results_dirs]
    labels: list[str] = [p.name or str(p) for p in dirs]

    # Build matrix: dataset_short -> label -> (status, value)
    rows: list[tuple[str, dict[str, str]]] = []
    coverage_by_label: dict[str, int] = {lab: 0 for lab in labels}

    for ds in selected:
        ds_short = short_by_full[ds]
        per_label: dict[str, str] = {}
        for lab, root in zip(labels, dirs):
            path = root / f"{ds_short}.json"
            if not path.exists():
                per_label[lab] = "MISSING"
                continue
            try:
                with open(path, "r") as f:
                    obj = json.load(f)
            except Exception:
                per_label[lab] = "ERROR"
                continue

            val = _extract_aggregated_metric(obj, metric)
            if val is None:
                per_label[lab] = "-"
            else:
                per_label[lab] = f"{val:.4f}"
                coverage_by_label[lab] += 1

        rows.append((ds_short, per_label))

    def _csv_escape(s: str) -> str:
        s = "" if s is None else str(s)
        if any(ch in s for ch in [",", '"', "\n", "\r"]):
            s = s.replace('"', '""')
            return f'"{s}"'
        return s

    if csv:
        header_cells = ["dataset"] + labels
        print(",".join(_csv_escape(x) for x in header_cells))
        for ds_short, per_label in rows:
            row_cells = [ds_short] + [per_label.get(lab, "") for lab in labels]
            print(",".join(_csv_escape(x) for x in row_cells))
        return

    # Pretty print aligned table.
    dataset_col_w = max(len("dataset"), max((len(ds_short) for ds_short, _ in rows), default=0))
    col_widths: dict[str, int] = {}
    for lab in labels:
        max_cell = max((len(r[lab]) for _, r in rows), default=0)
        col_widths[lab] = max(len(lab), max_cell, 8)

    print("\n" + "=" * 70)
    print(f"Compare results ({metric})")
    print("=" * 70)
    header = ["dataset".ljust(dataset_col_w)]
    header += [lab.rjust(col_widths[lab]) for lab in labels]
    print("  ".join(header))
    print("-" * (dataset_col_w + sum(col_widths[lab] for lab in labels) + 2 * len(labels)))

    for ds_short, per_label in rows:
        line = [ds_short.ljust(dataset_col_w)]
        line += [per_label[lab].rjust(col_widths[lab]) for lab in labels]
        print("  ".join(line))

    # Optional coverage summary.
    print("-" * (dataset_col_w + sum(col_widths[lab] for lab in labels) + 2 * len(labels)))
    cov_line = ["coverage".ljust(dataset_col_w)]
    for lab in labels:
        cov_line.append(f"{coverage_by_label[lab]}/{len(rows)}".rjust(col_widths[lab]))
    print("  ".join(cov_line))
    print("=" * 70 + "\n")


@app.command()
def report_llm_usage(
    results_dir: Annotated[
        str,
        typer.Option(
            "--results-dir",
            help="Pipeline results directory (e.g. results/AgenticPipelineV1__...).",
        ),
    ],
    all_datasets: Annotated[
        bool,
        typer.Option(
            "--all-datasets",
            "--all",
            help="Report across all available ViDoRe v3 datasets (overrides --datasets).",
        ),
    ] = False,
    datasets: Annotated[
        Optional[str],
        typer.Option(
            "--datasets",
            help="Optional comma-separated dataset filter (accepts either full names like 'vidore/vidore_v3_hr' "
            "or shorts like 'vidore_v3_hr'). Defaults to all datasets.",
        ),
    ] = None,
    traces_dir: Annotated[
        str,
        typer.Option(
            "--traces-dir",
            help="Default traces root directory to use when results JSON does not specify traces_dir.",
        ),
    ] = "traces",
    list_errors: Annotated[
        bool,
        typer.Option(
            "--list-errors",
            help="After the summary table, list all encountered LLM errors per dataset (query id + error string).",
        ),
    ] = False,
    max_error_len: Annotated[
        int,
        typer.Option(
            "--max-error-len",
            help=(
                "Maximum number of characters to print per error string"
                " when using --list-errors. Use 0 for no truncation."
            ),
        ),
    ] = 0,
):
    """
    Report LLM error counts and token usage per dataset for a pipeline results directory.

    Uses results JSON summary fields when present; otherwise falls back to scanning per-query traces.
    """
    results_root = Path(results_dir)
    if not results_root.exists() or not results_root.is_dir():
        raise typer.BadParameter(f"--results-dir must be an existing directory: {results_root}")

    # Canonical dataset list.
    ds_full = get_available_datasets()
    short_by_full = {ds: dataset_trace_dir(ds) for ds in ds_full}
    full_by_short = {v: k for k, v in short_by_full.items()}

    selected = list(ds_full)
    if not all_datasets and datasets:
        toks = [t.strip() for t in datasets.split(",") if t.strip()]
        if not toks:
            raise typer.BadParameter("--datasets was provided but empty.")
        selected = []
        for tok in toks:
            if tok in short_by_full:
                selected.append(tok)
            elif tok in full_by_short:
                selected.append(full_by_short[tok])
            else:
                raise typer.BadParameter(
                    f"Unknown dataset '{tok}'. Expected one of: {', '.join(short_by_full.values())}"
                )

    # Table rows: (dataset_short, nq, err_count, pt, ct, tt, wall_ms_q, avg_traj, source)
    rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    tot_err = 0
    tot_pt = 0
    tot_ct = 0
    tot_tt = 0
    tot_wall_ms = 0.0
    tot_nq = 0
    tot_traj_steps_sum = 0.0
    tot_traj_steps_count = 0

    inferred_trace_run_name = results_root.name
    trace_ctx_by_dataset: dict[str, tuple[str, str]] = {}
    results_obj_by_dataset: dict[str, Any] = {}

    for ds in selected:
        ds_short = short_by_full[ds]
        result_path = results_root / f"{ds_short}.json"

        # Prefer results JSON summary fields.
        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    obj = json.load(f)
            except Exception:
                obj = None
            results_obj_by_dataset[ds_short] = obj

            # Determine effective trace location (used for fallback and optional error listing).
            trace_run_name_eff = inferred_trace_run_name
            traces_dir_eff = traces_dir
            if isinstance(obj, dict):
                trn = obj.get("trace_run_name", None)
                tdir = obj.get("traces_dir", None)
                if isinstance(trn, str) and trn:
                    trace_run_name_eff = trn
                if isinstance(tdir, str) and tdir:
                    traces_dir_eff = tdir
            trace_ctx_by_dataset[ds_short] = (traces_dir_eff, trace_run_name_eff)

            summary = _try_extract_llm_summary_from_results(obj)
            if summary is not None:
                err_c, pt, ct, tt, avg_traj = summary
                wall_ms, nq = _extract_wall_time_and_nq(obj)
                nq_str = str(nq) if nq is not None else "-"
                wms_str = f"{wall_ms / nq:.1f}" if wall_ms is not None and nq else "-"
                if wall_ms is not None and nq:
                    tot_wall_ms += wall_ms
                if nq is not None:
                    tot_nq += nq
                traj_str = f"{avg_traj:.1f}" if avg_traj is not None else "-"
                if avg_traj is not None and nq:
                    tot_traj_steps_sum += avg_traj * nq
                    tot_traj_steps_count += nq
                rows.append((ds_short, nq_str, str(err_c), str(pt), str(ct), str(tt), wms_str, traj_str, "RESULTS"))
                tot_err += err_c
                tot_pt += pt
                tot_ct += ct
                tot_tt += tt
                continue

            # Fallback to traces using trace_run_name/traces_dir from the results file if available.
            traces_dir_eff, trace_run_name_eff = trace_ctx_by_dataset[ds_short]
            trace_summary = _compute_llm_summary_from_traces(traces_dir_eff, trace_run_name_eff, ds_short)
            if trace_summary is not None:
                err_c, pt, ct, tt, avg_traj, num_traces = trace_summary
                wall_ms, nq = _extract_wall_time_and_nq(obj)
                nq_eff = nq if nq is not None else num_traces
                nq_str = str(nq_eff)
                wms_str = f"{wall_ms / nq_eff:.1f}" if wall_ms is not None and nq_eff else "-"
                if wall_ms is not None and nq_eff:
                    tot_wall_ms += wall_ms
                tot_nq += nq_eff
                traj_str = f"{avg_traj:.1f}" if avg_traj is not None else "-"
                if avg_traj is not None and nq_eff:
                    tot_traj_steps_sum += avg_traj * nq_eff
                    tot_traj_steps_count += nq_eff
                rows.append((ds_short, nq_str, str(err_c), str(pt), str(ct), str(tt), wms_str, traj_str, "TRACES"))
                tot_err += err_c
                tot_pt += pt
                tot_ct += ct
                tot_tt += tt
            else:
                rows.append((ds_short, "-", "-", "-", "-", "-", "-", "-", "MISSING"))
            continue

        # Results JSON missing: try traces with inferred run name.
        trace_ctx_by_dataset[ds_short] = (traces_dir, inferred_trace_run_name)
        trace_summary = _compute_llm_summary_from_traces(traces_dir, inferred_trace_run_name, ds_short)
        if trace_summary is not None:
            err_c, pt, ct, tt, avg_traj, num_traces = trace_summary
            traj_str = f"{avg_traj:.1f}" if avg_traj is not None else "-"
            rows.append((ds_short, str(num_traces), str(err_c), str(pt), str(ct), str(tt), "-", traj_str, "TRACES"))
            tot_nq += num_traces
            tot_err += err_c
            tot_pt += pt
            tot_ct += ct
            tot_tt += tt
        else:
            rows.append((ds_short, "-", "-", "-", "-", "-", "-", "-", "MISSING"))

    # Pretty print.
    headers = (
        "dataset",
        "num_queries",
        "llm_errors",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "wall_ms/q",
        "avg_traj_steps",
        "source",
    )
    col_w = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    table_width = sum(col_w) + 2 * (len(col_w) - 1)
    title = f"LLM usage report: {results_root}"
    rule_width = max(70, len(title), table_width)

    print("\n" + "=" * rule_width)
    print(title)
    print("=" * rule_width)
    print(
        "  ".join(
            [
                headers[0].ljust(col_w[0]),
                headers[1].rjust(col_w[1]),
                headers[2].rjust(col_w[2]),
                headers[3].rjust(col_w[3]),
                headers[4].rjust(col_w[4]),
                headers[5].rjust(col_w[5]),
                headers[6].rjust(col_w[6]),
                headers[7].rjust(col_w[7]),
                headers[8].ljust(col_w[8]),
            ]
        )
    )
    print("-" * rule_width)

    for ds_short, nq_s, err_s, pt_s, ct_s, tt_s, wms_s, traj_s, src in rows:
        print(
            "  ".join(
                [
                    ds_short.ljust(col_w[0]),
                    nq_s.rjust(col_w[1]),
                    err_s.rjust(col_w[2]),
                    pt_s.rjust(col_w[3]),
                    ct_s.rjust(col_w[4]),
                    tt_s.rjust(col_w[5]),
                    wms_s.rjust(col_w[6]),
                    traj_s.rjust(col_w[7]),
                    src.ljust(col_w[8]),
                ]
            )
        )

    avg_wall_str = f"{tot_wall_ms / tot_nq:.1f}" if tot_nq > 0 else "-"
    avg_traj_str = f"{tot_traj_steps_sum / tot_traj_steps_count:.1f}" if tot_traj_steps_count > 0 else "-"
    print("-" * rule_width)
    print(
        "  ".join(
            [
                "TOTAL".ljust(col_w[0]),
                str(tot_nq).rjust(col_w[1]),
                str(tot_err).rjust(col_w[2]),
                str(tot_pt).rjust(col_w[3]),
                str(tot_ct).rjust(col_w[4]),
                str(tot_tt).rjust(col_w[5]),
                avg_wall_str.rjust(col_w[6]),
                avg_traj_str.rjust(col_w[7]),
                "".ljust(col_w[8]),
            ]
        )
    )
    print("=" * rule_width + "\n")

    if list_errors:
        print("=" * rule_width)
        print("LLM errors (query id -> error)")
        print("=" * rule_width)
        any_errs = False
        for ds in selected:
            ds_short = short_by_full[ds]
            traces_dir_eff, trace_run_name_eff = trace_ctx_by_dataset.get(
                ds_short, (traces_dir, inferred_trace_run_name)
            )
            errs_from_traces = _compute_llm_errors_from_traces(
                traces_dir_eff, trace_run_name_eff, ds_short, max_error_len=max_error_len
            )
            err_map = {qid: err for qid, err in (errs_from_traces or [])}

            # Also include any error ids recorded in the results JSON summary (even if the trace was deleted later).
            err_ids_from_results: list[str] = []
            pi = _extract_pipeline_infos(results_obj_by_dataset.get(ds_short, None))
            if isinstance(pi, dict):
                ids = pi.get("llm_error_query_ids", None)
                if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
                    err_ids_from_results = ids

            all_ids = set(err_map.keys()) | set(err_ids_from_results)
            if not all_ids:
                continue

            def _qid_key(x: str):
                return int(x) if x.isdigit() else x

            any_errs = True
            sorted_ids = sorted(all_ids, key=_qid_key)
            print(f"{ds_short}: {len(sorted_ids)}")
            for qid in sorted_ids:
                err = err_map.get(qid, None)
                if err is None:
                    err = "(trace missing or llm_error not present in trace file)"
                print(f"  {qid}: {err}")
        if not any_errs:
            print("(no llm_error entries found in traces)")
        print("=" * rule_width + "\n")


@app.command()
def purge_llm_error_traces(
    results_dir: Annotated[
        str,
        typer.Option(
            "--results-dir",
            help="Pipeline results directory (e.g. results/AgenticPipelineV1__...). Used to locate trace_run_name.",
        ),
    ],
    all_datasets: Annotated[
        bool,
        typer.Option(
            "--all-datasets",
            "--all",
            help="Purge across all available ViDoRe v3 datasets (overrides --datasets).",
        ),
    ] = False,
    datasets: Annotated[
        Optional[str],
        typer.Option(
            "--datasets",
            help="Optional comma-separated dataset filter (accepts either full names like 'vidore/vidore_v3_hr' "
            "or shorts like 'vidore_v3_hr'). Defaults to all datasets.",
        ),
    ] = None,
    traces_dir: Annotated[
        str,
        typer.Option(
            "--traces-dir",
            help="Default traces root directory to use when results JSON does not specify traces_dir.",
        ),
    ] = "traces",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="If enabled (default), print what would be deleted without deleting anything.",
        ),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Required to actually delete files when using --no-dry-run.",
        ),
    ] = False,
    print_query_ids: Annotated[
        bool,
        typer.Option(
            "--print-query-ids",
            help="Print the query ids that would be purged per dataset (useful for logging).",
        ),
    ] = False,
):
    """
    Delete per-query trace files where `pipeline_trace.llm_error` is present.

    This is a practical way to retrigger only queries that hit LLM failures, since evaluation
    uses per-query trace caching: missing traces are rerun, existing valid traces are skipped.
    """
    results_root = Path(results_dir)
    if not results_root.exists() or not results_root.is_dir():
        raise typer.BadParameter(f"--results-dir must be an existing directory: {results_root}")

    if (not dry_run) and (not yes):
        raise typer.BadParameter("Refusing to delete traces without --yes. Re-run with --no-dry-run --yes.")

    # Canonical dataset list.
    ds_full = get_available_datasets()
    short_by_full = {ds: dataset_trace_dir(ds) for ds in ds_full}
    full_by_short = {v: k for k, v in short_by_full.items()}

    selected = list(ds_full)
    if not all_datasets and datasets:
        toks = [t.strip() for t in datasets.split(",") if t.strip()]
        if not toks:
            raise typer.BadParameter("--datasets was provided but empty.")
        selected = []
        for tok in toks:
            if tok in short_by_full:
                selected.append(tok)
            elif tok in full_by_short:
                selected.append(full_by_short[tok])
            else:
                raise typer.BadParameter(
                    f"Unknown dataset '{tok}'. Expected one of: {', '.join(short_by_full.values())}"
                )

    inferred_trace_run_name = results_root.name
    total_marked = 0
    total_deleted = 0

    print("\n" + "=" * 70)
    mode = "DRY RUN (no deletions)" if dry_run else "DELETE"
    print(f"Purge LLM error traces: {results_root}  [{mode}]")
    print("=" * 70)

    for ds in selected:
        ds_short = short_by_full[ds]
        result_path = results_root / f"{ds_short}.json"

        trace_run_name_eff = inferred_trace_run_name
        traces_dir_eff = traces_dir

        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    obj = json.load(f)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                trn = obj.get("trace_run_name", None)
                tdir = obj.get("traces_dir", None)
                if isinstance(trn, str) and trn:
                    trace_run_name_eff = trn
                if isinstance(tdir, str) and tdir:
                    traces_dir_eff = tdir

        trace_root = Path(traces_dir_eff) / trace_run_name_eff / ds_short
        if not trace_root.exists() or not trace_root.is_dir():
            print(f"{ds_short}: no trace dir at {trace_root}")
            continue

        marked_paths: list[Path] = []
        marked_qids: list[str] = []

        for p in sorted(trace_root.glob("*.json")):
            try:
                with open(p, "r") as f:
                    obj = json.load(f)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            pipeline_trace = obj.get("pipeline_trace", None)
            if not isinstance(pipeline_trace, dict):
                continue
            if not isinstance(pipeline_trace.get("llm_error", None), str):
                continue

            marked_paths.append(p)
            qid = obj.get("query_id", None)
            marked_qids.append(str(qid) if qid is not None else p.stem)

        if not marked_paths:
            print(f"{ds_short}: 0 traces with llm_error")
            continue

        total_marked += len(marked_paths)
        print(f"{ds_short}: {len(marked_paths)} traces with llm_error ({trace_root})")
        if print_query_ids:
            print(f"  query_ids: {', '.join(marked_qids)}")

        if not dry_run:
            deleted = 0
            for p in marked_paths:
                try:
                    p.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"  WARNING: failed to delete {p}: {type(e).__name__}: {e}")
            total_deleted += deleted
            print(f"  deleted: {deleted}/{len(marked_paths)}")

    print("-" * 70)
    if dry_run:
        print(f"TOTAL marked for deletion: {total_marked}")
        print("Re-run with: --no-dry-run --yes  to delete.")
    else:
        print(f"TOTAL deleted: {total_deleted}/{total_marked}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    app()
