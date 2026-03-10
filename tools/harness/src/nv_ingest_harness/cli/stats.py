# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Statistical analysis of benchmark runs from history database."""

import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any

import click

from nv_ingest_harness.sinks.history import DEFAULT_DB_PATH

# Metrics to analyze with their direction (higher_is_better or lower_is_better)
METRICS_CONFIG = {
    "ingestion_time_s": {"direction": "lower", "format": ".2f", "unit": "s"},
    "pages_per_second": {"direction": "higher", "format": ".2f", "unit": "p/s"},
    "total_pages": {"direction": "exact", "format": "d", "unit": ""},
    "result_count": {"direction": "exact", "format": "d", "unit": ""},
    "failure_count": {"direction": "lower", "format": "d", "unit": ""},
    "text_chunks": {"direction": "higher", "format": "d", "unit": ""},
    "table_chunks": {"direction": "higher", "format": "d", "unit": ""},
    "chart_chunks": {"direction": "higher", "format": "d", "unit": ""},
    "recall_at_5": {"direction": "higher", "format": ".3f", "unit": ""},
    "recall_at_5_reranker": {"direction": "higher", "format": ".3f", "unit": ""},
    "retrieval_time_s": {"direction": "lower", "format": ".2f", "unit": "s"},
}


def get_runs_for_analysis(
    db_path: str,
    dataset: str | None = None,
    session_pattern: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Get runs from history DB with optional filters."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM runs WHERE 1=1"
    params: list[Any] = []

    if dataset:
        query += " AND dataset = ?"
        params.append(dataset)

    if session_pattern:
        query += " AND session_name LIKE ?"
        params.append(f"%{session_pattern}%")

    query += " ORDER BY timestamp DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def compute_stats(values: list[float | int]) -> dict[str, float]:
    """Compute statistical measures for a list of values."""
    if not values:
        return {}

    n = len(values)
    mean = statistics.mean(values)

    result = {
        "count": n,
        "mean": mean,
        "min": min(values),
        "max": max(values),
    }

    if n >= 2:
        result["stddev"] = statistics.stdev(values)
        result["cv"] = result["stddev"] / mean if mean != 0 else 0  # coefficient of variation
    else:
        result["stddev"] = 0.0
        result["cv"] = 0.0

    return result


def analyze_dataset(runs: list[dict[str, Any]], dataset: str) -> dict[str, dict[str, float]]:
    """Analyze all metrics for a dataset's runs."""
    analysis = {}

    for metric, config in METRICS_CONFIG.items():
        values = [r[metric] for r in runs if r.get(metric) is not None]
        if values:
            stats = compute_stats(values)
            stats["direction"] = config["direction"]
            stats["format"] = config["format"]
            stats["unit"] = config["unit"]
            analysis[metric] = stats

    return analysis


def suggest_baseline(stats: dict[str, float], sigma: float = 2.0) -> dict[str, Any]:
    """Suggest baseline threshold based on stats and direction."""
    direction = stats.get("direction", "higher")
    mean = stats["mean"]
    stddev = stats.get("stddev", 0)

    if direction == "higher":
        # For metrics where higher is better, set min threshold
        suggested = mean - sigma * stddev
        return {"min": max(0, suggested), "type": "min"}
    elif direction == "lower":
        # For metrics where lower is better, set max threshold
        suggested = mean + sigma * stddev
        return {"max": suggested, "warn_threshold": mean + stddev, "type": "max"}
    else:  # exact
        return {"expected": round(mean), "type": "expected"}


def format_value(value: float, fmt: str) -> str:
    """Format a value according to format spec."""
    if fmt == "d":
        return f"{int(value)}"
    return f"{value:{fmt}}"


def print_dataset_analysis(dataset: str, analysis: dict[str, dict[str, float]], sigma: float = 2.0) -> None:
    """Print analysis for a dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset.upper()}")
    print(f"{'='*70}")

    if not analysis:
        print("  No data available")
        return

    # Get run count from first metric
    first_metric = next(iter(analysis.values()))
    print(f"  Runs analyzed: {int(first_metric['count'])}")
    print()

    # Print metrics table
    print(f"  {'Metric':<28} {'Mean':>12} {'StdDev':>10} {'Min':>10} {'Max':>10} {'CV%':>8}")
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for metric, stats in analysis.items():
        fmt = stats["format"]
        unit = stats["unit"]
        mean_str = format_value(stats["mean"], fmt) + unit
        stddev_str = format_value(stats.get("stddev", 0), fmt) if stats.get("stddev") else "-"
        min_str = format_value(stats["min"], fmt) + unit
        max_str = format_value(stats["max"], fmt) + unit
        cv_str = f"{stats.get('cv', 0)*100:.1f}%" if stats.get("cv") else "-"

        print(f"  {metric:<28} {mean_str:>12} {stddev_str:>10} {min_str:>10} {max_str:>10} {cv_str:>8}")

    # Print suggested baselines
    print()
    print(f"  Suggested Baselines ({sigma}σ confidence):")
    print(f"  {'-'*50}")

    for metric, stats in analysis.items():
        suggestion = suggest_baseline(stats, sigma)
        fmt = stats["format"]
        unit = stats["unit"]

        if suggestion["type"] == "min":
            val = format_value(suggestion["min"], fmt)
            print(f"    {metric}: min={val}{unit}")
        elif suggestion["type"] == "max":
            val = format_value(suggestion["max"], fmt)
            warn = format_value(suggestion["warn_threshold"], fmt)
            print(f"    {metric}: max={val}{unit}, warn={warn}{unit}")
        else:
            val = format_value(suggestion["expected"], fmt)
            print(f"    {metric}: expected={val}{unit}")


def generate_baselines_dict(analysis_by_dataset: dict[str, dict], sigma: float = 2.0) -> dict[str, dict]:
    """Generate baselines dictionary suitable for baselines.py."""
    baselines = {}

    for dataset, analysis in analysis_by_dataset.items():
        dataset_baselines = {}

        for metric, stats in analysis.items():
            suggestion = suggest_baseline(stats, sigma)

            if suggestion["type"] == "min":
                dataset_baselines[metric] = {"min": round(suggestion["min"], 3)}
            elif suggestion["type"] == "max":
                dataset_baselines[metric] = {
                    "max": round(suggestion["max"], 2),
                    "warn_threshold": round(suggestion["warn_threshold"], 2),
                }
            else:
                dataset_baselines[metric] = {"expected": int(suggestion["expected"]), "required": True}

        # Always require zero failures
        if "failure_count" in dataset_baselines:
            dataset_baselines["failure_count"] = {"expected": 0, "required": True}

        baselines[dataset] = dataset_baselines

    return baselines


@click.command()
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True),
    default=None,
    help=f"Path to history database (default: {DEFAULT_DB_PATH})",
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Dataset(s) to analyze (can specify multiple, default: all)",
)
@click.option(
    "--session",
    "session_pattern",
    help="Filter by session name pattern (e.g., 'nightly_20260112' or 'baseline')",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=None,
    help="Limit to most recent N runs per dataset",
)
@click.option(
    "--sigma",
    type=float,
    default=2.0,
    help="Number of standard deviations for baseline thresholds (default: 2.0)",
)
@click.option(
    "--export-baselines",
    is_flag=True,
    help="Export suggested baselines as Python dict",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def main(
    db_path: str | None,
    dataset: tuple[str, ...],
    session_pattern: str | None,
    limit: int | None,
    sigma: float,
    export_baselines: bool,
    output_json: bool,
):
    """Analyze benchmark statistics from history database.

    Computes mean, stddev, min, max, and coefficient of variation for all metrics.
    Suggests baseline thresholds based on statistical analysis.

    Examples:

        # Analyze all datasets from recent nightly runs
        uv run nv-ingest-harness-stats

        # Analyze specific dataset with last 10 runs
        uv run nv-ingest-harness-stats -d bo767 -n 10

        # Analyze runs from tonight's baseline collection
        uv run nv-ingest-harness-stats --session nightly_20260112

        # Export suggested baselines as Python dict
        uv run nv-ingest-harness-stats --export-baselines
    """
    db = db_path or str(DEFAULT_DB_PATH)

    if not Path(db).exists():
        print(f"Error: History database not found at {db}")
        print("Run some benchmarks first to populate the database.")
        return 1

    # Get all runs
    all_runs = get_runs_for_analysis(db, session_pattern=session_pattern, limit=limit * 10 if limit else None)

    if not all_runs:
        print("No runs found matching criteria")
        return 1

    # Group by dataset
    runs_by_dataset: dict[str, list[dict]] = {}
    for run in all_runs:
        ds = run["dataset"]
        if dataset and ds not in dataset:
            continue
        if ds not in runs_by_dataset:
            runs_by_dataset[ds] = []
        if limit is None or len(runs_by_dataset[ds]) < limit:
            runs_by_dataset[ds].append(run)

    if not runs_by_dataset:
        print(f"No runs found for specified datasets: {dataset}")
        return 1

    # Analyze each dataset
    analysis_by_dataset = {}
    for ds in sorted(runs_by_dataset.keys()):
        runs = runs_by_dataset[ds]
        analysis_by_dataset[ds] = analyze_dataset(runs, ds)

    # Output
    if output_json:
        output = {
            "datasets": analysis_by_dataset,
            "suggested_baselines": generate_baselines_dict(analysis_by_dataset, sigma),
        }
        print(json.dumps(output, indent=2))
    elif export_baselines:
        baselines = generate_baselines_dict(analysis_by_dataset, sigma)
        print("# Suggested baselines based on statistical analysis")
        print(f"# Sigma: {sigma} (confidence interval)")
        print()
        print("DATASET_BASELINES = {")
        for ds, metrics in baselines.items():
            print(f'    "{ds}": {{')
            for metric, thresholds in metrics.items():
                print(f'        "{metric}": {thresholds},')
            print("    },")
        print("}")
    else:
        print("\n" + "=" * 70)
        print("  BENCHMARK STATISTICS ANALYSIS")
        print("=" * 70)
        if session_pattern:
            print(f"  Session filter: *{session_pattern}*")
        if limit:
            print(f"  Limited to: {limit} most recent runs per dataset")
        print(f"  Sigma for baselines: {sigma}")

        for ds in sorted(analysis_by_dataset.keys()):
            print_dataset_analysis(ds, analysis_by_dataset[ds], sigma)

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
