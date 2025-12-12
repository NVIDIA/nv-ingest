#!/usr/bin/env python3
"""
Sweep runner for nv-ingest integration tests.

Orchestrates multiple test runs across parameter values with repetitions,
then generates aggregated results and analysis.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import click

from nv_ingest_harness.utils.cases import now_timestr


def run_single_test(run_py_path: str, test_args: list[str]) -> tuple[int, dict | None]:
    """Run a single test and return its exit code and results."""
    cmd = [sys.executable, run_py_path] + test_args
    print(f"  Running: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"  ✓ Completed in {elapsed:.1f}s\n")
    else:
        print(f"  ✗ Failed with code {result.returncode} after {elapsed:.1f}s\n")

    return result.returncode, elapsed


def collect_results(artifacts_dir: str) -> dict | None:
    """Load results.json from an artifacts directory."""
    results_path = Path(artifacts_dir) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def write_summary_csv(sweep_results: list[dict], output_path: str, sweep_param: str):
    """Write aggregated results to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        # Determine columns from first result
        if not sweep_results:
            return

        # Core columns
        fieldnames = [
            "sweep_param",
            "value",
            "repetition",
            "test_name",
            "ingestion_time_s",
            "pages_per_second",
            "text_chunks",
            "table_chunks",
            "chart_chunks",
            "result_count",
            "failure_count",
            "return_code",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in sweep_results:
            row = {
                "sweep_param": sweep_param,
                "value": result.get("sweep_value", ""),
                "repetition": result.get("repetition", ""),
                "test_name": result.get("test_name", ""),
                "ingestion_time_s": result.get("ingestion_time_s", ""),
                "pages_per_second": result.get("pages_per_second", ""),
                "text_chunks": result.get("text_chunks", ""),
                "table_chunks": result.get("table_chunks", ""),
                "chart_chunks": result.get("chart_chunks", ""),
                "result_count": result.get("result_count", ""),
                "failure_count": result.get("failure_count", ""),
                "return_code": result.get("return_code", ""),
            }
            writer.writerow(row)


def write_analysis(sweep_results: list[dict], output_path: str, sweep_param: str, sweep_name: str, dataset: str):
    """Write human-readable analysis summary."""
    # Group by sweep value
    from collections import defaultdict
    import statistics

    by_value = defaultdict(list)
    for result in sweep_results:
        if result.get("return_code") == 0:  # Only successful runs
            by_value[result["sweep_value"]].append(result)

    with open(output_path, "w") as f:
        f.write(f"Sweep Results: {sweep_name}\n")
        f.write(f"Parameter: {sweep_param}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Repetitions: {len(sweep_results) // len(by_value) if by_value else 0}\n")
        f.write("\n")

        f.write("Average Results by Value:\n")
        f.write(f"{'Value':<10} | {'Time (s)':<15} | {'Pages/sec':<12} | {'Chunks':<10} | {'Status'}\n")
        f.write("-" * 70 + "\n")

        # Find best performance (lowest time)
        best_time = float("inf")
        best_value = None

        for value in sorted(by_value.keys()):
            results = by_value[value]

            times = [r.get("ingestion_time_s", 0) for r in results if r.get("ingestion_time_s")]
            pages_per_sec = [r.get("pages_per_second", 0) for r in results if r.get("pages_per_second")]
            chunks = [r.get("text_chunks", 0) + r.get("table_chunks", 0) + r.get("chart_chunks", 0) for r in results]

            if times:
                avg_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                avg_pps = statistics.mean(pages_per_sec) if pages_per_sec else 0
                avg_chunks = statistics.mean(chunks) if chunks else 0

                time_str = f"{avg_time:.1f} ± {std_time:.1f}"
                pps_str = f"{avg_pps:.1f}"
                chunks_str = f"{int(avg_chunks)}"

                status = ""
                if avg_time < best_time:
                    best_time = avg_time
                    best_value = value

                f.write(f"{str(value):<10} | {time_str:<15} | {pps_str:<12} | {chunks_str:<10} | {status}\n")

        if best_value is not None:
            f.write("\n")
            f.write(f"Best performance: {sweep_param}={best_value} ({best_time:.1f}s avg)\n")

        # Summary statistics
        f.write("\n")
        f.write("Summary:\n")
        total_runs = len(sweep_results)
        successful_runs = sum(1 for r in sweep_results if r.get("return_code") == 0)
        failed_runs = total_runs - successful_runs

        f.write(f"  Total runs: {total_runs}\n")
        f.write(f"  Successful: {successful_runs}\n")
        f.write(f"  Failed: {failed_runs}\n")


@click.command()
@click.option("--sweep-param", required=True, help="Parameter to sweep (e.g., pdf_split_page_count)")
@click.option("--sweep-values", required=True, help="Comma-separated values to sweep (e.g., 16,32,64,128)")
@click.option("--repetitions", type=int, default=1, help="Number of repetitions per value")
@click.option("--name", required=True, help="Name for this sweep (used in output directory)")
@click.option("--case", default="e2e", help="Test case to run")
@click.option("--dataset", help="Dataset path or shortcut")
@click.option("--api-version", help="API version (v1 or v2)")
@click.option("--managed", is_flag=True, help="Use managed infrastructure")
@click.option("--doc-analysis", is_flag=True, help="Show per-document analysis")
@click.option("--continue-on-failure", is_flag=True, default=True, help="Continue sweep if a run fails")
def main(
    sweep_param,
    sweep_values,
    repetitions,
    name,
    case,
    dataset,
    api_version,
    managed,
    doc_analysis,
    continue_on_failure,
):
    """Run a parameter sweep across multiple test configurations."""

    # Parse sweep values
    values = [v.strip() for v in sweep_values.split(",")]

    # Convert to appropriate types (assume int for now, could be smarter)
    try:
        values = [int(v) for v in values]
    except ValueError:
        pass  # Keep as strings if not ints

    # Setup paths
    script_dir = Path(__file__).parent
    run_py = script_dir / "run.py"
    sweep_results_dir = script_dir / "sweep_results"
    sweep_results_dir.mkdir(exist_ok=True)

    # Create timestamped sweep directory
    timestamp = now_timestr()
    sweep_dir = sweep_results_dir / f"{name}_{timestamp}"
    sweep_dir.mkdir(exist_ok=True)
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Print sweep info
    total_runs = len(values) * repetitions
    print("=" * 80)
    print(f"Sweep: {name}")
    print(f"Parameter: {sweep_param}")
    print(f"Values: {values}")
    print(f"Repetitions: {repetitions}")
    print(f"Total runs: {total_runs} ({len(values)} values × {repetitions} reps)")
    print(f"Infrastructure: {'managed' if managed else 'attached'}")
    print(f"Results: {sweep_dir}")
    print("=" * 80)
    print()

    # Build base arguments
    base_args = ["--case", case]
    if dataset:
        base_args.extend(["--dataset", dataset])
    if api_version:
        base_args.extend(["--api-version", api_version])
    if managed:
        base_args.append("--managed")
        # Only start services once, keep them up
        base_args.append("--keep-up")
    if doc_analysis:
        base_args.append("--doc-analysis")

    # Override artifacts directory to put runs in sweep directory
    base_args.extend(["--artifacts-dir", str(runs_dir)])

    # Run sweep
    sweep_results = []
    run_number = 0

    for value in values:
        for rep in range(1, repetitions + 1):
            run_number += 1

            # Generate test name
            test_name = f"{sweep_param}{value}_rep{rep}"

            print(f"[{run_number}/{total_runs}] {test_name}")

            # Build arguments for this run
            run_args = base_args.copy()
            run_args.extend(["--test-name", test_name])
            run_args.extend([f"--{sweep_param.replace('_', '-')}", str(value)])

            # Run test
            return_code, elapsed = run_single_test(str(run_py), run_args)

            # Find and collect results
            # Results will be in runs_dir/{test_name}_{timestamp}/
            run_artifacts = None
            for subdir in runs_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(test_name):
                    run_artifacts = subdir
                    break

            # Load results
            result_data = {
                "sweep_value": value,
                "repetition": rep,
                "test_name": test_name,
                "return_code": return_code,
                "elapsed_s": elapsed,
            }

            if run_artifacts:
                results = collect_results(run_artifacts)
                if results:
                    # Extract key metrics from nested structure
                    if "results" in results:
                        result_data.update(results["results"])
                    # Also include some top-level metadata
                    for key in ["api_version", "pdf_split_page_count"]:
                        if key in results:
                            result_data[key] = results[key]

            sweep_results.append(result_data)

            # Check if we should continue
            if return_code != 0 and not continue_on_failure:
                print("\n✗ Run failed. Stopping sweep (use --continue-on-failure to continue on errors)")
                break

        # Break outer loop too if we stopped
        if run_number < total_runs and not continue_on_failure:
            break

    # Stop services if managed
    if managed:
        print("\nStopping services...")
        stop_args = ["--managed"]
        run_single_test(str(run_py), stop_args + ["--case", "e2e", "--no-build"])

    # Generate summary
    print("\n" + "=" * 80)
    print("Generating summary...")

    summary_csv = sweep_dir / "summary.csv"
    analysis_txt = sweep_dir / "analysis.txt"

    write_summary_csv(sweep_results, str(summary_csv), sweep_param)
    write_analysis(sweep_results, str(analysis_txt), sweep_param, name, dataset or "from config")

    print("\n✓ Sweep completed!")
    print(f"  Results: {sweep_dir}")
    print(f"  Summary: {summary_csv}")
    print(f"  Analysis: {analysis_txt}")
    print("=" * 80)

    # Show quick summary
    print("\nQuick Summary:")
    with open(analysis_txt) as f:
        print(f.read())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
