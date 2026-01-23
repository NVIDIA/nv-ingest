# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Nightly benchmark orchestrator."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from nv_ingest_harness.config import load_config
from nv_ingest_harness.reporting.baselines import validate_results, check_all_passed, get_expected_counts
from nv_ingest_harness.reporting.environment import get_environment_data
from nv_ingest_harness.sinks import SlackSink, HistorySink
from nv_ingest_harness.utils.cases import now_timestr
from nv_ingest_harness.utils.session import create_session_dir, write_session_summary

DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "nightly_config.yaml"


def load_nightly_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Nightly config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_harness(
    dataset: str,
    case: str = "e2e",
    session_dir: Path | None = None,
    managed: bool = False,
    deployment_type: str = "compose",
) -> tuple[int, Path | None]:
    """Run a single harness test.

    Args:
        dataset: Dataset name to run
        case: Test case type (e2e or e2e_recall)
        session_dir: Session directory for artifacts
        managed: If True, pass --managed flag to run.py for service lifecycle management
        deployment_type: Deployment type (compose or helm)
    """
    cmd = [
        sys.executable,
        "-m",
        "nv_ingest_harness.cli.run",
        f"--case={case}",
        f"--dataset={dataset}",
        f"--deployment-type={deployment_type}",
    ]

    if managed:
        cmd.append("--managed")

    if session_dir:
        cmd.append(f"--session-dir={str(session_dir)}")

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    if session_dir:
        session_dir = Path(session_dir)

        artifact_paths_file = session_dir / ".artifact_paths.json"
        if artifact_paths_file.exists():
            with open(artifact_paths_file) as f:
                artifact_paths = json.load(f)
            artifact_dir_str = artifact_paths.get(dataset)
            if artifact_dir_str:
                artifact_dir = Path(artifact_dir_str)
                if artifact_dir.exists() and (artifact_dir / "results.json").exists():
                    return result.returncode, artifact_dir

        cfg = load_config(case=case, dataset=dataset)
        artifact_name = cfg.test_name or dataset
        candidate = session_dir / artifact_name
        if candidate.exists() and (candidate / "results.json").exists():
            return result.returncode, candidate

        candidates = [d for d in session_dir.iterdir() if d.is_dir() and (d / "results.json").exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return result.returncode, candidates[0]
    else:
        artifacts_root = Path(__file__).parents[3] / "artifacts"
        if artifacts_root.exists():
            matching_dirs = sorted(
                [
                    d
                    for d in artifacts_root.iterdir()
                    if d.is_dir() and d.name.startswith(dataset) and (d / "results.json").exists()
                ],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if matching_dirs:
                return result.returncode, matching_dirs[0]

    return result.returncode, None


def load_results(artifact_dir: Path) -> dict:
    results_file = artifact_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to nightly config YAML (default: nightly_config.yaml)",
)
@click.option(
    "--deployment-type",
    type=click.Choice(["compose", "helm"], case_sensitive=False),
    default="compose",
    help="Deployment type for service management (compose or helm)",
)
@click.option(
    "--skip-slack",
    is_flag=True,
    help="Disable Slack posting (overrides config)",
)
@click.option(
    "--skip-history",
    is_flag=True,
    help="Disable SQLite history storage (overrides config)",
)
@click.option(
    "--managed",
    is_flag=True,
    help="Enable managed service lifecycle (start/stop services for each dataset)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be done without running benchmarks",
)
@click.option(
    "--replay",
    "replay_dirs",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Replay results from artifact directories to Slack (can specify multiple)",
)
@click.option(
    "--note",
    type=str,
    default=None,
    help="Optional note to attach to the session summary and Slack output",
)
def main(
    config_path: Path | None,
    deployment_type: str,
    skip_slack: bool,
    skip_history: bool,
    managed: bool,
    dry_run: bool,
    replay_dirs: tuple[Path, ...],
    note: str | None,
):
    """Run nightly benchmarks and post results."""
    if replay_dirs:
        return _replay_results(replay_dirs)

    config_file = config_path or DEFAULT_CONFIG_PATH
    try:
        config = load_nightly_config(config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    runs_config = config.get("runs", {})
    recall_config = config.get("recall", {})
    sinks_config = config.get("sinks", {})

    e2e_datasets = runs_config.get("e2e", [])
    recall_datasets = runs_config.get("e2e_recall", [])
    reranker_mode = recall_config.get("reranker_mode", "both")
    session_name = f"nightly_{now_timestr()}"
    session_dir = create_session_dir(session_name)

    print("\n" + "=" * 60)
    print("nv-ingest Nightly Benchmark")
    print(f"Session: {session_name}")
    print(f"Session Dir: {session_dir}")
    print(f"Config: {config_file}")
    print("=" * 60)
    print(f"E2E datasets: {e2e_datasets}")
    print(f"Recall datasets: {recall_datasets}")
    print(f"Reranker mode: {reranker_mode}")
    print(f"Deployment type: {deployment_type}")
    print(f"Managed services: {managed}")
    if note:
        print(f"Note: {note}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN - not executing benchmarks")
        print(f"Would create session dir: {session_dir}")
        return 0

    os.environ["RERANKER_MODE"] = reranker_mode
    # Use local reranker container instead of build API
    reranker_endpoint = recall_config.get("reranker_endpoint", "http://localhost:8020/v1/ranking")
    os.environ["RERANKER_NIM_ENDPOINT"] = reranker_endpoint
    # Pass recall_top_k from nightly config to harness
    recall_top_k = recall_config.get("top_k", 10)
    os.environ["RECALL_TOP_K"] = str(recall_top_k)

    env_data = get_environment_data()
    if note:
        env_data["note"] = note
    print("Environment:")
    for key, val in env_data.items():
        print(f"  {key}: {val}")

    sinks = []
    slack_config = sinks_config.get("slack", {})
    if slack_config.get("enabled", True) and not skip_slack:
        sinks.append(SlackSink(slack_config))

    history_config = sinks_config.get("history", {})
    if history_config.get("enabled", True) and not skip_history:
        sinks.append(HistorySink(history_config))

    for sink in sinks:
        sink.initialize(session_name, env_data)

    all_results = []

    for dataset in e2e_datasets:
        print(f"\n--- Running e2e for {dataset} ---")
        rc, artifact_dir = run_harness(
            dataset,
            case="e2e",
            session_dir=session_dir,
            managed=managed,
            deployment_type=deployment_type,
        )
        result = _process_result(dataset, rc, artifact_dir, case="e2e")
        all_results.append(result)

        for sink in sinks:
            sink.process_result(result)

    for dataset in recall_datasets:
        print(f"\n--- Running e2e_recall for {dataset} ---")
        rc, artifact_dir = run_harness(
            dataset,
            case="e2e_recall",
            session_dir=session_dir,
            managed=managed,
            deployment_type=deployment_type,
        )
        result = _process_result(dataset, rc, artifact_dir, case="e2e_recall")
        all_results.append(result)

        for sink in sinks:
            sink.process_result(result)

    for sink in sinks:
        sink.finalize()

    # Write session summary
    summary_path = write_session_summary(
        session_dir=session_dir,
        session_name=session_name,
        results=all_results,
        # nightly-specific extensions
        config_file=str(config_file),
        environment=env_data,
        note=note,
        datasets={
            "e2e": e2e_datasets,
            "e2e_recall": recall_datasets,
        },
    )

    print("\n" + "=" * 60)
    print("Nightly Benchmark Complete")
    print(f"Session: {session_dir}")
    print("=" * 60)
    for result in all_results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {result['dataset']}: {status}")
    print(f"\nSession summary: {summary_path}")
    print("=" * 60)

    return 0 if all(r["success"] for r in all_results) else 1


def _process_result(dataset: str, rc: int, artifact_dir: Path | None, case: str) -> dict[str, Any]:
    if not artifact_dir:
        return {
            "dataset": dataset if case == "e2e" else f"{dataset}_recall",
            "success": False,
            "return_code": rc,
            "artifact_dir": None,
            "metrics": {},
            "requirements_status": [],
        }

    raw_results = load_results(artifact_dir)

    if "results" in raw_results:
        metrics = raw_results["results"]
    elif "ingestion_results" in raw_results:
        metrics = dict(raw_results.get("ingestion_results", {}))
        for mode, scores in raw_results.get("recall_results", {}).items():
            if isinstance(scores, dict):
                for k, score in scores.items():
                    suffix = "reranker" if mode == "with_reranker" else "no_reranker"
                    metrics[f"recall_multimodal_@{k}_{suffix}"] = score
    else:
        metrics = {k: v for k, v in raw_results.items() if not k.startswith("test_")}

    validation = validate_results(dataset, metrics)
    success = check_all_passed(validation) and rc == 0
    expected = get_expected_counts(dataset)

    return {
        "dataset": dataset if case == "e2e" else f"{dataset}_recall",
        "success": success,
        "return_code": rc,
        "artifact_dir": str(artifact_dir),
        "metrics": metrics,
        "requirements_status": validation,
        "expected_result_count": expected.get("result_count"),
        "expected_total_pages": expected.get("total_pages"),
    }


def _replay_results(artifact_dirs: tuple[Path, ...]) -> int:
    """Replay results from artifact directories to Slack."""
    print("\n" + "=" * 60)
    print("Replaying Results to Slack")
    print("=" * 60)

    if not os.environ.get("SLACK_WEBHOOK_URL"):
        print("ERROR: SLACK_WEBHOOK_URL not set")
        return 1

    all_results = []
    env_data = get_environment_data()

    for artifact_dir in artifact_dirs:
        artifact_path = Path(artifact_dir)
        results_file = artifact_path / "results.json"

        if not results_file.exists():
            print(f"Warning: No results.json in {artifact_path}, skipping")
            continue

        print(f"Loading: {artifact_path.name}")
        raw_results = load_results(artifact_path)

        dataset = raw_results.get("test_config", {}).get("test_name")
        if not dataset:
            dir_name = artifact_path.name
            dataset = dir_name.rsplit("_", 3)[0] if "_" in dir_name else dir_name

        case = raw_results.get("case", "e2e")
        rc = raw_results.get("return_code", 0)

        result = _process_result(dataset, rc, artifact_path, case)
        all_results.append(result)
        print(f"  {result['dataset']}: {'✓ PASS' if result['success'] else '✗ FAIL'}")

    if not all_results:
        print("ERROR: No valid results found to replay")
        return 1

    slack_sink = SlackSink({"enabled": True})
    session_name = f"replay_{now_timestr()}"
    slack_sink.initialize(session_name, env_data)

    for result in all_results:
        slack_sink.process_result(result)

    slack_sink.finalize()

    print("\n" + "=" * 60)
    print("Replay Complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
