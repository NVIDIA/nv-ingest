# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
import json
import os
import pty
import re
import select
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from nemo_retriever.harness.artifacts import (
    create_run_artifact_dir,
    create_session_dir,
    last_commit,
    now_timestr,
    write_json,
    write_session_summary,
)
from nemo_retriever.harness.config import (
    DEFAULT_NIGHTLY_CONFIG_PATH,
    HarnessConfig,
    TUNING_FIELDS,
    load_harness_config,
    load_runs_config,
)
from nemo_retriever.harness.parsers import StreamMetrics

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _resolve_lancedb_uri(cfg: HarnessConfig, artifact_dir: Path) -> str:
    raw = str(cfg.lancedb_uri or "lancedb")
    if raw == "lancedb":
        return str((artifact_dir / "lancedb").resolve())
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def _build_command(cfg: HarnessConfig, artifact_dir: Path, run_id: str) -> tuple[list[str], Path, Path]:
    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if cfg.write_detection_file:
        detection_summary_file = artifact_dir / "detection_summary.json"
    else:
        # Keep detection summary out of top-level artifacts unless explicitly requested.
        detection_summary_file = runtime_dir / ".detection_summary.json"
    query_csv = Path(cfg.query_csv) if cfg.query_csv else (artifact_dir / "__query_csv_missing__.csv")

    cmd = [
        sys.executable,
        "-m",
        "nemo_retriever.examples.batch_pipeline",
        str(Path(cfg.dataset_dir).resolve()),
        "--input-type",
        cfg.input_type,
        "--query-csv",
        str(query_csv),
        "--no-recall-details",
        "--pdf-extract-workers",
        str(cfg.pdf_extract_workers),
        "--pdf-extract-num-cpus",
        str(cfg.pdf_extract_num_cpus),
        "--pdf-extract-batch-size",
        str(cfg.pdf_extract_batch_size),
        "--pdf-split-batch-size",
        str(cfg.pdf_split_batch_size),
        "--page-elements-batch-size",
        str(cfg.page_elements_batch_size),
        "--page-elements-workers",
        str(cfg.page_elements_workers),
        "--ocr-workers",
        str(cfg.ocr_workers),
        "--ocr-batch-size",
        str(cfg.ocr_batch_size),
        "--embed-workers",
        str(cfg.embed_workers),
        "--embed-batch-size",
        str(cfg.embed_batch_size),
        "--page-elements-cpus-per-actor",
        str(cfg.page_elements_cpus_per_actor),
        "--ocr-cpus-per-actor",
        str(cfg.ocr_cpus_per_actor),
        "--embed-cpus-per-actor",
        str(cfg.embed_cpus_per_actor),
        "--gpu-page-elements",
        str(cfg.gpu_page_elements),
        "--gpu-ocr",
        str(cfg.gpu_ocr),
        "--gpu-embed",
        str(cfg.gpu_embed),
        "--embed-model-name",
        cfg.embed_model_name,
        "--runtime-metrics-dir",
        str(runtime_dir),
        "--runtime-metrics-prefix",
        run_id,
        "--detection-summary-file",
        str(detection_summary_file),
        "--lancedb-uri",
        _resolve_lancedb_uri(cfg, artifact_dir),
    ]

    if cfg.ray_address:
        cmd += ["--ray-address", cfg.ray_address]
    if cfg.hybrid:
        cmd += ["--hybrid"]

    return cmd, runtime_dir, detection_summary_file


def _evaluate_run_outcome(
    process_rc: int, recall_required: bool, recall_metrics: dict[str, float]
) -> tuple[int, str, bool]:
    if process_rc != 0:
        reason = f"subprocess_exit_{process_rc}"
        return process_rc, reason, False
    if recall_required and not recall_metrics:
        return 98, "missing_recall_metrics", False
    return 0, "", True


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _consume_parseable_output(metrics: StreamMetrics, parse_buffer: str) -> str:
    while "\n" in parse_buffer:
        line, parse_buffer = parse_buffer.split("\n", 1)
        cleaned = ANSI_ESCAPE_RE.sub("", line)
        metrics.consume(cleaned + "\n")
    return parse_buffer


def _run_subprocess_with_tty(cmd: list[str], metrics: StreamMetrics) -> int:
    """
    Run command in a pseudo-terminal so Ray renders rich progress.

    We still parse lines from the PTY stream to extract benchmark metrics.
    """
    master_fd, slave_fd = pty.openpty()
    parse_buffer = ""
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=None,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    try:
        while True:
            read_fds, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd not in read_fds:
                if proc.poll() is not None:
                    break
                continue

            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                # PTY EOF on Linux often appears as EIO.
                if exc.errno == errno.EIO:
                    break
                raise

            if not chunk:
                break

            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()

            parse_buffer += text.replace("\r", "\n")
            parse_buffer = _consume_parseable_output(metrics, parse_buffer)

        if parse_buffer:
            cleaned_tail = ANSI_ESCAPE_RE.sub("", parse_buffer)
            metrics.consume(cleaned_tail)

        return proc.wait()
    finally:
        os.close(master_fd)


def _run_single(cfg: HarnessConfig, artifact_dir: Path, run_id: str) -> dict[str, Any]:
    cmd, runtime_dir, detection_summary_file = _build_command(cfg, artifact_dir, run_id)
    command_text = " ".join(shlex.quote(token) for token in cmd)
    (artifact_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    typer.echo(f"\n=== Running {run_id} ===")
    typer.echo(command_text)

    metrics = StreamMetrics()
    process_rc = _run_subprocess_with_tty(cmd, metrics)
    runtime_summary_path = runtime_dir / f"{run_id}.runtime.summary.json"
    runtime_summary = _read_json_if_exists(runtime_summary_path)
    detection_summary = _read_json_if_exists(detection_summary_file)
    if not cfg.write_detection_file and detection_summary_file.exists():
        detection_summary_file.unlink()

    recall_metrics_normalized: dict[str, float] = {}
    for key, val in metrics.recall_metrics.items():
        normalized = key.replace("@", "_").replace("-", "_")
        recall_metrics_normalized[f"recall_{normalized}"] = val

    effective_rc, failure_reason, success = _evaluate_run_outcome(
        process_rc=process_rc,
        recall_required=bool(cfg.recall_required),
        recall_metrics=metrics.recall_metrics,
    )

    result_payload: dict[str, Any] = {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "success": success,
        "return_code": effective_rc,
        "failure_reason": failure_reason or None,
        "test_config": {
            "dataset_label": cfg.dataset_label,
            "dataset_dir": cfg.dataset_dir,
            "preset": cfg.preset,
            "query_csv": cfg.query_csv,
            "input_type": cfg.input_type,
            "recall_required": cfg.recall_required,
            "ray_address": cfg.ray_address,
            "hybrid": cfg.hybrid,
            "embed_model_name": cfg.embed_model_name,
            "write_detection_file": cfg.write_detection_file,
            "lancedb_uri": _resolve_lancedb_uri(cfg, artifact_dir),
            "tuning": {field: getattr(cfg, field) for field in sorted(TUNING_FIELDS)},
        },
        "metrics": {
            "files": metrics.files,
            "pages": metrics.pages,
            "ingest_secs": metrics.ingest_secs,
            "pages_per_sec_ingest": metrics.pages_per_sec_ingest,
            **recall_metrics_normalized,
        },
        "runtime_summary": runtime_summary,
        "detection_summary": detection_summary,
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
        },
    }
    if cfg.write_detection_file:
        result_payload["artifacts"]["detection_summary_file"] = str(detection_summary_file.resolve())

    write_json(artifact_dir / "results.json", result_payload)
    return result_payload


def _run_entry(
    *,
    run_name: str | None,
    config_file: str | None,
    session_dir: Path | None,
    dataset: str | None,
    preset: str | None,
    sweep_overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    recall_required: bool | None = None,
) -> dict[str, Any]:
    cfg = load_harness_config(
        config_file=config_file,
        dataset=dataset,
        preset=preset,
        sweep_overrides=sweep_overrides,
        cli_overrides=cli_overrides,
        cli_recall_required=recall_required,
    )

    if session_dir is None:
        artifact_dir = create_run_artifact_dir(cfg.dataset_label, run_name=run_name, base_dir=cfg.artifacts_dir)
    else:
        resolved_run_name = run_name or cfg.dataset_label
        artifact_dir = session_dir / resolved_run_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_name = run_name or cfg.dataset_label
    result = _run_single(cfg, artifact_dir, run_id=resolved_run_name)
    return {
        "run_name": resolved_run_name,
        "dataset": cfg.dataset_label,
        "preset": cfg.preset,
        "artifact_dir": str(artifact_dir.resolve()),
        "success": bool(result["success"]),
        "return_code": int(result["return_code"]),
        "failure_reason": result.get("failure_reason"),
        "metrics": dict(result.get("metrics", {})),
    }


def execute_runs(
    *,
    runs: list[dict[str, Any]],
    config_file: str | None,
    session_prefix: str,
    preset_override: str | None,
    base_artifacts_dir: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    session_dir = create_session_dir(session_prefix, base_dir=base_artifacts_dir)
    run_results: list[dict[str, Any]] = []

    for idx, run in enumerate(runs):
        run_name = str(run.get("name") or f"run_{idx + 1:03d}")
        run_result = _run_entry(
            run_name=run_name,
            config_file=config_file,
            session_dir=session_dir,
            dataset=run.get("dataset"),
            preset=run.get("preset") if preset_override is None else preset_override,
            sweep_overrides=run.get("overrides") if isinstance(run.get("overrides"), dict) else run,
            recall_required=run.get("recall_required"),
        )
        run_results.append(run_result)

    return session_dir, run_results


def run_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name from config or direct path."),
    preset: str | None = typer.Option(None, "--preset", help="Preset override."),
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    run_name: str | None = typer.Option(None, "--run-name", help="Optional run name label."),
    override: list[str] = typer.Option([], "--override", help="Override values with KEY=VALUE."),
    recall_required: bool | None = typer.Option(
        None, "--recall-required/--no-recall-required", help="Override recall-required gate for this run."
    ),
) -> None:
    result = _run_entry(
        run_name=run_name,
        config_file=config,
        session_dir=None,
        dataset=dataset,
        preset=preset,
        cli_overrides=override,
        recall_required=recall_required,
    )
    typer.echo(
        f"\nResult: {'PASS' if result['success'] else 'FAIL'} | "
        f"return_code={result['return_code']} | artifact_dir={result['artifact_dir']}"
    )
    raise typer.Exit(code=0 if result["success"] else 1)


def sweep_command(
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    runs_config: str = typer.Option(str(DEFAULT_NIGHTLY_CONFIG_PATH), "--runs-config", help="Path to sweep runs YAML."),
    preset: str | None = typer.Option(None, "--preset", help="Force preset for all sweep runs."),
    session_prefix: str = typer.Option("sweep", "--session-prefix", help="Session directory prefix."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan without executing."),
) -> None:
    runs = load_runs_config(runs_config)
    if dry_run:
        typer.echo("Sweep dry run:")
        for idx, run in enumerate(runs):
            typer.echo(
                f"  {idx + 1:03d}: name={run.get('name')} dataset={run.get('dataset')} preset={run.get('preset')}"
            )
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix=session_prefix,
        preset_override=preset,
    )
    summary_path = write_session_summary(
        session_dir,
        run_results,
        session_type="sweep",
        config_path=str(Path(runs_config).expanduser().resolve()),
    )

    typer.echo(f"\nSweep session: {session_dir}")
    typer.echo(f"Session summary: {summary_path}")
    failed = [r for r in run_results if not r["success"]]
    raise typer.Exit(code=0 if not failed else 1)
