# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
from importlib import metadata
import json
import os
import pty
import re
import select
import shlex
import socket
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
from nemo_retriever.harness.recall_adapters import prepare_recall_query_file
from nemo_retriever.utils.input_files import resolve_input_files

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _collect_gpu_metadata() -> tuple[int | None, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    combined_output = f"{result.stdout}\n{result.stderr}"
    if "No devices were found" in combined_output:
        return 0, None
    if result.returncode != 0:
        return None, None
    if not output_lines:
        return 0, None
    return len(output_lines), output_lines[0]


def _collect_run_metadata() -> dict[str, Any]:
    try:
        host = socket.gethostname().strip() or "unknown"
    except OSError:
        host = "unknown"

    version_info = getattr(sys, "version_info", None)
    if version_info is None:
        python_version = "unknown"
    else:
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"

    gpu_count, cuda_driver = _collect_gpu_metadata()
    return {
        "host": host,
        "gpu_count": gpu_count,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "python_version": python_version,
    }


def _normalize_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in tags or []:
        tag = str(raw).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)

    return normalized


def _normalize_recall_metric_key(key: str) -> str:
    metric = str(key).strip().lower()
    if metric.startswith("recall@"):
        return "recall_" + metric.split("@", 1)[1]
    return metric.replace("@", "_").replace("-", "_")


def _safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                count = int(len(doc))
            except Exception:
                count = int(doc.get_page_count())  # type: ignore[attr-defined]
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return max(count, 0)
    except Exception:
        return None


def _resolve_summary_metrics(
    cfg: HarnessConfig,
    metrics_payload: dict[str, Any],
    runtime_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary_metrics: dict[str, Any] = {
        "pages": metrics_payload.get("pages"),
        "ingest_secs": metrics_payload.get("ingest_secs"),
        "pages_per_sec_ingest": metrics_payload.get("pages_per_sec_ingest"),
        "recall_5": metrics_payload.get("recall_5"),
    }

    if summary_metrics["pages"] is None and isinstance(runtime_summary, dict):
        runtime_pages = runtime_summary.get("num_pages")
        if runtime_pages is None:
            runtime_pages = runtime_summary.get("input_pages")
        if runtime_pages is not None:
            try:
                summary_metrics["pages"] = int(runtime_pages)
            except (TypeError, ValueError):
                summary_metrics["pages"] = None

    if summary_metrics["pages"] is None and cfg.input_type == "pdf":
        total_pages = 0
        counted_any = False
        for path in resolve_input_files(Path(cfg.dataset_dir), cfg.input_type):
            page_count = _safe_pdf_page_count(path)
            if page_count is None:
                continue
            counted_any = True
            total_pages += page_count
        if counted_any:
            summary_metrics["pages"] = total_pages

    if summary_metrics["pages_per_sec_ingest"] is None:
        pages = summary_metrics.get("pages")
        ingest_secs = summary_metrics.get("ingest_secs")
        if pages is not None and ingest_secs not in {None, 0, 0.0}:
            try:
                summary_metrics["pages_per_sec_ingest"] = round(float(pages) / float(ingest_secs), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                summary_metrics["pages_per_sec_ingest"] = None

    return summary_metrics


def _resolve_lancedb_uri(cfg: HarnessConfig, artifact_dir: Path) -> str:
    raw = str(cfg.lancedb_uri or "lancedb")
    if raw == "lancedb":
        return str((artifact_dir / "lancedb").resolve())
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def _command_worker_count_value(value: int | None) -> str:
    if value is not None:
        return str(value)
    return "0"


def _build_command(cfg: HarnessConfig, artifact_dir: Path, run_id: str) -> tuple[list[str], Path, Path, Path]:
    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if cfg.write_detection_file:
        detection_summary_file = artifact_dir / "detection_summary.json"
    else:
        # Keep detection summary out of top-level artifacts unless explicitly requested.
        detection_summary_file = runtime_dir / ".detection_summary.json"
    query_csv = prepare_recall_query_file(
        query_csv=Path(cfg.query_csv) if cfg.query_csv else None,
        recall_adapter=cfg.recall_adapter,
        output_dir=runtime_dir,
    )

    cmd = [
        sys.executable,
        "-m",
        "nemo_retriever.examples.batch_pipeline",
        str(Path(cfg.dataset_dir).resolve()),
        "--input-type",
        cfg.input_type,
        "--query-csv",
        str(query_csv),
        "--recall-match-mode",
        cfg.recall_match_mode,
        "--no-recall-details",
        "--pdf-extract-tasks",
        _command_worker_count_value(cfg.pdf_extract_workers),
        "--pdf-extract-cpus-per-task",
        str(cfg.pdf_extract_num_cpus),
        "--pdf-extract-batch-size",
        str(cfg.pdf_extract_batch_size),
        "--pdf-split-batch-size",
        str(cfg.pdf_split_batch_size),
        "--page-elements-batch-size",
        str(cfg.page_elements_batch_size),
        "--page-elements-actors",
        _command_worker_count_value(cfg.page_elements_workers),
        "--ocr-actors",
        _command_worker_count_value(cfg.ocr_workers),
        "--ocr-batch-size",
        str(cfg.ocr_batch_size),
        "--embed-actors",
        _command_worker_count_value(cfg.embed_workers),
        "--embed-batch-size",
        str(cfg.embed_batch_size),
        "--page-elements-cpus-per-actor",
        str(cfg.page_elements_cpus_per_actor),
        "--ocr-cpus-per-actor",
        str(cfg.ocr_cpus_per_actor),
        "--embed-cpus-per-actor",
        str(cfg.embed_cpus_per_actor),
        "--page-elements-gpus-per-actor",
        str(cfg.gpu_page_elements),
        "--ocr-gpus-per-actor",
        str(cfg.gpu_ocr),
        "--embed-gpus-per-actor",
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

    return cmd, runtime_dir, detection_summary_file, query_csv


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


def _run_single(cfg: HarnessConfig, artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict[str, Any]:
    cmd, runtime_dir, detection_summary_file, effective_query_csv = _build_command(cfg, artifact_dir, run_id)
    command_text = " ".join(shlex.quote(token) for token in cmd)
    (artifact_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    typer.echo(f"\n=== Running {run_id} ===")
    typer.echo(command_text)

    metrics = StreamMetrics()
    process_rc = _run_subprocess_with_tty(cmd, metrics)
    run_metadata = _collect_run_metadata()
    runtime_summary_path = runtime_dir / f"{run_id}.runtime.summary.json"
    runtime_summary = _read_json_if_exists(runtime_summary_path)
    detection_summary = _read_json_if_exists(detection_summary_file)
    if not cfg.write_detection_file and detection_summary_file.exists():
        detection_summary_file.unlink()

    recall_metrics_normalized: dict[str, float] = {}
    for key, val in metrics.recall_metrics.items():
        recall_metrics_normalized[_normalize_recall_metric_key(key)] = val

    effective_rc, failure_reason, success = _evaluate_run_outcome(
        process_rc=process_rc,
        recall_required=bool(cfg.recall_required),
        recall_metrics=metrics.recall_metrics,
    )

    metrics_payload = {
        "files": metrics.files,
        "pages": metrics.pages,
        "ingest_secs": metrics.ingest_secs,
        "pages_per_sec_ingest": metrics.pages_per_sec_ingest,
        **recall_metrics_normalized,
    }
    summary_metrics = _resolve_summary_metrics(cfg, metrics_payload, runtime_summary)

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
            "effective_query_csv": str(effective_query_csv),
            "input_type": cfg.input_type,
            "recall_required": cfg.recall_required,
            "recall_match_mode": cfg.recall_match_mode,
            "recall_adapter": cfg.recall_adapter,
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
            "rows_processed": metrics.rows_processed,
            "rows_per_sec_ingest": metrics.rows_per_sec_ingest,
            **recall_metrics_normalized,
        },
        "summary_metrics": summary_metrics,
        "run_metadata": run_metadata,
        "runtime_summary": runtime_summary,
        "detection_summary": detection_summary,
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
        },
    }
    if cfg.write_detection_file:
        result_payload["artifacts"]["detection_summary_file"] = str(detection_summary_file.resolve())
    if tags:
        result_payload["tags"] = list(tags)

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
    tags: list[str] | None = None,
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
    normalized_tags = _normalize_tags(tags)
    result = _run_single(cfg, artifact_dir, run_id=resolved_run_name, tags=normalized_tags)
    run_result = {
        "run_name": resolved_run_name,
        "dataset": cfg.dataset_label,
        "preset": cfg.preset,
        "artifact_dir": str(artifact_dir.resolve()),
        "success": bool(result["success"]),
        "return_code": int(result["return_code"]),
        "failure_reason": result.get("failure_reason"),
        "metrics": dict(result.get("summary_metrics", result.get("metrics", {}))),
    }
    if normalized_tags:
        run_result["tags"] = normalized_tags
    return run_result


def execute_runs(
    *,
    runs: list[dict[str, Any]],
    config_file: str | None,
    session_prefix: str,
    preset_override: str | None,
    base_artifacts_dir: str | None = None,
    tags: list[str] | None = None,
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
            tags=tags,
        )
        run_results.append(run_result)

    return session_dir, run_results


def run_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name from config or direct path."),
    preset: str | None = typer.Option(None, "--preset", help="Preset override."),
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    run_name: str | None = typer.Option(None, "--run-name", help="Optional run name label."),
    override: list[str] = typer.Option([], "--override", help="Override values with KEY=VALUE."),
    tag: list[str] = typer.Option([], "--tag", help="Run tag to persist in harness artifacts. Repeatable."),
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
        tags=tag,
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
    tag: list[str] = typer.Option([], "--tag", help="Session tag to persist on each run. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan without executing."),
) -> None:
    normalized_tags = _normalize_tags(tag)
    runs = load_runs_config(runs_config)
    if dry_run:
        typer.echo("Sweep dry run:")
        for idx, run in enumerate(runs):
            tag_text = f" tags={normalized_tags}" if normalized_tags else ""
            effective_preset = preset if preset is not None else run.get("preset")
            plan_line = (
                f"  {idx + 1:03d}: name={run.get('name')} "
                f"dataset={run.get('dataset')} preset={effective_preset}{tag_text}"
            )
            typer.echo(plan_line)
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix=session_prefix,
        preset_override=preset,
        tags=normalized_tags,
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
