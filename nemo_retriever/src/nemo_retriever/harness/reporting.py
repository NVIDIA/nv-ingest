# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

SESSION_SUMMARY_BASENAME = "session_summary.json"


def _resolve_session_summary_path(path_or_dir: Path) -> Path:
    candidate = path_or_dir.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.is_dir():
        candidate = candidate / SESSION_SUMMARY_BASENAME

    if candidate.name != SESSION_SUMMARY_BASENAME:
        raise FileNotFoundError(f"Expected a session directory or '{SESSION_SUMMARY_BASENAME}' file, got: {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"Session summary not found: {candidate}")
    return candidate


def _load_session_summary(path_or_dir: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = _resolve_session_summary_path(path_or_dir)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {summary_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Session summary must be a JSON object: {summary_path}")
    return summary_path, payload


def _load_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Session summary 'results' must be a list")
    return [item for item in results if isinstance(item, dict)]


def _collect_recall_keys(results: list[dict[str, Any]]) -> list[str]:
    keys = {
        key
        for result in results
        for key in dict(result.get("metrics", {})).keys()
        if isinstance(key, str) and key.startswith("recall_")
    }
    return sorted(keys)


def _metric_float(result: dict[str, Any], key: str) -> float | None:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_int(result: dict[str, Any], key: str) -> int | None:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _format_delta(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "-"
    delta = right - left
    return f"{delta:+.3f}".rstrip("0").rstrip(".")


def _format_table(headers: list[str], rows: list[list[str]], right_align: set[int] | None = None) -> str:
    right_align = right_align or set()
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_row(row: list[str]) -> str:
        rendered: list[str] = []
        for idx, cell in enumerate(row):
            if idx in right_align:
                rendered.append(cell.rjust(widths[idx]))
            else:
                rendered.append(cell.ljust(widths[idx]))
        return " | ".join(rendered)

    divider = "-+-".join("-" * width for width in widths)
    lines = [_render_row(headers), divider]
    lines.extend(_render_row(row) for row in rows)
    return "\n".join(lines)


def format_session_summary(summary_path: Path, payload: dict[str, Any]) -> str:
    results = _load_results(payload)
    recall_keys = _collect_recall_keys(results)
    include_tags = any(result.get("tags") for result in results)

    headers = ["run_name", "status", "files", "pages", "pps"]
    headers.extend(recall_keys)
    if include_tags:
        headers.append("tags")
    headers.append("artifact_dir")

    rows: list[list[str]] = []
    for result in results:
        row = [
            _format_value(result.get("run_name")),
            "PASS" if bool(result.get("success")) else "FAIL",
            _format_value(_metric_int(result, "files")),
            _format_value(_metric_int(result, "pages")),
            _format_value(_metric_float(result, "pages_per_sec_ingest")),
        ]
        row.extend(_format_value(_metric_float(result, key)) for key in recall_keys)
        if include_tags:
            row.append(_format_value(result.get("tags")))
        row.append(_format_value(result.get("artifact_dir")))
        rows.append(row)

    header_lines = [
        f"Session summary: {summary_path}",
        (
            f"type={_format_value(payload.get('session_type'))} "
            f"all_passed={_format_value(payload.get('all_passed'))} "
            f"runs={len(results)}"
        ),
    ]
    if not rows:
        return "\n".join(header_lines + ["No runs found."])

    right_align = {2, 3, 4}
    right_align.update(idx for idx, header in enumerate(headers) if header.startswith("recall_"))
    return "\n".join(header_lines + ["", _format_table(headers, rows, right_align=right_align)])


def format_session_compare(
    left_path: Path, left_payload: dict[str, Any], right_path: Path, right_payload: dict[str, Any]
) -> str:
    left_results = _load_results(left_payload)
    right_results = _load_results(right_payload)

    left_map = {str(item.get("run_name")): item for item in left_results if item.get("run_name")}
    right_map = {str(item.get("run_name")): item for item in right_results if item.get("run_name")}
    run_names = sorted(set(left_map) | set(right_map))
    recall_keys = _collect_recall_keys(left_results + right_results)

    headers = ["run_name", "left_status", "right_status", "left_pps", "right_pps", "delta_pps"]
    headers.extend(f"delta_{key}" for key in recall_keys)

    rows: list[list[str]] = []
    for run_name in run_names:
        left_result = left_map.get(run_name)
        right_result = right_map.get(run_name)
        left_pps = _metric_float(left_result or {}, "pages_per_sec_ingest")
        right_pps = _metric_float(right_result or {}, "pages_per_sec_ingest")

        row = [
            run_name,
            "PASS" if left_result and bool(left_result.get("success")) else ("FAIL" if left_result else "MISSING"),
            "PASS" if right_result and bool(right_result.get("success")) else ("FAIL" if right_result else "MISSING"),
            _format_value(left_pps),
            _format_value(right_pps),
            _format_delta(left_pps, right_pps),
        ]
        for key in recall_keys:
            row.append(_format_delta(_metric_float(left_result or {}, key), _metric_float(right_result or {}, key)))
        rows.append(row)

    header_lines = [
        f"Compare: {left_path} -> {right_path}",
        f"runs={len(run_names)}",
    ]
    if not rows:
        return "\n".join(header_lines + ["No comparable runs found."])

    right_align = {3, 4, 5}
    right_align.update(idx for idx, header in enumerate(headers) if header.startswith("delta_recall_"))
    return "\n".join(header_lines + ["", _format_table(headers, rows, right_align=right_align)])


def summary_command(
    session: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Session directory or session_summary.json.",
    )
) -> None:
    try:
        summary_path, payload = _load_session_summary(session)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(format_session_summary(summary_path, payload))


def compare_command(
    left: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Left session directory or session_summary.json.",
    ),
    right: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Right session directory or session_summary.json.",
    ),
) -> None:
    try:
        left_path, left_payload = _load_session_summary(left)
        right_path, right_payload = _load_session_summary(right)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(format_session_compare(left_path, left_payload, right_path, right_payload))
