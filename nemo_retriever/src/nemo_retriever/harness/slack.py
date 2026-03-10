from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_retriever.harness.artifacts import now_timestr

DEFAULT_USERNAME = "nemo_retriever Nightly"
DEFAULT_ICON_EMOJI = ":satellite:"
_BLANK_ROW = [
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
]
METRIC_LABELS = {
    "pages_per_sec_ingest": "pages/s",
    "ingest_secs": "ingest_s",
    "pages": "pages",
    "recall_5": "recall@5",
}


@dataclass
class NightlyRunReport:
    run_name: str
    dataset: str
    preset: str | None
    success: bool
    return_code: int | None
    failure_reason: str | None
    artifact_dir: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    latest_commit: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NightlySessionReport:
    session_name: str
    session_dir: Path
    session_type: str
    timestamp: str | None
    latest_commit: str | None
    all_passed: bool
    results: list[NightlyRunReport]


def _normalize_metrics(raw_metrics: Any) -> dict[str, Any]:
    if not isinstance(raw_metrics, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in raw_metrics.items():
        metric_key = str(key).strip()
        if metric_key.startswith("recall_recall_"):
            metric_key = "recall_" + metric_key.removeprefix("recall_recall_")
        normalized[metric_key] = value
    return normalized


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_results_payload(artifact_dir: Path | None) -> dict[str, Any]:
    if artifact_dir is None:
        return {}
    results_path = artifact_dir / "results.json"
    if not results_path.exists():
        return {}
    return _read_json_dict(results_path)


def _load_preferred_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary_metrics = payload.get("summary_metrics")
    if isinstance(summary_metrics, dict) and summary_metrics:
        return _normalize_metrics(summary_metrics)
    return _normalize_metrics(payload.get("metrics", {}))


def _normalize_run_report(summary_entry: dict[str, Any]) -> NightlyRunReport:
    artifact_dir_str = summary_entry.get("artifact_dir")
    artifact_dir = Path(artifact_dir_str).expanduser().resolve() if artifact_dir_str else None
    results_payload = _load_results_payload(artifact_dir)
    test_config = results_payload.get("test_config", {})
    run_metadata = results_payload.get("run_metadata", {})

    metrics = summary_entry.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        metrics = _load_preferred_metrics(results_payload)
    if not isinstance(metrics, dict):
        metrics = {}

    if not isinstance(test_config, dict):
        test_config = {}
    if not isinstance(run_metadata, dict):
        run_metadata = {}

    return NightlyRunReport(
        run_name=str(summary_entry.get("run_name") or (artifact_dir.name if artifact_dir else "unknown_run")),
        dataset=str(summary_entry.get("dataset") or test_config.get("dataset_label") or "unknown_dataset"),
        preset=str(summary_entry.get("preset") or test_config.get("preset")) if (summary_entry.get("preset") or test_config.get("preset")) else None,
        success=bool(summary_entry.get("success")),
        return_code=int(summary_entry["return_code"]) if summary_entry.get("return_code") is not None else None,
        failure_reason=str(summary_entry.get("failure_reason")) if summary_entry.get("failure_reason") else None,
        artifact_dir=artifact_dir,
        metrics=_normalize_metrics(metrics),
        latest_commit=str(results_payload.get("latest_commit")) if results_payload.get("latest_commit") else None,
        run_metadata=dict(run_metadata),
    )


def load_session_report(session_summary_path: Path) -> NightlySessionReport:
    resolved_summary_path = Path(session_summary_path).expanduser().resolve()
    if resolved_summary_path.is_dir():
        resolved_summary_path = resolved_summary_path / "session_summary.json"

    payload = _read_json_dict(resolved_summary_path)
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError(f"'results' must be a list in {resolved_summary_path}")

    return NightlySessionReport(
        session_name=resolved_summary_path.parent.name,
        session_dir=resolved_summary_path.parent,
        session_type=str(payload.get("session_type") or "nightly"),
        timestamp=str(payload.get("timestamp")) if payload.get("timestamp") else None,
        latest_commit=str(payload.get("latest_commit")) if payload.get("latest_commit") else None,
        all_passed=bool(payload.get("all_passed")),
        results=[_normalize_run_report(dict(item)) for item in raw_results if isinstance(item, dict)],
    )


def load_replay_report(replay_paths: list[Path]) -> NightlySessionReport:
    if not replay_paths:
        raise ValueError("At least one replay path is required")

    resolved_paths = [Path(path).expanduser().resolve() for path in replay_paths]
    session_dirs = [path for path in resolved_paths if path.is_dir() and (path / "session_summary.json").exists()]
    if session_dirs:
        if len(resolved_paths) != 1:
            raise ValueError("Replay accepts either one session directory or one or more run directories")
        return load_session_report(session_dirs[0] / "session_summary.json")

    run_reports: list[NightlyRunReport] = []
    latest_commit: str | None = None

    for path in resolved_paths:
        results_path = path / "results.json" if path.is_dir() else path
        if results_path.name != "results.json":
            raise ValueError(f"Replay path must be a run directory, session directory, or results.json file: {path}")
        payload = _read_json_dict(results_path)
        artifact_dir = results_path.parent
        test_config = payload.get("test_config", {})
        if not isinstance(test_config, dict):
            test_config = {}
        run_metadata = payload.get("run_metadata", {})
        if not isinstance(run_metadata, dict):
            run_metadata = {}

        latest_commit = latest_commit or (str(payload.get("latest_commit")) if payload.get("latest_commit") else None)
        run_reports.append(
            NightlyRunReport(
                run_name=artifact_dir.name,
                dataset=str(test_config.get("dataset_label") or artifact_dir.name),
                preset=str(test_config.get("preset")) if test_config.get("preset") else None,
                success=bool(payload.get("success")),
                return_code=int(payload["return_code"]) if payload.get("return_code") is not None else None,
                failure_reason=str(payload.get("failure_reason")) if payload.get("failure_reason") else None,
                artifact_dir=artifact_dir,
                metrics=_load_preferred_metrics(payload),
                latest_commit=str(payload.get("latest_commit")) if payload.get("latest_commit") else None,
                run_metadata=dict(run_metadata),
            )
        )

    session_dir = resolved_paths[0].parent if resolved_paths else Path.cwd()
    return NightlySessionReport(
        session_name=f"replay_{now_timestr()}",
        session_dir=session_dir,
        session_type="nightly_replay",
        timestamp=now_timestr(),
        latest_commit=latest_commit,
        all_passed=all(run.success for run in run_reports),
        results=run_reports,
    )


def _format_metric_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "N/A"

    if metric_name.endswith("_secs") or metric_name.endswith("_time_s") or metric_name == "ingest_secs":
        seconds = float(value)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        formatted = f"{seconds:.2f}s"
        if hours > 0 or minutes > 0:
            formatted += " ("
            if hours > 0:
                formatted += f"{hours}h : "
            formatted += f"{minutes:02}m : {secs:05.2f}s)"
        return formatted
    if metric_name.endswith("_per_sec_ingest"):
        return f"{float(value):.2f}"
    if metric_name.startswith("recall_"):
        return f"{float(value):.3f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _format_metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name)


def _select_metric_names(metrics: dict[str, Any], metric_keys: list[str]) -> list[str]:
    metric_names: list[str] = []
    seen: set[str] = set()

    for key in metric_keys:
        if key in metrics and metrics[key] is not None:
            metric_names.append(key)
            seen.add(key)

    return metric_names


def _two_column_row(left: str, right: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": left}]}],
        },
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
        },
    ]


def _two_column_row_bold(left: str, right: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "rich_text",
            "elements": [
                {"type": "rich_text_section", "elements": [{"type": "text", "text": left, "style": {"bold": True}}]}
            ],
        },
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
        },
    ]


def build_slack_payload(report: NightlySessionReport, slack_config: dict[str, Any]) -> dict[str, Any]:
    metric_keys = [str(key) for key in slack_config.get("metric_keys", [])]
    post_artifact_paths = bool(slack_config.get("post_artifact_paths", True))
    passed_count = sum(1 for run in report.results if run.success)
    total_count = len(report.results)
    overall_status = (
        f"PASS ({passed_count}/{total_count})" if report.all_passed else f"FAIL ({passed_count}/{total_count} passed)"
    )
    first_metadata = next((run.run_metadata for run in report.results if run.run_metadata), {})

    rows: list[list[dict[str, Any]]] = []
    rows.append(_two_column_row_bold("OVERALL STATUS", overall_status))
    for run in report.results:
        run_status = "PASS" if run.success else "FAIL"
        rows.append(_two_column_row_bold(f"-    {run.dataset}", run_status))

    rows.append(_BLANK_ROW)
    rows.append(_two_column_row_bold("ENVIRONMENT", " "))
    rows.append(_two_column_row("-    session", report.session_name))
    rows.append(_two_column_row("-    session_dir", str(report.session_dir)))
    if report.latest_commit:
        rows.append(_two_column_row("-    git_commit", report.latest_commit))
    for key in ["host", "gpu_count", "cuda_driver", "ray_version", "python_version"]:
        if key not in first_metadata or first_metadata[key] is None:
            continue
        rows.append(_two_column_row(f"-    {key}", _format_metric_value(key, first_metadata[key])))

    rows.append(_BLANK_ROW)
    rows.append(_two_column_row_bold("RESULTS", " "))
    for run in report.results:
        run_status = "PASS" if run.success else "FAIL"
        rows.append(_two_column_row_bold(run.dataset, run_status))
        if not run.success and run.return_code is not None:
            rows.append(_two_column_row("-    return_code", str(run.return_code)))
        for metric_name in _select_metric_names(run.metrics, metric_keys):
            rows.append(
                _two_column_row(
                    f"-    {_format_metric_label(metric_name)}",
                    _format_metric_value(metric_name, run.metrics[metric_name]),
                )
            )
        if run.failure_reason:
            rows.append(_two_column_row("-    failure_reason", run.failure_reason))
        if post_artifact_paths and run.artifact_dir is not None:
            rows.append(_two_column_row("-    artifact_dir", str(run.artifact_dir)))
        rows.append(_BLANK_ROW)

    if report.results:
        rows.pop(-1)

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": str(slack_config.get("title") or "nemo_retriever Nightly Harness")},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Session: `{report.session_name}`"},
        },
        {"type": "divider"},
        {"type": "table", "rows": rows},
    ]

    return {
        "username": DEFAULT_USERNAME,
        "icon_emoji": DEFAULT_ICON_EMOJI,
        "blocks": blocks,
    }


def post_slack_payload(payload: dict[str, Any], webhook_url: str) -> None:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests is required for Slack posting") from exc

    response = requests.post(
        webhook_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Slack post failed with status={response.status_code}: {response.text}")


def post_report_to_slack(
    report: NightlySessionReport,
    slack_config: dict[str, Any],
    *,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    effective_webhook = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not effective_webhook:
        raise RuntimeError("SLACK_WEBHOOK_URL is not set")

    payload = build_slack_payload(report, slack_config)
    post_slack_payload(payload, effective_webhook)
    return payload
