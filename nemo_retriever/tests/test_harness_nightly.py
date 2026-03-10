import json
from pathlib import Path

from typer.testing import CliRunner

from nemo_retriever.harness import app as harness_app
from nemo_retriever.harness import nightly as harness_nightly
from nemo_retriever.harness.slack import build_slack_payload, load_replay_report, load_session_report

RUNNER = CliRunner()


def _write_nightly_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "preset: single_gpu",
                "runs:",
                "  - name: jp20_single",
                "    dataset: jp20",
                "slack:",
                "  enabled: true",
                "  title: Test Nightly",
                "  post_artifact_paths: true",
                "  metric_keys:",
                "    - pages",
                "    - ingest_secs",
                "    - pages_per_sec_ingest",
                "    - recall_5",
            ]
        ),
        encoding="utf-8",
    )


def test_build_slack_payload_includes_metrics_and_artifact_paths(tmp_path: Path) -> None:
    session_dir = tmp_path / "nightly_20260306_000000_UTC"
    run_dir = session_dir / "jp20_single"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "success": True,
                "return_code": 0,
                "latest_commit": "abc1234",
                "summary_metrics": {
                    "pages": 801,
                    "ingest_secs": 144.2,
                    "pages_per_sec_ingest": 5.55,
                    "recall_5": 0.9123,
                },
                "run_metadata": {
                    "host": "builder-01",
                    "gpu_count": 2,
                    "cuda_driver": "550.54.15",
                },
                "test_config": {
                    "dataset_label": "jp20",
                    "preset": "single_gpu",
                },
            }
        ),
        encoding="utf-8",
    )
    session_summary = session_dir / "session_summary.json"
    session_summary.write_text(
        json.dumps(
            {
                "session_type": "nightly",
                "timestamp": "20260306_000000_UTC",
                "latest_commit": "abc1234",
                "all_passed": True,
                "results": [
                    {
                        "run_name": "jp20_single",
                        "dataset": "jp20",
                        "preset": "single_gpu",
                        "artifact_dir": str(run_dir),
                        "success": True,
                        "return_code": 0,
                        "failure_reason": None,
                        "metrics": {
                            "pages": 801,
                            "ingest_secs": 144.2,
                            "pages_per_sec_ingest": 5.55,
                            "recall_5": 0.9123,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = load_session_report(session_summary)
    payload = build_slack_payload(
        report,
        {
            "title": "Test Nightly",
            "enabled": True,
            "post_artifact_paths": True,
            "metric_keys": ["pages", "ingest_secs", "pages_per_sec_ingest", "recall_5"],
        },
    )

    assert payload["blocks"][0]["text"]["text"] == "Test Nightly"
    assert payload["blocks"][1]["text"]["text"] == "Session: `nightly_20260306_000000_UTC`"
    table_rows = payload["blocks"][3]["rows"]
    flattened = " | ".join(
        element["elements"][0]["elements"][0]["text"]
        for row in table_rows
        for element in row
        if element.get("elements")
    )
    assert "OVERALL STATUS" in flattened
    assert "pages" in flattened
    assert "801" in flattened
    assert "pages/s" in flattened
    assert "5.55" in flattened
    assert "recall@5" in flattened
    assert "0.912" in flattened
    assert str(run_dir) in flattened
    assert "builder-01" in flattened


def test_cli_nightly_skips_slack_when_webhook_missing(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "nightly.yaml"
    _write_nightly_config(config_path)
    session_dir = tmp_path / "nightly_20260306_010000_UTC"
    session_dir.mkdir()
    summary_path = session_dir / "session_summary.json"

    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        harness_nightly,
        "execute_runs",
        lambda **_kwargs: (
            session_dir,
            [
                {
                    "run_name": "jp20_single",
                    "dataset": "jp20",
                    "preset": "single_gpu",
                    "artifact_dir": str((session_dir / "jp20_single").resolve()),
                    "success": True,
                    "return_code": 0,
                    "failure_reason": None,
                    "metrics": {"pages": 1, "ingest_secs": 1.0, "pages_per_sec_ingest": 1.0, "recall_5": 0.5},
                }
            ],
        ),
    )
    monkeypatch.setattr(harness_nightly, "write_session_summary", lambda *_args, **_kwargs: summary_path)

    result = RUNNER.invoke(harness_app, ["nightly", "--runs-config", str(config_path)])

    assert result.exit_code == 0
    assert "SLACK_WEBHOOK_URL is not set; skipping post." in result.stdout


def test_cli_nightly_replay_posts_run_dir(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "nightly.yaml"
    _write_nightly_config(config_path)
    run_dir = tmp_path / "jp20_single"
    run_dir.mkdir()
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "success": True,
                "return_code": 0,
                "failure_reason": None,
                "latest_commit": "abc1234",
                "summary_metrics": {"pages": 801, "ingest_secs": 144.2, "pages_per_sec_ingest": 5.55, "recall_5": 0.9},
                "run_metadata": {"host": "builder-01", "gpu_count": 1},
                "test_config": {"dataset_label": "jp20", "preset": "single_gpu"},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_post(report, slack_config, *, webhook_url=None):
        captured["report"] = report
        captured["slack_config"] = slack_config
        captured["webhook_url"] = webhook_url
        return {}

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/services/example")
    monkeypatch.setattr(harness_nightly, "post_report_to_slack", _fake_post)

    result = RUNNER.invoke(
        harness_app,
        ["nightly", "--runs-config", str(config_path), "--replay", str(run_dir)],
    )

    assert result.exit_code == 0
    replay_report = captured["report"]
    assert replay_report.session_type == "nightly_replay"
    assert replay_report.results[0].dataset == "jp20"
    assert replay_report.results[0].artifact_dir == run_dir.resolve()
    assert captured["slack_config"]["title"] == "Test Nightly"
    assert captured["webhook_url"] == "https://hooks.slack.test/services/example"


def test_load_replay_report_accepts_run_results_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "bo20_single"
    run_dir.mkdir()
    results_path = run_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "success": False,
                "return_code": 98,
                "failure_reason": "missing_recall_metrics",
                "latest_commit": "abc1234",
                "summary_metrics": {"pages": 42, "ingest_secs": 12.0, "pages_per_sec_ingest": 3.5},
                "metrics": {"recall_recall_5": 0.81},
                "test_config": {"dataset_label": "bo20", "preset": "single_gpu"},
            }
        ),
        encoding="utf-8",
    )

    report = load_replay_report([results_path])

    assert report.results[0].run_name == "bo20_single"
    assert report.results[0].success is False
    assert report.results[0].failure_reason == "missing_recall_metrics"
    assert report.results[0].metrics["pages"] == 42
