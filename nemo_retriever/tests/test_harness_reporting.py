# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from typer.testing import CliRunner

from nemo_retriever.harness.cli import app as harness_app

RUNNER = CliRunner()


def _write_session_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_summary_command_accepts_session_dir_and_prints_tags(tmp_path: Path) -> None:
    session_dir = tmp_path / "nightly_20260305"
    summary_path = session_dir / "session_summary.json"
    _write_session_summary(
        summary_path,
        {
            "session_type": "nightly",
            "all_passed": True,
            "results": [
                {
                    "run_name": "jp20_single",
                    "success": True,
                    "metrics": {
                        "files": 20,
                        "pages": 3181,
                        "pages_per_sec_ingest": 12.5,
                        "recall_5": 0.9,
                    },
                    "artifact_dir": "/tmp/jp20_single",
                    "tags": ["nightly", "candidate"],
                }
            ],
        },
    )

    result = RUNNER.invoke(harness_app, ["summary", str(session_dir)])

    assert result.exit_code == 0
    assert "Session summary:" in result.output
    assert "jp20_single" in result.output
    assert "nightly,candidate" in result.output
    assert "recall_5" in result.output


def test_compare_command_prints_deltas_and_missing_runs(tmp_path: Path) -> None:
    left_summary = tmp_path / "left" / "session_summary.json"
    right_summary = tmp_path / "right" / "session_summary.json"
    _write_session_summary(
        left_summary,
        {
            "session_type": "nightly",
            "all_passed": True,
            "results": [
                {
                    "run_name": "bo20_single",
                    "success": True,
                    "metrics": {"pages_per_sec_ingest": 5.0},
                    "artifact_dir": "/tmp/bo20_left",
                },
                {
                    "run_name": "jp20_single",
                    "success": True,
                    "metrics": {"pages_per_sec_ingest": 10.0, "recall_5": 0.7},
                    "artifact_dir": "/tmp/jp20_left",
                },
            ],
        },
    )
    _write_session_summary(
        right_summary,
        {
            "session_type": "nightly",
            "all_passed": False,
            "results": [
                {
                    "run_name": "jp20_single",
                    "success": False,
                    "metrics": {"pages_per_sec_ingest": 12.0, "recall_5": 0.8},
                    "artifact_dir": "/tmp/jp20_right",
                }
            ],
        },
    )

    result = RUNNER.invoke(harness_app, ["compare", str(left_summary), str(right_summary)])

    assert result.exit_code == 0
    assert "delta_pps" in result.output
    assert "delta_recall_5" in result.output
    assert "bo20_single" in result.output
    assert "MISSING" in result.output
    assert "jp20_single" in result.output
    assert "+2" in result.output
