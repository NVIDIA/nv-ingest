# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess

import pytest

from nemo_retriever.tools.skill_eval import artifacts


def test_session_summary_preserves_format_and_run_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "now_timestr", lambda: "20261008_120000_UTC")
    monkeypatch.setattr(artifacts, "last_commit", lambda: "latest")
    session = artifacts.create_session_dir("skilleval", str(tmp_path))
    results = [{"success": True, "path": Path("output.txt")}, {"success": False}]

    summary = artifacts.write_session_summary(
        session, results, session_type="skill_eval", config_path="config.yaml", run_commit="executed"
    )

    assert session == tmp_path / "skilleval_20261008_120000_UTC"
    assert summary.name == "session_summary.json"
    assert json.loads(summary.read_text()) == {
        "session_type": "skill_eval",
        "timestamp": "20261008_120000_UTC",
        "run_commit": "executed",
        "latest_commit": "latest",
        "config_path": "config.yaml",
        "all_passed": False,
        "results": [{"success": True, "path": "output.txt"}, {"success": False}],
    }
    assert artifacts.DEFAULT_ARTIFACTS_ROOT == Path(__file__).resolve().parents[2] / "artifacts"


def test_last_commit_preserves_full_sha(monkeypatch):
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    monkeypatch.setattr(
        artifacts.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, sha + "\n")
    )
    assert artifacts.last_commit() == sha


@pytest.mark.parametrize("ref_kind", ["loose", "packed", "detached", "gitfile", "absent"])
def test_last_commit_falls_back_without_git(monkeypatch, tmp_path, ref_kind):
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    monkeypatch.setattr(artifacts, "NEMO_RETRIEVER_ROOT", tmp_path / "nemo_retriever")

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(artifacts.subprocess, "run", unavailable)
    if ref_kind != "absent":
        git_dir = tmp_path / ("git-metadata" if ref_kind == "gitfile" else ".git")
        git_dir.mkdir()
        if ref_kind == "gitfile":
            (tmp_path / ".git").write_text("gitdir: git-metadata\n")
        if ref_kind in {"detached", "gitfile"}:
            (git_dir / "HEAD").write_text(sha)
        else:
            (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
            if ref_kind == "packed":
                (git_dir / "packed-refs").write_text(f"# packed refs\n{sha} refs/heads/main\n")
            else:
                (git_dir / "refs/heads").mkdir(parents=True)
                (git_dir / "refs/heads/main").write_text(sha)
    assert artifacts.last_commit() == ("unknown" if ref_kind == "absent" else sha)


def test_atomic_write_failure_preserves_previous_summary(monkeypatch, tmp_path):
    target = tmp_path / "session_summary.json"
    target.write_text('{"old": true}\n')

    def fail_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        artifacts.write_json(target, {"new": True})
    assert target.read_text() == '{"old": true}\n'
    assert list(tmp_path.glob(".session_summary.json.*.tmp")) == []


def test_serialization_failure_does_not_publish_summary(tmp_path):
    class Unserializable:
        def __str__(self):
            raise ValueError("cannot serialize")

    with pytest.raises(ValueError, match="cannot serialize"):
        artifacts.write_json(tmp_path / "session_summary.json", {"value": Unserializable()})
    assert list(tmp_path.iterdir()) == []
