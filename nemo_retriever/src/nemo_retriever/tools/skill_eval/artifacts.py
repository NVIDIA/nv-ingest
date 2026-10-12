# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEMO_RETRIEVER_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACTS_ROOT = NEMO_RETRIEVER_ROOT / "artifacts"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


def now_timestr() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def _normalize_commit(value: str | None) -> str | None:
    text = (value or "").strip()
    if not _COMMIT_RE.match(text):
        return None
    return text.lower()


def _resolve_git_dir(repo_root: Path) -> Path | None:
    dot_git = repo_root / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    try:
        raw = dot_git.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw.startswith("gitdir:"):
        return None
    gitdir_text = raw.split(":", 1)[1].strip()
    git_dir = Path(gitdir_text).expanduser()
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    return git_dir


def _read_packed_ref(git_dir: Path, ref_name: str) -> str | None:
    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return None
    try:
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            commit, _sep, ref = line.partition(" ")
            if ref.strip() == ref_name:
                normalized = _normalize_commit(commit)
                if normalized is not None:
                    return normalized
    except Exception:
        return None
    return None


def _read_head_commit(repo_root: Path) -> str | None:
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    try:
        head_value = head_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if head_value.startswith("ref:"):
        ref_name = head_value.split(":", 1)[1].strip()
        ref_path = git_dir / ref_name
        if ref_path.exists():
            try:
                normalized = _normalize_commit(ref_path.read_text(encoding="utf-8"))
                if normalized is not None:
                    return normalized
            except Exception:
                pass
        return _read_packed_ref(git_dir, ref_name)

    return _normalize_commit(head_value)


def last_commit() -> str:
    repo_root = NEMO_RETRIEVER_ROOT.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        result = None

    if result is not None and result.returncode == 0:
        normalized = _normalize_commit(result.stdout)
        if normalized is not None:
            return normalized

    fallback = _read_head_commit(repo_root)
    if fallback is not None:
        return fallback
    return "unknown"


def get_artifacts_root(base_dir: str | None = None) -> Path:
    if base_dir:
        return Path(base_dir).expanduser().resolve()
    return DEFAULT_ARTIFACTS_ROOT


def create_session_dir(prefix: str, base_dir: str | None = None) -> Path:
    root = get_artifacts_root(base_dir)
    session_dir = root / f"{prefix}_{now_timestr()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        rendered = json.dumps(jsonable(payload), indent=2, sort_keys=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
        assert temporary_path is not None
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_session_summary(
    session_dir: Path,
    run_results: list[dict[str, Any]],
    *,
    session_type: str,
    config_path: str,
    run_commit: str | None = None,
) -> Path:
    payload = {
        "session_type": session_type,
        "timestamp": now_timestr(),
        # ``run_commit`` is the commit the evaluation actually executed with
        # (captured at run start); ``latest_commit`` is HEAD at summary-write
        # time, which can differ when a summary is regenerated later.
        "run_commit": run_commit or last_commit(),
        "latest_commit": last_commit(),
        "config_path": config_path,
        "all_passed": all(bool(item.get("success")) for item in run_results),
        "results": run_results,
    }
    out_path = session_dir / "session_summary.json"
    write_json(out_path, payload)
    return out_path
