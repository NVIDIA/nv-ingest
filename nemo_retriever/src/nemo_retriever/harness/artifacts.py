# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEMO_RETRIEVER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_ROOT = NEMO_RETRIEVER_ROOT / "artifacts"


def now_timestr() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def last_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(NEMO_RETRIEVER_ROOT.parent),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    if result.returncode != 0:
        return "unknown"
    return (result.stdout or "").strip() or "unknown"


def get_artifacts_root(base_dir: str | None = None) -> Path:
    if base_dir:
        return Path(base_dir).expanduser().resolve()
    return DEFAULT_ARTIFACTS_ROOT


def create_run_artifact_dir(dataset_label: str, run_name: str | None = None, base_dir: str | None = None) -> Path:
    root = get_artifacts_root(base_dir)
    label = run_name or dataset_label or "run"
    out_dir = root / f"{label}_{now_timestr()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def create_session_dir(prefix: str, base_dir: str | None = None) -> Path:
    root = get_artifacts_root(base_dir)
    session_dir = root / f"{prefix}_{now_timestr()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_session_summary(
    session_dir: Path,
    run_results: list[dict[str, Any]],
    *,
    session_type: str,
    config_path: str,
) -> Path:
    payload = {
        "session_type": session_type,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "config_path": config_path,
        "all_passed": all(bool(item.get("success")) for item in run_results),
        "results": run_results,
    }
    out_path = session_dir / "session_summary.json"
    write_json(out_path, payload)
    return out_path
