# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Shared session management utilities for harness CLIs."""

import json
from pathlib import Path

from nv_ingest_harness.utils.cases import now_timestr, last_commit

__all__ = [
    "create_session_dir",
    "get_artifact_path",
    "write_session_summary",
    "get_default_artifacts_root",
]


def get_default_artifacts_root() -> Path:
    """Get the default artifacts root directory."""
    return Path(__file__).resolve().parents[3] / "artifacts"


def create_session_dir(session_name: str, base_dir: Path | None = None) -> Path:
    """
    Create a session directory for grouping related runs.

    Args:
        session_name: Name for the session (e.g., 'nightly_20260112_120000_UTC')
        base_dir: Base directory for artifacts. If None, uses default artifacts root.

    Returns:
        Path to the created session directory.
    """
    artifacts_root = base_dir or get_default_artifacts_root()
    session_dir = artifacts_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_artifact_path(
    session_dir: Path | str | None,
    dataset_name: str,
    base_dir: Path | str | None = None,
) -> Path:
    """
    Get artifact path for a dataset.

    If session_dir is provided, creates path within the session.
    Otherwise, creates a timestamped directory in base_dir.

    Args:
        session_dir: Session directory path, or None for standalone artifacts.
        dataset_name: Name of the dataset (e.g., 'bo767').
        base_dir: Base directory for artifacts when not using session_dir.

    Returns:
        Path to the artifact directory (created if needed).
    """
    dataset_name = dataset_name or "unknown"

    if session_dir:
        path = Path(session_dir) / dataset_name
    else:
        root = Path(base_dir) if base_dir else get_default_artifacts_root()
        timestamp = now_timestr()
        dirname = f"{dataset_name}_{timestamp}" if dataset_name else timestamp
        path = root / dirname

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_session_summary(
    session_dir: Path | str,
    session_name: str,
    results: list[dict],
    **extras,
) -> Path:
    """
    Write session summary with base schema + optional extensions.

    Base schema:
    {
        "session_name": str,
        "timestamp": str,
        "latest_commit": str,
        "results": list[dict],
        "all_passed": bool,
    }

    Additional fields can be passed via **extras (e.g., case, environment, config_file).

    Args:
        session_dir: Path to the session directory.
        session_name: Name of the session.
        results: List of result dictionaries for each dataset run.
        **extras: Additional fields to include in the summary.

    Returns:
        Path to the written session_summary.json file.
    """
    session_dir = Path(session_dir)

    # Determine all_passed based on result structure
    # Support both run.py style (rc) and nightly.py style (success)
    def is_passed(r: dict) -> bool:
        if "success" in r:
            return r["success"]
        if "rc" in r:
            return r["rc"] == 0
        return r.get("status") == "success"

    # Build base schema
    summary = {
        "session_name": session_name,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "results": results,
        "all_passed": all(is_passed(r) for r in results),
    }

    # Add extension fields
    summary.update(extras)

    summary_path = session_dir / "session_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path
