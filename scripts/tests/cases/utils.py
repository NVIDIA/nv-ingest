from datetime import datetime, timezone
from pathlib import Path
import subprocess
import os

__all__ = [
    "now_timestr",
    "get_repo_root",
    "default_collection_name",
    "last_commit",
]


def now_timestr() -> str:
    """Returns UTC timestamp"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%Z")


def get_repo_root() -> str:
    """Get the root dir of the git repo a script belongs to"""

    start = Path.cwd()
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return root
    except subprocess.CalledProcessError:
        pass

    # Fallback: walk up looking for a .git directory/file
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return str(parent)

    raise RuntimeError(f"Not inside a Git repository starting from {start}")


def default_collection_name() -> str:
    """Derive test name from dataset directory or use explicit TEST_NAME"""

    dataset_dir = os.getenv("DATASET_DIR", "")
    if dataset_dir:
        dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    else:
        dataset_name = "e2e"

    test_name = os.getenv("TEST_NAME", dataset_name)
    return f"{test_name}_{now_timestr()}"


def last_commit() -> str:
    """Returns the commit hash of the last commit"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
