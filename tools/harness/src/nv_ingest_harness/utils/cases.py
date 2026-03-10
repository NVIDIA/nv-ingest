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


def default_collection_name(config=None) -> str:
    """
    Derive collection name from config or fallback to ENV vars.

    Returns deterministic collection name (no timestamp) to enable:
    - Reusing collections across test runs
    - Running recall evaluation after e2e ingestion
    - Consistent collection naming with recall patterns

    Args:
        config: TestConfig object (preferred). If None, falls back to ENV vars.

    Returns:
        Collection name string (e.g., 'bo767', 'earnings_consulting')
    """
    if config:
        # Use config object (preferred)
        dataset_name = os.path.basename(config.dataset_dir.rstrip("/"))
        test_name = config.test_name or dataset_name
    else:
        # Fallback to ENV vars for backward compatibility
        dataset_dir = os.getenv("DATASET_DIR", "")
        if dataset_dir:
            dataset_name = os.path.basename(dataset_dir.rstrip("/"))
        else:
            dataset_name = "e2e"
        test_name = os.getenv("TEST_NAME", dataset_name)

    # Return deterministic name (no timestamp) for consistency with recall patterns
    return test_name


def last_commit() -> str:
    """Returns the commit hash of the last commit"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
