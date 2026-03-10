from pathlib import Path

from nv_ingest_harness.utils.cases import get_repo_root


def get_lancedb_path(config, collection_name: str) -> str:
    base_dir = config.lancedb_dir
    if base_dir is None:
        base_dir = Path(get_repo_root()) / "tools" / "harness" / "lancedb"
    else:
        base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / collection_name)
