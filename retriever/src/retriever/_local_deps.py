from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_nv_ingest_api_importable() -> None:
    """Make `nv_ingest_api` importable in a monorepo checkout.

    In CI / production, `nv-ingest-api` should be installed as a normal dependency.
    During local development inside this repo, it may not be installed yet, but the
    sources are available at `../../api/src` relative to this file.

    By default in this monorepo, we prefer the in-tree `api/src` over any installed
    `nv-ingest-api` wheel so fixes can be iterated quickly without reinstalling.
    Set `RETRIEVER_PREFER_INSTALLED_NV_INGEST_API=1` to disable this behavior.
    """
    repo_root = Path(__file__).resolve().parents[3]  # retriever/src/retriever -> repo root
    api_src = repo_root / "api" / "src"
    prefer_installed = os.getenv("RETRIEVER_PREFER_INSTALLED_NV_INGEST_API", "").strip() in {"1", "true", "yes"}
    if api_src.is_dir() and not prefer_installed:
        sys.path.insert(0, str(api_src))

    try:
        import nv_ingest_api  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    if api_src.is_dir():
        sys.path.insert(0, str(api_src))

