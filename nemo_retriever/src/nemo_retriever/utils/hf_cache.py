from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ENV_HF_CACHE_BASE_DIR = "NEMO_RETRIEVER_HF_CACHE_DIR"


def resolve_hf_cache_dir(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Resolve Hugging Face cache dir from explicit arg, env, then default."""
    candidate = explicit_hf_cache_dir or os.getenv(ENV_HF_CACHE_BASE_DIR)
    if candidate:
        return str(Path(candidate).expanduser())
    return str(Path.home() / ".cache" / "huggingface")


def configure_global_hf_cache_base(explicit_hf_cache_dir: Optional[str] = None) -> str:
    """Apply resolved HF cache base to standard Hugging Face env vars."""
    cache_base = resolve_hf_cache_dir(explicit_hf_cache_dir)
    os.environ.setdefault("HF_HOME", cache_base)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_base) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_base) / "transformers"))
    return cache_base
