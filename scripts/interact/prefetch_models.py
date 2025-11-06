#!/usr/bin/env python3
"""
Utility to pre-download Hugging Face model repositories into the same cache layout
used by the vLLM-backed Nemotron VLM actor.

- Populates HF caches under a chosen cache directory (default: /workspace/models)
- Safe to run multiple times; only missing blobs are fetched
- Helps eliminate first-run timeouts when launching the vLLM server

Typical usage (inside the ray-gpu-worker container):
  python scripts/interact/prefetch_models.py \
    nvidia/Nemotron-Nano-12B-v2-VL-BF16 \
    nvidia/Nemotron-Nano-VL-12B-V2-FP8 \
    nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD \
    --cache-dir /workspace/models --max-workers 10

If a model requires auth, set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) in the environment.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import click


def _require_hf() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as e:
        click.echo(
            "huggingface_hub is not installed in this environment. "
            "Please install it in nv_ingest_runtime: pip install huggingface_hub",
            err=True,
        )
        raise SystemExit(1) from e


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def _set_cache_env(cache_dir: str) -> None:
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_dir, "hf", "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("VLLM_CACHE_DIR", os.path.join(cache_dir, "vllm"))


def _ensure_dirs(cache_dir: str) -> None:
    for sub in ("hf", os.path.join("hf", "hub"), "transformers", "vllm"):
        os.makedirs(os.path.join(cache_dir, sub), exist_ok=True)


def _prefetch_one(
    repo_id: str,
    cache_dir: str,
    revision: Optional[str],
    allow_patterns: Optional[Sequence[str]],
    ignore_patterns: Optional[Sequence[str]],
    max_workers: int,
    token: Optional[str],
) -> str:
    from huggingface_hub import snapshot_download

    target_cache = os.path.join(cache_dir, "hf")
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=target_cache,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_files_only=False,
        token=token,
        max_workers=max_workers,
        resume_download=True,
    )
    return local_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("models", nargs=-1, required=True)
@click.option(
    "--cache-dir",
    type=str,
    default="/workspace/models",
    show_default=True,
    help="Base cache directory shared with vLLM actor",
)
@click.option("--revision", type=str, default=None, help="Optional git revision (branch/tag/commit) for all models")
@click.option(
    "--token", type=str, default=None, help="Hugging Face token (falls back to HUGGINGFACE_HUB_TOKEN/HF_TOKEN)"
)
@click.option("--max-workers", type=int, default=8, show_default=True, help="Parallel download workers")
@click.option("--allow", "allow_patterns", multiple=True, help="Glob patterns to include (can be repeated)")
@click.option("--ignore", "ignore_patterns", multiple=True, help="Glob patterns to exclude (can be repeated)")
@click.option(
    "--print-paths", is_flag=True, default=False, show_default=True, help="Print resolved local paths for each model"
)
def main(
    models: Iterable[str],
    cache_dir: str,
    revision: Optional[str],
    token: Optional[str],
    max_workers: int,
    allow_patterns: Sequence[str],
    ignore_patterns: Sequence[str],
    print_paths: bool,
) -> None:
    """Pre-download one or more Hugging Face model repos into the vLLM cache directory."""
    _require_hf()
    _ensure_dirs(cache_dir)
    _set_cache_env(cache_dir)

    resolved_token = _resolve_token(token)

    # Reasonable default allow list for large model repos if the user specifies none.
    # By default, we download everything. You may pass --allow to narrow.
    allow = list(allow_patterns) if allow_patterns else None
    ignore = list(ignore_patterns) if ignore_patterns else None

    for rid in models:
        click.echo(f"Prefetching {rid} -> cache={cache_dir} (revision={revision or 'default'}) ...")
        path = _prefetch_one(
            repo_id=rid,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=allow,
            ignore_patterns=ignore,
            max_workers=max_workers,
            token=resolved_token,
        )
        if print_paths:
            click.echo(f"  local path: {path}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
