# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.traceback import install
from tqdm import tqdm

from retriever.ingest_config import load_ingest_config_section
from retriever.text_embed.processor import embed_text_from_primitives_df, maybe_inject_local_hf_embedder

logger = logging.getLogger(__name__)
console = Console()
install(show_locals=False)

app = typer.Typer(
    help="Text embedding stage: generate embeddings for primitives (and optional helpers for local runs)."
)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as file_handle:
        json.dump(obj, file_handle, ensure_ascii=False, indent=2)
        file_handle.flush()
        try:
            import os

            os.fsync(file_handle.fileno())
        except Exception:
            pass
    tmp.replace(path)


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    return str(obj)


def _embedding_input_path_for_input_json(input_json: Path, *, output_dir: Optional[Path]) -> Path:
    out_name = input_json.with_suffix(".embedding_input.txt").name
    return (output_dir / out_name) if output_dir is not None else input_json.with_suffix(".embedding_input.txt")


def _embeddings_out_path_for_input_json(input_json: Path, *, output_dir: Optional[Path]) -> Path:
    out_name = input_json.with_suffix(".text_embeddings.json").name
    return (output_dir / out_name) if output_dir is not None else input_json.with_suffix(".text_embeddings.json")


@app.command()
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    pattern: str = typer.Option(
        "*.pdf_extraction.infographic.table.json",
        "--pattern",
        help="Glob pattern for input JSON files (non-recursive unless --recursive is set).",
    ),
    recursive: bool = typer.Option(False, "--recursive/--no-recursive", help="Scan subdirectories for inputs too."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="If set, write outputs into this directory (instead of alongside inputs).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Uses section: embedding."
        ),
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key override for embedding service."),
    endpoint_url: Optional[str] = typer.Option(
        None,
        "--endpoint-url",
        help=(
            "Optional embedding service endpoint override (e.g. 'http://embedding:8000/v1'). "
            "Use 'none' to disable remote embeddings and force local HF fallback."
        ),
    ),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Optional embedding model name override."),
    dimensions: Optional[int] = typer.Option(None, "--dimensions", help="Optional embedding dimensions override."),
    use_local_hf_if_no_endpoint: bool = typer.Option(
        True,
        "--local-hf-fallback/--no-local-hf-fallback",
        help="If no embedding endpoint is configured, run local HuggingFace embeddings instead.",
    ),
    local_hf_device: Optional[str] = typer.Option(
        None,
        "--local-hf-device",
        help="Device for local HF embeddings (e.g. 'cuda', 'cpu', 'cuda:0').",
    ),
    local_hf_cache_dir: Optional[Path] = typer.Option(
        None,
        "--local-hf-cache-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional HuggingFace cache directory for local embeddings.",
    ),
    local_hf_batch_size: int = typer.Option(
        64,
        "--local-hf-batch-size",
        min=1,
        help="Batch size for local HF embedding inference.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Optionally limit number of input files."),
    write_embedding_input: bool = typer.Option(
        True,
        "--write-embedding-input/--no-write-embedding-input",
        help="Write *.embedding_input.txt containing the exact combined text used for embedding.",
    ),
) -> None:
    # Local import to avoid a circular import at module load.
    from retriever.text_embed.config import load_text_embedding_schema_from_dict

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="embedding")
    schema = load_text_embedding_schema_from_dict(cfg_dict)

    # Task-config overrides (takes precedence over schema defaults).
    task_cfg: Dict[str, Any] = {}
    if api_key is not None:
        task_cfg["api_key"] = api_key
    if endpoint_url is not None:
        value = endpoint_url.strip()
        task_cfg["endpoint_url"] = None if value.lower() in ("", "none", "null") else value
    elif not cfg_dict.get("embedding_nim_endpoint"):
        # No CLI endpoint and no config-file endpoint — override the schema default
        # so maybe_inject_local_hf_embedder() will inject the local HF fallback.
        task_cfg["endpoint_url"] = None
    if model_name is not None:
        task_cfg["model_name"] = model_name
    if dimensions is not None:
        task_cfg["dimensions"] = int(dimensions)
    task_cfg["use_local_hf_if_no_endpoint"] = bool(use_local_hf_if_no_endpoint)
    if local_hf_device is not None:
        task_cfg["local_hf_device"] = str(local_hf_device)
    if local_hf_cache_dir is not None:
        task_cfg["local_hf_cache_dir"] = str(local_hf_cache_dir)
    if local_hf_batch_size is not None:
        task_cfg["local_hf_batch_size"] = int(local_hf_batch_size)

    inputs = sorted((input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)))
    if limit is not None:
        inputs = inputs[: int(limit)]

    if not inputs:
        console.print(f"[red]No input files found[/red] input_dir={input_dir} pattern={pattern} recursive={recursive}")
        raise typer.Exit(code=2)

    processed = 0
    skipped = 0
    failed = 0

    console.print(
        f"[bold cyan]Text embeddings[/bold cyan] inputs={len(inputs)} input_dir={input_dir} recursive={recursive} "
        f"output_dir={output_dir if output_dir is not None else '(alongside inputs)'}"
    )

    # Pre-warm: inject the local HF embedder (if applicable) into task_cfg ONCE,
    # so per-file copies inside embed_text_from_primitives_df reuse it instead
    # of re-loading the model for every file.
    console.print("[bold yellow]Loading embedding model...[/bold yellow]")
    maybe_inject_local_hf_embedder(task_cfg, schema)
    console.print("[bold green]Embedding model ready.[/bold green]")

    for input_path in tqdm(inputs, desc="Text embeddings", unit="json"):
        out_txt = _embedding_input_path_for_input_json(input_path, output_dir=output_dir)
        out_json = _embeddings_out_path_for_input_json(input_path, output_dir=output_dir)

        if out_json.exists() and not overwrite:
            skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            df_in = pd.read_json(input_path)

            df_out, info = embed_text_from_primitives_df(
                df_in,
                transform_config=schema,
                task_config=task_cfg,
            )

            payload: Dict[str, Any] = {
                "schema_version": 1,
                "stage": "text_embedding",
                "input_json": str(input_path),
                "pattern": str(pattern),
                "embedding_input_txt": str(out_txt) if write_embedding_input else None,
                "outputs": {"text_embeddings_json": str(out_json)},
                "df_records": _to_jsonable(df_out.to_dict(orient="records")),
                "info": _to_jsonable(info),
                "timing": {"seconds": float(time.perf_counter() - t0)},
            }
            _atomic_write_json(out_json, payload)
            processed += 1
        except Exception as e:
            failed += 1
            logger.exception("Text embedding failed for %s", input_path)
            console.print(f"[red]Failed[/red] input={input_path} err={e}")

    console.print(f"[green]Done[/green] processed={processed} skipped={skipped} failed={failed}")


def main() -> None:
    app()
