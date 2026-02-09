from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import typer
from rich.console import Console
from rich.traceback import install
from tqdm import tqdm

from retriever._local_deps import ensure_nv_ingest_api_importable
from retriever.ingest_config import load_ingest_config_section

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

from retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_to_lancedb

logger = logging.getLogger(__name__)
console = Console()
install(show_locals=False)

app = typer.Typer(
    help="Text embedding stage: generate embeddings for primitives (and optional helpers for local runs)."
)


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


@traceable_func(trace_name="retriever::text_embedding")
def embed_text_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    transform_config: TextEmbeddingSchema,
    task_config: Optional[Dict[str, Any]] = None,
    lancedb: Optional[LanceDBConfig] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate embeddings for supported content types and write to metadata."""
    _validate_primitives_df(df_primitives)

    if task_config is None:
        task_config = {}
    else:
        # Avoid mutating the caller's dict (we may inject a local embedder callable).
        task_config = dict(task_config)

    # Auto-fallback: if no embedding endpoint is configured, inject a local HF embedder callable.
    _maybe_inject_local_hf_embedder(task_config, transform_config)

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = transform_create_text_embeddings_internal(
            df_primitives,
            task_config=task_config,
            transform_config=transform_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Text embedding failed")
        raise

    if lancedb is not None:
        try:
            write_embeddings_to_lancedb(out_df, cfg=lancedb)
        except Exception:
            logger.exception("Failed writing embeddings to LanceDB")
            raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info


def _maybe_inject_local_hf_embedder(task_config: Dict[str, Any], transform_config: TextEmbeddingSchema) -> None:
    """
    If no remote embedding endpoint is configured, inject a local HF embedder into task_config.

    This keeps the DataFrame embedding logic centralized in `nv_ingest_api.internal.transform.embed_text`
    while allowing retriever-local runs to operate without an embedding microservice.
    """
    # Respect explicit caller-provided embedder.
    if callable(task_config.get("embedder")):
        return

    # Resolve endpoint_url with explicit None override support.
    if "endpoint_url" in task_config:
        endpoint_url = task_config.get("endpoint_url")
    else:
        endpoint_url = getattr(transform_config, "embedding_nim_endpoint", None)

    endpoint_url = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
    has_endpoint = bool(endpoint_url)

    use_local = bool(task_config.get("use_local_hf_if_no_endpoint", True))
    if has_endpoint or not use_local:
        return

    # Lazy import: only load torch/HF when we truly need local embeddings.
    from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

    local_device = task_config.get("local_hf_device")
    local_cache_dir = task_config.get("local_hf_cache_dir")
    local_batch_size = int(task_config.get("local_hf_batch_size") or 64)

    embedder = LlamaNemotronEmbed1BV2Embedder(device=local_device, hf_cache_dir=local_cache_dir, normalize=True)

    def _embed(texts):
        vecs = embedder.embed(texts, batch_size=local_batch_size)
        return vecs.tolist()

    # Force the API transform to use the callable path by explicitly overriding endpoint_url to None.
    task_config["endpoint_url"] = None
    task_config["embedder"] = _embed
    task_config["local_batch_size"] = local_batch_size


# --------------------------------------------------------------------------------------
# CLI helper: embed text from local JSON sidecars
#
# This is used by `retriever local stage5 ...` via a thin proxy wrapper.
# --------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    tmp.replace(path)


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

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
        v = endpoint_url.strip()
        task_cfg["endpoint_url"] = None if v.lower() in ("", "none", "null") else v
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

    for in_path in tqdm(inputs, desc="Text embeddings", unit="json"):
        out_txt = _embedding_input_path_for_input_json(in_path, output_dir=output_dir)
        out_json = _embeddings_out_path_for_input_json(in_path, output_dir=output_dir)

        if out_json.exists() and not overwrite:
            skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            df_in = pd.read_json(in_path)

            df_out, info = embed_text_from_primitives_df(
                df_in,
                transform_config=schema,
                task_config=task_cfg,
            )

            payload: Dict[str, Any] = {
                "schema_version": 1,
                "stage": "text_embedding",
                "input_json": str(in_path),
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
            logger.exception("Text embedding failed for %s", in_path)
            console.print(f"[red]Failed[/red] input={in_path} err={e}")

    console.print(f"[green]Done[/green] processed={processed} skipped={skipped} failed={failed}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
