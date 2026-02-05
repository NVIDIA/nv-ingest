from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.traceback import install
from tqdm import tqdm

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

from retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_to_lancedb

logger = logging.getLogger(__name__)
console = Console()
install(show_locals=False)

app = typer.Typer(help="Text embedding stage: generate embeddings for primitives (and optional helpers for local runs).")


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


# --------------------------------------------------------------------------------------
# CLI helper: embed text from local JSON sidecars
#
# This is used by `retriever local stage5 ...` via a thin proxy wrapper.
# --------------------------------------------------------------------------------------


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise typer.BadParameter(f"Failed reading YAML config {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"YAML config must be a mapping/object at top-level: {path}")
    return data


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


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSON file and return a DataFrame-shaped list of primitive records.

    Supports wrapper payloads with keys like:
      - extracted_df_records
      - primitives
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        raise typer.BadParameter(f"Failed reading JSON {path}: {e}") from e

    if isinstance(obj, dict):
        for k in ("extracted_df_records", "primitives"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    return []


def _extract_metadata_texts(records: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Extract:
      - metadata.table_metadata.table_content
      - metadata.content
    """
    pieces: List[str] = []
    for rec in records:
        meta = rec.get("metadata")
        if not isinstance(meta, dict):
            continue

        table_meta = meta.get("table_metadata")
        if isinstance(table_meta, dict):
            tc = table_meta.get("table_content")
            if isinstance(tc, str) and tc.strip():
                pieces.append(tc.strip())

        c = meta.get("content")
        if isinstance(c, str) and c.strip():
            pieces.append(c.strip())

    # De-dupe consecutive repeats (common when both fields contain same content)
    out: List[str] = []
    prev: Optional[str] = None
    for p in pieces:
        if prev is not None and p == prev:
            continue
        out.append(p)
        prev = p
    return out


def _embedding_input_path_for_input_json(input_json: Path, *, output_dir: Optional[Path]) -> Path:
    out_name = input_json.with_suffix(".embedding_input.txt").name
    return (output_dir / out_name) if output_dir is not None else input_json.with_suffix(".embedding_input.txt")


def _embeddings_out_path_for_input_json(input_json: Path, *, output_dir: Optional[Path]) -> Path:
    out_name = input_json.with_suffix(".text_embeddings.json").name
    return (output_dir / out_name) if output_dir is not None else input_json.with_suffix(".text_embeddings.json")


def _make_embedding_df(
    *,
    combined_text: str,
    source_id: str,
    input_json: Path,
    embedding_input_txt: Optional[Path],
) -> pd.DataFrame:
    # Minimal row satisfying nv-ingest-api's embedding transform expectations:
    # it reads metadata.content and metadata.content_metadata.type.
    metadata: Dict[str, Any] = {
        "content": combined_text,
        "content_metadata": {"type": "text"},
        "source_metadata": {"source_id": source_id},
        "custom_content": {
            "input_json": str(input_json),
            "embedding_input_txt": str(embedding_input_txt) if embedding_input_txt is not None else None,
        },
    }
    return pd.DataFrame([{"document_type": "TEXT", "metadata": metadata}])


@app.command()
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    pattern: str = typer.Option(
        "*.pdf_extraction.table.chart.json",
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
        help="Optional YAML config for TextEmbeddingSchema.",
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key override for embedding service."),
    endpoint_url: Optional[str] = typer.Option(
        None,
        "--endpoint-url",
        help="Optional embedding service endpoint override (e.g. 'http://embedding:8000/v1').",
    ),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Optional embedding model name override."),
    dimensions: Optional[int] = typer.Option(None, "--dimensions", help="Optional embedding dimensions override."),
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

    cfg_dict: Dict[str, Any] = _read_yaml_mapping(config) if config is not None else {}
    schema = load_text_embedding_schema_from_dict(cfg_dict)

    # Task-config overrides (takes precedence over schema defaults).
    task_cfg: Dict[str, Any] = {}
    if api_key is not None:
        task_cfg["api_key"] = api_key
    if endpoint_url is not None:
        task_cfg["endpoint_url"] = endpoint_url
    if model_name is not None:
        task_cfg["model_name"] = model_name
    if dimensions is not None:
        task_cfg["dimensions"] = int(dimensions)

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
            records = _read_json_records(in_path)
            pieces = _extract_metadata_texts(records)
            combined = "\n\n".join(pieces).strip()

            if write_embedding_input:
                out_txt.parent.mkdir(parents=True, exist_ok=True)
                out_txt.write_text((combined + "\n") if combined else "", encoding="utf-8", errors="replace")
                embedding_input_txt: Optional[Path] = out_txt
            else:
                embedding_input_txt = None

            df_in = _make_embedding_df(
                combined_text=combined,
                source_id=str(in_path),
                input_json=in_path,
                embedding_input_txt=embedding_input_txt,
            )

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
                "embedding_input_stats": {"num_pieces": int(len(pieces)), "num_chars": int(len(combined))},
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
