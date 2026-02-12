from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import typer
from rich.console import Console

from retriever.ingest_config import load_ingest_config_section

from nv_ingest_api.internal.extract.image.table_extractor import extract_table_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

from retriever.table.config import load_table_extractor_schema_from_dict

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Table extraction: enrich table primitives with OCR/structure-derived table_content.")


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


@traceable_func(trace_name="retriever::table_extraction")
def extract_table_data_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    extractor_config: TableExtractorSchema,
    task_config: Optional[Dict[str, Any]] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich table primitives in-place by running OCR / YOLOX and writing table_content."""
    _validate_primitives_df(df_primitives)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = extract_table_data_from_image_internal(
            df_extraction_ledger=df_primitives,
            task_config=task_config,
            extraction_config=extractor_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Table extraction failed")
        raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info


def _read_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".jsonl", ".json"}:
        # jsonl: one object per line; json: list[object] or object
        #
        # Special-case: allow reading PDF stage sidecar JSON payloads like:
        #   { ..., "extracted_df_records": [ {"metadata": ...}, ... ] }
        # so downstream stages can consume `<pdf>.pdf_extraction.json` directly.
        txt = path.read_text(encoding="utf-8")
        if suf == ".jsonl":
            recs = [json.loads(line) for line in txt.splitlines() if line.strip()]
        else:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                for k in ("extracted_df_records", "primitives"):
                    v = obj.get(k)
                    if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                        recs = v
                        break
                else:
                    recs = [obj]
            else:
                recs = obj if isinstance(obj, list) else [obj]

        # If jsonl contains exactly one wrapper payload, unwrap it too.
        if (
            isinstance(recs, list)
            and len(recs) == 1
            and isinstance(recs[0], dict)
            and not ("metadata" in recs[0] and "document_type" in recs[0])
        ):
            one = recs[0]
            for k in ("extracted_df_records", "primitives"):
                v = one.get(k)
                if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                    recs = v
                    break
        return pd.DataFrame(recs)
    raise typer.BadParameter(f"Unsupported input format: {path} (expected .parquet, .jsonl, or .json)")


def _write_df(df: pd.DataFrame, path: Path) -> None:
    suf = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suf == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suf == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    if suf == ".json":
        path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    raise typer.BadParameter(f"Unsupported output format: {path} (expected .parquet, .jsonl, or .json)")


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Input primitives as a DataFrame file (.parquet, .jsonl, or .json). Must include a 'metadata' column.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path (.parquet, .jsonl, or .json). Defaults to <input>.+table<ext>.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Uses section: table."
        ),
    ),
) -> None:
    """
    Load a primitives DataFrame, run table enrichment, and write the enriched DataFrame.
    """
    df = _read_df(input_path)
    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="table")
    schema = load_table_extractor_schema_from_dict(cfg_dict)

    out_df, _info = extract_table_data_from_primitives_df(df, extractor_config=schema, task_config={})

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".table" + input_path.suffix)

    _write_df(out_df, output_path)
    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
