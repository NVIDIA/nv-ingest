from __future__ import annotations

"""
Local pipeline stage6: generate text embeddings and upload to LanceDB.

This stage is intentionally a thin wrapper around `retriever.text_embed.stage`.

It runs the embedding stage (same behavior/flags), then ingests the produced
`*.text_embeddings.json` artifacts into LanceDB by reading `df_records[*].metadata.embedding`
and writing it to the LanceDB `vector` column.
"""

from pathlib import Path
from typing import Optional

import typer

from retriever.text_embed import stage as text_embed_stage
from retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_json_to_lancedb

app = typer.Typer(help="Stage 6: run text embeddings and upload vectors to LanceDB for recall.")


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
    # LanceDB upload options
    lancedb_uri: str = typer.Option(..., "--lancedb-uri", help="LanceDB URI (e.g. './lancedb')."),
    lancedb_table: str = typer.Option("embeddings", "--lancedb-table", help="LanceDB table name."),
    lancedb_mode: str = typer.Option(
        "append",
        "--lancedb-mode",
        help="Write mode: append|overwrite.",
        show_default=True,
        case_sensitive=False,
    ),
) -> None:
    """
    Run the embedding stage over local JSON sidecars and upload resulting vectors to LanceDB.
    """
    # 1) Generate embeddings JSON artifacts (delegate to embedding stage).
    text_embed_stage.run(
        input_dir=input_dir,
        pattern=pattern,
        recursive=recursive,
        output_dir=output_dir,
        config=config,
        api_key=api_key,
        endpoint_url=endpoint_url,
        model_name=model_name,
        dimensions=dimensions,
        overwrite=overwrite,
        limit=limit,
        write_embedding_input=write_embedding_input,
    )

    # 2) Upload produced `*.text_embeddings.json` into LanceDB.
    # The embedding stage's output file name is derived from the input file path stem.
    inputs = sorted((Path(input_dir).rglob(pattern) if recursive else Path(input_dir).glob(pattern)))
    if limit is not None:
        inputs = inputs[: int(limit)]

    mode = str(lancedb_mode).lower().strip()
    if mode not in {"append", "overwrite"}:
        raise typer.BadParameter("--lancedb-mode must be 'append' or 'overwrite'")

    cfg = LanceDBConfig(uri=str(lancedb_uri), table=str(lancedb_table), mode=mode)

    written_total = 0
    missing = 0
    failed = 0
    for in_path in inputs:
        out_json = text_embed_stage._embeddings_out_path_for_input_json(in_path, output_dir=output_dir)
        if not out_json.exists():
            missing += 1
            continue
        try:
            written_total += int(write_embeddings_json_to_lancedb(out_json, cfg=cfg))
        except Exception:
            failed += 1
            continue

    typer.echo(
        f"lancedb_upload: inputs={len(inputs)} wrote_rows={written_total} missing_outputs={missing} failed={failed} "
        f"uri={lancedb_uri} table={lancedb_table} mode={mode}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

