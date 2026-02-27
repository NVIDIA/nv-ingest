# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from nemo_retriever.vector_store.lancedb_store import LanceDBConfig, write_text_embeddings_dir_to_lancedb

console = Console()
app = typer.Typer(help="Vector store stage: upload stage5 embeddings to a vector DB (LanceDB).")


@app.command()
def run(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing `*.text_embeddings.json` files (from `retriever local stage5`).",
    ),
    recursive: bool = typer.Option(False, "--recursive/--no-recursive", help="Scan subdirectories too."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Optionally limit number of input files."),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI (directory path)."),
    table_name: str = typer.Option("nv-ingest", "--table-name", help="LanceDB table name."),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--append",
        help="Overwrite table (default) or append to existing.",
    ),
    create_index: bool = typer.Option(
        True,
        "--create-index/--no-create-index",
        help="Create a vector index after upload.",
    ),
    index_type: str = typer.Option("IVF_HNSW_SQ", "--index-type", help="LanceDB index type."),
    metric: str = typer.Option("l2", "--metric", help="Distance metric for the index."),
    num_partitions: int = typer.Option(16, "--num-partitions", min=1, help="Index partitions."),
    num_sub_vectors: int = typer.Option(256, "--num-sub-vectors", min=1, help="Index sub-vectors."),
) -> None:
    """
    Upload embeddings from `*.text_embeddings.json` into LanceDB.

    Each stored row includes:
      - `vector`: the embedding
      - `pdf_basename`: source filename stem
      - `page_number`: page number from `metadata.content_metadata.page_number`
      - `path` / `source_id`: source identifiers
    """
    cfg = LanceDBConfig(
        uri=str(lancedb_uri),
        table_name=str(table_name),
        overwrite=bool(overwrite),
        create_index=bool(create_index),
        index_type=str(index_type),
        metric=str(metric),
        num_partitions=int(num_partitions),
        num_sub_vectors=int(num_sub_vectors),
    )

    info = write_text_embeddings_dir_to_lancedb(
        Path(input_dir),
        cfg=cfg,
        recursive=bool(recursive),
        limit=limit,
    )
    console.print(
        f"[green]Done[/green] files={info['n_files']} processed={info['processed']} skipped={info['skipped']} "
        f"failed={info['failed']} lancedb_uri={cfg.uri} table={cfg.table_name}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
