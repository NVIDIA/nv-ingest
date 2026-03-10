# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console

from nemo_retriever.ingest_config import load_ingest_config_section
from nemo_retriever.io.dataframe import read_dataframe, write_dataframe

from nemo_retriever.infographic.config import load_infographic_extractor_schema_from_dict
from nemo_retriever.infographic.processor import extract_infographic_data_from_primitives_df

app = typer.Typer(help="Infographic extraction: enrich infographic primitives with OCR-derived table_content.")
console = Console()


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
        help="Output path (.parquet, .jsonl, or .json). Defaults to <input>.+infographic<ext>.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Uses section: infographic."
        ),
    ),
) -> None:
    """
    Load a primitives DataFrame, run infographic enrichment, and write the enriched DataFrame.
    """
    try:
        df = read_dataframe(input_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="infographic")
    schema = load_infographic_extractor_schema_from_dict(cfg_dict)
    out_df, _info = extract_infographic_data_from_primitives_df(df, extractor_config=schema, task_config={})

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".infographic" + input_path.suffix)

    try:
        write_dataframe(out_df, output_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


def main() -> None:
    app()
