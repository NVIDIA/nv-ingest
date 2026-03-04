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

from nemo_retriever.table.config import load_table_extractor_schema_from_dict, load_table_structure_ocr_config_from_dict
from nemo_retriever.table.processor import extract_table_data_from_primitives_df
from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

app = typer.Typer(help="Table extraction: enrich table primitives with OCR/structure-derived table_content.")
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
    try:
        df = read_dataframe(input_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="table")
    schema = load_table_extractor_schema_from_dict(cfg_dict)
    out_df, _info = extract_table_data_from_primitives_df(df, extractor_config=schema, task_config={})

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".table" + input_path.suffix)

    try:
        write_dataframe(out_df, output_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


@app.command("run-structure-ocr")
def run_structure_ocr(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Input primitives as a DataFrame file (.parquet, .jsonl, or .json). "
        "Must include page_elements_v3 and page_image columns.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path (.parquet, .jsonl, or .json). Defaults to <input>.+table_structure_ocr<ext>.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=("Optional YAML config file for table-structure+OCR stage. " "Uses section: table_structure_ocr."),
    ),
) -> None:
    """
    Load a primitives DataFrame, run table-structure + OCR enrichment, and write the output.
    """
    try:
        df = read_dataframe(input_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="table_structure_ocr")
    stage_cfg = load_table_structure_ocr_config_from_dict(cfg_dict)

    ts_url = stage_cfg.table_structure_invoke_url
    ocr_url = stage_cfg.ocr_invoke_url

    ts_model = None
    ocr_model_inst = None
    if not ts_url:
        from nemo_retriever.model.local import NemotronTableStructureV1

        ts_model = NemotronTableStructureV1()
    if not ocr_url:
        from nemo_retriever.model.local import NemotronOCRV1

        ocr_model_inst = NemotronOCRV1()

    out_df = table_structure_ocr_page_elements(
        df,
        table_structure_model=ts_model,
        ocr_model=ocr_model_inst,
        table_structure_invoke_url=ts_url,
        ocr_invoke_url=ocr_url,
        api_key=stage_cfg.api_key,
        request_timeout_s=stage_cfg.request_timeout_s,
    )

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".table_structure_ocr" + input_path.suffix)

    try:
        write_dataframe(out_df, output_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


def main() -> None:
    app()
