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

from nemo_retriever.chart.config import load_chart_extractor_schema_from_dict
from nemo_retriever.chart.processor import extract_chart_data_from_primitives_df

app = typer.Typer(help="Chart Extraction Stage")
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
        help="Output path (.parquet, .jsonl, or .json). Defaults to <input>.+chart<ext>.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Uses section: chart."
        ),
    ),
) -> None:
    """
    Load a primitives DataFrame, run chart enrichment, and write the enriched DataFrame.
    """
    try:
        df = read_dataframe(input_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    cfg_dict: Dict[str, Any] = load_ingest_config_section(config, section="chart")
    schema = load_chart_extractor_schema_from_dict(cfg_dict)
    out_df, _info = extract_chart_data_from_primitives_df(df, extractor_config=schema, task_config={})

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".chart" + input_path.suffix)

    try:
        write_dataframe(out_df, output_path)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    console.print(f"[green]Done[/green] wrote={output_path} rows={len(out_df)}")


@app.command("graphic-elements")
def render_graphic_elements(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config file. If set, values are loaded from YAML; "
        "explicitly passed CLI flags override YAML.",
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan recursively for *.pdf (can be provided via --config).",
    ),
    method: str = typer.Option("pdfium", "--method", help="PDF extraction method."),
    auth_token: Optional[str] = typer.Option(None, "--auth-token", help="Auth token for NIM-backed services."),
    yolox_grpc_endpoint: Optional[str] = typer.Option(None, "--yolox-grpc-endpoint"),
    yolox_http_endpoint: Optional[str] = typer.Option(None, "--yolox-http-endpoint"),
    nemotron_parse_grpc_endpoint: Optional[str] = typer.Option(None, "--nemotron-parse-grpc-endpoint"),
    nemotron_parse_http_endpoint: Optional[str] = typer.Option(None, "--nemotron-parse-http-endpoint"),
    nemotron_parse_model_name: Optional[str] = typer.Option(None, "--nemotron-parse-model-name"),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text"),
    extract_images: bool = typer.Option(False, "--extract-images/--no-extract-images"),
    extract_tables: bool = typer.Option(False, "--extract-tables/--no-extract-tables"),
    extract_charts: bool = typer.Option(False, "--extract-charts/--no-extract-charts"),
    extract_infographics: bool = typer.Option(False, "--extract-infographics/--no-extract-infographics"),
    extract_page_as_image: bool = typer.Option(False, "--extract-page-as-image/--no-extract-page-as-image"),
    text_depth: str = typer.Option("page", "--text-depth"),
    write_json_outputs: bool = typer.Option(True, "--write-json-outputs/--no-write-json-outputs"),
    json_output_dir: Optional[Path] = typer.Option(None, "--json-output-dir", file_okay=False, dir_okay=True),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of PDFs processed."),
) -> None:
    _ = (
        config,
        input_dir,
        method,
        auth_token,
        yolox_grpc_endpoint,
        yolox_http_endpoint,
        nemotron_parse_grpc_endpoint,
        nemotron_parse_http_endpoint,
        nemotron_parse_model_name,
        extract_text,
        extract_images,
        extract_tables,
        extract_charts,
        extract_infographics,
        extract_page_as_image,
        text_depth,
        write_json_outputs,
        json_output_dir,
        limit,
    )
    typer.echo("graphic-elements command is not implemented yet.")


def main() -> None:
    app()
