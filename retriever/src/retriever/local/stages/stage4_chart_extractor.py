# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local pipeline stage: chart extraction wrapper.

This stage mostly delegates to `retriever.chart.stage`, but adds a local-only
convenience: `--input` may be either:

- A single primitives DF file (e.g. `*.pdf_extraction.infographics.table.json`, `.jsonl`, `.parquet`)
- A directory, in which case we iterate over all stage3 table outputs in that
  directory (files ending with `pdf_extraction.infographics.table.json`) and run
  chart extraction on each.

This makes it easy to run stage4 across a whole folder of stage3 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from retriever.chart import stage as chart_stage
from retriever.io.stage_files import build_stage_output_path, find_stage_inputs

app = typer.Typer(help="Stage 4: chart extractor (wrapper around `retriever.chart.stage`).")


def _iter_stage3_table_outputs(input_dir: Path) -> list[Path]:
    return find_stage_inputs(input_dir, suffix="pdf_extraction.infographic.table.json")


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help=(
            "Input primitives as a DataFrame file (.parquet, .jsonl, or .json), or a directory. "
            "If a directory, we process every `*pdf_extraction.infographics.table.json` file in it."
        ),
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path for single-file input (.parquet, .jsonl, or .json). Defaults to <input>.+chart<ext>.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        dir_okay=True,
        file_okay=False,
        help=(
            "Output directory for directory input. Defaults to the input directory. "
            "Output filenames follow the single-file default naming."
        ),
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config for ChartExtractorSchema.",
    ),
) -> None:
    """
    Run chart enrichment on a single primitives file, or batch-run it across a directory of stage3 outputs.
    """
    if input_path.is_file():
        if output_dir is not None:
            raise typer.BadParameter("--output-dir is only valid when --input is a directory.")
        chart_stage.run(input_path=input_path, output_path=output_path, config=config)
        return

    # Directory mode
    if output_path is not None:
        raise typer.BadParameter("--output is only valid when --input is a file.")

    files = _iter_stage3_table_outputs(input_path)
    if not files:
        raise typer.BadParameter(f"No `*pdf_extraction.infographics.table.json` files found in directory: {input_path}")

    out_dir = output_dir or input_path
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        out = build_stage_output_path(p, stage_suffix=".chart", output_dir=out_dir) if output_dir is not None else None
        chart_stage.run(input_path=p, output_path=out, config=config)


# Preserve access to other chart-stage commands under local stage4 (e.g. `graphic-elements`).
app.command("graphic-elements")(chart_stage.render_graphic_elements)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
