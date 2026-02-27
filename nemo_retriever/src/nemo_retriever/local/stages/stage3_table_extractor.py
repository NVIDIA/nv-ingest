# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local pipeline stage: table extraction wrapper.

Adds local-only conveniences (e.g. dumping OCR input images) while delegating
the actual extraction to `nemo_retriever.table.stage`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from nemo_retriever.table import stage as table_stage
from nemo_retriever.io.stage_files import build_stage_output_path, find_stage_inputs

app = typer.Typer(help="Stage 3: table extractor (wrapper around `nemo_retriever.table.stage`).")


def _iter_pdf_extraction_infographics_json_files(input_dir: Path) -> list[Path]:
    return find_stage_inputs(input_dir, suffix="pdf_extraction.infographic.json")


def _run_one(
    *,
    input_path: Path,
    output_path: Optional[Path],
    config: Optional[Path],
    output_ocr_input_images: bool,
) -> None:
    """
    Run table enrichment for a single input file, optionally dumping OCR input images.
    """
    # Stage3 knows where the input DF file lives; communicate that to lower layers via env vars.
    # This keeps the API extractors reusable across callsites (Typer, Ray, etc).
    env_backup = {
        "NV_INGEST_DUMP_OCR_INPUT_IMAGES": os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES"),
        "NV_INGEST_DUMP_OCR_INPUT_IMAGES_DIR": os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES_DIR"),
        "NV_INGEST_DUMP_OCR_INPUT_IMAGES_PREFIX": os.getenv("NV_INGEST_DUMP_OCR_INPUT_IMAGES_PREFIX"),
    }
    try:
        if output_ocr_input_images:
            os.environ["NV_INGEST_DUMP_OCR_INPUT_IMAGES"] = "1"
            os.environ["NV_INGEST_DUMP_OCR_INPUT_IMAGES_DIR"] = str(input_path.parent)
            os.environ["NV_INGEST_DUMP_OCR_INPUT_IMAGES_PREFIX"] = input_path.name

        # Delegate to the canonical table stage implementation.
        table_stage.run(input_path=input_path, output_path=output_path, config=config)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
            "If a directory, we process every `*pdf_extraction.infographics.json` file in it."
        ),
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path for single-file input (.parquet, .jsonl, or .json). Defaults to <input>.+table<ext>.",
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
        help="Optional YAML config for TableExtractorSchema.",
    ),
    output_ocr_input_images: bool = typer.Option(
        False,
        "--output-ocr-input-images/--no-output-ocr-input-images",
        help=(
            "If enabled, writes every image passed to OCR into the input file's directory. "
            "Filenames: <input_filename>.page_<PPPP>_<IIII>.(hf|nim).png"
        ),
    ),
) -> None:
    """
    Run table enrichment and optionally dump OCR input images.
    """
    if input_path.is_file():
        if output_dir is not None:
            raise typer.BadParameter("--output-dir is only valid when --input is a directory.")
        _run_one(
            input_path=input_path,
            output_path=output_path,
            config=config,
            output_ocr_input_images=output_ocr_input_images,
        )
        return

    # Directory mode
    if output_path is not None:
        raise typer.BadParameter("--output is only valid when --input is a file.")

    files = _iter_pdf_extraction_infographics_json_files(input_path)
    if not files:
        raise typer.BadParameter(f"No `*pdf_extraction.infographics.json` files found in directory: {input_path}")

    out_dir = output_dir or input_path
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        out = build_stage_output_path(p, stage_suffix=".table", output_dir=out_dir) if output_dir is not None else None
        _run_one(
            input_path=p,
            output_path=out,
            config=config,
            output_ocr_input_images=output_ocr_input_images,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
