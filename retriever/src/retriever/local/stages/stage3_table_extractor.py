"""
Local pipeline stage: table extraction wrapper.

Adds local-only conveniences (e.g. dumping OCR input images) while delegating
the actual extraction to `retriever.table.stage`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from retriever.table import stage as table_stage

app = typer.Typer(help="Stage 3: table extractor (wrapper around `retriever.table.stage`).")


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Input primitives as a DataFrame file (.parquet, .jsonl, or .json).",
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()

