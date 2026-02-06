"""
Local pipeline stage: infographic extraction wrapper.

This stage mostly delegates to `retriever.infographic.stage`, but adds a local-only
convenience: `--input` may be either:

- A single primitives DF file (e.g. `<pdf>.pdf_extraction.json`, `.jsonl`, `.parquet`)
- A directory, in which case we iterate over all `*pdf_extraction.json` files in that
  directory and run the extractor on each.

This makes it easy to run stage2 across a whole folder of stage1 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from retriever.infographic import stage as infographic_stage

app = typer.Typer(help="Stage 2: infographic extractor (wrapper around `retriever.infographic.stage`).")


def _iter_pdf_extraction_json_files(input_dir: Path) -> list[Path]:
    # Match both:
    # - `pdf_extraction.json` (common when each doc has its own folder)
    # - `<doc>.pdf_extraction.json` (common when all outputs live in one folder)
    files = [p for p in input_dir.iterdir() if p.is_file() and p.name.endswith("pdf_extraction.json")]
    return sorted(files)


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=True,
        file_okay=True,
        help=(
            "Input primitives DataFrame file (.parquet, .jsonl, or .json), or a directory. "
            "If a directory, we process every `*pdf_extraction.json` file in it."
        ),
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        file_okay=True,
        help="Output path for single-file input. Defaults to <input>.+infographic<ext>.",
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
        help="Optional YAML config for InfographicExtractorSchema.",
    ),
) -> None:
    """
    Run infographic enrichment on a single primitives file, or batch-run it across a directory of
    stage1 outputs.
    """
    if input_path.is_file():
        if output_dir is not None:
            raise typer.BadParameter("--output-dir is only valid when --input is a directory.")
        infographic_stage.run(input_path=input_path, output_path=output_path, config=config)
        return

    # Directory mode
    if output_path is not None:
        raise typer.BadParameter("--output is only valid when --input is a file.")

    files = _iter_pdf_extraction_json_files(input_path)
    if not files:
        raise typer.BadParameter(f"No `*pdf_extraction.json` files found in directory: {input_path}")

    out_dir = output_dir or input_path
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        # Mirror `retriever.infographic.stage.run` default output naming, but place into out_dir.
        out = out_dir / (p.stem + ".infographic" + p.suffix)
        infographic_stage.run(input_path=p, output_path=out, config=config)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

