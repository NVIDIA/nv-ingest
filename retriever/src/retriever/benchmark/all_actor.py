from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import extract_actor, ocr_actor, page_elements_actor, split_actor

app = typer.Typer(help="Run all actor-stage benchmarks in sequence.")


@app.command("run")
def run(
    pdf_path: Path = typer.Option(..., "--pdf-path", exists=True, dir_okay=False, file_okay=True, help="Input PDF."),
    rows_split: int = typer.Option(256, "--rows-split", min=1, help="Synthetic rows per split benchmark trial."),
    rows_extract: int = typer.Option(256, "--rows-extract", min=1, help="Synthetic rows per extract benchmark trial."),
    rows_page_elements: int = typer.Option(
        256, "--rows-page-elements", min=1, help="Synthetic rows per page-elements benchmark trial."
    ),
    rows_ocr: int = typer.Option(128, "--rows-ocr", min=1, help="Synthetic rows per OCR benchmark trial."),
    workers: str = typer.Option("1,2", "--workers", help="Comma-separated worker counts used by all stages."),
    batch_sizes: str = typer.Option("1,2,4,8", "--batch-sizes", help="Comma-separated batch sizes used by all stages."),
    dpi: int = typer.Option(200, "--dpi", min=72, help="Seed extraction DPI for extraction/detection/OCR stages."),
    inference_batch_size: int = typer.Option(
        8, "--inference-batch-size", min=1, help="Internal model inference batch size for page-elements stage."
    ),
    num_gpus: float = typer.Option(1.0, "--num-gpus", min=0.0, help="GPUs reserved per page-elements/OCR actor."),
    num_cpus: float = typer.Option(1.0, "--num-cpus", min=0.0, help="CPUs reserved per page-elements/OCR actor."),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables", help="OCR table detections."),
    extract_charts: bool = typer.Option(False, "--extract-charts/--no-extract-charts", help="OCR chart detections."),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", help="OCR infographic detections."
    ),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray address (default local)."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Optional directory for per-stage JSON benchmark summaries."
    ),
) -> None:
    output_dir_resolved = output_dir.expanduser().resolve() if output_dir is not None else None
    if output_dir_resolved is not None:
        output_dir_resolved.mkdir(parents=True, exist_ok=True)

    split_json = (output_dir_resolved / "split.json") if output_dir_resolved is not None else None
    extract_json = (output_dir_resolved / "extract.json") if output_dir_resolved is not None else None
    page_elements_json = (output_dir_resolved / "page_elements.json") if output_dir_resolved is not None else None
    ocr_json = (output_dir_resolved / "ocr.json") if output_dir_resolved is not None else None

    typer.echo("=== benchmark: split ===")
    split_actor.run(
        pdf_path=pdf_path,
        rows=int(rows_split),
        workers=workers,
        batch_sizes=batch_sizes,
        ray_address=ray_address,
        output_json=split_json,
    )

    typer.echo("=== benchmark: extract ===")
    extract_actor.run(
        pdf_path=pdf_path,
        rows=int(rows_extract),
        workers=workers,
        batch_sizes=batch_sizes,
        dpi=int(dpi),
        extract_text=True,
        extract_page_as_image=True,
        ray_address=ray_address,
        output_json=extract_json,
    )

    typer.echo("=== benchmark: page-elements ===")
    page_elements_actor.run(
        pdf_path=pdf_path,
        rows=int(rows_page_elements),
        workers=workers,
        batch_sizes=batch_sizes,
        inference_batch_size=int(inference_batch_size),
        num_gpus=float(num_gpus),
        num_cpus=float(num_cpus),
        dpi=int(dpi),
        ray_address=ray_address,
        output_json=page_elements_json,
    )

    typer.echo("=== benchmark: ocr ===")
    ocr_actor.run(
        pdf_path=pdf_path,
        rows=int(rows_ocr),
        workers=workers,
        batch_sizes=batch_sizes,
        extract_tables=bool(extract_tables),
        extract_charts=bool(extract_charts),
        extract_infographics=bool(extract_infographics),
        num_gpus=float(num_gpus),
        num_cpus=float(num_cpus),
        dpi=int(dpi),
        ray_address=ray_address,
        output_json=ocr_json,
    )

    typer.echo("Completed all benchmark stages.")
