from __future__ import annotations

from pathlib import Path
from typing import Optional

import ray.data as rd
import typer

from retriever.page_elements import PageElementDetectionActor

from .common import (
    benchmark_sweep,
    make_seed_page_elements_row,
    maybe_init_ray,
    maybe_write_results_json,
    parse_csv_ints,
)

app = typer.Typer(help="Benchmark `PageElementDetectionActor` throughput (rows/sec).")


@app.command("run")
def run(
    pdf_path: Path = typer.Option(..., "--pdf-path", exists=True, dir_okay=False, file_okay=True, help="Input PDF."),
    rows: int = typer.Option(256, "--rows", min=1, help="How many page rows to benchmark per trial."),
    workers: str = typer.Option("1,2", "--workers", help="Comma-separated actor counts to try."),
    batch_sizes: str = typer.Option("1,2,4,8,16", "--batch-sizes", help="Comma-separated Ray batch sizes to try."),
    inference_batch_size: int = typer.Option(
        8, "--inference-batch-size", min=1, help="Internal model inference batch size inside the actor."
    ),
    num_gpus: float = typer.Option(1.0, "--num-gpus", min=0.0, help="GPUs reserved per actor."),
    num_cpus: float = typer.Option(1.0, "--num-cpus", min=0.0, help="CPUs reserved per actor."),
    dpi: int = typer.Option(200, "--dpi", min=72, help="Seed extraction DPI."),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray address (default local)."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional output JSON summary path."),
) -> None:
    maybe_init_ray(ray_address)
    worker_grid = parse_csv_ints(workers, name="workers")
    batch_grid = parse_csv_ints(batch_sizes, name="batch_sizes")
    seed_row = make_seed_page_elements_row(pdf_path, dpi=int(dpi))

    def _map(ds: rd.Dataset, worker_count: int, batch_size: int) -> rd.Dataset:
        return ds.map_batches(
            PageElementDetectionActor,
            batch_size=int(batch_size),
            batch_format="pandas",
            num_cpus=float(num_cpus),
            num_gpus=float(num_gpus),
            compute=rd.ActorPoolStrategy(size=int(worker_count)),
            fn_constructor_kwargs={"inference_batch_size": int(inference_batch_size)},
        )

    best, results = benchmark_sweep(
        stage_name="page_elements",
        seed_row=seed_row,
        rows=int(rows),
        workers=worker_grid,
        batch_sizes=batch_grid,
        map_builder=_map,
    )
    typer.echo(
        f"BEST page_elements: workers={best.workers} batch_size={best.batch_size} "
        f"rows={best.rows} elapsed={best.elapsed_seconds:.3f}s rows_per_second={best.rows_per_second:.2f}"
    )
    maybe_write_results_json(output_json, best=best, results=results)
