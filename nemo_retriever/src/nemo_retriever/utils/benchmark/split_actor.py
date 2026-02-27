# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ray.data as rd
import typer

from nemo_retriever.pdf.split import PDFSplitActor

from .common import (
    benchmark_sweep,
    make_seed_split_row,
    maybe_init_ray,
    maybe_write_results_json,
    parse_csv_ints,
)

app = typer.Typer(help="Benchmark `PDFSplitActor` throughput (rows/sec).")


@app.command("run")
def run(
    pdf_path: Path = typer.Option(..., "--pdf-path", exists=True, dir_okay=False, file_okay=True, help="Input PDF."),
    rows: int = typer.Option(
        256, "--rows", min=1, help="How many input rows to synthesize from the seed PDF for each benchmark trial."
    ),
    workers: str = typer.Option("1,2,4", "--workers", help="Comma-separated worker counts to try."),
    batch_sizes: str = typer.Option("1,2,4,8,16", "--batch-sizes", help="Comma-separated Ray batch sizes to try."),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray address (default local)."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional output JSON summary path."),
) -> None:
    maybe_init_ray(ray_address)
    worker_grid = parse_csv_ints(workers, name="workers")
    batch_grid = parse_csv_ints(batch_sizes, name="batch_sizes")
    seed_row = make_seed_split_row(pdf_path)

    def _map(ds: rd.Dataset, worker_count: int, batch_size: int) -> rd.Dataset:
        return ds.map_batches(
            PDFSplitActor(),
            batch_size=int(batch_size),
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
            compute=rd.TaskPoolStrategy(size=int(worker_count)),
        )

    best, results = benchmark_sweep(
        stage_name="pdf_split",
        seed_row=seed_row,
        rows=int(rows),
        workers=worker_grid,
        batch_sizes=batch_grid,
        map_builder=_map,
    )

    typer.echo(
        f"BEST pdf_split: workers={best.workers} batch_size={best.batch_size} "
        f"rows={best.rows} elapsed={best.elapsed_seconds:.3f}s rows_per_second={best.rows_per_second:.2f}"
    )
    maybe_write_results_json(output_json, best=best, results=results)
