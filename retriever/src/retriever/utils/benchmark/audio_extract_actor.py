# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark MediaChunkActor + ASRActor throughput (chunk rows per second).

Supports --mock-asr to avoid loading Parakeet/GPU; measures chunking + actor overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import ray.data as rd
import typer

from retriever.audio.asr_actor import ASRActor
from retriever.audio.asr_actor import asr_params_from_env
from retriever.audio.chunk_actor import MediaChunkActor
from retriever.audio.media_interface import is_media_available
from retriever.params import AudioChunkParams

from .common import (
    benchmark_sweep,
    maybe_init_ray,
    maybe_write_results_json,
    parse_csv_ints,
)


def make_seed_audio_row(audio_path: Path) -> Dict[str, Any]:
    """One row per source file (path only); MediaChunkActor will read and chunk."""
    p = audio_path.expanduser().resolve()
    if not p.is_file():
        raise typer.BadParameter(f"Audio path does not exist: {p}")
    return {"path": str(p)}


class MockASRActor:
    """Returns fixed transcript per chunk row so benchmark runs without Parakeet/GPU."""

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )
        out = batch_df.drop(columns=["bytes"], errors="ignore").copy()
        out["text"] = "mock transcript"
        return out


app = typer.Typer(help="Benchmark audio extraction (MediaChunkActor + ASRActor) throughput (chunk rows/sec).")


@app.command("run")
def run(
    audio_path: Path = typer.Option(
        ...,
        "--audio-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Input audio file (e.g. WAV, MP3).",
    ),
    rows: int = typer.Option(
        16,
        "--rows",
        min=1,
        help="Number of source-file rows to replicate for each trial.",
    ),
    workers: str = typer.Option(
        "1,2",
        "--workers",
        help="Comma-separated worker counts to try.",
    ),
    batch_sizes: str = typer.Option(
        "2,4,8",
        "--batch-sizes",
        help="Comma-separated Ray batch sizes to try.",
    ),
    mock_asr: bool = typer.Option(
        True,
        "--mock-asr/--no-mock-asr",
        help="Use mock ASR (no GPU/Parakeet); when False, uses local or remote ASR from env.",
    ),
    split_type: str = typer.Option(
        "size",
        "--split-type",
        help="Chunk split type: size, time, or frame.",
    ),
    split_interval: int = typer.Option(
        450,
        "--split-interval",
        min=1,
        help="Chunk split interval (bytes/seconds/frames).",
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="Ray address (default local).",
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json",
        help="Optional output JSON summary path.",
    ),
) -> None:
    if not is_media_available():
        raise typer.BadParameter("Audio benchmark requires ffmpeg on PATH.")

    if split_type not in ("size", "time", "frame"):
        raise typer.BadParameter("--split-type must be one of: size, time, frame")

    maybe_init_ray(ray_address)
    worker_grid = parse_csv_ints(workers, name="workers")
    batch_grid = parse_csv_ints(batch_sizes, name="batch_sizes")
    seed_row = make_seed_audio_row(audio_path)

    chunk_params = AudioChunkParams(
        split_type=split_type,
        split_interval=split_interval,
    )

    def _map(ds: rd.Dataset, worker_count: int, batch_size: int) -> rd.Dataset:
        chunk_actor = MediaChunkActor(params=chunk_params)
        if mock_asr:
            asr_actor = MockASRActor()
        else:
            asr_actor = ASRActor(params=asr_params_from_env())

        ds = ds.map_batches(
            chunk_actor,
            batch_size=int(batch_size),
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
            compute=rd.TaskPoolStrategy(size=int(worker_count)),
        )
        ds = ds.map_batches(
            asr_actor,
            batch_size=int(batch_size),
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0.25 if not mock_asr else 0,
            compute=rd.TaskPoolStrategy(size=int(worker_count)),
        )
        return ds

    best, results = benchmark_sweep(
        stage_name="audio_extract",
        seed_row=seed_row,
        rows=int(rows),
        workers=worker_grid,
        batch_sizes=batch_grid,
        map_builder=_map,
    )
    typer.echo(
        f"BEST audio_extract: workers={best.workers} batch_size={best.batch_size} "
        f"chunk_rows={best.rows} elapsed={best.elapsed_seconds:.3f}s rows_per_second={best.rows_per_second:.2f}"
    )
    maybe_write_results_json(output_json, best=best, results=results)
