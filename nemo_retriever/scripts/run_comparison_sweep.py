#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run compare-from-pre-embed over a parameter grid (gpu_util x max_rows) and append
each run's metrics to a CSV (and optionally JSONL) for later plotting.

FlashInfer: not a CLI flag. Run this script twice — once with flashinfer-cubin
installed, once after uninstalling it — and append to the same --output-csv so
the file contains both flashinfer_cubin=true and false rows.

Example (with cubin, from repo root):
  unset RAY_ADDRESS
  uv run python nemo_retriever/scripts/run_comparison_sweep.py run-sweep \\
    --pre-embed-dir /path/to/bo767_pre_embed \\
    --query-csv data/bo767_query_gt.csv \\
    --embed-model-path /path/to/llama-nemotron-embed-1b-v2/main \\
    --output-csv comparison_sweep.csv

Then (without cubin):
  uv pip uninstall flashinfer-cubin
  uv run python nemo_retriever/scripts/run_comparison_sweep.py run-sweep ... --output-csv comparison_sweep.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow importing vllm_embedding_comparison from this directory (nemo_retriever/scripts)
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import typer
from vllm_embedding_comparison import (
    _append_comparison_row_csv,
    _append_comparison_row_json,
    _detect_flashinfer_cubin,
    run_compare_from_pre_embed,
)

app = typer.Typer(help="Sweep compare-from-pre-embed over gpu_util and max_rows; append to CSV.")

# Defaults from plan
DEFAULT_GPU_UTILS = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_MAX_ROWS = [1000, 2000, 5000, 10000]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@app.command()
def run_sweep(
    pre_embed_dir: Path = typer.Option(..., "--pre-embed-dir", path_type=Path),
    query_csv: Path = typer.Option(..., "--query-csv", path_type=Path),
    embed_model_path: Path | None = typer.Option(None, "--embed-model-path", path_type=Path),
    output_csv: Path = typer.Option(..., "--output-csv", path_type=Path),
    output_json: Path | None = typer.Option(None, "--output-json", path_type=Path),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri"),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2", "--embed-model-name"
    ),
    gpu_utils: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_GPU_UTILS),
        "--gpu-utils",
        help="Comma-separated gpu_memory_utilization values (e.g. 0.4,0.5,0.6,0.7,0.8).",
    ),
    max_rows_list: str = typer.Option(
        ",".join(str(x) for x in DEFAULT_MAX_ROWS),
        "--max-rows-list",
        help="Comma-separated max_rows values (e.g. 1000,2000,5000,10000).",
    ),
    sort_key_column: str | None = typer.Option(None, "--sort-key"),
    ray_address: str | None = typer.Option(None, "--ray-address"),
) -> None:
    """Run grid of compare-from-pre-embed; append one row per run to output_csv (and optional output_json)."""
    pre_embed_dir = Path(pre_embed_dir)
    query_csv = Path(query_csv)
    output_csv = Path(output_csv)
    if not pre_embed_dir.is_dir():
        raise typer.BadParameter(f"Pre-embed directory not found: {pre_embed_dir}")
    if not query_csv.is_file():
        raise typer.BadParameter(f"Query CSV not found: {query_csv}")

    os.environ.pop("RAY_ADDRESS", None)

    gpu_util_values = _parse_float_list(gpu_utils)
    max_rows_values = _parse_int_list(max_rows_list)
    total = len(gpu_util_values) * len(max_rows_values)
    typer.echo(f"flashinfer_cubin={_detect_flashinfer_cubin()} (detected at start)")
    typer.echo(f"Grid: {len(gpu_util_values)} gpu_utils x {len(max_rows_values)} max_rows = {total} runs")
    typer.echo(f"Output CSV: {output_csv}")

    run_id = 0
    for gpu_util in gpu_util_values:
        for max_rows in max_rows_values:
            run_id += 1
            typer.echo(f"[{run_id}/{total}] gpu_memory_utilization={gpu_util}, max_rows={max_rows}")
            try:
                row = run_compare_from_pre_embed(
                    pre_embed_dir=pre_embed_dir,
                    query_csv=query_csv,
                    lancedb_uri=lancedb_uri,
                    ray_address=ray_address,
                    embed_model_path=str(embed_model_path) if embed_model_path else None,
                    embed_model_name=embed_model_name,
                    max_rows=max_rows,
                    gpu_memory_utilization=gpu_util,
                    enforce_eager=False,
                    compile_cache_dir=None,
                    sort_key_column=sort_key_column,
                )
                _append_comparison_row_csv(row, output_csv)
                if output_json is not None:
                    _append_comparison_row_json(row, Path(output_json))
            except Exception as e:
                typer.echo(f"Run failed: {e}", err=True)
                raise

    typer.echo(f"Wrote {total} rows to {output_csv}")


if __name__ == "__main__":
    app()
