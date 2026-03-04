#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read the CSV produced by run_comparison_sweep.py and generate figures for
ingest time vs scale/GPU util, recall comparison, and FlashInfer impact.

Requires: pandas, matplotlib (pip install matplotlib if needed).

Example (from repo root):
  uv run python nemo_retriever/scripts/plot_comparison_sweep.py run \
    --input-csv comparison_sweep.csv \
    --output-dir ./comparison_plots
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Plot comparison sweep CSV (ingest time, recall, FlashInfer impact).")


def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError("plot_comparison_sweep.py requires matplotlib. Install with: pip install matplotlib")


@app.command()
def run(
    input_csv: Path = typer.Option(..., "--input-csv", path_type=Path),
    output_dir: Path = typer.Option(
        Path("comparison_plots"),
        "--output-dir",
        path_type=Path,
        help="Directory to write PNG/PDF figures.",
    ),
) -> None:
    """Generate 2–4 figures from the sweep CSV."""
    _check_matplotlib()
    import matplotlib.pyplot as plt

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    if not input_csv.is_file():
        raise typer.BadParameter(f"Input CSV not found: {input_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    # Normalize column names (allow both with/without underscores)
    df = df.rename(columns=lambda c: c.strip())
    required = [
        "gpu_memory_utilization",
        "max_rows",
        "baseline_ingest_s",
        "vllm_ingest_s",
        "baseline_recall_at_1",
        "baseline_recall_at_5",
        "baseline_recall_at_10",
        "vllm_recall_at_1",
        "vllm_recall_at_5",
        "vllm_recall_at_10",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"CSV missing columns: {missing}. Expected columns from run_comparison_sweep.")

    # Normalize flashinfer_cubin (CSV may have "True"/"False" strings)
    if "flashinfer_cubin" in df.columns:
        df["flashinfer_cubin"] = df["flashinfer_cubin"].map(
            lambda x: str(x).lower() in ("true", "1", "yes") if pd.notna(x) else False
        )
    has_flashinfer = "flashinfer_cubin" in df.columns and df["flashinfer_cubin"].nunique() > 1

    # For figures 1–3 use one row per (max_rows, gpu_util) to avoid duplicate lines when both flashinfer states present
    df_main = df.drop_duplicates(subset=["max_rows", "gpu_memory_utilization"], keep="first")

    # 1) Ingest time vs max_rows (baseline vs vLLM); one baseline series, vLLM by gpu_util
    fig, ax = plt.subplots(figsize=(8, 5))
    base = df_main.groupby("max_rows", as_index=False).agg({"baseline_ingest_s": "first"}).sort_values("max_rows")
    ax.plot(base["max_rows"], base["baseline_ingest_s"], "o-", label="Baseline (HF)", linewidth=2, alpha=0.9)
    for gpu in sorted(df_main["gpu_memory_utilization"].unique()):
        sub = df_main[df_main["gpu_memory_utilization"] == gpu].sort_values("max_rows")
        if sub.empty:
            continue
        ax.plot(
            sub["max_rows"],
            sub["vllm_ingest_s"],
            "s--",
            label=f"vLLM (gpu_util={gpu})",
            alpha=0.8,
        )
    ax.set_xlabel("max_rows")
    ax.set_ylabel("Ingest time (s)")
    ax.set_title("Ingest time vs benchmark scale (max_rows)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = output_dir / f"ingest_vs_max_rows.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None)
        typer.echo(f"Wrote {out}")
    plt.close(fig)

    # 2) Ingest time vs gpu_memory_utilization (baseline flat per max_rows, vLLM by max_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for max_r in sorted(df_main["max_rows"].unique()):
        sub = df_main[df_main["max_rows"] == max_r].sort_values("gpu_memory_utilization")
        if sub.empty:
            continue
        ax.plot(
            sub["gpu_memory_utilization"],
            sub["baseline_ingest_s"],
            "o-",
            label=f"Baseline (max_rows={max_r})",
            alpha=0.8,
        )
        ax.plot(
            sub["gpu_memory_utilization"],
            sub["vllm_ingest_s"],
            "s--",
            label=f"vLLM (max_rows={max_r})",
            alpha=0.8,
        )
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("Ingest time (s)")
    ax.set_title("Ingest time vs vLLM GPU utilization")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = output_dir / f"ingest_vs_gpu_util.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None)
        typer.echo(f"Wrote {out}")
    plt.close(fig)

    # 3) Recall comparison: recall@10 vs max_rows (baseline vs vLLM)
    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu in sorted(df_main["gpu_memory_utilization"].unique())[:5]:
        s = df_main[df_main["gpu_memory_utilization"] == gpu].sort_values("max_rows")
        if s.empty:
            continue
        ax.plot(s["max_rows"], s["baseline_recall_at_10"], "o-", label=f"Baseline (gpu_util={gpu})", alpha=0.8)
        ax.plot(s["max_rows"], s["vllm_recall_at_10"], "s--", label=f"vLLM (gpu_util={gpu})", alpha=0.8)
    ax.set_xlabel("max_rows")
    ax.set_ylabel("recall@10")
    ax.set_title("Recall@10 vs benchmark scale")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = output_dir / f"recall10_vs_max_rows.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None)
        typer.echo(f"Wrote {out}")
    plt.close(fig)

    # 4) FlashInfer impact (if we have both true/false)
    if has_flashinfer:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (cubin_val, label) in enumerate([(True, "flashinfer_cubin=True"), (False, "flashinfer_cubin=False")]):
            sub = df[df["flashinfer_cubin"] == cubin_val].sort_values(["max_rows", "gpu_memory_utilization"])
            if sub.empty:
                continue
            ax = axes[idx]
            for gpu in sorted(sub["gpu_memory_utilization"].unique()):
                s = sub[sub["gpu_memory_utilization"] == gpu].sort_values("max_rows")
                ax.plot(s["max_rows"], s["baseline_ingest_s"], "o-", label=f"Baseline (gpu={gpu})", alpha=0.8)
                ax.plot(s["max_rows"], s["vllm_ingest_s"], "s--", label=f"vLLM (gpu={gpu})", alpha=0.8)
            ax.set_xlabel("max_rows")
            ax.set_ylabel("Ingest time (s)")
            ax.set_title(label)
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = output_dir / f"flashinfer_impact.{ext}"
            fig.savefig(out, dpi=150 if ext == "png" else None)
            typer.echo(f"Wrote {out}")
        plt.close(fig)
    else:
        typer.echo("No flashinfer_cubin variation in CSV; skipping FlashInfer impact figure.")

    typer.echo(f"Done. Figures in {output_dir}")


if __name__ == "__main__":
    app()
