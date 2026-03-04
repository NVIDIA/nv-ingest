#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
jd5 recall barometer: ingest PDFs, run recall, report metrics.

Use this script to ensure the vLLM embedding path does not veer from the
current implementation. Run once with local HF embeddings (baseline), once
with vLLM endpoint + --embed-use-vllm-compat, and/or once with vLLM offline
(--embed-use-vllm-offline), then compare recall@1/@5/@10.

Query CSV format: must contain columns "query" and either "pdf_page" or
"pdf" + "page". Use "pdf" = PDF basename (with or without .pdf); page should
match the convention used when writing to LanceDB (e.g. 1-based).

Example (baseline, local HF):
  uv run python nemo_retriever/scripts/jd5_recall_barometer.py run \\
    --pdf-dir /raid/charlesb/datasets/jd5/data \\
    --query-csv /raid/charlesb/datasets/jd5/jd5_query_gt.csv

Example (vLLM server; requires vllm==0.11.0 and model config_vllm.json for
llama-nemotron-embed-1b-v2 — see model README):
  uv run python nemo_retriever/scripts/jd5_recall_barometer.py run \\
    --pdf-dir /raid/charlesb/datasets/jd5/data \\
    --query-csv /raid/charlesb/datasets/jd5/jd5_query_gt.csv \\
    --embed-invoke-url http://localhost:8000/v1 \\
    --embed-use-vllm-compat

Example (vLLM offline, no server):
  uv run python nemo_retriever/scripts/jd5_recall_barometer.py run \\
    --pdf-dir /raid/charlesb/datasets/jd5/data \\
    --query-csv /raid/charlesb/datasets/jd5/jd5_query_gt.csv \\
    --embed-use-vllm-offline

Example (compare baseline vs vLLM server):
  uv run python nemo_retriever/scripts/jd5_recall_barometer.py compare \\
    --pdf-dir /raid/charlesb/datasets/jd5/data \\
    --query-csv /raid/charlesb/datasets/jd5/jd5_query_gt.csv \\
    --embed-invoke-url http://localhost:8000/v1

Example (three-way compare: baseline, vLLM server, vLLM offline):
  uv run python nemo_retriever/scripts/jd5_recall_barometer.py compare \\
    --pdf-dir /raid/charlesb/datasets/jd5/data \\
    --query-csv /raid/charlesb/datasets/jd5/jd5_query_gt.csv \\
    --embed-invoke-url http://localhost:8000/v1 \\
    --embed-use-vllm-offline
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root or nemo_retriever root (package lives at .../src/nemo_retriever)
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    if _root.name == "scripts":
        _root = _root.parent
    _src = _root / "src"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import typer
from nemo_retriever import create_ingestor
from nemo_retriever.ingest_modes.batch import EmbedServiceActor
from nemo_retriever.params import (
    EmbedParams,
    ExtractParams,
    IngestorCreateParams,
    IngestExecuteParams,
    ModelRuntimeParams,
    VdbUploadParams,
)
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer(help="jd5 recall barometer: ingest PDFs, run recall, report recall@1/@5/@10.")

JD5_PDF_DIR = "/raid/charlesb/datasets/jd5/data"
JD5_QUERY_CSV = "/raid/charlesb/datasets/jd5/jd5_query_gt.csv"
LANCEDB_URI = "lancedb"
TABLE_BASELINE = "nv-ingest-baseline"
TABLE_VLLM = "nv-ingest-vllm"
TABLE_VLLM_OFFLINE = "nv-ingest-vllm-offline"


def _run_ingest_and_recall(
    pdf_dir: Path,
    query_csv: Path,
    lancedb_uri: str,
    table_name: str,
    embed_invoke_url: str | None,
    embed_use_vllm_compat: bool,
    embed_use_vllm_offline: bool,
    embed_model_name: str,
    embed_model_path: str | None = None,
    run_mode: str = "inprocess",
    ray_address: str | None = None,
) -> dict:
    """Run ingest (inprocess or batch) then recall; return metrics."""
    pdf_dir = Path(pdf_dir)
    query_csv = Path(query_csv)
    if not pdf_dir.is_dir():
        raise typer.BadParameter(f"PDF directory not found: {pdf_dir}")
    if not query_csv.is_file():
        raise typer.BadParameter(f"Query CSV not found: {query_csv}")

    glob_pattern = str(pdf_dir / "*.pdf")
    create_kwargs: dict = {"run_mode": run_mode}
    if run_mode == "batch":
        create_kwargs["params"] = IngestorCreateParams(ray_address=ray_address)
    ingestor = create_ingestor(**create_kwargs)
    ingestor = (
        ingestor.files(glob_pattern)
        .extract(
            ExtractParams(
                method="pdfium",
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
            )
        )
        .embed(
            EmbedParams(
                model_name=embed_model_name,
                embed_invoke_url=embed_invoke_url,
                embed_use_vllm_compat=embed_use_vllm_compat,
                embed_use_vllm_offline=embed_use_vllm_offline,
                embed_model_path=embed_model_path,
            )
        )
        .vdb_upload(
            VdbUploadParams(
                lancedb={
                    "lancedb_uri": lancedb_uri,
                    "table_name": table_name,
                    "overwrite": True,
                    "create_index": True,
                }
            )
        )
    )
    start = time.perf_counter()
    ingestor.ingest(
        params=IngestExecuteParams(parallel=True, max_workers=8, show_progress=True),
    )
    ingest_elapsed = time.perf_counter() - start

    cfg = RecallConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=table_name,
        embedding_model=embed_model_name,
        embedding_http_endpoint=embed_invoke_url,
        embedding_use_vllm_compat=embed_use_vllm_compat,
        embedding_use_vllm_offline=embed_use_vllm_offline,
        embedding_vllm_model_path=embed_model_path,
        top_k=10,
        ks=(1, 5, 10),
    )
    _df, _gold, _raw, _keys, metrics = retrieve_and_score(
        query_csv=query_csv,
        cfg=cfg,
        vector_column_name="vector",
    )
    return {
        "metrics": metrics,
        "ingest_elapsed_s": ingest_elapsed,
        "n_queries": len(_df),
    }


@app.command("run")
def run(
    pdf_dir: Path = typer.Option(
        JD5_PDF_DIR,
        "--pdf-dir",
        path_type=Path,
        help="Directory containing PDFs to ingest.",
    ),
    query_csv: Path = typer.Option(
        JD5_QUERY_CSV,
        "--query-csv",
        path_type=Path,
        help="Query ground-truth CSV (columns: query, pdf_page OR query, pdf, page).",
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", help="LanceDB URI (directory path)."),
    table_name: str = typer.Option(
        "nv-ingest",
        "--table-name",
        help="LanceDB table name.",
    ),
    run_mode: str = typer.Option(
        "inprocess",
        "--run-mode",
        help="Ingest run mode: inprocess or batch (batch uses Ray Data).",
    ),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray cluster address for batch (e.g. auto)."),
    embed_invoke_url: str | None = typer.Option(
        None,
        "--embed-invoke-url",
        help="Embedding endpoint (e.g. http://localhost:8000/v1). Omit for local HF.",
    ),
    embed_use_vllm_compat: bool = typer.Option(
        False,
        "--embed-use-vllm-compat/--no-embed-use-vllm-compat",
        help="Use vLLM-compatible HTTP payload. Set when endpoint is a vLLM server.",
    ),
    embed_use_vllm_offline: bool = typer.Option(
        False,
        "--embed-use-vllm-offline/--no-embed-use-vllm-offline",
        help="Use vLLM offline Python API (no server).",
    ),
    embed_model_path: str | None = typer.Option(
        None,
        "--embed-model-path",
        help="Local path to model for vLLM offline (optional; else uses --embed-model-name).",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
) -> None:
    """Run ingest + recall once; print recall@1, @5, @10."""
    if run_mode not in ("inprocess", "batch"):
        raise typer.BadParameter("run_mode must be 'inprocess' or 'batch'")
    result = _run_ingest_and_recall(
        pdf_dir=pdf_dir,
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_invoke_url=embed_invoke_url,
        embed_use_vllm_compat=embed_use_vllm_compat,
        embed_use_vllm_offline=embed_use_vllm_offline,
        embed_model_name=embed_model_name,
        embed_model_path=embed_model_path,
        run_mode=run_mode,
        ray_address=ray_address,
    )
    m = result["metrics"]
    print(f"Queries: {result['n_queries']}")
    print(f"Ingest (s): {result['ingest_elapsed_s']:.2f}")
    print(f"recall@1:  {m.get('recall@1', 0):.4f}")
    print(f"recall@5:  {m.get('recall@5', 0):.4f}")
    print(f"recall@10: {m.get('recall@10', 0):.4f}")


@app.command("check-server")
def check_server(
    embed_invoke_url: str = typer.Argument(
        ...,
        help="vLLM embedding base URL (e.g. http://localhost:8000/v1).",
    ),
) -> None:
    """Probe the embedding server; print the first model id or an error."""
    from nemo_retriever.text_embed.vllm_http import get_model_id_from_server

    model_id = get_model_id_from_server(embed_invoke_url)
    if model_id:
        typer.echo(f"OK: server reachable, model id: {model_id}")
    else:
        typer.echo(
            "Could not reach server or no models listed. "
            f"Ensure vLLM is running (e.g. vllm serve <path> --runner pooling ...) and try: {embed_invoke_url}/models",
            err=True,
        )
        raise typer.Exit(1)


@app.command("compare")
def compare(
    pdf_dir: Path = typer.Option(
        JD5_PDF_DIR,
        "--pdf-dir",
        path_type=Path,
        help="Directory containing PDFs to ingest.",
    ),
    query_csv: Path = typer.Option(
        JD5_QUERY_CSV,
        "--query-csv",
        path_type=Path,
        help="Query ground-truth CSV.",
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", help="LanceDB URI."),
    run_mode: str = typer.Option(
        "inprocess",
        "--run-mode",
        help="Ingest run mode: inprocess or batch (batch uses Ray Data).",
    ),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray cluster address for batch (e.g. auto)."),
    embed_invoke_url: str | None = typer.Option(
        None,
        "--embed-invoke-url",
        help="vLLM embedding endpoint (e.g. http://localhost:8000/v1). Include to run vLLM server path.",
    ),
    embed_use_vllm_offline: bool = typer.Option(
        False,
        "--embed-use-vllm-offline/--no-embed-use-vllm-offline",
        help="Include vLLM offline path in comparison.",
    ),
    embed_model_path: str | None = typer.Option(
        None,
        "--embed-model-path",
        help="Local path to model for vLLM offline (optional).",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
) -> None:
    """Run baseline (local HF); optionally vLLM server and/or vLLM offline; print metrics side-by-side."""
    if not embed_invoke_url and not embed_use_vllm_offline:
        typer.echo(
            "For compare, set at least one of: --embed-invoke-url (vLLM server) or --embed-use-vllm-offline.",
            err=True,
        )
        raise typer.Exit(1)
    if run_mode not in ("inprocess", "batch"):
        raise typer.BadParameter("run_mode must be 'inprocess' or 'batch'")
    pdf_dir = Path(pdf_dir)
    query_csv = Path(query_csv)

    vllm_model = embed_model_name
    if embed_invoke_url:
        from nemo_retriever.text_embed.vllm_http import get_model_id_from_server

        discovered = get_model_id_from_server(embed_invoke_url)
        if discovered:
            typer.echo(f"Using vLLM model id from server: {discovered}")
            vllm_model = discovered
        else:
            typer.echo("Warning: could not discover model from server, using --embed-model-name.", err=True)

    _run = lambda **kw: _run_ingest_and_recall(  # noqa: E731
        pdf_dir=pdf_dir,
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        run_mode=run_mode,
        ray_address=ray_address,
        **kw,
    )
    typer.echo("Running baseline (local HF)...")
    baseline = _run(
        table_name=TABLE_BASELINE,
        embed_invoke_url=None,
        embed_use_vllm_compat=False,
        embed_use_vllm_offline=False,
        embed_model_name=embed_model_name,
        embed_model_path=None,
    )
    results = [("Baseline (HF)", baseline)]

    if embed_invoke_url:
        typer.echo("Running vLLM server...")
        vllm = _run(
            table_name=TABLE_VLLM,
            embed_invoke_url=embed_invoke_url,
            embed_use_vllm_compat=True,
            embed_use_vllm_offline=False,
            embed_model_name=vllm_model,
            embed_model_path=None,
        )
        results.append(("vLLM server", vllm))

    if embed_use_vllm_offline:
        typer.echo("Running vLLM offline...")
        vllm_off = _run(
            table_name=TABLE_VLLM_OFFLINE,
            embed_invoke_url=None,
            embed_use_vllm_compat=False,
            embed_use_vllm_offline=True,
            embed_model_name=embed_model_name,
            embed_model_path=embed_model_path,
        )
        results.append(("vLLM offline", vllm_off))

    # Print side-by-side table
    names = [r[0] for r in results]
    col_w = max(16, max(len(n) for n in names))
    header = f"{'Metric':<12} " + " ".join(f"{n:<{col_w}}" for n in names)
    print("\n--- Comparison ---")
    print(header)
    print("-" * len(header))
    for k in ("recall@1", "recall@5", "recall@10"):
        row = f"{k:<12} " + " ".join(f"{r[1]['metrics'].get(k, 0):<{col_w}.4f}" for r in results)
        print(row)
    ingest_row = "Ingest (s)   " + " ".join(f"{r[1]['ingest_elapsed_s']:.2f}".ljust(col_w) for r in results)
    print(f"\n{ingest_row}")


def _run_recall_only(
    query_csv: Path,
    lancedb_uri: str,
    table_name: str,
    embed_model_name: str,
    embed_invoke_url: str | None = None,
    embed_use_vllm_offline: bool = False,
    embed_model_path: str | None = None,
) -> dict:
    """Run recall only (no ingest); return metrics."""
    cfg = RecallConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=table_name,
        embedding_model=embed_model_name,
        embedding_http_endpoint=embed_invoke_url,
        embedding_use_vllm_compat=False,
        embedding_use_vllm_offline=embed_use_vllm_offline,
        embedding_vllm_model_path=embed_model_path,
        top_k=10,
        ks=(1, 5, 10),
    )
    _df, _gold, _raw, _keys, metrics = retrieve_and_score(
        query_csv=query_csv,
        cfg=cfg,
        vector_column_name="vector",
    )
    return {"metrics": metrics, "n_queries": len(_df)}


@app.command("save-pre-embed")
def save_pre_embed(
    pdf_dir: Path = typer.Option(
        JD5_PDF_DIR,
        "--pdf-dir",
        path_type=Path,
        help="Directory containing PDFs to ingest.",
    ),
    pre_embed_dir: Path = typer.Option(
        ...,
        "--pre-embed-dir",
        path_type=Path,
        help="Output directory for pre-embed parquet (post-explode, pre-embed).",
    ),
    run_mode: str = typer.Option(
        "batch",
        "--run-mode",
        help="Run mode (batch only for save-pre-embed).",
    ),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray cluster address (e.g. auto)."),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name (used for modality; no model loaded).",
    ),
) -> None:
    """Run pipeline through explode and write pre-embed parquet. No embed model loaded."""
    if run_mode != "batch":
        raise typer.BadParameter("save-pre-embed requires --run-mode batch")
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise typer.BadParameter(f"PDF directory not found: {pdf_dir}")

    glob_pattern = str(pdf_dir / "*.pdf")
    create_kwargs: dict = {"run_mode": "batch"}
    if ray_address is not None:
        create_kwargs["params"] = IngestorCreateParams(ray_address=ray_address)
    ingestor = create_ingestor(**create_kwargs)
    ingestor = (
        ingestor.files(glob_pattern)
        .extract(
            ExtractParams(
                method="pdfium",
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
            )
        )
        .prepare_for_embed(EmbedParams(model_name=embed_model_name))
        .save_intermediate_results(str(pre_embed_dir))
    )
    typer.echo(f"Wrote pre-embed parquet to {pre_embed_dir}")


def _detect_flashinfer_cubin() -> bool:
    """Return True if flashinfer_cubin is importable (prebuilt FlashInfer kernels)."""
    try:
        import flashinfer_cubin  # noqa: F401

        return True
    except ImportError:
        return False


# CSV/JSON output column order for sweep and single-run logging
_COMPARISON_ROW_KEYS = [
    "gpu_memory_utilization",
    "max_rows",
    "enforce_eager",
    "flashinfer_cubin",
    "baseline_recall_at_1",
    "baseline_recall_at_5",
    "baseline_recall_at_10",
    "baseline_ingest_s",
    "vllm_recall_at_1",
    "vllm_recall_at_5",
    "vllm_recall_at_10",
    "vllm_ingest_s",
]


def run_compare_from_pre_embed(
    *,
    pre_embed_dir: Path,
    query_csv: Path,
    lancedb_uri: str,
    ray_address: str | None,
    embed_model_path: str | None,
    embed_model_name: str,
    max_rows: int | None,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    compile_cache_dir: Path | None,
    sort_key_column: str | None = None,
) -> dict:
    """Run baseline + vLLM offline from pre-embed parquet; return one row of metrics as a dict.

    Used by compare-from-pre-embed (CLI) and by the sweep script. Does not print;
    caller may append the returned dict to --output-csv / --output-json.

    Ray is started first. A long-lived EmbedServiceActor is created per backend (baseline,
    then vLLM), warmed up with one batch, then the timed ingest pipeline calls the service
    so model load is excluded from the timed run. The service is torn down after each backend.
    """
    import ray
    import ray.data as rd

    flashinfer_cubin = _detect_flashinfer_cubin()
    create_kwargs: dict = {"run_mode": "batch"}
    if ray_address is not None:
        create_kwargs["params"] = IngestorCreateParams(ray_address=ray_address)

    # Start Ray cluster first so init time is not included in ingest timing.
    # We use only the CLI --ray-address (not RAY_ADDRESS env); None => "local" so we start our own cluster.
    ray_addr = ray_address or "local"
    if not ray.is_initialized():
        ray.init(address=ray_addr, ignore_reinit_error=True, log_to_driver=True)
    print(
        f"Ray address: {ray_addr} (RAY_ADDRESS env is {'set' if os.environ.get('RAY_ADDRESS') else 'unset'})",
        flush=True,
    )
    _ = ray.cluster_resources()
    # EmbedServiceActor requires num_gpus=1; without a GPU the actor stays pending forever.
    available = ray.available_resources()
    cluster = ray.cluster_resources()
    num_gpus = available.get("GPU", 0)
    print(f"Ray cluster_resources={cluster} available_resources={available} (embed service needs 1 GPU)", flush=True)
    if num_gpus < 1:
        raise RuntimeError(
            "Embed service requires at least 1 GPU. Ray reports 0 GPUs available "
            f"(cluster_resources={cluster}, available_resources={available}). "
            "Ensure the cluster has a GPU and Ray can see it "
            "(e.g. run with CUDA_VISIBLE_DEVICES set, or use a Ray cluster started with --num-gpus=1)."
        )

    def _limit(ds: "rd.Dataset", n: int, sort_key: str | None) -> "rd.Dataset":
        if sort_key is not None:
            ds = ds.sort(sort_key)
        return ds.limit(n)

    def _warmup_batch(pre_embed_dir: Path, sort_key: str | None):
        ds = rd.read_parquet(str(pre_embed_dir))
        ds = _limit(ds, 1, sort_key)
        return next(iter(ds.iter_batches(batch_format="pandas", batch_size=1)))

    # Baseline: long-lived service -> warmup -> timed ingest -> teardown
    try:
        ray.get_actor("embed_service_baseline")
        ray.kill(ray.get_actor("embed_service_baseline"))
    except ValueError:
        pass
    baseline_params = EmbedParams(
        model_name=embed_model_name,
        embed_use_vllm_offline=False,
        embed_model_path=None,
    )
    service_baseline = EmbedServiceActor.options(name="embed_service_baseline").remote(baseline_params)
    warmup_df = _warmup_batch(pre_embed_dir, sort_key_column)
    _warmup_timeout_s = 300
    try:
        ray.get(service_baseline.embed_batch.remote(warmup_df), timeout=_warmup_timeout_s)
    except ray.exceptions.GetTimeoutError:
        ray.kill(service_baseline)
        raise RuntimeError(
            f"Baseline embed service warmup did not complete within {_warmup_timeout_s}s. "
            "Actor may be stuck pending (no GPU placement). Check ray.available_resources() and cluster GPU config."
        ) from None
    ds_baseline = rd.read_parquet(str(pre_embed_dir))
    if max_rows is not None:
        ds_baseline = _limit(ds_baseline, max_rows, sort_key_column)
    ingestor_b = create_ingestor(**create_kwargs)
    ingestor_b._rd_dataset = ds_baseline
    ingestor_b._input_documents = []
    t0 = time.perf_counter()
    ingestor_b.embed_only(
        EmbedParams(
            model_name=embed_model_name,
            embed_use_vllm_offline=False,
            embed_model_path=None,
        ),
        embedding_service_name="embed_service_baseline",
    ).vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": lancedb_uri,
                "table_name": TABLE_BASELINE,
                "overwrite": True,
                "create_index": True,
            }
        )
    ).ingest(
        params=IngestExecuteParams(parallel=True, max_workers=8, show_progress=True)
    )
    baseline_ingest_s = time.perf_counter() - t0
    ray.kill(service_baseline)
    baseline_recall = _run_recall_only(
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_BASELINE,
        embed_model_name=embed_model_name,
        embed_use_vllm_offline=False,
    )

    # vLLM offline: long-lived service -> warmup -> timed ingest -> teardown
    try:
        ray.get_actor("embed_service_vllm")
        ray.kill(ray.get_actor("embed_service_vllm"))
    except ValueError:
        pass
    vllm_runtime = ModelRuntimeParams(
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        compile_cache_dir=str(compile_cache_dir) if compile_cache_dir else None,
    )
    vllm_params = EmbedParams(
        model_name=embed_model_name,
        embed_use_vllm_offline=True,
        embed_model_path=embed_model_path,
        runtime=vllm_runtime,
    )
    service_vllm = EmbedServiceActor.options(name="embed_service_vllm").remote(vllm_params)
    try:
        ray.get(service_vllm.embed_batch.remote(warmup_df), timeout=_warmup_timeout_s)
    except ray.exceptions.GetTimeoutError:
        ray.kill(service_vllm)
        raise RuntimeError(
            f"vLLM embed service warmup did not complete within {_warmup_timeout_s}s. "
            "Actor may be stuck pending (no GPU placement). Check ray.available_resources() and cluster GPU config."
        ) from None
    ds_vllm = rd.read_parquet(str(pre_embed_dir))
    if max_rows is not None:
        ds_vllm = _limit(ds_vllm, max_rows, sort_key_column)
    ingestor_v = create_ingestor(**create_kwargs)
    ingestor_v._rd_dataset = ds_vllm
    ingestor_v._input_documents = []
    t0 = time.perf_counter()
    ingestor_v.embed_only(
        EmbedParams(
            model_name=embed_model_name,
            embed_use_vllm_offline=True,
            embed_model_path=embed_model_path,
            runtime=vllm_runtime,
        ),
        embedding_service_name="embed_service_vllm",
    ).vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": lancedb_uri,
                "table_name": TABLE_VLLM_OFFLINE,
                "overwrite": True,
                "create_index": True,
            }
        )
    ).ingest(
        params=IngestExecuteParams(parallel=True, max_workers=8, show_progress=True)
    )
    vllm_ingest_s = time.perf_counter() - t0
    ray.kill(service_vllm)
    vllm_recall = _run_recall_only(
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_VLLM_OFFLINE,
        embed_model_name=embed_model_name,
        embed_use_vllm_offline=True,
        embed_model_path=embed_model_path,
    )

    return {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_rows": max_rows,
        "enforce_eager": enforce_eager,
        "flashinfer_cubin": flashinfer_cubin,
        "baseline_recall_at_1": baseline_recall["metrics"].get("recall@1"),
        "baseline_recall_at_5": baseline_recall["metrics"].get("recall@5"),
        "baseline_recall_at_10": baseline_recall["metrics"].get("recall@10"),
        "baseline_ingest_s": baseline_ingest_s,
        "vllm_recall_at_1": vllm_recall["metrics"].get("recall@1"),
        "vllm_recall_at_5": vllm_recall["metrics"].get("recall@5"),
        "vllm_recall_at_10": vllm_recall["metrics"].get("recall@10"),
        "vllm_ingest_s": vllm_ingest_s,
    }


def _append_comparison_row_csv(row: dict, path: Path) -> None:
    """Append one row (with keys in _COMPARISON_ROW_KEYS) to a CSV file; create with header if missing."""
    path = Path(path)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COMPARISON_ROW_KEYS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in _COMPARISON_ROW_KEYS})


def _append_comparison_row_json(row: dict, path: Path) -> None:
    """Append one row as a JSON line to a JSONL file."""
    with path.open("a") as f:
        f.write(json.dumps({k: row.get(k) for k in _COMPARISON_ROW_KEYS}) + "\n")


@app.command("compare-from-pre-embed")
def compare_from_pre_embed(
    pre_embed_dir: Path = typer.Option(
        ...,
        "--pre-embed-dir",
        path_type=Path,
        help="Directory containing pre-embed parquet (from save-pre-embed).",
    ),
    query_csv: Path = typer.Option(
        JD5_QUERY_CSV,
        "--query-csv",
        path_type=Path,
        help="Query ground-truth CSV.",
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", help="LanceDB URI."),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray cluster address for batch (e.g. auto)."),
    embed_model_path: str | None = typer.Option(
        None,
        "--embed-model-path",
        help="Local path to model for vLLM offline (optional).",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="If set, use only this many rows from pre-embed for both baseline and vLLM (faster iteration).",
    ),
    gpu_memory_utilization: float = typer.Option(
        0.55,
        "--gpu-memory-utilization",
        help="vLLM: fraction of GPU memory for KV cache (default 0.55). Increase (e.g. 0.6–0.7) for throughput.",
    ),
    enforce_eager: bool = typer.Option(
        False,
        "--enforce-eager/--no-enforce-eager",
        help="vLLM: if set, disable CUDA graphs/torch.compile (slower but avoids noexec/compile issues).",
    ),
    compile_cache_dir: str | None = typer.Option(
        None,
        "--compile-cache-dir",
        path_type=Path,
        help="vLLM: dir for torch inductor/Triton cache when enforce_eager=False (must be on non-noexec fs).",
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        path_type=Path,
        help="If set, append one row of metrics (params + recall + ingest s) to this CSV for sweep/plotting.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        path_type=Path,
        help="If set, append one row as a JSON line (JSONL) to this file.",
    ),
    sort_key_column: str | None = typer.Option(
        None,
        "--sort-key",
        help="If set, sort pre-embed dataset by this column before limit (deterministic same rows for baseline and "
        "vLLM).",
    ),
) -> None:
    """Load pre-embed parquet, run baseline + vLLM offline embed+vdb, then recall; print comparison.

    For faster iteration use --max-rows (e.g. with bo767 pre-embed at
    /raid/charlesb/datasets/bo767_pre_embed) to compare on a subset.
    Use --output-csv to append one row for data collection sweeps.
    """
    pre_embed_dir = Path(pre_embed_dir)
    query_csv = Path(query_csv)
    if not pre_embed_dir.is_dir():
        raise typer.BadParameter(f"Pre-embed directory not found: {pre_embed_dir}")
    if not query_csv.is_file():
        raise typer.BadParameter(f"Query CSV not found: {query_csv}")

    if max_rows is not None:
        typer.echo(f"Using up to {max_rows} rows (--max-rows={max_rows})")
    if sort_key_column is not None:
        typer.echo(f"Sorting by {sort_key_column!r} before limit for deterministic row set.")

    typer.echo("Running baseline (local HF) then vLLM offline from pre-embed...")
    row = run_compare_from_pre_embed(
        pre_embed_dir=pre_embed_dir,
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        ray_address=ray_address,
        embed_model_path=embed_model_path,
        embed_model_name=embed_model_name,
        max_rows=max_rows,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        compile_cache_dir=Path(compile_cache_dir) if compile_cache_dir else None,
        sort_key_column=sort_key_column,
    )

    # Build results for human-readable table from returned row
    results = [
        (
            "Baseline (HF)",
            {
                "metrics": {
                    "recall@1": row["baseline_recall_at_1"],
                    "recall@5": row["baseline_recall_at_5"],
                    "recall@10": row["baseline_recall_at_10"],
                },
                "ingest_elapsed_s": row["baseline_ingest_s"],
            },
        ),
        (
            "vLLM offline",
            {
                "metrics": {
                    "recall@1": row["vllm_recall_at_1"],
                    "recall@5": row["vllm_recall_at_5"],
                    "recall@10": row["vllm_recall_at_10"],
                },
                "ingest_elapsed_s": row["vllm_ingest_s"],
            },
        ),
    ]
    names = [r[0] for r in results]
    col_w = max(16, max(len(n) for n in names))
    header = f"{'Metric':<12} " + " ".join(f"{n:<{col_w}}" for n in names)
    print("\n--- Comparison (from pre-embed) ---")
    print(header)
    print("-" * len(header))
    for k in ("recall@1", "recall@5", "recall@10"):
        row_line = f"{k:<12} " + " ".join(f"{r[1]['metrics'].get(k, 0):<{col_w}.4f}" for r in results)
        print(row_line)
    ingest_row = "Ingest (s)   " + " ".join(f"{r[1]['ingest_elapsed_s']:.2f}".ljust(col_w) for r in results)
    print(f"\n{ingest_row}")

    if output_csv is not None:
        _append_comparison_row_csv(row, Path(output_csv))
        typer.echo(f"Appended row to {output_csv}")
    if output_json is not None:
        _append_comparison_row_json(row, Path(output_json))
        typer.echo(f"Appended row to {output_json}")


def _metrics_equal(a: dict, b: dict, tol: float = 1e-5) -> bool:
    """Return True if recall@1/5/10 in a and b are within tol."""
    for k in ("recall@1", "recall@5", "recall@10"):
        va = a.get(k)
        vb = b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return False
        if abs(float(va) - float(vb)) > tol:
            return False
    return True


@app.command("validate-recall-parity")
def validate_recall_parity(
    pdf_dir: Path = typer.Option(
        JD5_PDF_DIR,
        "--pdf-dir",
        path_type=Path,
        help="Directory containing PDFs.",
    ),
    query_csv: Path = typer.Option(
        JD5_QUERY_CSV,
        "--query-csv",
        path_type=Path,
        help="Query ground-truth CSV.",
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", help="LanceDB URI."),
    pre_embed_dir: Path = typer.Option(
        ...,
        "--pre-embed-dir",
        path_type=Path,
        help="Directory to write/read pre-embed parquet (will be overwritten).",
    ),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray cluster address for batch (e.g. auto)."),
    embed_model_path: str | None = typer.Option(
        None,
        "--embed-model-path",
        help="Local path to model for vLLM offline (optional).",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
    tolerance: float = typer.Option(
        1e-5,
        "--tolerance",
        help="Max allowed difference in recall metrics between full and pre-embed paths.",
    ),
) -> None:
    """Run full compare, then save-pre-embed + compare-from-pre-embed; exit non-zero if recall differs."""
    pdf_dir = Path(pdf_dir)
    query_csv = Path(query_csv)
    pre_embed_dir = Path(pre_embed_dir)
    if not pdf_dir.is_dir():
        raise typer.BadParameter(f"PDF directory not found: {pdf_dir}")
    if not query_csv.is_file():
        raise typer.BadParameter(f"Query CSV not found: {query_csv}")

    typer.echo("Step 1: Full compare (baseline + vLLM offline)...")
    full_baseline = _run_ingest_and_recall(
        pdf_dir=pdf_dir,
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_BASELINE,
        embed_invoke_url=None,
        embed_use_vllm_compat=False,
        embed_use_vllm_offline=False,
        embed_model_name=embed_model_name,
        embed_model_path=None,
        run_mode="batch",
        ray_address=ray_address,
    )
    full_vllm = _run_ingest_and_recall(
        pdf_dir=pdf_dir,
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_VLLM_OFFLINE,
        embed_invoke_url=None,
        embed_use_vllm_compat=False,
        embed_use_vllm_offline=True,
        embed_model_name=embed_model_name,
        embed_model_path=embed_model_path,
        run_mode="batch",
        ray_address=ray_address,
    )

    typer.echo("Step 2: Save pre-embed parquet...")
    pre_embed_dir.mkdir(parents=True, exist_ok=True)
    create_kwargs: dict = {"run_mode": "batch"}
    if ray_address is not None:
        create_kwargs["params"] = IngestorCreateParams(ray_address=ray_address)
    ingestor = create_ingestor(**create_kwargs)
    ingestor = (
        ingestor.files(str(pdf_dir / "*.pdf"))
        .extract(
            ExtractParams(
                method="pdfium",
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
            )
        )
        .prepare_for_embed(EmbedParams(model_name=embed_model_name))
        .save_intermediate_results(str(pre_embed_dir))
    )

    typer.echo("Step 3: Compare from pre-embed...")
    import ray.data as rd

    create_kwargs2: dict = {"run_mode": "batch"}
    if ray_address is not None:
        create_kwargs2["params"] = IngestorCreateParams(ray_address=ray_address)

    ds_b = rd.read_parquet(str(pre_embed_dir))
    ingestor_b = create_ingestor(**create_kwargs2)
    ingestor_b._rd_dataset = ds_b
    ingestor_b._input_documents = []
    ingestor_b.embed_only(
        EmbedParams(model_name=embed_model_name, embed_use_vllm_offline=False, embed_model_path=None)
    ).vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": lancedb_uri,
                "table_name": TABLE_BASELINE,
                "overwrite": True,
                "create_index": True,
            }
        )
    ).ingest(
        params=IngestExecuteParams(parallel=True, max_workers=8, show_progress=True)
    )
    preembed_baseline = _run_recall_only(
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_BASELINE,
        embed_model_name=embed_model_name,
        embed_use_vllm_offline=False,
    )

    ds_v = rd.read_parquet(str(pre_embed_dir))
    ingestor_v = create_ingestor(**create_kwargs2)
    ingestor_v._rd_dataset = ds_v
    ingestor_v._input_documents = []
    ingestor_v.embed_only(
        EmbedParams(
            model_name=embed_model_name,
            embed_use_vllm_offline=True,
            embed_model_path=embed_model_path,
        )
    ).vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": lancedb_uri,
                "table_name": TABLE_VLLM_OFFLINE,
                "overwrite": True,
                "create_index": True,
            }
        )
    ).ingest(
        params=IngestExecuteParams(parallel=True, max_workers=8, show_progress=True)
    )
    preembed_vllm = _run_recall_only(
        query_csv=query_csv,
        lancedb_uri=lancedb_uri,
        table_name=TABLE_VLLM_OFFLINE,
        embed_model_name=embed_model_name,
        embed_use_vllm_offline=True,
        embed_model_path=embed_model_path,
    )

    typer.echo("Step 4: Compare metrics...")
    ok = True
    if not _metrics_equal(full_baseline["metrics"], preembed_baseline["metrics"], tol=tolerance):
        typer.echo(
            f"Baseline recall mismatch: full={full_baseline['metrics']} vs pre-embed={preembed_baseline['metrics']}",
            err=True,
        )
        ok = False
    if not _metrics_equal(full_vllm["metrics"], preembed_vllm["metrics"], tol=tolerance):
        typer.echo(
            f"vLLM recall mismatch: full={full_vllm['metrics']} vs pre-embed={preembed_vllm['metrics']}",
            err=True,
        )
        ok = False
    if ok:
        typer.echo("Recall parity: OK (full pipeline vs pre-embed path match within tolerance).")
    else:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
