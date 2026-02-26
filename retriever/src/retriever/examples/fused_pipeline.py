# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused ingestion pipeline with optional recall evaluation.
Run with: uv run python -m retriever.examples.fused_pipeline <input-dir>
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import lancedb
import ray
import typer
from retriever import create_ingestor
from retriever.params import EmbedParams
from retriever.params import ExtractParams
from retriever.params import IngestExecuteParams
from retriever.params import IngestorCreateParams
from retriever.params import VdbUploadParams
from retriever.examples.batch_pipeline import (
    LANCEDB_TABLE,
    LANCEDB_URI,
    _configure_logging,
    _ensure_lancedb_table,
    _estimate_processed_pages,
    _gold_to_doc_page,
    _hit_key_and_distance,
    _is_hit_at_k,
    _print_detection_summary,
    _print_pages_per_second,
    _write_detection_summary,
    _collect_detection_summary,
)
from retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs to ingest.",
        path_type=Path,
        exists=True,
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    start_ray: bool = typer.Option(
        False,
        "--start-ray",
        help=(
            "Start a Ray head node (ray start --head) and connect to it. "
            "Dashboard at http://127.0.0.1:8265. Ignores --ray-address."
        ),
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help=(
            "Path to query CSV for recall evaluation. Default: bo767_query_gt.csv "
            "(current directory). Recall is skipped if the file does not exist."
        ),
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help=(
            "Do not print per-query retrieval details (query, gold, hits). "
            "Only the missed-gold summary and recall metrics are printed."
        ),
    ),
    pdf_extract_workers: int = typer.Option(
        12,
        "--pdf-extract-workers",
        min=1,
        help="Number of CPU workers for PDF extraction stage.",
    ),
    pdf_extract_num_cpus: float = typer.Option(
        2.0,
        "--pdf-extract-num-cpus",
        min=0.1,
        help="CPUs reserved per PDF extraction task.",
    ),
    pdf_extract_batch_size: int = typer.Option(
        4,
        "--pdf-extract-batch-size",
        min=1,
        help="Batch size for PDF extraction stage.",
    ),
    pdf_split_batch_size: int = typer.Option(
        1,
        "--pdf-split-batch-size",
        min=1,
        help="Batch size for PDF split stage.",
    ),
    fused_workers: int = typer.Option(
        1,
        "--fused-workers",
        min=1,
        help="Actor count for fused model stage.",
    ),
    fused_batch_size: int = typer.Option(
        64,
        "--fused-batch-size",
        min=1,
        help="Ray Data batch size for fused model stage.",
    ),
    fused_cpus_per_actor: float = typer.Option(
        1.0,
        "--fused-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per fused actor.",
    ),
    fused_gpus_per_actor: float = typer.Option(
        1.0,
        "--fused-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per fused actor.",
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None,
        "--runtime-metrics-dir",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        help="Optional directory where Ray runtime metrics are written per run.",
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(
        None,
        "--runtime-metrics-prefix",
        help="Optional filename prefix for per-run metrics artifacts.",
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        path_type=Path,
        dir_okay=False,
        help="Optional file to collect all pipeline + Ray driver logs for this run.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver (recommended with --log-file).",
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Optional JSON file path to write end-of-run detection counts summary.",
    ),
) -> None:
    log_handle, original_stdout, original_stderr = _configure_logging(log_file)
    try:
        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        if start_ray:
            subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
            ray_address = "auto"

        input_dir = Path(input_dir)
        pdf_glob = str(input_dir / "*.pdf")

        ingestor = create_ingestor(
            run_mode="fused",
            params=IngestorCreateParams(ray_address=ray_address, ray_log_to_driver=ray_log_to_driver),
        )
        ingestor = (
            ingestor.files(pdf_glob)
            .extract(
                ExtractParams(
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    batch_tuning={
                        "pdf_extract_workers": int(pdf_extract_workers),
                        "pdf_extract_num_cpus": float(pdf_extract_num_cpus),
                        "pdf_split_batch_size": int(pdf_split_batch_size),
                        "pdf_extract_batch_size": int(pdf_extract_batch_size),
                    },
                )
            )
            .embed(
                EmbedParams(
                    model_name="nemo_retriever_v1",
                    fused_tuning={
                        "fused_workers": int(fused_workers),
                        "fused_batch_size": int(fused_batch_size),
                        "fused_cpus_per_actor": float(fused_cpus_per_actor),
                        "fused_gpus_per_actor": float(fused_gpus_per_actor),
                    },
                )
            )
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": lancedb_uri,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                )
            )
        )

        print("Running extraction...")
        ingest_start = time.perf_counter()
        ingestor.ingest(
            params=IngestExecuteParams(
                runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
                runtime_metrics_prefix=runtime_metrics_prefix,
            )
        )
        ingest_elapsed_s = time.perf_counter() - ingest_start
        processed_pages = _estimate_processed_pages(lancedb_uri, LANCEDB_TABLE)
        detection_summary = _collect_detection_summary(lancedb_uri, LANCEDB_TABLE)
        print("Extraction complete.")
        _print_detection_summary(detection_summary)
        if detection_summary_file is not None:
            _write_detection_summary(detection_summary_file, detection_summary)
            print(f"Wrote detection summary JSON to {Path(detection_summary_file).expanduser().resolve()}")

        ray.shutdown()

        query_csv = Path(query_csv)
        if not query_csv.exists():
            print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return

        db = lancedb.connect(lancedb_uri)
        table = None
        open_err: Optional[Exception] = None
        for _ in range(3):
            try:
                table = db.open_table(LANCEDB_TABLE)
                open_err = None
                break
            except Exception as e:
                open_err = e
                _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)
                time.sleep(2)
        if table is None:
            raise RuntimeError(
                f"Recall stage requires LanceDB table {LANCEDB_TABLE!r} at {lancedb_uri!r}, " f"but it was not found."
            ) from open_err
        try:
            if int(table.count_rows()) == 0:
                print(f"LanceDB table {LANCEDB_TABLE!r} exists but is empty; skipping recall evaluation.")
                _print_pages_per_second(processed_pages, ingest_elapsed_s)
                return
        except Exception:
            pass

        unique_basenames = table.to_pandas()["pdf_basename"].unique()
        print(f"Unique basenames: {unique_basenames}")

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            top_k=10,
            ks=(1, 5, 10),
        )

        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

        if not no_recall_details:
            print("\nPer-query retrieval details:")
        missed_gold: list[tuple[str, str]] = []
        for i, (q, g, hits) in enumerate(
            zip(
                _df_query["query"].astype(str).tolist(),
                _gold,
                _raw_hits,
            )
        ):
            doc, page = _gold_to_doc_page(g)

            scored_hits: list[tuple[str, float | None]] = []
            for h in hits:
                key, dist = _hit_key_and_distance(h)
                if key:
                    scored_hits.append((key, dist))

            top_keys = [k for (k, _d) in scored_hits]
            hit = _is_hit_at_k(g, top_keys, cfg.top_k)

            if not no_recall_details:
                print(f"\nQuery {i}: {q}")
                print(f"  Gold: {g}  (file: {doc}.pdf, page: {page})")
                print(f"  Hit@{cfg.top_k}: {hit}")
                print("  Top hits:")
                if not scored_hits:
                    print("    (no hits)")
                else:
                    for rank, (key, dist) in enumerate(scored_hits[: int(cfg.top_k)], start=1):
                        if dist is None:
                            print(f"    {rank:02d}. {key}")
                        else:
                            print(f"    {rank:02d}. {key}  distance={dist:.6f}")

            if not hit:
                missed_gold.append((f"{doc}.pdf", str(page)))

        missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
        print("\nMissed gold (unique pdf/page):")
        if not missed_unique:
            print("  (none)")
        else:
            for pdf, page in missed_unique:
                print(f"  {pdf} page {page}")
        print(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")

        print("\nRecall metrics (matching retriever.recall.core):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
    finally:
        # Restore real stdio before closing the mirror file so exception hooks
        # and late flushes never write to a closed stream wrapper.
        import sys

        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            try:
                log_handle.flush()
            finally:
                log_handle.close()


if __name__ == "__main__":
    app()
