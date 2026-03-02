# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified ingestion pipeline: inprocess (local) or online (REST API) with optional recall.

- Inprocess: runs the full pipeline locally (no server).
- Online: submits each document to the online ingest REST service (start with
  `retriever online serve`). Uses the same LanceDB for recall evaluation.

Run with:
  uv run python -m nemo_retriever.examples.online_pipeline <input-dir>
  uv run python -m nemo_retriever.examples.online_pipeline <input-dir>
  --run-mode online --base-url http://localhost:7670
"""

import json
from pathlib import Path

import lancedb
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


def _gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def _is_hit_at_k(golden_key: str, retrieved_keys: list[str], k: int) -> bool:
    doc, page = _gold_to_doc_page(golden_key)
    specific_page = f"{doc}_{page}"
    entire_document = f"{doc}_-1"
    top = (retrieved_keys or [])[: int(k)]
    return (specific_page in top) or (entire_document in top)


def _hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:
    try:
        res = json.loads(hit.get("metadata", "{}"))
        source = json.loads(hit.get("source", "{}"))
    except Exception:
        return None, None

    source_id = source.get("source_id")
    page_number = res.get("page_number")
    if not source_id or page_number is None:
        return None, float(hit.get("_distance")) if "_distance" in hit else None

    key = f"{Path(str(source_id)).stem}_{page_number}"
    dist = float(hit.get("_distance")) if "_distance" in hit else None
    return key, dist


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs or .txt files to ingest.",
        path_type=Path,
        exists=True,
    ),
    run_mode: str = typer.Option(
        "inprocess",
        "--run-mode",
        "-m",
        help="'inprocess' (local pipeline) or 'online' (submit to REST API).",
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the online ingest service (used when run-mode=online).",
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', or 'doc'. Online mode supports PDF only.",
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Recall is skipped if the file does not exist.",
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help="Do not print per-query retrieval details.",
    ),
) -> None:
    if run_mode not in ("inprocess", "online"):
        raise typer.BadParameter("run_mode must be 'inprocess' or 'online'")

    if run_mode == "online" and input_type != "pdf":
        typer.echo("Online mode currently supports PDF only; use --input-type pdf.", err=True)
        raise typer.Exit(1)

    _ = input_type

    input_dir = Path(input_dir)

    if run_mode == "online":
        ingestor = create_ingestor(run_mode="online", params=IngestorCreateParams(base_url=base_url))
        glob_pattern = str(input_dir / "*.pdf")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract(ExtractParams(method="pdfium", extract_text=True, extract_tables=True, extract_charts=True))
            .embed(EmbedParams(model_name="nemo_retriever_v1"))
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": LANCEDB_URI,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": False,
                        "create_index": True,
                    }
                )
            )
        )
        typer.echo("Submitting documents to online ingest service...")
        results = ingestor.ingest()
        ok_count = sum(1 for r in results if r.get("ok"))
        total_rows = sum(int(r.get("rows_written", 0)) for r in results if r.get("ok"))
        total_sec = sum(float(r.get("total_duration_sec", 0)) for r in results if r.get("ok"))
        typer.echo(f"Online ingest: {ok_count}/{len(results)} docs OK, {total_rows} rows, {total_sec:.2f}s total.")
        for r in results:
            if not r.get("ok"):
                typer.echo(f"  FAIL {r.get('source_path', '?')}: {r.get('error', 'unknown')}", err=True)
    else:
        # Inprocess: same as inprocess_pipeline
        if input_type == "txt":
            glob_pattern = str(input_dir / "*.txt")
            ingestor = create_ingestor(run_mode="inprocess")
            ingestor = (
                ingestor.files(glob_pattern)
                .extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(EmbedParams(model_name="nemo_retriever_v1"))
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": LANCEDB_URI,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": False,
                            "create_index": True,
                        }
                    )
                )
            )
        elif input_type == "doc":
            doc_globs = [str(input_dir / "*.docx"), str(input_dir / "*.pptx")]
            ingestor = create_ingestor(run_mode="inprocess")
            ingestor = (
                ingestor.files(doc_globs)
                .extract(ExtractParams(method="pdfium", extract_text=True, extract_tables=True, extract_charts=True))
                .embed(EmbedParams(model_name="nemo_retriever_v1"))
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": LANCEDB_URI,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": False,
                            "create_index": True,
                        }
                    )
                )
            )
        else:
            glob_pattern = str(input_dir / "*.pdf")
            ingestor = create_ingestor(run_mode="inprocess")
            ingestor = (
                ingestor.files(glob_pattern)
                .extract(ExtractParams(method="pdfium", extract_text=True, extract_tables=True, extract_charts=True))
                .embed(EmbedParams(model_name="nemo_retriever_v1"))
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": LANCEDB_URI,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": False,
                            "create_index": True,
                        }
                    )
                )
            )
        typer.echo("Running inprocess extraction...")
        ingestor.ingest(params=IngestExecuteParams(show_progress=True))
        typer.echo("Extraction complete.")

    # Recall evaluation (same for both modes)
    query_csv = Path(query_csv)
    if not query_csv.exists():
        typer.echo(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        return

    db = lancedb.connect(f"./{LANCEDB_URI}")
    table = db.open_table(LANCEDB_TABLE)
    unique_basenames = table.to_pandas()["pdf_basename"].unique()
    typer.echo(f"Unique basenames in VDB: {unique_basenames}")

    cfg = RecallConfig(
        lancedb_uri=str(LANCEDB_URI),
        lancedb_table=str(LANCEDB_TABLE),
        embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k=10,
        ks=(1, 5, 10),
    )

    _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

    if not no_recall_details:
        typer.echo("\nPer-query retrieval details:")
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
            ext = ".txt" if input_type == "txt" else (".docx" if input_type == "doc" else ".pdf")
            typer.echo(f"\nQuery {i}: {q}")
            typer.echo(f"  Gold: {g}  (file: {doc}{ext}, page: {page})")
            typer.echo(f"  Hit@{cfg.top_k}: {hit}")
            typer.echo("  Top hits:")
            if not scored_hits:
                typer.echo("    (no hits)")
            else:
                for rank, (key, dist) in enumerate(scored_hits[: int(cfg.top_k)], start=1):
                    if dist is None:
                        typer.echo(f"    {rank:02d}. {key}")
                    else:
                        typer.echo(f"    {rank:02d}. {key}  distance={dist:.6f}")
        if not hit:
            ext = ".txt" if input_type == "txt" else (".docx" if input_type == "doc" else ".pdf")
            missed_gold.append((f"{doc}{ext}", str(page)))

    missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
    typer.echo("\nMissed gold (unique doc/page):")
    if not missed_unique:
        typer.echo("  (none)")
    else:
        for doc_page, page in missed_unique:
            typer.echo(f"  {doc_page} page {page}")
    typer.echo(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")
    typer.echo("\nRecall metrics:")
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    app()
