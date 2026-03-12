# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified ingestion pipeline: inprocess (local) or online (REST API) with optional recall.

- Inprocess: runs the full pipeline locally (no server).
- Online sync: submits each document to the online ingest REST service
  synchronously (start with ``retriever online serve``).
- Online async: submits documents asynchronously and polls for results.

Supports all file types: PDF, DOCX, PPTX, TXT, HTML, images, and audio.

Run with:
  uv run python -m nemo_retriever.examples.online_pipeline <input-dir>
  uv run python -m nemo_retriever.examples.online_pipeline <input-dir> \
    --run-mode online --base-url http://localhost:7670
  uv run python -m nemo_retriever.examples.online_pipeline <input-dir> \
    --run-mode online --async-mode --base-url http://localhost:7670
"""

from pathlib import Path

import typer
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"

_INPUT_TYPE_GLOBS = {
    "pdf": ["*.pdf"],
    "txt": ["*.txt"],
    "html": ["*.html", "*.htm"],
    "doc": ["*.docx", "*.pptx"],
    "image": ["*.png", "*.jpg", "*.jpeg", "*.tiff"],
    "audio": ["*.mp3", "*.wav", "*.m4a"],
}


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing documents to ingest.",
        path_type=Path,
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
        help="Input format: 'pdf', 'txt', 'html', 'doc', 'image', or 'audio'.",
    ),
    async_mode: bool = typer.Option(
        False,
        "--async-mode",
        help="Use async ingest (online mode only). Submits all docs, then polls.",
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Recall is skipped if the file does not exist.",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF text extraction method: 'pdfium', 'pdfium_hybrid', or 'ocr'.",
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help="Do not print per-query retrieval details.",
    ),
    query_text: str = typer.Option(
        "",
        "--query",
        "-q",
        help="If set, run a retrieval query after ingest (online mode uses /query endpoint).",
    ),
) -> None:
    if run_mode not in ("inprocess", "online"):
        raise typer.BadParameter("run_mode must be 'inprocess' or 'online'")

    input_path = Path(input_path)
    if input_path.is_file():
        file_patterns = [str(input_path)]
    elif input_path.is_dir():
        exts = _INPUT_TYPE_GLOBS.get(input_type, ["*.pdf"])
        file_patterns = [str(input_path / e) for e in exts]
    else:
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    if run_mode == "online":
        _run_online(
            file_patterns=file_patterns,
            base_url=base_url,
            method=method,
            async_mode=async_mode,
            input_type=input_type,
        )
    else:
        _run_inprocess(
            file_patterns=file_patterns,
            method=method,
            input_type=input_type,
        )

    # Optional query after ingest
    if query_text.strip():
        _run_query(
            query_text=query_text.strip(),
            run_mode=run_mode,
            base_url=base_url,
        )

    # Recall evaluation (same for both modes)
    query_csv = Path(query_csv)
    if not query_csv.exists():
        typer.echo(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        return

    _run_recall(query_csv, input_type, no_recall_details)


def _run_online(
    *,
    file_patterns: list[str],
    base_url: str,
    method: str,
    async_mode: bool,
    input_type: str,
) -> None:
    ingestor = create_ingestor(run_mode="online", params=IngestorCreateParams(base_url=base_url))
    ingestor = (
        ingestor.files(file_patterns)
        .extract(ExtractParams(method=method, extract_text=True, extract_tables=True, extract_charts=True))
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

    if async_mode:
        typer.echo("Submitting documents asynchronously to online ingest service...")
        ingestor.ingest_async()
        typer.echo("Polling for results...")
        results = ingestor.poll_results(timeout=600.0, poll_interval=2.0)
    else:
        typer.echo("Submitting documents to online ingest service...")
        results = ingestor.ingest()

    ok_count = sum(1 for r in results if r.get("ok"))
    total_rows = sum(int(r.get("rows_written", 0)) for r in results if r.get("ok"))
    total_sec = sum(float(r.get("total_duration_sec", 0)) for r in results if r.get("ok"))
    typer.echo(f"Online ingest: {ok_count}/{len(results)} docs OK, {total_rows} rows, {total_sec:.2f}s total.")
    for r in results:
        if not r.get("ok"):
            typer.echo(f"  FAIL {r.get('source_path', '?')}: {r.get('error', 'unknown')}", err=True)


def _run_inprocess(
    *,
    file_patterns: list[str],
    method: str,
    input_type: str,
) -> None:
    ingestor = create_ingestor(run_mode="inprocess")
    if input_type == "txt":
        ingestor = (
            ingestor.files(file_patterns)
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
    else:
        ingestor = (
            ingestor.files(file_patterns)
            .extract(ExtractParams(method=method, extract_text=True, extract_tables=True, extract_charts=True))
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


def _run_query(*, query_text: str, run_mode: str, base_url: str) -> None:
    if run_mode == "online":
        import requests

        url = f"{base_url.rstrip('/')}/query"
        resp = requests.post(url, json={"query": query_text, "top_k": 10}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        typer.echo(f"\nQuery: {query_text}")
        typer.echo(f"Hits: {len(hits)}")
        for i, hit in enumerate(hits[:10], 1):
            source = hit.get("source", "?")
            page = hit.get("page_number", "?")
            text_preview = (hit.get("text", "") or "")[:120]
            typer.echo(f"  {i:02d}. {source} p{page}")
            if text_preview:
                typer.echo(f"      {text_preview}...")
    else:
        from nemo_retriever.retriever import Retriever

        retriever = Retriever(lancedb_uri=LANCEDB_URI, lancedb_table=LANCEDB_TABLE)
        hits = retriever.query(query_text)
        typer.echo(f"\nQuery: {query_text}")
        typer.echo(f"Hits: {len(hits)}")
        for i, hit in enumerate(hits[:10], 1):
            source = hit.get("source", "?")
            page = hit.get("page_number", "?")
            text_preview = (hit.get("text", "") or "")[:120]
            typer.echo(f"  {i:02d}. {source} p{page}")
            if text_preview:
                typer.echo(f"      {text_preview}...")


def _run_recall(query_csv: Path, input_type: str, no_recall_details: bool) -> None:
    import lancedb
    from nemo_retriever.recall.core import (
        RecallConfig,
        gold_to_doc_page,
        hit_key_and_distance,
        is_hit_at_k,
        retrieve_and_score,
    )

    db = lancedb.connect(f"./{LANCEDB_URI}")
    table = db.open_table(LANCEDB_TABLE)
    unique_basenames = table.to_pandas()["pdf_basename"].unique()
    typer.echo(f"Unique basenames in VDB: {unique_basenames}")

    cfg = RecallConfig(
        lancedb_uri=str(LANCEDB_URI),
        lancedb_table=str(LANCEDB_TABLE),
        embedding_model="nvidia/llama-nemotron-embed-1b-v2",
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
        doc, page = gold_to_doc_page(g)
        scored_hits: list[tuple[str, float | None]] = []
        for h in hits:
            key, dist = hit_key_and_distance(h)
            if key:
                scored_hits.append((key, dist))
        top_keys = [k for (k, _d) in scored_hits]
        hit = is_hit_at_k(g, top_keys, cfg.top_k, match_mode="pdf_page")
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
