# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online service CLI: start the Ray Serve + FastAPI REST API and submit documents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    help="Online service: Ray Serve + FastAPI REST API (serve) and CLI to submit documents (submit).",
)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host for the HTTP server."),
    port: int = typer.Option(7670, "--port", help="Port for the HTTP server."),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray cluster address (default: local)."),
    extract_method: Optional[str] = typer.Option(
        None,
        "--extract-method",
        help="PDF extraction method (pdfium, pdfium_hybrid, ocr). Env: RETRIEVER_EXTRACT_METHOD",
    ),
    embed_model: Optional[str] = typer.Option(
        None,
        "--embed-model",
        help="Embedding model name. Env: RETRIEVER_EMBED_MODEL",
    ),
    embed_endpoint: Optional[str] = typer.Option(
        None,
        "--embed-endpoint",
        help="Remote embedding NIM endpoint URL. Env: RETRIEVER_EMBED_ENDPOINT",
    ),
    lancedb_uri: Optional[str] = typer.Option(
        None,
        "--lancedb-uri",
        help="LanceDB URI (directory path). Env: RETRIEVER_LANCEDB_URI",
    ),
    lancedb_table: Optional[str] = typer.Option(
        None,
        "--lancedb-table",
        help="LanceDB table name. Env: RETRIEVER_LANCEDB_TABLE",
    ),
) -> None:
    """Start the Retriever Ray Serve + FastAPI REST API.

    Pipeline configuration can be set via CLI options or environment variables.
    CLI options take precedence over environment variables. If neither is set,
    built-in defaults are used.

    Environment variables:
      RETRIEVER_EXTRACT_METHOD, RETRIEVER_EMBED_MODEL, RETRIEVER_EMBED_ENDPOINT,
      RETRIEVER_LANCEDB_URI, RETRIEVER_LANCEDB_TABLE,
      RETRIEVER_QUERY_EMBEDDER, RETRIEVER_QUERY_EMBED_ENDPOINT
    """
    import os

    try:
        import ray
        from ray import serve
        from ray.serve.config import HTTPOptions
    except ImportError as e:
        typer.echo(f"Ray Serve is required: {e}", err=True)
        raise typer.Exit(1)

    if extract_method:
        os.environ["RETRIEVER_EXTRACT_METHOD"] = extract_method
    if embed_model:
        os.environ["RETRIEVER_EMBED_MODEL"] = embed_model
    if embed_endpoint:
        os.environ["RETRIEVER_EMBED_ENDPOINT"] = embed_endpoint
    if lancedb_uri:
        os.environ["RETRIEVER_LANCEDB_URI"] = lancedb_uri
    if lancedb_table:
        os.environ["RETRIEVER_LANCEDB_TABLE"] = lancedb_table

    from nemo_retriever.adapters.service.app import RetrieverAPIDeployment

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    serve.start(http_options=HTTPOptions(host=host, port=port))

    serve.run(
        RetrieverAPIDeployment.bind(),
        name="retriever_api",
        route_prefix="/",
        blocking=True,
    )


@app.command("submit")
def submit_cmd(
    files: List[Path] = typer.Argument(
        ...,
        help="Files to submit to the online ingest endpoint (PDF, TXT, HTML, images, audio).",
        path_type=Path,
        exists=True,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running service.",
    ),
    async_mode: bool = typer.Option(
        False,
        "--async",
        "-a",
        help="Use async ingest (POST /ingest/async) instead of sync.",
    ),
    poll_timeout: float = typer.Option(
        600.0,
        "--poll-timeout",
        help="Timeout in seconds for polling async results.",
    ),
) -> None:
    """Submit files to the online ingest endpoint."""
    from nemo_retriever import create_ingestor
    from nemo_retriever.params import IngestorCreateParams

    ingestor = create_ingestor(
        run_mode="online",
        params=IngestorCreateParams(base_url=base_url),
    )
    file_paths = [str(f.resolve()) for f in files]
    ingestor = ingestor.files(file_paths)

    if async_mode:
        typer.echo(f"Submitting {len(file_paths)} file(s) asynchronously...")
        ingestor.ingest_async()
        typer.echo("All files submitted. Polling for results...")
        results = ingestor.poll_results(timeout=poll_timeout)
    else:
        typer.echo(f"Submitting {len(file_paths)} file(s) synchronously...")
        results = ingestor.ingest()

    ok_count = sum(1 for r in results if r.get("ok"))
    typer.echo(f"\nResults: {ok_count}/{len(results)} succeeded.")
    for r in results:
        src = r.get("source_path", "?")
        if r.get("ok"):
            rows = r.get("rows_written", 0)
            dur = r.get("total_duration_sec", 0)
            typer.echo(f"  OK {src}: {rows} rows, {dur:.2f}s")
        else:
            typer.echo(f"  FAIL {src}: {r.get('error', 'unknown')}", err=True)

    if ok_count < len(results):
        sys.exit(1)


@app.command("stream-pdf")
def stream_pdf_cmd(
    files: List[Path] = typer.Argument(
        ...,
        help="PDF files to stream (POST /stream-pdf, print NDJSON per page).",
        path_type=Path,
        exists=True,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running service.",
    ),
) -> None:
    """Submit PDFs to POST /stream-pdf and print the streaming page-by-page text (NDJSON)."""
    import requests

    url = f"{base_url.rstrip('/')}/stream-pdf"
    for path in files:
        path = path.resolve()
        typer.echo(f"--- {path} ---", err=True)
        try:
            with path.open("rb") as file_handle:
                response = requests.post(
                    url,
                    files={"file": (path.name, file_handle, "application/pdf")},
                    stream=True,
                    timeout=60,
                )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    page = obj.get("page", "?")
                    text = obj.get("text", "")
                    typer.echo(f"Page {page}:")
                    typer.echo(text if text else "(no text)")
                except json.JSONDecodeError:
                    typer.echo(line)
        except requests.RequestException as e:
            typer.echo(f"Error: {e}", err=True)
            sys.exit(1)
        typer.echo("", err=True)


@app.command("query")
def query_cmd(
    query_text: str = typer.Argument(..., help="Query string to search."),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running service.",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return."),
) -> None:
    """Submit a query to the retrieval endpoint."""
    import requests

    url = f"{base_url.rstrip('/')}/query"
    try:
        resp = requests.post(url, json={"query": query_text, "top_k": top_k}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        typer.echo(f"Query: {query_text}")
        typer.echo(f"Hits: {len(hits)}")
        for i, hit in enumerate(hits[:top_k], 1):
            source = hit.get("source", "?")
            page = hit.get("page_number", "?")
            text_preview = (hit.get("text", "") or "")[:120]
            distance = hit.get("_distance")
            dist_str = f"  distance={distance:.6f}" if distance is not None else ""
            typer.echo(f"  {i:02d}. {source} p{page}{dist_str}")
            if text_preview:
                typer.echo(f"      {text_preview}...")
    except requests.RequestException as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    app()
