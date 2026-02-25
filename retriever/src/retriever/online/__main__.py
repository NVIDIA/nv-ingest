"""
Online ingest CLI: start the Ray Serve REST API or submit documents to it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    help="Online ingest: low-latency REST API (serve) and CLI to submit documents (submit).",
)


def _default_extract_kwargs() -> dict:
    return {
        "method": "pdfium",
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_infographics": False,
        "extract_page_as_image": True,
        "dpi": 200,
    }


def _default_embed_kwargs() -> dict:
    return {
        "model_name": os.environ.get("ONLINE_EMBED_MODEL", "nemo_retriever_v1"),
        "embedding_endpoint": os.environ.get("ONLINE_EMBED_ENDPOINT", "").strip() or None,
        "embed_invoke_url": os.environ.get("ONLINE_EMBED_INVOKE_URL", "").strip() or None,
    }


def _default_vdb_kwargs() -> dict:
    return {
        "lancedb_uri": os.environ.get("ONLINE_LANCEDB_URI", "lancedb"),
        "table_name": os.environ.get("ONLINE_LANCEDB_TABLE", "nv-ingest"),
        "overwrite": False,
        "create_index": True,
    }


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host for the HTTP server."),
    port: int = typer.Option(7670, "--port", help="Port for the HTTP server."),
    lancedb_uri: Optional[str] = typer.Option(None, "--lancedb-uri", help="LanceDB URI (default: env ONLINE_LANCEDB_URI or 'lancedb')."),
    table_name: Optional[str] = typer.Option(None, "--table-name", help="LanceDB table name (default: env ONLINE_LANCEDB_TABLE or 'nv-ingest')."),
    embed_endpoint: Optional[str] = typer.Option(None, "--embed-endpoint", help="Embedding NIM endpoint URL (optional)."),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray cluster address (default: local)."),
) -> None:
    """Start the online ingest Ray Serve REST API (POST /ingest for documents)."""
    try:
        import ray
        from ray import serve
    except ImportError as e:
        typer.echo(f"Ray Serve is required for online mode: {e}", err=True)
        raise typer.Exit(1)

    from retriever.ingest_modes.serve import OnlineIngestDeployment

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    vdb_kwargs = _default_vdb_kwargs()
    if lancedb_uri is not None:
        vdb_kwargs["lancedb_uri"] = lancedb_uri
    if table_name is not None:
        vdb_kwargs["table_name"] = table_name

    embed_kwargs = _default_embed_kwargs()
    if embed_endpoint is not None:
        embed_kwargs["embedding_endpoint"] = embed_endpoint
        embed_kwargs["embed_invoke_url"] = embed_endpoint

    extract_kwargs = _default_extract_kwargs()

    serve.run(
        OnlineIngestDeployment.options(name="online_ingest").bind(
            extract_kwargs=extract_kwargs,
            embed_kwargs=embed_kwargs,
            vdb_kwargs=vdb_kwargs,
        ),
        host=host,
        port=port,
        name="online_ingest",
    )


@app.command("submit")
def submit_cmd(
    files: List[Path] = typer.Argument(
        ...,
        help="PDF files to submit to the online ingest service.",
        path_type=Path,
        exists=True,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running online ingest service.",
    ),
    show_metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Print per-document and per-stage metrics."),
) -> None:
    """Submit one or more PDFs to the online ingest REST endpoint."""
    import httpx

    base_url = base_url.rstrip("/")
    ingest_url = f"{base_url}/ingest"

    total_duration = 0.0
    total_rows = 0
    failed = 0

    for path in files:
        path = Path(path)
        if not path.is_file():
            typer.echo(f"Skipping (not a file): {path}", err=True)
            failed += 1
            continue

        try:
            with open(path, "rb") as f:
                body = f.read()
        except Exception as e:
            typer.echo(f"Failed to read {path}: {e}", err=True)
            failed += 1
            continue

        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    ingest_url,
                    files={"file": (path.name, body, "application/pdf")},
                    headers={"X-Source-Path": str(path.resolve())},
                )
        except Exception as e:
            typer.echo(f"Request failed for {path}: {e}", err=True)
            failed += 1
            continue

        if resp.status_code != 200:
            typer.echo(f"{path}: HTTP {resp.status_code} - {resp.text[:500]}", err=True)
            failed += 1
            continue

        data = resp.json()
        if data.get("ok"):
            total_duration += float(data.get("total_duration_sec", 0))
            total_rows += int(data.get("rows_written", 0))
            typer.echo(f"OK {path.name}  rows={data.get('rows_written', 0)}  total_sec={data.get('total_duration_sec', 0):.2f}")
            if show_metrics and data.get("stages"):
                for s in data["stages"]:
                    typer.echo(f"  - {s.get('stage', '?')}: {s.get('duration_sec', 0):.3f}s")
        else:
            typer.echo(f"FAIL {path.name}: {data.get('error', 'unknown')}", err=True)
            failed += 1

    if len(files) > 1:
        typer.echo(f"Total: {total_rows} rows, {total_duration:.2f}s, {failed} failed.")

    if failed > 0:
        sys.exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
