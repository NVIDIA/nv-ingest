# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online service CLI: start the Ray Serve + FastAPI REST API.
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
) -> None:
    """Start the Retriever Ray Serve + FastAPI REST API (e.g. GET /version)."""
    try:
        import ray
        from ray import serve
        from ray.serve.config import HTTPOptions
    except ImportError as e:
        typer.echo(f"Ray Serve is required: {e}", err=True)
        raise typer.Exit(1)

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
        help="PDF files to submit (ingest endpoint not yet implemented).",
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
    """Submit PDFs to the online ingest endpoint (not yet implemented). Use GET /version to check the service."""
    _ = base_url
    typer.echo("Submit (POST /ingest) is not yet implemented. Use GET /version to verify the service is running.")
    for path in files:
        typer.echo(f"  Would submit: {path}")
    sys.exit(0)


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


def main() -> None:
    app()
