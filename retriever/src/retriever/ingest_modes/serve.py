"""
Ray Serve deployment for online ingest: low-latency REST API that runs the
same pipeline as inprocess on a single document per request, with metrics.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .inprocess import (
    InProcessIngestor,
    pages_df_from_pdf_bytes,
    run_pipeline_tasks_on_df,
)


def _build_ingestor_tasks(
    extract_kwargs: Optional[Dict[str, Any]] = None,
    embed_kwargs: Optional[Dict[str, Any]] = None,
    vdb_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Build (per_doc_tasks, post_tasks) using the same pipeline as InProcessIngestor."""
    ingestor = InProcessIngestor()
    ingestor.extract(**(extract_kwargs or {}))
    ingestor.embed(**(embed_kwargs or {}))
    ingestor.vdb_upload(**(vdb_kwargs or {}))
    return ingestor.get_pipeline_tasks()


def _ingest_document(
    pdf_bytes: bytes,
    source_path: str,
    per_doc_tasks: list,
    post_tasks: list,
) -> Dict[str, Any]:
    """
    Run the pipeline on one document and return result plus metrics.
    """
    t0_total = time.perf_counter()
    try:
        pages_df = pages_df_from_pdf_bytes(pdf_bytes, source_path)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "source_path": source_path,
            "total_duration_sec": time.perf_counter() - t0_total,
            "stages": [],
            "rows_written": 0,
        }

    result, stage_metrics = run_pipeline_tasks_on_df(pages_df, per_doc_tasks, post_tasks)
    total_sec = time.perf_counter() - t0_total

    rows_written = 0
    if hasattr(result, "index") and result is not None:
        rows_written = int(len(result.index))

    return {
        "ok": True,
        "source_path": source_path,
        "total_duration_sec": total_sec,
        "stages": stage_metrics,
        "rows_written": rows_written,
        "pages_processed": int(len(pages_df.index)),
    }


try:
    from ray import serve
except ImportError:
    serve = None  # type: ignore[assignment]


if serve is not None:

    @serve.deployment(
        name="online_ingest",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
    )
    class OnlineIngestDeployment:
        """
        Ray Serve deployment for online document ingestion.

        Accepts POST /ingest with a PDF file (multipart/form-data "file" or
        raw body). Returns JSON with metrics (total_duration_sec, per-stage
        duration_sec, rows_written).
        """

        def __init__(
            self,
            extract_kwargs: Optional[Dict[str, Any]] = None,
            embed_kwargs: Optional[Dict[str, Any]] = None,
            vdb_kwargs: Optional[Dict[str, Any]] = None,
        ):
            self._extract_kwargs = extract_kwargs or {}
            self._embed_kwargs = embed_kwargs or {}
            self._vdb_kwargs = vdb_kwargs or {}
            self._per_doc_tasks, self._post_tasks = _build_ingestor_tasks(
                self._extract_kwargs,
                self._embed_kwargs,
                self._vdb_kwargs,
            )

        async def __call__(self, request: Request) -> Response:
            if request.url.path == "/ingest" or request.url.path.strip("/") == "ingest":
                return await self._handle_ingest(request)
            if request.url.path in ("/", "/health"):
                return JSONResponse({"status": "ok", "service": "online_ingest"})
            return JSONResponse({"error": "Not found"}, status_code=404)

        async def _handle_ingest(self, request: Request) -> Response:
            if request.method != "POST":
                return JSONResponse({"error": "Method not allowed"}, status_code=405)

            pdf_bytes: Optional[bytes] = None
            source_path = "document.pdf"

            content_type = request.headers.get("content-type", "") or ""

            if "multipart/form-data" in content_type:
                form = await request.form()
                file_part = form.get("file")
                if file_part is not None and hasattr(file_part, "read"):
                    pdf_bytes = await file_part.read()
                    if hasattr(file_part, "filename") and file_part.filename:
                        source_path = str(file_part.filename)
                filename = form.get("filename")
                if isinstance(filename, str) and filename.strip():
                    source_path = filename.strip()
            elif "application/pdf" in content_type:
                pdf_bytes = await request.body()
                x_path = request.headers.get("x-source-path") or request.headers.get("X-Source-Path")
                if isinstance(x_path, str) and x_path.strip():
                    source_path = x_path.strip()
            elif "application/json" in content_type:
                try:
                    body = await request.json()
                    b64 = body.get("content") or body.get("pdf_base64")
                    if b64:
                        pdf_bytes = base64.b64decode(b64)
                    source_path = str(body.get("source_path", body.get("filename", source_path)))
                except Exception as e:
                    return JSONResponse(
                        {"ok": False, "error": f"Invalid JSON or base64: {e}"},
                        status_code=400,
                    )
            else:
                # Try reading raw body (e.g. binary POST)
                pdf_bytes = await request.body()
                if not pdf_bytes:
                    return JSONResponse(
                        {"error": "No document body. Use multipart/form-data with 'file', or application/pdf, or JSON with 'content'/'pdf_base64'."},
                        status_code=400,
                    )

            if not pdf_bytes or len(pdf_bytes) == 0:
                return JSONResponse({"ok": False, "error": "Empty document"}, status_code=400)

            try:
                out = _ingest_document(
                    pdf_bytes,
                    source_path,
                    self._per_doc_tasks,
                    self._post_tasks,
                )
            except Exception as e:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": str(e),
                        "source_path": source_path,
                    },
                    status_code=500,
                )

            return JSONResponse(out)
