# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Serve application integrated with FastAPI.

Provides REST endpoints for:
- Document ingestion (sync and async) across all supported file types
- Retrieval queries against LanceDB with optional reranking
- Utility endpoints (version, stream-pdf)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

try:
    from ray import serve
except ImportError:
    serve = None  # type: ignore[assignment]

app = None

if serve is not None:
    from fastapi import FastAPI, File, Header, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse

    from .models import (
        AsyncIngestResponse,
        IngestResponse,
        JobStatusResponse,
        QueriesRequest,
        QueriesResponse,
        QueryRequest,
        QueryResponse,
        StageMetric,
    )

    app = FastAPI(title="Retriever", description="Retriever online service")

    STREAM_PDF_MAX_BYTES = 50 * 1024 * 1024

    def _pdf_page_text_stream(pdf_bytes: bytes):
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(pdf_bytes)
        except Exception:
            doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
        try:
            for i in range(len(doc)):
                page = doc.get_page(i)
                tp = page.get_textpage()
                text = tp.get_text_bounded()
                if text is None:
                    text = ""
                yield (i + 1, text)
        finally:
            try:
                doc.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Utility endpoints
    # ------------------------------------------------------------------

    @app.get("/version")
    def version() -> dict:
        """Return the running application version and build metadata."""
        try:
            from nemo_retriever.version import get_version_info

            return get_version_info()
        except Exception:
            return {"version": "unknown", "git_sha": "unknown", "build_date": "unknown", "full_version": "unknown"}

    @app.post("/stream-pdf")
    def stream_pdf(
        file: UploadFile = File(..., description="PDF file to extract text from (page-by-page stream)."),
    ):
        """Stream back pdfium get_text (per page) as NDJSON."""
        if file.content_type and file.content_type not in (
            "application/pdf",
            "application/octet-stream",
        ):
            raise HTTPException(400, detail="Expected a PDF file (application/pdf).")
        try:
            import pypdfium2 as pdfium  # noqa: F401
        except ImportError as e:
            raise HTTPException(503, detail="PDF processing (pypdfium2) is not available.") from e

        raw = file.file.read()
        if len(raw) > STREAM_PDF_MAX_BYTES:
            raise HTTPException(413, detail=f"PDF larger than {STREAM_PDF_MAX_BYTES // (1024 * 1024)} MiB not allowed.")

        def ndjson_stream():
            for page_num, text in _pdf_page_text_stream(raw):
                line = json.dumps({"page": page_num, "text": text}, ensure_ascii=False) + "\n"
                yield line.encode("utf-8")

        return StreamingResponse(
            ndjson_stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "inline; filename=pages.ndjson"},
        )

    # ------------------------------------------------------------------
    # Ray Serve deployment
    # ------------------------------------------------------------------

    @serve.deployment(
        name="retriever_api",
        num_replicas=1,
        ray_actor_options={"num_cpus": 2, "num_gpus": 2},
    )
    @serve.ingress(app)
    class RetrieverAPIDeployment:
        """Ray Serve deployment with ingest and retrieval endpoints.

        On startup, builds the inprocess pipeline task chain (loading GPU
        models into memory) and instantiates a Retriever for query serving.
        Pipeline configuration is read from environment variables.
        """

        def __init__(self) -> None:
            from nemo_retriever.ingest_modes.inprocess import (
                InProcessIngestor,
                run_pipeline_tasks_on_df,
            )
            from nemo_retriever.params import EmbedParams, ExtractParams, VdbUploadParams

            extract_method = os.environ.get("RETRIEVER_EXTRACT_METHOD", "pdfium")
            embed_model = os.environ.get("RETRIEVER_EMBED_MODEL", None)
            embed_endpoint = os.environ.get("RETRIEVER_EMBED_ENDPOINT", None)
            lancedb_uri = os.environ.get("RETRIEVER_LANCEDB_URI", "lancedb")
            lancedb_table = os.environ.get("RETRIEVER_LANCEDB_TABLE", "nv-ingest")

            # Build the pipeline by configuring an InProcessIngestor.
            # PDF pipeline tasks: extract -> page_elements -> OCR -> embed -> VDB upload
            ingestor = InProcessIngestor()
            ingestor.extract(
                ExtractParams(
                    method=extract_method,
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=True,
                )
            )

            embed_kwargs: dict[str, Any] = {}
            if embed_model:
                embed_kwargs["model_name"] = embed_model
            if embed_endpoint:
                embed_kwargs["embedding_endpoint"] = embed_endpoint
            ingestor.embed(EmbedParams(**embed_kwargs))

            ingestor.vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": lancedb_uri,
                        "table_name": lancedb_table,
                        "overwrite": False,
                        "create_index": True,
                    }
                )
            )

            self._per_doc_tasks, self._post_tasks = ingestor.get_pipeline_tasks()
            self._run_pipeline = run_pipeline_tasks_on_df

            # For TXT/HTML/audio files the pipeline tasks from the PDF ingestor
            # are not appropriate (they include pdf_extraction). We keep a
            # trimmed version that has only GPU tasks (embed + VDB upload).
            self._non_pdf_per_doc_tasks = [
                (f, k) for f, k in self._per_doc_tasks if getattr(f, "__name__", "") != "pdf_extraction"
            ]

            # Retriever for query endpoints
            from nemo_retriever.retriever import Retriever

            retriever_kwargs: dict[str, Any] = {
                "lancedb_uri": lancedb_uri,
                "lancedb_table": lancedb_table,
            }
            retriever_embedder = os.environ.get("RETRIEVER_QUERY_EMBEDDER", None)
            retriever_embed_endpoint = os.environ.get("RETRIEVER_QUERY_EMBED_ENDPOINT", None)
            if retriever_embedder:
                retriever_kwargs["embedder"] = retriever_embedder
            if retriever_embed_endpoint:
                retriever_kwargs["embedding_endpoint"] = retriever_embed_endpoint
            self._retriever = Retriever(**retriever_kwargs)

            # Shared semaphore: only one pipeline execution at a time (GPU safety)
            self._pipeline_sem = asyncio.Semaphore(1)

            # Job manager for async ingest
            from nemo_retriever.adapters.service.job_manager import JobManager

            self._job_manager = JobManager(
                run_fn=self._run_ingest_pipeline,
                pipeline_semaphore=self._pipeline_sem,
            )

            print(
                f"RetrieverAPIDeployment initialized: "
                f"method={extract_method}, lancedb={lancedb_uri}/{lancedb_table}, "
                f"per_doc_tasks={len(self._per_doc_tasks)}, post_tasks={len(self._post_tasks)}"
            )

        # ------------------------------------------------------------------
        # Internal pipeline runner
        # ------------------------------------------------------------------

        def _run_ingest_pipeline(
            self,
            initial_df: Any,
            source_path: str,
            file_category: str,
        ) -> tuple[bool, dict[str, Any]]:
            """Execute the pipeline on a single document DataFrame.

            Returns ``(ok, response_dict)`` for use by both the sync endpoint
            and the async JobManager callback.
            """
            import pandas as pd

            t0 = time.perf_counter()
            try:
                if file_category == "pdf":
                    per_doc = self._per_doc_tasks
                else:
                    per_doc = self._non_pdf_per_doc_tasks

                result, metrics = self._run_pipeline(initial_df, per_doc, self._post_tasks)
                rows_written = 0
                if isinstance(result, pd.DataFrame):
                    rows_written = len(result)

                total_sec = time.perf_counter() - t0
                return True, {
                    "ok": True,
                    "source_path": source_path,
                    "total_duration_sec": round(total_sec, 4),
                    "stages": metrics,
                    "rows_written": rows_written,
                }
            except Exception as exc:
                total_sec = time.perf_counter() - t0
                return False, {
                    "ok": False,
                    "source_path": source_path,
                    "total_duration_sec": round(total_sec, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        def _run_ingest_pipeline_stages(
            self,
            initial_df: Any,
            source_path: str,
            file_category: str,
        ):
            """Generator that yields an NDJSON line after each pipeline stage.

            Used by the streaming endpoint to give clients real-time progress.
            Yields ``bytes`` lines (UTF-8 encoded JSON + newline).
            """
            import pandas as pd
            from nemo_retriever.pdf.extract import pdf_extraction

            t0 = time.perf_counter()
            per_doc = self._per_doc_tasks if file_category == "pdf" else self._non_pdf_per_doc_tasks

            current: Any = initial_df
            stages: List[Dict[str, Any]] = []
            try:
                for func, kwargs in per_doc:
                    stage_name = getattr(func, "__name__", "unknown")
                    ts = time.perf_counter()
                    if func is pdf_extraction:
                        current = func(pdf_binary=current, **kwargs)
                    else:
                        current = func(current, **kwargs)
                    dur = round(time.perf_counter() - ts, 4)
                    stages.append({"stage": stage_name, "duration_sec": dur})

                    row_count = len(current) if isinstance(current, pd.DataFrame) else 0
                    line = (
                        json.dumps(
                            {
                                "event": "stage_complete",
                                "stage": stage_name,
                                "duration_sec": dur,
                                "rows": row_count,
                                "source_path": source_path,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    yield line.encode("utf-8")

                if self._post_tasks and current is not None:
                    combined = current if isinstance(current, pd.DataFrame) else pd.concat(current, ignore_index=True)
                    for func, kwargs in self._post_tasks:
                        stage_name = getattr(func, "__name__", "unknown")
                        ts = time.perf_counter()
                        combined = func(combined, **kwargs)
                        dur = round(time.perf_counter() - ts, 4)
                        stages.append({"stage": stage_name, "duration_sec": dur})

                        row_count = len(combined) if isinstance(combined, pd.DataFrame) else 0
                        line = (
                            json.dumps(
                                {
                                    "event": "stage_complete",
                                    "stage": stage_name,
                                    "duration_sec": dur,
                                    "rows": row_count,
                                    "source_path": source_path,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        yield line.encode("utf-8")
                    current = combined

                rows_written = len(current) if isinstance(current, pd.DataFrame) else 0

                if isinstance(current, pd.DataFrame) and len(current) > 0:
                    has_text = "text" in current.columns
                    has_page = "page_number" in current.columns
                    for _, row in current.iterrows():
                        text_val = str(row["text"]) if has_text else ""
                        preview = text_val[:25] + "..." if len(text_val) > 25 else text_val
                        row_event = {
                            "event": "row",
                            "source_path": source_path,
                            "page_number": int(row["page_number"]) if has_page else None,
                            "text": text_val,
                            "text_preview": preview,
                        }
                        yield (json.dumps(row_event, ensure_ascii=False) + "\n").encode("utf-8")

                total_sec = round(time.perf_counter() - t0, 4)
                line = (
                    json.dumps(
                        {
                            "event": "complete",
                            "ok": True,
                            "source_path": source_path,
                            "total_duration_sec": total_sec,
                            "rows_written": rows_written,
                            "stages": stages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                yield line.encode("utf-8")

            except Exception as exc:
                total_sec = round(time.perf_counter() - t0, 4)
                line = (
                    json.dumps(
                        {
                            "event": "error",
                            "ok": False,
                            "source_path": source_path,
                            "total_duration_sec": total_sec,
                            "error": f"{type(exc).__name__}: {exc}",
                            "stages": stages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                yield line.encode("utf-8")

        # ------------------------------------------------------------------
        # Ingest endpoints
        # ------------------------------------------------------------------

        @app.post("/ingest", response_model=IngestResponse)
        def ingest(
            self,
            file: UploadFile = File(..., description="Document to ingest."),
            x_source_path: Optional[str] = Header(None, alias="X-Source-Path"),
        ) -> IngestResponse:
            """Synchronous single-document ingest.

            Accepts any supported file type (PDF, DOCX, PPTX, TXT, HTML,
            images, audio/video). Runs the full pipeline and returns metrics.
            """
            from nemo_retriever.adapters.service.file_routing import (
                bytes_to_df,
                detect_file_category,
            )

            raw = file.file.read()
            filename = x_source_path or file.filename or "document.pdf"
            try:
                category = detect_file_category(filename)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc))

            try:
                initial_df = bytes_to_df(raw, filename, category)
            except Exception as exc:
                raise HTTPException(
                    422,
                    detail=f"Failed to load {filename!r} as {category}: {type(exc).__name__}: {exc}",
                )

            ok, result = self._run_ingest_pipeline(initial_df, filename, category)
            stages = [StageMetric(stage=s["stage"], duration_sec=s["duration_sec"]) for s in result.get("stages", [])]
            return IngestResponse(
                ok=ok,
                source_path=result.get("source_path"),
                total_duration_sec=result.get("total_duration_sec", 0.0),
                stages=stages,
                rows_written=result.get("rows_written", 0),
                error=result.get("error"),
            )

        @app.post("/ingest/async", response_model=AsyncIngestResponse)
        async def ingest_async(
            self,
            file: UploadFile = File(..., description="Document to ingest asynchronously."),
            x_source_path: Optional[str] = Header(None, alias="X-Source-Path"),
        ) -> AsyncIngestResponse:
            """Async single-document ingest.

            Returns a ``job_id`` immediately. Use ``GET /ingest/status/{job_id}``
            to poll and ``GET /ingest/result/{job_id}`` to retrieve results.
            """
            from nemo_retriever.adapters.service.file_routing import (
                bytes_to_df,
                detect_file_category,
            )

            raw = await file.read()
            filename = x_source_path or file.filename or "document.pdf"
            try:
                category = detect_file_category(filename)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc))

            try:
                initial_df = await asyncio.to_thread(bytes_to_df, raw, filename, category)
            except Exception as exc:
                raise HTTPException(
                    422,
                    detail=f"Failed to load {filename!r} as {category}: {type(exc).__name__}: {exc}",
                )

            job_id = self._job_manager.create_job()
            asyncio.create_task(self._job_manager.submit(job_id, initial_df, filename, category))
            return AsyncIngestResponse(job_id=job_id)

        @app.post("/ingest/stream")
        async def ingest_stream(
            self,
            file: UploadFile = File(..., description="Document to ingest with streaming progress."),
            x_source_path: Optional[str] = Header(None, alias="X-Source-Path"),
        ):
            """Streaming single-document ingest.

            Runs the full pipeline and streams back NDJSON lines as each
            pipeline stage completes. The final line has ``event: complete``
            (or ``event: error``). Respects the pipeline semaphore so it
            queues behind any running async jobs.
            """
            from nemo_retriever.adapters.service.file_routing import (
                bytes_to_df,
                detect_file_category,
            )

            raw = await file.read()
            filename = x_source_path or file.filename or "document.pdf"
            try:
                category = detect_file_category(filename)
            except ValueError as exc:
                raise HTTPException(400, detail=str(exc))

            try:
                initial_df = await asyncio.to_thread(bytes_to_df, raw, filename, category)
            except Exception as exc:
                raise HTTPException(
                    422,
                    detail=f"Failed to load {filename!r} as {category}: {type(exc).__name__}: {exc}",
                )

            sem = self._pipeline_sem
            stage_gen = self._run_ingest_pipeline_stages

            async def _streaming_wrapper():
                async with sem:
                    q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    def _produce():
                        try:
                            for chunk in stage_gen(initial_df, filename, category):
                                loop.call_soon_threadsafe(q.put_nowait, chunk)
                        finally:
                            loop.call_soon_threadsafe(q.put_nowait, None)

                    fut = asyncio.ensure_future(asyncio.to_thread(_produce))
                    while True:
                        chunk = await q.get()
                        if chunk is None:
                            break
                        yield chunk
                    await fut

            return StreamingResponse(
                _streaming_wrapper(),
                media_type="application/x-ndjson",
            )

        @app.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
        def ingest_status(self, job_id: str) -> JobStatusResponse:
            """Poll the status of an async ingest job."""
            status = self._job_manager.get_status(job_id)
            if status is None:
                raise HTTPException(404, detail=f"Job {job_id!r} not found.")
            return JobStatusResponse(**status)

        @app.get("/ingest/result/{job_id}", response_model=IngestResponse)
        def ingest_result(self, job_id: str) -> IngestResponse:
            """Retrieve the result of a completed async ingest job."""
            status = self._job_manager.get_status(job_id)
            if status is None:
                raise HTTPException(404, detail=f"Job {job_id!r} not found.")
            if status["status"] not in ("completed", "failed"):
                raise HTTPException(
                    409,
                    detail=f"Job {job_id!r} is still {status['status']}. Poll /ingest/status/{job_id} until complete.",
                )
            result = self._job_manager.get_result(job_id)
            if result is None:
                raise HTTPException(500, detail=f"Job {job_id!r} has no result data.")

            stages = [StageMetric(stage=s["stage"], duration_sec=s["duration_sec"]) for s in result.get("stages", [])]
            return IngestResponse(
                ok=result.get("ok", False),
                source_path=result.get("source_path"),
                total_duration_sec=result.get("total_duration_sec", 0.0),
                stages=stages,
                rows_written=result.get("rows_written", 0),
                error=result.get("error"),
            )

        # ------------------------------------------------------------------
        # Query endpoints
        # ------------------------------------------------------------------

        @app.post("/query", response_model=QueryResponse)
        def query(self, body: QueryRequest) -> QueryResponse:
            """Run retrieval for a single query string."""
            try:
                hits = self._retriever.query(
                    body.query,
                    lancedb_uri=body.lancedb_uri,
                    lancedb_table=body.lancedb_table,
                )
            except Exception as exc:
                raise HTTPException(500, detail=f"Query failed: {type(exc).__name__}: {exc}")

            serializable_hits = _serialize_hits(hits)
            return QueryResponse(query=body.query, hits=serializable_hits)

        @app.post("/queries", response_model=QueriesResponse)
        def queries(self, body: QueriesRequest) -> QueriesResponse:
            """Run retrieval for multiple query strings."""
            try:
                all_hits = self._retriever.queries(
                    body.queries,
                    lancedb_uri=body.lancedb_uri,
                    lancedb_table=body.lancedb_table,
                )
            except Exception as exc:
                raise HTTPException(500, detail=f"Queries failed: {type(exc).__name__}: {exc}")

            results = []
            for q, hits in zip(body.queries, all_hits):
                serializable_hits = _serialize_hits(hits)
                results.append(QueryResponse(query=q, hits=serializable_hits))
            return QueriesResponse(results=results)

    def _serialize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Best-effort conversion of LanceDB hit dicts to JSON-serializable form."""
        import numpy as np

        out = []
        for h in hits:
            cleaned: Dict[str, Any] = {}
            for k, v in h.items():
                if isinstance(v, np.generic):
                    cleaned[k] = v.item()
                elif isinstance(v, np.ndarray):
                    cleaned[k] = v.tolist()
                else:
                    cleaned[k] = v
            out.append(cleaned)
        return out
