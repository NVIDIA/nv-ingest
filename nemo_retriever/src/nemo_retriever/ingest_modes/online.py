# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online runmode.

Low-latency request/response serving: documents are submitted to a Ray Serve
REST API (see ingest_modes.serve) which runs the same pipeline as inprocess
on a single document per request. This module provides the client-side
Ingestor that delegates to that REST endpoint.

Supports both synchronous (``ingest()``) and asynchronous (``ingest_async()``
+ ``poll_results()``) submission for all supported file types.
"""

from __future__ import annotations

import glob
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from ..ingestor import Ingestor
from ..params import EmbedParams
from ..params import ExtractParams
from ..params import IngestExecuteParams
from ..params import VdbUploadParams

_EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def _mime_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _EXTENSION_MIME.get(ext, "application/octet-stream")


class OnlineIngestor(Ingestor):
    """
    Client-side ingestor that submits documents to the online ingest REST API.

    Use ``create_ingestor(run_mode="online", base_url="http://localhost:7670")``
    then ``.files(...).extract(...).embed(...).vdb_upload().ingest()``.

    ``extract``/``embed``/``vdb_upload`` record configuration for API
    compatibility; the actual pipeline runs on the server.

    Supports two modes of operation:

    - **Sync**: ``ingest()`` POSTs each file to ``/ingest`` and blocks until
      the response is received.
    - **Async**: ``ingest_async()`` POSTs each file to ``/ingest/async``,
      collects job IDs, then ``poll_results()`` polls until all jobs complete.
    """

    RUN_MODE = "online"

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        base_url: str = "http://localhost:7670",
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._input_documents: List[str] = self._documents
        self._pending_jobs: List[Dict[str, str]] = []

    def files(self, documents: Union[str, List[str]]) -> "OnlineIngestor":
        """Add local file paths or globs for submission to the online service."""
        if isinstance(documents, str):
            documents = [documents]

        for pattern in documents:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"Invalid document pattern: {pattern!r}")

            matches = glob.glob(pattern, recursive=True)
            if matches:
                files = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not files:
                    raise FileNotFoundError(f"Pattern resolved, but no files found: {pattern!r}")
                self._input_documents.extend(files)
                continue

            if os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
                continue

            raise FileNotFoundError(f"No local files found for: {pattern!r}")

        return self

    def buffers(self, buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]]) -> "OnlineIngestor":
        """Add in-memory buffers for submission. Name is used as source_path."""
        if isinstance(buffers, (list, tuple)) and len(buffers) == 2 and not isinstance(buffers[0], (list, tuple)):
            buffers = [buffers]
        for name, buf in buffers:
            self._buffers.append((str(name), buf))
        return self

    def load(self) -> "OnlineIngestor":
        return self

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "OnlineIngestor":
        _ = params or ExtractParams(**kwargs)
        return self

    def extract_image_files(self, params: ExtractParams | None = None, **kwargs: Any) -> "OnlineIngestor":
        _ = params or ExtractParams(**kwargs)
        return self

    def extract_txt(self, params: Any = None, **kwargs: Any) -> "OnlineIngestor":
        return self

    def extract_audio(self, params: Any = None, asr_params: Any = None, **kwargs: Any) -> "OnlineIngestor":
        return self

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "OnlineIngestor":
        _ = params or EmbedParams(**kwargs)
        return self

    def vdb_upload(self, params: VdbUploadParams | None = None, **kwargs: Any) -> "OnlineIngestor":
        _ = params or VdbUploadParams(
            purge_results_after_upload=bool(kwargs.get("purge_results_after_upload", True)),
            lancedb={k: v for k, v in kwargs.items() if k != "purge_results_after_upload"},
        )
        return self

    # ------------------------------------------------------------------
    # Sync ingest
    # ------------------------------------------------------------------

    def ingest(self, params: IngestExecuteParams | None = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Submit each configured file (and buffer) to ``POST /ingest`` synchronously.

        Returns a list of response dicts (one per document) with ``ok``,
        ``total_duration_sec``, ``stages``, ``rows_written``, and optional ``error``.
        """
        _ = params or IngestExecuteParams(**kwargs)
        import httpx

        ingest_url = f"{self._base_url}/ingest"
        results: List[Dict[str, Any]] = []

        for path in list(self._input_documents):
            result = self._submit_file_sync(ingest_url, path=path)
            results.append(result)

        for name, buf in self._buffers:
            buf.seek(0)
            body = buf.read()
            result = self._submit_bytes_sync(ingest_url, body, name)
            results.append(result)

        return results

    def _submit_file_sync(self, url: str, *, path: str) -> Dict[str, Any]:
        import httpx

        try:
            with open(path, "rb") as f:
                body = f.read()
        except Exception as e:
            return {"ok": False, "error": str(e), "source_path": path}
        return self._submit_bytes_sync(url, body, path)

    def _submit_bytes_sync(self, url: str, body: bytes, source_path: str) -> Dict[str, Any]:
        import httpx

        mime = _mime_for_path(source_path)
        headers = {"X-Source-Path": source_path}
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    url,
                    files={"file": (os.path.basename(source_path), body, mime)},
                    headers=headers,
                )
        except Exception as e:
            return {"ok": False, "error": str(e), "source_path": source_path}

        try:
            data = resp.json()
        except Exception:
            data = {"ok": False, "error": resp.text[:500], "source_path": source_path}
        if resp.status_code != 200:
            data["ok"] = False
            data.setdefault("error", f"HTTP {resp.status_code}")
        return data

    # ------------------------------------------------------------------
    # Async ingest
    # ------------------------------------------------------------------

    def ingest_async(self, *, return_failures: bool = False, return_traces: bool = False) -> "OnlineIngestor":
        """Submit each configured file to ``POST /ingest/async``.

        Job IDs are stored internally. Call ``poll_results()`` to wait for
        completion and retrieve the results.
        """
        import httpx

        async_url = f"{self._base_url}/ingest/async"

        for path in list(self._input_documents):
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except Exception as e:
                self._pending_jobs.append({"source_path": path, "job_id": "", "error": str(e)})
                continue
            self._submit_async(async_url, body, path)

        for name, buf in self._buffers:
            buf.seek(0)
            body = buf.read()
            self._submit_async(async_url, body, name)

        return self

    def _submit_async(self, url: str, body: bytes, source_path: str) -> None:
        import httpx

        mime = _mime_for_path(source_path)
        headers = {"X-Source-Path": source_path}
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    url,
                    files={"file": (os.path.basename(source_path), body, mime)},
                    headers=headers,
                )
            if resp.status_code == 200:
                data = resp.json()
                self._pending_jobs.append({
                    "source_path": source_path,
                    "job_id": data.get("job_id", ""),
                })
            else:
                self._pending_jobs.append({
                    "source_path": source_path,
                    "job_id": "",
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                })
        except Exception as e:
            self._pending_jobs.append({
                "source_path": source_path,
                "job_id": "",
                "error": str(e),
            })

    def poll_results(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Poll ``GET /ingest/status/{job_id}`` until all jobs complete.

        Once a job reaches ``completed`` or ``failed``, fetches the result
        from ``GET /ingest/result/{job_id}``.

        Returns a list of result dicts (same shape as ``ingest()``).
        """
        import httpx

        results: List[Dict[str, Any]] = []
        remaining = [j for j in self._pending_jobs if j.get("job_id")]
        immediate_failures = [j for j in self._pending_jobs if not j.get("job_id")]

        for fail in immediate_failures:
            results.append({
                "ok": False,
                "source_path": fail.get("source_path", ""),
                "error": fail.get("error", "No job_id"),
            })

        deadline = time.monotonic() + timeout
        done_ids: set[str] = set()

        while remaining and time.monotonic() < deadline:
            still_pending = []
            for job in remaining:
                if job["job_id"] in done_ids:
                    continue
                try:
                    with httpx.Client(timeout=30.0) as client:
                        status_resp = client.get(
                            f"{self._base_url}/ingest/status/{job['job_id']}"
                        )
                    if status_resp.status_code != 200:
                        still_pending.append(job)
                        continue
                    status = status_resp.json()
                    if status.get("status") in ("completed", "failed"):
                        result = self._fetch_result(job["job_id"], job.get("source_path", ""))
                        results.append(result)
                        done_ids.add(job["job_id"])
                    else:
                        still_pending.append(job)
                except Exception:
                    still_pending.append(job)

            remaining = still_pending
            if remaining:
                time.sleep(poll_interval)

        for job in remaining:
            results.append({
                "ok": False,
                "source_path": job.get("source_path", ""),
                "error": f"Timed out after {timeout}s (job_id={job['job_id']})",
            })

        self._pending_jobs.clear()
        return results

    def _fetch_result(self, job_id: str, source_path: str) -> Dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(f"{self._base_url}/ingest/result/{job_id}")
            if resp.status_code == 200:
                return resp.json()
            return {"ok": False, "source_path": source_path, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "source_path": source_path, "error": str(e)}

    # ------------------------------------------------------------------
    # Streaming ingest
    # ------------------------------------------------------------------

    def ingest_stream(
        self,
        params: IngestExecuteParams | None = None,
        **kwargs: Any,
    ):
        """Submit each file to ``POST /ingest/stream`` and yield NDJSON events.

        For every file, yields one dict per pipeline stage with
        ``event="stage_complete"``, followed by a final ``event="complete"``
        or ``event="error"`` dict. The caller receives results in real-time
        as each pipeline stage finishes on the server.

        Example::

            for event in ingestor.ingest_stream():
                if event["event"] == "stage_complete":
                    print(f"  {event['stage']} done in {event['duration_sec']}s")
                elif event["event"] == "complete":
                    print(f"Done: {event['rows_written']} rows")
        """
        _ = params or IngestExecuteParams(**kwargs)
        import httpx

        stream_url = f"{self._base_url}/ingest/stream"

        for path in list(self._input_documents):
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except Exception as e:
                yield {"event": "error", "ok": False, "source_path": path, "error": str(e)}
                continue
            yield from self._stream_file(stream_url, body, path)

        for name, buf in self._buffers:
            buf.seek(0)
            body = buf.read()
            yield from self._stream_file(stream_url, body, name)

    def _stream_file(self, url: str, body: bytes, source_path: str):
        """POST a file to the streaming endpoint and yield parsed NDJSON events."""
        import httpx

        mime = _mime_for_path(source_path)
        headers = {"X-Source-Path": source_path}
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    url,
                    files={"file": (os.path.basename(source_path), body, mime)},
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        yield {
                            "event": "error",
                            "ok": False,
                            "source_path": source_path,
                            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        }
                        return
                    buffer = ""
                    for chunk in resp.iter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if line:
                                try:
                                    yield json.loads(line)
                                except json.JSONDecodeError:
                                    pass
        except Exception as e:
            yield {"event": "error", "ok": False, "source_path": source_path, "error": str(e)}

    # ------------------------------------------------------------------
    # Async polling (generator)
    # ------------------------------------------------------------------

    def poll_results_stream(
        self,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ):
        """Like ``poll_results()`` but yields each result as soon as it is ready.

        Use after ``ingest_async()``. Instead of waiting for all jobs to
        complete and returning a list, this generator yields one result dict
        at a time, allowing the caller to act on completed documents while
        others are still processing.

        Example::

            ingestor.ingest_async()
            for result in ingestor.poll_results_stream(timeout=1200):
                status = "OK" if result["ok"] else "FAIL"
                print(f"[{status}] {result.get('source_path')}")
        """
        import httpx

        remaining = [j for j in self._pending_jobs if j.get("job_id")]

        for fail in [j for j in self._pending_jobs if not j.get("job_id")]:
            yield {
                "ok": False,
                "source_path": fail.get("source_path", ""),
                "error": fail.get("error", "No job_id"),
            }

        deadline = time.monotonic() + timeout
        done_ids: set[str] = set()

        while remaining and time.monotonic() < deadline:
            still_pending = []
            for job in remaining:
                if job["job_id"] in done_ids:
                    continue
                try:
                    with httpx.Client(timeout=30.0) as client:
                        status_resp = client.get(
                            f"{self._base_url}/ingest/status/{job['job_id']}"
                        )
                    if status_resp.status_code != 200:
                        still_pending.append(job)
                        continue
                    status = status_resp.json()
                    if status.get("status") in ("completed", "failed"):
                        result = self._fetch_result(job["job_id"], job.get("source_path", ""))
                        done_ids.add(job["job_id"])
                        yield result
                    else:
                        still_pending.append(job)
                except Exception:
                    still_pending.append(job)

            remaining = still_pending
            if remaining:
                time.sleep(poll_interval)

        for job in remaining:
            yield {
                "ok": False,
                "source_path": job.get("source_path", ""),
                "error": f"Timed out after {timeout}s (job_id={job['job_id']})",
            }

        self._pending_jobs.clear()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, str]:
        """Return per-document job status by querying active jobs."""
        import httpx

        statuses: Dict[str, str] = {}
        for job in self._pending_jobs:
            source = job.get("source_path", "")
            job_id = job.get("job_id", "")
            if not job_id:
                statuses[source] = "submit_failed"
                continue
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(f"{self._base_url}/ingest/status/{job_id}")
                if resp.status_code == 200:
                    statuses[source] = resp.json().get("status", "unknown")
                else:
                    statuses[source] = "unknown"
            except Exception:
                statuses[source] = "unreachable"
        return statuses

    # ------------------------------------------------------------------
    # No-op builder methods for API compatibility
    # ------------------------------------------------------------------

    def all_tasks(self) -> "OnlineIngestor":
        return self

    def dedup(self) -> "OnlineIngestor":
        return self

    def filter(self) -> "OnlineIngestor":
        return self

    def split(self) -> "OnlineIngestor":
        return self

    def store(self) -> "OnlineIngestor":
        return self

    def store_embed(self) -> "OnlineIngestor":
        return self

    def udf(self, *args: Any, **kwargs: Any) -> "OnlineIngestor":
        return self

    def save_intermediate_results(self, output_dir: str) -> "OnlineIngestor":
        return self

    def save_to_disk(self, *args: Any) -> "OnlineIngestor":
        return self

    def caption(self) -> "OnlineIngestor":
        return self

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "OnlineIngestor":
        return self

    def completed_jobs(self) -> int:
        return sum(1 for j in self._pending_jobs if not j.get("job_id"))

    def failed_jobs(self) -> int:
        return sum(1 for j in self._pending_jobs if j.get("error"))

    def cancelled_jobs(self) -> int:
        return 0

    def remaining_jobs(self) -> int:
        return sum(1 for j in self._pending_jobs if j.get("job_id") and not j.get("error"))
