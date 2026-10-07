# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client helpers for retriever service ingest and query calls.

Ingest uploads whole documents via ``POST /v1/ingest/job/{job_id}/document``
(after opening a job aggregate with ``POST /v1/ingest/job``), tracks
completion via the per-job ``GET /v1/ingest/job/{job_id}/events`` SSE
stream (with ``POST /v1/ingest/status/batch`` bulk-poll fallback), and
surfaces results through both materialized and streaming interfaces.

The SSE connection is opened **before** uploads begin so that completion
events for fast-finishing documents are never missed.  A ``seen_terminal``
buffer reconciles events that arrive before the client registers the
corresponding ``document_id`` from the upload response.

API compatibility
-----------------
The Retriever Service v2 refactor (multi-pod) removed the legacy
single-shot ``POST /v1/ingest`` and the firehose
``GET /v1/ingest/events`` routes in favor of the job-scoped ingest API
used here.  Older SDK builds may still call the legacy routes; the
server now returns ``410 Gone`` with a migration body for those.  Ingest
entry points detect the matching failure mode on their first call to
``POST /v1/ingest/job`` and raise
:class:`RetrieverServiceCompatibilityError` so callers see a single,
actionable "SDK and service versions are out of sync" message instead
of an empty/no-completion result. Query uses ``POST /v1/query`` and
validates that response with the shared query schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, NamedTuple, TypeVar

import httpx
from pydantic import ValidationError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from nemo_retriever.common.schemas.collections import (
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
    QueryHit,
)
from nemo_retriever.common.schemas.responses import (
    JobAggregateResponse,
    JobDocumentsPage,
)
from nemo_retriever.service.errors import (
    RetrieverServiceConflictError,
    RetrieverServiceError,
    RetrieverServiceNotFoundError,
    RetrieverServiceValidationError,
)
from nemo_retriever.service.query_schema import AgenticAnswerResponse, QueryResponse

logger = logging.getLogger(__name__)

_BULK_POLL_INTERVAL_S = 5.0
_BULK_POLL_TIMEOUT_S = 1800.0
_MAX_UPLOAD_RETRIES = 10
_DEFAULT_RETRY_AFTER = 2.0
_DEFAULT_REQUEST_TIMEOUT_S = 300.0
_DEFAULT_AGENTIC_REQUEST_TIMEOUT_S = 1800.0

_T = TypeVar("_T")

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)


class _CreatedJob(NamedTuple):
    job_id: str
    trace_id: str | None = None


class _UploadOutcome(NamedTuple):
    filename: str
    document_id: str = ""
    attempt_id: str = ""
    error: str | None = None


class InMemoryUpload(NamedTuple):
    """Document bytes that should be uploaded without a temporary file."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"
    classification_filename: str | None = None
    metadata: dict[str, Any] | None = None


class FileUpload(NamedTuple):
    """Disk-backed document streamed during upload."""

    path: Path
    filename: str
    content_type: str = "application/octet-stream"
    classification_filename: str | None = None
    metadata: dict[str, Any] | None = None


UploadInput = Path | InMemoryUpload | FileUpload


def _upload_filename(source: UploadInput) -> str:
    return source.name if isinstance(source, Path) else source.filename


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------


class RetrieverServiceCompatibilityError(RuntimeError):
    """Raised when the SDK and the retriever service disagree on the API.

    The Retriever Service v2 refactor removed the legacy
    ``POST /v1/ingest`` / ``GET /v1/ingest/events`` routes in favor of
    job-scoped routes (``POST /v1/ingest/job`` +
    ``POST /v1/ingest/job/{job_id}/document`` +
    ``GET /v1/ingest/job/{job_id}/events``).  Whenever the very first
    call from an ingest entry point — opening a job aggregate via
    ``POST /v1/ingest/job`` — returns ``404`` (route missing) or
    ``410`` (route removed, with migration body), the deployed
    nrl-service is older than this SDK build.  Raising a dedicated
    error type lets callers surface a single, actionable message
    instead of the previous silent "no document_complete event"
    failure mode that 26.08.1 customers reported.
    """


def _is_api_mismatch_status(status: int) -> bool:
    """Return ``True`` for HTTP status codes that signal a route mismatch.

    The new client points at ``POST /v1/ingest/job`` (and friends).
    Servers that predate the multi-pod refactor return ``404`` for that
    path; servers carrying the explicit legacy stubs added alongside
    this client return ``410 Gone`` with a migration body.  Either is a
    deterministic "wrong service version" signal.
    """
    return status in (404, 410)


def _compat_error_message(
    *,
    url: str,
    status: int,
    body: str,
) -> str:
    """Build the customer-facing message attached to compatibility errors."""
    body_clip = (body or "(empty)").strip()[:500]
    return (
        f"Retriever service rejected {url} with HTTP {status}. "
        "This signals an SDK/service version mismatch: this Python "
        "SDK targets the job-scoped ingest API "
        "(POST /v1/ingest/job + POST /v1/ingest/job/{job_id}/document "
        "+ GET /v1/ingest/job/{job_id}/events) introduced in 26.08.1, "
        "but the deployed nrl-service does not advertise that route. "
        "Upgrade the chart/image to a 26.08.1+ build, or downgrade the "
        "Python SDK to match the deployed service version. Server "
        f"response body: {body_clip}"
    )


# ------------------------------------------------------------------
# Progress bar helper
# ------------------------------------------------------------------


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


# ------------------------------------------------------------------
# Document-level tracker
# ------------------------------------------------------------------


class DocumentTracker:
    """Lightweight tracker for document-level completion."""

    def __init__(self) -> None:
        self.pending: set[str] = set()
        self.completed: int = 0
        self.failed: int = 0
        self.errors: list[tuple[str, str]] = []
        self.results: list[dict[str, Any]] = []

    @property
    def done_count(self) -> int:
        return self.completed + self.failed

    def mark_completed(self, doc_id: str, event: dict[str, Any]) -> None:
        self.pending.discard(doc_id)
        self.completed += 1
        self.results.append(event)

    def mark_failed(self, doc_id: str, error: str | None, event: dict[str, Any]) -> None:
        self.pending.discard(doc_id)
        self.failed += 1
        if error:
            self.errors.append((doc_id, error))
        self.results.append(event)


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------


class RetrieverServiceClient:
    """Manage scoped collections, ingest documents, and query a retriever service.

    Ingest opens a job aggregate with ``POST /v1/ingest/job`` (sized to
    the number of files), then uses
    ``POST /v1/ingest/job/{job_id}/document`` for each upload. Query uses
    ``POST /v1/query`` directly.

    Ingest entry points translate ``404`` or ``410`` from job-scoped
    routes into :class:`RetrieverServiceCompatibilityError` so callers
    see a clear SDK/service version-mismatch message.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7670",
        max_concurrency: int = 8,
        *,
        api_token: str | None = None,
        scope: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._api_token = (api_token or "").strip() or None
        self._scope = (scope or "").strip() or None

    @property
    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._scope:
            headers["X-NRL-Scope"] = self._scope
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    @staticmethod
    def _raise_for_response(resp: httpx.Response, operation: str) -> None:
        if resp.status_code < 400:
            return
        detail = resp.text[:1000] if resp.text else "(empty)"
        error_type: type[RetrieverServiceError]
        if resp.status_code == 404:
            error_type = RetrieverServiceNotFoundError
        elif resp.status_code == 409:
            error_type = RetrieverServiceConflictError
        elif resp.status_code in (400, 422):
            error_type = RetrieverServiceValidationError
        else:
            error_type = RetrieverServiceError
        raise error_type(
            f"{operation} failed: HTTP {resp.status_code}: {detail}",
            status_code=resp.status_code,
        )

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        request_timeout_s: float = _DEFAULT_REQUEST_TIMEOUT_S,
        **kwargs: Any,
    ) -> Any:
        # Construct the client inside the coroutine. ``_run`` may drive this on
        # a worker thread's event loop, and a client bound to a different loop
        # fails there — so do not hoist it to ``__init__`` to pool connections.
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(request_timeout_s, connect=30.0),
                headers=self._auth_headers,
            ) as client:
                resp = await client.request(method, f"{self._base_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise RetrieverServiceError(f"{method} {path} transport failure: {exc}") from exc
        self._raise_for_response(resp, f"{method} {path}")
        try:
            return resp.json() if resp.content else None
        except ValueError as exc:
            raise RetrieverServiceError(f"{method} {path} returned malformed JSON") from exc

    @staticmethod
    def _run(coro: Coroutine[Any, Any, _T]) -> _T:
        """Drive an async operation to completion from synchronous code.

        Each operation is implemented once, asynchronously; the synchronous
        methods are thin facades over it. When the caller already runs inside
        an event loop we hand the coroutine to a worker thread, because
        ``asyncio.run`` refuses to nest.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    @staticmethod
    def _model(model: Any, payload: Any, operation: str) -> Any:
        try:
            return model.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise RetrieverServiceError(f"{operation} returned an invalid response: {exc}") from exc

    # ------------------------------------------------------------------
    # Collection and document lifecycle
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> CollectionInfo:
        """Create a logical collection in the client's configured scope."""

        return self._run(
            self.acreate_collection(
                name,
                description=description,
                metadata=metadata,
                expires_at=expires_at,
            )
        )

    async def acreate_collection(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> CollectionInfo:
        """Asynchronously create a logical collection."""

        body = {
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "expires_at": expires_at,
        }
        return self._model(
            CollectionInfo,
            await self._arequest("POST", "/v1/collections", json=body),
            "create collection",
        )

    def get_collection(self, name: str) -> CollectionInfo:
        """Return one logical collection from the configured scope."""

        return self._run(self.aget_collection(name))

    async def aget_collection(self, name: str) -> CollectionInfo:
        """Asynchronously return one logical collection."""

        return self._model(
            CollectionInfo,
            await self._arequest("GET", f"/v1/collections/{name}"),
            "get collection",
        )

    def list_collections(self, *, limit: int = 100, continuation_token: str | None = None) -> CollectionPage:
        """List logical collections in the configured scope."""

        return self._run(self.alist_collections(limit=limit, continuation_token=continuation_token))

    async def alist_collections(
        self,
        *,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> CollectionPage:
        """Asynchronously list logical collections."""

        params = {
            "limit": limit,
            "continuation_token": continuation_token,
        }
        return self._model(
            CollectionPage,
            await self._arequest("GET", "/v1/collections", params=params),
            "list collections",
        )

    def update_collection(self, name: str, **changes: Any) -> CollectionInfo:
        """Update mutable properties of a logical collection."""

        return self._run(self.aupdate_collection(name, **changes))

    async def aupdate_collection(self, name: str, **changes: Any) -> CollectionInfo:
        """Asynchronously update a logical collection."""

        return self._model(
            CollectionInfo,
            await self._arequest("PATCH", f"/v1/collections/{name}", json=changes),
            "update collection",
        )

    def delete_collection(self, name: str, *, if_exists: bool = False) -> CollectionDeleteResult:
        """Request deletion of a logical collection and its VectorDB-owned data."""

        return self._run(self.adelete_collection(name, if_exists=if_exists))

    async def adelete_collection(self, name: str, *, if_exists: bool = False) -> CollectionDeleteResult:
        """Asynchronously delete a logical collection."""

        body = await self._arequest("DELETE", f"/v1/collections/{name}", params={"if_exists": if_exists})
        return self._model(CollectionDeleteResult, body, "delete collection")

    def list_documents(
        self,
        collection_name: str,
        *,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> DocumentPage:
        """List committed documents in a logical collection."""

        return self._run(
            self.alist_documents(
                collection_name,
                limit=limit,
                continuation_token=continuation_token,
            )
        )

    async def alist_documents(
        self,
        collection_name: str,
        *,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> DocumentPage:
        """Asynchronously list committed collection documents."""

        params = {
            "limit": limit,
            "continuation_token": continuation_token,
        }
        return self._model(
            DocumentPage,
            await self._arequest("GET", f"/v1/collections/{collection_name}/documents", params=params),
            "list documents",
        )

    def get_document(self, collection_name: str, document_id: str) -> DocumentInfo:
        """Return one committed collection document."""

        return self._run(self.aget_document(collection_name, document_id))

    async def aget_document(self, collection_name: str, document_id: str) -> DocumentInfo:
        """Asynchronously return one committed collection document."""

        return self._model(
            DocumentInfo,
            await self._arequest("GET", f"/v1/collections/{collection_name}/documents/{document_id}"),
            "get document",
        )

    def delete_document(
        self,
        collection_name: str,
        document_id: str,
        *,
        if_exists: bool = False,
    ) -> DocumentDeleteResult:
        """Request deletion of one document and its collection chunks."""

        return self._run(self.adelete_document(collection_name, document_id, if_exists=if_exists))

    async def adelete_document(
        self,
        collection_name: str,
        document_id: str,
        *,
        if_exists: bool = False,
    ) -> DocumentDeleteResult:
        """Asynchronously delete one collection document."""

        return self._model(
            DocumentDeleteResult,
            await self._arequest(
                "DELETE",
                f"/v1/collections/{collection_name}/documents/{document_id}",
                params={"if_exists": if_exists},
            ),
            "delete document",
        )

    def get_job(self, job_id: str) -> JobAggregateResponse:
        """Return aggregate ingestion status for a job."""

        return self._run(self.aget_job(job_id))

    async def aget_job(self, job_id: str) -> JobAggregateResponse:
        """Asynchronously return aggregate ingestion status."""

        return self._model(
            JobAggregateResponse,
            await self._arequest("GET", f"/v1/ingest/job/{job_id}"),
            "get job",
        )

    def list_job_documents(self, job_id: str, *, offset: int = 0, limit: int = 100) -> JobDocumentsPage:
        """List per-document ingestion status for a job."""

        return self._run(self.alist_job_documents(job_id, offset=offset, limit=limit))

    async def alist_job_documents(self, job_id: str, *, offset: int = 0, limit: int = 100) -> JobDocumentsPage:
        """Asynchronously list per-document ingestion status."""

        return self._model(
            JobDocumentsPage,
            await self._arequest(
                "GET",
                f"/v1/ingest/job/{job_id}/documents",
                params={"offset": offset, "limit": limit},
            ),
            "list job documents",
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str | list[str],
        *,
        top_k: int,
        collection_name: str | None = None,
    ) -> list[list[dict[str, Any]]] | list[QueryHit]:
        """Search ingested documents through ``POST /v1/query``.

        Note:
            ``top_k`` is required here but defaults to 10 on :meth:`aquery`.
            That asymmetry is part of the released signature; do not unify it.
        """
        return self._run(self.aquery(query, top_k=top_k, collection_name=collection_name))

    def _query_hit(self, hit: dict[str, Any]) -> QueryHit:
        return self._model(
            QueryHit,
            {
                **hit,
                "bbox": hit.get("bbox_xyxy_norm"),
                "source": hit.get("source"),
                "metadata": hit.get("metadata") or {},
            },
            "collection query",
        )

    async def aquery(
        self,
        query: str | list[str],
        *,
        top_k: int = 10,
        collection_name: str | None = None,
    ) -> list[list[dict[str, Any]]] | list[QueryHit]:
        """Asynchronously search through ``POST /v1/query``."""

        payload: dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if collection_name:
            payload["collection_name"] = collection_name
        body = await self._arequest("POST", "/v1/query", json=payload)
        try:
            parsed = QueryResponse.model_validate(body).hits_by_query(
                expected_results=len(query) if isinstance(query, list) else 1
            )
        except (ValidationError, ValueError) as exc:
            raise RetrieverServiceError(f"Service query returned invalid response: {exc}") from exc
        if collection_name and isinstance(query, str):
            return [self._query_hit(hit) for hit in parsed[0]]
        return parsed

    def agentic_answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        request_timeout_s: float = _DEFAULT_AGENTIC_REQUEST_TIMEOUT_S,
    ) -> AgenticAnswerResponse:
        """Research and answer a query with the service's integrated ReAct agent.

        Parameters
        ----------
        query:
            Question for the retrieval agent.
        top_k:
            Documents requested from each retrieval step.
        request_timeout_s:
            End-to-end HTTP timeout. Defaults to 30 minutes because an agentic
            run can make many sequential LLM and retrieval calls.

        Returns
        -------
        AgenticAnswerResponse
            Integrated answer, validated citations, hydrated citation hits,
            status, errors, and provider-reported usage.
        """
        return self._run(
            self.aagentic_answer(
                query,
                top_k=top_k,
                request_timeout_s=request_timeout_s,
            )
        )

    async def aagentic_answer(
        self,
        query: str,
        *,
        top_k: int = 5,
        request_timeout_s: float = _DEFAULT_AGENTIC_REQUEST_TIMEOUT_S,
    ) -> AgenticAnswerResponse:
        """Asynchronously run integrated agentic answer mode.

        Parameters
        ----------
        query:
            Question for the retrieval agent.
        top_k:
            Documents requested from each retrieval step.
        request_timeout_s:
            End-to-end HTTP timeout. Defaults to 30 minutes because an agentic
            run can make many sequential LLM and retrieval calls.

        Returns
        -------
        AgenticAnswerResponse
            Integrated answer, validated citations, hydrated citation hits,
            status, errors, and provider-reported usage.
        """
        body = await self._arequest(
            "POST",
            "/v1/answer",
            request_timeout_s=request_timeout_s,
            json={"query": str(query), "top_k": int(top_k), "mode": "agentic"},
        )
        return self._model(AgenticAnswerResponse, body, "agentic answer")

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    async def _create_job(
        self,
        client: httpx.AsyncClient,
        *,
        expected_documents: int,
        label: str | None = None,
        retain_results: bool = False,
        collection_name: str | None = None,
        operation: str = "append",
        target_document_id: str | None = None,
        idempotency_key: str | None = None,
        document_manifest: list[dict[str, str]] | None = None,
    ) -> _CreatedJob:
        """Open a server-side job aggregate and return its client-visible metadata.

        Every upload made through this client must reference a job (J3+).
        We open one job per ``ingest_documents`` / ``aingest_documents_stream``
        call sized to the number of files supplied.
        """
        url = f"{self._base_url}/v1/ingest/job"
        payload: dict[str, Any] = {
            "expected_documents": expected_documents,
            "retain_results": retain_results,
        }
        if label is not None:
            payload["label"] = label
        if collection_name is not None:
            payload["collection_name"] = collection_name
        payload["operation"] = operation
        if target_document_id is not None:
            payload["target_document_id"] = target_document_id
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        if document_manifest:
            payload["document_manifest"] = document_manifest
        try:
            resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RetrieverServiceError(f"Job creation transport failure: {exc}") from exc
        # A 404/410 here means the deployed service does not advertise
        # the job-scoped ingest API.  Surface a dedicated compatibility
        # error instead of a generic HTTPStatusError so callers see one
        # actionable message — see the 26.08.1 release-integration
        # regression report.
        if _is_api_mismatch_status(resp.status_code):
            raise RetrieverServiceCompatibilityError(
                _compat_error_message(
                    url=url,
                    status=resp.status_code,
                    body=resp.text if resp.text else "",
                )
            )
        self._raise_for_response(resp, "create ingestion job")
        try:
            body = resp.json()
        except ValueError as exc:
            raise RetrieverServiceError("Job creation returned malformed JSON") from exc
        job_id = body.get("job_id")
        if not job_id:
            raise RuntimeError(f"Job creation returned no job_id: {body!r}")
        trace_id = body.get("trace_id")
        return _CreatedJob(
            job_id=job_id,
            trace_id=trace_id if isinstance(trace_id, str) and trace_id else None,
        )

    async def asubmit_documents(
        self,
        collection_name: str,
        files: list[str | Path],
        *,
        idempotency_key: str | None = None,
        operation: str = "append",
        target_document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
    ) -> JobAggregateResponse:
        """Create a job and upload its files, returning after acceptance.

        Processing continues on the service. Poll :meth:`get_job` or
        :meth:`aget_job` until the aggregate reaches a terminal state.
        """
        paths = [Path(file) for file in files]
        if not paths:
            raise RetrieverServiceValidationError("At least one file is required")
        timeout = httpx.Timeout(timeout=None, connect=30.0)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        manifest = []
        for position, path in enumerate(paths):
            content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_entry_id = hashlib.sha256(f"{position}\0{path.name}\0{content_sha256}".encode("utf-8")).hexdigest()
            manifest.append(
                {
                    "manifest_entry_id": manifest_entry_id,
                    "filename": path.name,
                    "content_sha256": content_sha256,
                }
            )
        async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=self._auth_headers) as client:
            created = await self._create_job(
                client,
                expected_documents=len(paths),
                collection_name=collection_name,
                operation=operation,
                target_document_id=target_document_id,
                idempotency_key=idempotency_key,
                document_manifest=manifest,
            )
            sem = asyncio.Semaphore(self._max_concurrency)

            async def upload(path: Path, entry: dict[str, str]) -> None:
                async with sem:
                    await self._upload_one(
                        client,
                        path,
                        job_id=created.job_id,
                        metadata=metadata,
                        pipeline_spec=pipeline_spec,
                        manifest_entry_id=entry["manifest_entry_id"],
                    )

            await asyncio.gather(*(upload(path, entry) for path, entry in zip(paths, manifest, strict=True)))
            resp = await client.get(f"{self._base_url}/v1/ingest/job/{created.job_id}")
            self._raise_for_response(resp, "get accepted job")
            try:
                payload = resp.json()
            except ValueError as exc:
                raise RetrieverServiceError("Accepted job response was malformed JSON") from exc
            return self._model(JobAggregateResponse, payload, "get accepted job")

    def submit_documents(
        self,
        collection_name: str,
        files: list[str | Path],
        **kwargs: Any,
    ) -> JobAggregateResponse:
        """Synchronous accepted-not-completed collection ingestion."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asubmit_documents(collection_name, files, **kwargs))
        raise RuntimeError("submit_documents cannot run inside an event loop; use asubmit_documents")

    def replace_document(
        self,
        collection_name: str,
        document_id: str,
        file: str | Path,
        *,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> JobAggregateResponse:
        """Replace one stable document with a newly ingested file."""

        return self.submit_documents(
            collection_name,
            [file],
            operation="replace",
            target_document_id=document_id,
            idempotency_key=idempotency_key,
            **kwargs,
        )

    async def areplace_document(
        self,
        collection_name: str,
        document_id: str,
        file: str | Path,
        *,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> JobAggregateResponse:
        """Asynchronously replace one stable document."""

        return await self.asubmit_documents(
            collection_name,
            [file],
            operation="replace",
            target_document_id=document_id,
            idempotency_key=idempotency_key,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def _upload_one(
        self,
        client: httpx.AsyncClient,
        source: UploadInput,
        *,
        job_id: str,
        metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
        manifest_entry_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a file under an existing job, with retry on 429 + transient errors.

        Posts to ``POST /v1/ingest/job/{job_id}/document``; ``pipeline_spec``
        (when provided) is attached to ``metadata`` under the ``pipeline``
        key so the server can validate and apply it. Returns the parsed
        JSON response (contains ``document_id`` and ``job_id``).
        """
        file_path: Path | None = None
        file_bytes: bytes | None = None
        if isinstance(source, Path):
            file_path = source
            filename = source.name
            content_type = "application/octet-stream"
            classification_filename = None
            source_metadata = None
        elif isinstance(source, FileUpload):
            file_path = source.path
            filename = source.filename
            content_type = source.content_type
            classification_filename = source.classification_filename
            source_metadata = source.metadata
        else:
            file_bytes = source.content
            filename = source.filename
            content_type = source.content_type
            classification_filename = source.classification_filename
            source_metadata = source.metadata
        meta_payload: dict[str, Any] = dict(metadata or {})
        if classification_filename is not None:
            meta_payload.setdefault("filename", classification_filename)
        if source_metadata:
            document_metadata = dict(meta_payload.get("metadata") or {})
            document_metadata.update(source_metadata)
            meta_payload["metadata"] = document_metadata
        if pipeline_spec is not None:
            meta_payload["pipeline"] = pipeline_spec
        meta_json = json.dumps(meta_payload)
        transport_attempts = 0
        url = f"{self._base_url}/v1/ingest/job/{job_id}/document"

        for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
            try:
                data = {
                    "metadata": meta_json,
                    **({"manifest_entry_id": manifest_entry_id} if manifest_entry_id else {}),
                }
                if file_path is None:
                    resp = await client.post(url, files={"file": (filename, file_bytes, content_type)}, data=data)
                else:
                    with file_path.open("rb") as file_obj:
                        resp = await client.post(url, files={"file": (filename, file_obj, content_type)}, data=data)
            except _TRANSIENT_ERRORS as exc:
                transport_attempts += 1
                if transport_attempts > 5:
                    raise RetrieverServiceError(f"Upload of {filename} transport failure after retries: {exc}") from exc
                delay = min(_DEFAULT_RETRY_AFTER * (2 ** (transport_attempts - 1)), 60.0)
                logger.debug(
                    "Transient %s uploading %s, retry in %.1fs",
                    type(exc).__name__,
                    filename,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 429:
                delay = float(resp.headers.get("retry-after", _DEFAULT_RETRY_AFTER))
                logger.debug("429 for %s, retry in %.1fs (attempt %d)", filename, delay, attempt)
                await asyncio.sleep(delay)
                continue

            # Same compatibility-mismatch translation as `_create_job`.
            # If the service did not have the job-scoped upload route at
            # job-create time it would have already failed there; a
            # 404/410 here usually means a rolling upgrade pointed the
            # client at a stale pod after the job was created on a
            # newer one.  Either way, the actionable advice is the same.
            if _is_api_mismatch_status(resp.status_code):
                raise RetrieverServiceCompatibilityError(
                    _compat_error_message(
                        url=url,
                        status=resp.status_code,
                        body=resp.text if resp.text else "",
                    )
                )

            self._raise_for_response(resp, f"upload {filename}")
            try:
                return resp.json()
            except ValueError as exc:
                raise RetrieverServiceError(f"Upload of {filename} returned malformed JSON") from exc

        raise RetrieverServiceError(f"Upload of {filename} failed after {_MAX_UPLOAD_RETRIES} retries")

    async def _upload_source(
        self,
        client: httpx.AsyncClient,
        source: UploadInput,
        *,
        job_id: str,
        pipeline_spec: dict[str, Any] | None,
        semaphore: asyncio.Semaphore,
    ) -> _UploadOutcome:
        """Upload one source and capture the common success or failure outcome."""
        filename = _upload_filename(source)
        async with semaphore:
            try:
                response = await self._upload_one(client, source, job_id=job_id, pipeline_spec=pipeline_spec)
            except Exception as exc:
                logger.error("Upload failed for %s: %s", filename, exc)
                return _UploadOutcome(filename=filename, error=str(exc))
        return _UploadOutcome(
            filename=filename,
            document_id=response.get("document_id", ""),
            attempt_id=response.get("attempt_id", ""),
        )

    # ------------------------------------------------------------------
    # SSE consumer
    # ------------------------------------------------------------------

    async def _consume_sse(
        self,
        client: httpx.AsyncClient,
        pending: set[str],
        uploads_done: asyncio.Event,
        tracker: DocumentTracker,
        *,
        job_id: str,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Consume the per-job SSE stream until all pending items resolve.

        Subscribes to ``GET /v1/ingest/job/{job_id}/events`` (J4+). The
        firehose endpoint was removed deliberately — every public SSE
        consumer must declare the job it is observing. ``pending`` is
        the set of document ids we expect to terminate; reconciliation
        handles the race where an event arrives before the upload
        response added the id to ``pending``.
        """
        url = f"{self._base_url}/v1/ingest/job/{job_id}/events"
        seen_terminal: set[str] = set()
        seen_events: dict[str, dict[str, Any]] = {}

        def _reconcile() -> None:
            late = pending & seen_terminal
            if not late:
                return
            for doc_id in late:
                evt = seen_events.get(doc_id, {})
                status = evt.get("status", "completed")
                if status == "failed":
                    tracker.mark_failed(doc_id, evt.get("error"), evt)
                else:
                    tracker.mark_completed(doc_id, evt)
                if on_event:
                    on_event(evt)
            pending.difference_update(late)
            logger.debug("Reconciled %d late items", len(late))

        def _is_done() -> bool:
            if not uploads_done.is_set():
                return False
            _reconcile()
            return not pending

        _JOB_LIFECYCLE_EVENTS = {
            "job_created",
            "job_started",
            "job_progress",
            "job_finalized",
            "job_partial",
            "job_failed",
        }

        try:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"SSE endpoint returned HTTP {response.status_code}")

                event_type = ""
                data_buf = ""

                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_buf = line[5:].strip()
                    elif line == "" and data_buf:
                        try:
                            event = json.loads(data_buf)
                        except json.JSONDecodeError:
                            data_buf = ""
                            event_type = ""
                            continue

                        # Capture event name (SSE 'event:' field) inline so
                        # downstream consumers can demux without re-parsing
                        # the wire format.
                        if event_type and "event" not in event:
                            event["event"] = event_type
                        evt_name = event.get("event", event_type)
                        data_buf = ""
                        event_type = ""

                        if evt_name in _JOB_LIFECYCLE_EVENTS:
                            if on_event:
                                on_event(event)
                            if _is_done():
                                break
                            continue

                        item_id = event.get("id", "")
                        status = event.get("status", evt_name)

                        seen_terminal.add(item_id)
                        seen_events[item_id] = event

                        if item_id not in pending:
                            if _is_done():
                                break
                            continue

                        pending.discard(item_id)
                        error_msg = event.get("error")
                        if status == "failed":
                            tracker.mark_failed(item_id, error_msg, event)
                        else:
                            tracker.mark_completed(item_id, event)

                        if on_event:
                            on_event(event)

                        if _is_done():
                            break
                    elif line.startswith(":"):
                        if uploads_done.is_set():
                            _reconcile()
                            if pending:
                                logger.info(
                                    "SSE keepalive arrived with %d items pending "
                                    "after uploads completed; switching to bulk poll",
                                    len(pending),
                                )
                            break

        except Exception as exc:
            logger.warning("SSE stream error: %s: %s", type(exc).__name__, exc)

        if pending:
            logger.info(
                "SSE closed with %d items pending — falling back to bulk poll",
                len(pending),
            )
            await self._bulk_poll_fallback(client, pending, tracker, on_event)

    # ------------------------------------------------------------------
    # Bulk-poll fallback
    # ------------------------------------------------------------------

    async def _bulk_poll_fallback(
        self,
        client: httpx.AsyncClient,
        pending: set[str],
        tracker: DocumentTracker,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Poll ``POST /v1/ingest/status/batch`` until all pending items resolve."""
        url = f"{self._base_url}/v1/ingest/status/batch"
        t0 = time.perf_counter()
        consecutive_errors = 0

        while pending and (time.perf_counter() - t0) < _BULK_POLL_TIMEOUT_S:
            try:
                resp = await client.post(url, json={"ids": list(pending)})
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    logger.error("Bulk poll: too many errors, giving up")
                    break
                logger.warning("Bulk poll error (%s), attempt %d", exc, consecutive_errors)
                await asyncio.sleep(_BULK_POLL_INTERVAL_S * min(consecutive_errors, 6))
                continue

            if resp.status_code != 200:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    break
                await asyncio.sleep(_BULK_POLL_INTERVAL_S)
                continue

            consecutive_errors = 0
            body = resp.json()
            items = body.get("items", {})
            for doc_id, info in items.items():
                if doc_id not in pending:
                    continue
                status = info.get("status", "")
                if status in ("completed", "failed"):
                    pending.discard(doc_id)
                    event = {
                        "id": doc_id,
                        "status": status,
                        "result_rows": info.get("result_rows", 0),
                    }
                    error_msg = info.get("error")
                    if error_msg:
                        event["error"] = error_msg
                    if status == "failed":
                        tracker.mark_failed(doc_id, error_msg, event)
                    else:
                        tracker.mark_completed(doc_id, event)
                    if on_event:
                        on_event(event)

            if pending:
                await asyncio.sleep(_BULK_POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # Public: materialized ingest (upload all, wait for all, return list)
    # ------------------------------------------------------------------

    async def ingest_documents(
        self,
        files: list[UploadInput],
        *,
        on_file_submitted: Callable[[str, str], Any] | None = None,
        show_progress: bool = True,
        pipeline_spec: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Upload documents and wait for all to complete.

        Opens the SSE stream first, then uploads concurrently.  Returns
        one result dict per document containing ``id``, ``status``,
        ``result_rows``, ``elapsed_s``, and optionally ``error``.

        Parameters
        ----------
        on_file_submitted
            Called with ``(filename, document_id)`` after each upload.
        show_progress
            Show Rich progress bars during upload and SSE tracking.
        pipeline_spec
            Optional :class:`PipelineSpec` dict attached to every upload's
            ``metadata`` form blob (see
            :meth:`aingest_documents_stream` for details).
        """
        tracker = DocumentTracker()
        pending = tracker.pending
        uploads_done = asyncio.Event()
        document_ids: list[str] = []

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(timeout=None, connect=30.0)

        t0 = time.monotonic()

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=pool_limits,
            headers=self._auth_headers,
        ) as client:
            created_job = await self._create_job(client, expected_documents=len(files))
            job_id = created_job.job_id
            upload_sem = asyncio.Semaphore(self._max_concurrency)
            upload_failures: list[tuple[str, str]] = []

            async def _upload_one_file(source: UploadInput) -> None:
                outcome = await self._upload_source(
                    client,
                    source,
                    job_id=job_id,
                    pipeline_spec=pipeline_spec,
                    semaphore=upload_sem,
                )
                if outcome.error is not None:
                    upload_failures.append((outcome.filename, outcome.error))
                elif outcome.document_id:
                    pending.add(outcome.attempt_id or outcome.document_id)
                    document_ids.append(outcome.document_id)
                    if on_file_submitted:
                        on_file_submitted(outcome.filename, outcome.document_id)

            progress_ctx = _make_progress() if show_progress else None

            async def _upload_all() -> None:
                tasks = [asyncio.create_task(_upload_one_file(f)) for f in files]
                if progress_ctx:
                    task_id = progress_ctx.add_task("[blue]Uploading", total=len(files))
                    done = 0
                    for coro in asyncio.as_completed(tasks):
                        await coro
                        done += 1
                        progress_ctx.update(task_id, completed=done)
                else:
                    await asyncio.gather(*tasks)
                uploads_done.set()

            if progress_ctx:
                with progress_ctx:
                    sse_task = asyncio.create_task(
                        self._consume_sse(client, pending, uploads_done, tracker, job_id=job_id)
                    )
                    await asyncio.sleep(0.3)
                    await _upload_all()
                    await sse_task
            else:
                sse_task = asyncio.create_task(self._consume_sse(client, pending, uploads_done, tracker, job_id=job_id))
                await asyncio.sleep(0.3)
                await _upload_all()
                await sse_task

        elapsed = time.monotonic() - t0

        for fname, err in upload_failures:
            tracker.errors.append((fname, f"upload failed: {err}"))

        if show_progress:
            mins, secs = divmod(elapsed, 60)
            pps_str = f"{len(files) / elapsed:.1f}" if elapsed > 0 else "N/A"
            print(
                f"\n  Completed: {tracker.completed}  Failed: {tracker.failed}  "
                f"Upload errors: {len(upload_failures)}"
            )
            print(f"  Wall time: {int(mins)}m {secs:.1f}s  ({pps_str} docs/sec)")
            if tracker.errors:
                print(f"\n  Errors ({len(tracker.errors)}):")
                for eid, emsg in tracker.errors[:20]:
                    print(f"    {eid}: {emsg[:100]}")

        return tracker.results

    # ------------------------------------------------------------------
    # Public: streaming ingest (yields events as they arrive)
    # ------------------------------------------------------------------

    async def aingest_documents_stream(
        self,
        files: list[UploadInput],
        *,
        pipeline_spec: dict[str, Any] | None = None,
        retain_results: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator: upload files, yield events as documents complete.

        Opens SSE before uploading.  Yields dicts with:

        * ``{"event": "upload_complete", "filename": ..., "document_id": ...}``
        * ``{"event": "document_complete", "document_id": ..., "status": ...,
              "result_rows": ..., "elapsed_s": ..., "error": ...}``

        When *pipeline_spec* is provided it is attached to each upload's
        ``metadata`` form blob under the ``pipeline`` key so the server can
        validate and apply per-request pipeline overrides.
        """
        tracker = DocumentTracker()
        pending = tracker.pending
        uploads_done = asyncio.Event()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(timeout=None, connect=30.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=pool_limits,
            headers=self._auth_headers,
        ) as client:
            created_job = await self._create_job(
                client,
                expected_documents=len(files),
                retain_results=retain_results,
            )
            job_id = created_job.job_id
            event: dict[str, Any] = {
                "event": "job_created",
                "job_id": job_id,
                "expected_documents": len(files),
            }
            if created_job.trace_id is not None:
                event["trace_id"] = created_job.trace_id
            yield event
            upload_sem = asyncio.Semaphore(self._max_concurrency)

            async def _upload_one_file(source: UploadInput) -> None:
                outcome = await self._upload_source(
                    client,
                    source,
                    job_id=job_id,
                    pipeline_spec=pipeline_spec,
                    semaphore=upload_sem,
                )
                if outcome.error is not None:
                    await event_queue.put(
                        {
                            "event": "upload_failed",
                            "filename": outcome.filename,
                            "error": outcome.error,
                            "job_id": job_id,
                        }
                    )
                elif outcome.document_id:
                    pending.add(outcome.attempt_id or outcome.document_id)
                    await event_queue.put(
                        {
                            "event": "upload_complete",
                            "filename": outcome.filename,
                            "document_id": outcome.document_id,
                            "job_id": job_id,
                        }
                    )

            async def _upload_all() -> None:
                tasks = [asyncio.create_task(_upload_one_file(f)) for f in files]
                await asyncio.gather(*tasks, return_exceptions=True)
                uploads_done.set()

            def _on_sse_event(event: dict[str, Any]) -> None:
                event_name = event.get("event")
                if event_name in {
                    "job_created",
                    "job_started",
                    "job_progress",
                    "job_finalized",
                    "job_partial",
                    "job_failed",
                }:
                    payload = dict(event)
                    payload.setdefault("job_id", job_id)
                    event_queue.put_nowait(payload)
                    return
                doc_id = event.get("id", "")
                status = event.get("status", "completed")
                if status not in ("completed", "failed"):
                    return
                event_queue.put_nowait(
                    {
                        "event": "document_complete",
                        "document_id": doc_id,
                        "status": status,
                        "result_rows": event.get("result_rows", 0),
                        "elapsed_s": event.get("elapsed_s"),
                        "error": event.get("error"),
                        "job_id": job_id,
                    }
                )

            async def _sse_then_signal() -> None:
                await self._consume_sse(
                    client,
                    pending,
                    uploads_done,
                    tracker,
                    job_id=job_id,
                    on_event=_on_sse_event,
                )
                await event_queue.put(None)

            sse_task = asyncio.create_task(_sse_then_signal())
            await asyncio.sleep(0.3)
            upload_task = asyncio.create_task(_upload_all())

            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event

            await upload_task
            await sse_task
