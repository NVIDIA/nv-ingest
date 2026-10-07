# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest endpoints: general, per-page, and whole-document upload.

Each endpoint is mode-aware:

* **gateway** — record :class:`IngestMetrics` (the single authoritative store),
  then proxy the raw HTTP request to the correct backend worker pod.
* **standalone** — record :class:`IngestMetrics` *and* enqueue locally.
* **realtime / batch** — enqueue work to the local pipeline pool only.
  ``IngestMetrics`` is not initialised in worker modes, so ``get_metrics()``
  returns ``None`` and no per-item tracking occurs.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator
from starlette.responses import StreamingResponse

from nemo_retriever.common.schemas.collections import IngestOperation
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.schemas.requests import IngestRequest, JobCreateRequest
from nemo_retriever.common.schemas.responses import (
    DocumentIngestAccepted,
    DocumentStatusResponse,
    IngestAccepted,
    JobAggregateResponse,
    JobCreatedResponse,
    JobDocumentsPage,
    JobStatusResponse,
    PageIngestAccepted,
    SidecarUploadResponse,
)
from nemo_retriever.service.query_schema import AgenticAnswerResponse, QueryRequest, QueryResponse
from nemo_retriever.common.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.models.llm.types import (
    AnswerRequest as CoreAnswerRequest,
    AnswerResult,
    RetrievalResult,
    build_answer_result,
)
from nemo_retriever.service.services.event_bus import get_event_bus
from nemo_retriever.service.services.job_tracker import (
    DocumentRecord,
    JobAggregate,
    MarkOutcome,
    get_job_tracker,
)
from nemo_retriever.service.services.metrics import get_metrics
from nemo_retriever.service.services.pipeline_pool import (
    DocumentWriteContext,
    PoolType,
    WorkItem,
    get_pipeline_pool,
)
from nemo_retriever.service.services.prometheus import (
    INGEST_BYTES_TOTAL,
    INGEST_DOCUMENTS_TOTAL,
    INGEST_PAGES_TOTAL,
    INGEST_REQUESTS_TOTAL,
)
from nemo_retriever.service.services.proxy import get_proxy
from nemo_retriever.service.metrics_otel import record_ingest_accepted
from nemo_retriever.service.services.worker_result_store import (
    ResultStoreTemporarilyUnavailable,
    get_result_data,
    store_result_data,
)
from nemo_retriever.service.utils.file_type import (
    FileCategory,
    FileClassification,
    FileClassifier,
    enforce_media_dependencies,
)

_RETRY_AFTER_SECONDS = "5"
_RESULT_RETRY_AFTER_SECONDS = 60
_DRY_RUN_HEADER = "X-Nemo-Dry-Run"
_PAGE_THRESHOLD_FOR_BATCH = 5

# SSE keepalive cadence; tests monkey-patch this to a short value so
# they don't have to wait the production 30 s before the generator
# re-checks ``request.is_disconnected()`` and exits cleanly.
SSE_KEEPALIVE_TIMEOUT_S = 30.0


class ServiceAnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=1000)
    mode: Literal["classic", "agentic"] = "classic"
    include_chunks: bool = False
    include_metadata: bool = False
    reasoning_enabled: bool | None = None
    reference: str | None = None
    judge: bool = False

    @model_validator(mode="after")
    def _validate_judge_reference(self) -> "ServiceAnswerRequest":
        if self.judge and self.reference is None:
            raise ValueError("judge requires reference")
        if self.mode == "agentic" and (
            self.judge
            or self.reference is not None
            or self.reasoning_enabled is not None
            or self.include_chunks
            or self.include_metadata
        ):
            raise ValueError(
                "mode='agentic' cannot be combined with judge, reference, reasoning_enabled, "
                "include_chunks, or include_metadata"
            )
        return self


logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mode(request: Request) -> str:
    return request.app.state.config.mode


def _is_dry_run(request: Request) -> bool:
    """Return ``True`` when the client sends the dry-run header.

    When present (any truthy value), page and whole-document routes skip
    queue admission and return an immediate 202.
    """
    val = request.headers.get(_DRY_RUN_HEADER, "").strip().lower()
    return val not in ("", "0", "false", "no")


def _role(request: Request) -> str:
    return getattr(request.app.state, "prometheus_role", "standalone")


def _is_gateway(request: Request) -> bool:
    return _mode(request) == "gateway"


def _is_worker(request: Request) -> bool:
    """Return True for split-mode worker pods (``realtime`` or ``batch``).

    Workers don't own the ``JobTracker`` aggregate — the gateway does.
    They receive work by claiming it from the gateway broker rather than
    over these routes.
    """
    return _mode(request) in ("realtime", "batch")


def _job_retain_results(job_id: str | None) -> bool:
    if not job_id:
        return False
    tracker = get_job_tracker()
    if tracker is None:
        return False
    return tracker.should_retain_results(job_id)


def _internal_auth_headers(request: Request) -> dict[str, str]:
    """Return service credentials for pod-to-pod callback traffic."""
    from nemo_retriever.service.auth import internal_auth_headers

    return internal_auth_headers(request.app.state.config.vectordb.internal_api_token)


def _sidecar_owner_fingerprint(request: Request) -> str | None:
    value = getattr(request.state, "caller_fingerprint", None)
    return str(value) if value else None


def _bind_sidecar_for_admission(
    request: Request, pipeline_spec: dict[str, Any] | None
) -> tuple[Any | None, Any | None, dict[str, Any] | None]:
    """Consume an owner-authorized sidecar before publishing work."""
    if pipeline_spec is None or not (vdb := pipeline_spec.get("vdb_upload_params")):
        return None, None, pipeline_spec
    sidecar_id = vdb.get("meta_dataframe_id")
    if not sidecar_id:
        return None, None, pipeline_spec
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store
    from nemo_retriever.service.services.work_queue import SidecarAttachment

    store = get_sidecar_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Sidecar store not initialised")
    entry = store.consume(str(sidecar_id), owner_token=_sidecar_owner_fingerprint(request))
    if entry is None:
        raise HTTPException(status_code=404, detail="Sidecar was not found, expired, or is not owned by this caller")
    attachment = SidecarAttachment(payload=entry.payload, filename=entry.filename, content_type=entry.content_type)
    if _is_gateway(request):
        return entry, attachment, pipeline_spec
    from nemo_retriever.service.services.pipeline_executor import inject_sidecar_attachment

    return (
        entry,
        None,
        inject_sidecar_attachment(
            pipeline_spec, payload=entry.payload, filename=entry.filename, content_type=entry.content_type
        ),
    )


def _proxied_response(response: httpx.Response) -> Response:
    """Relay an upstream VectorDB response to the caller unchanged."""
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _vectordb_get(request: Request, url: str, *, scope: str, failure_detail: str) -> httpx.Response:
    """Read one scoped resource from the internal VectorDB service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.get(
                url,
                headers={"X-NRL-Scope": scope, **_internal_auth_headers(request)},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, failure_detail) from exc


def _job_idempotency_fingerprint(body: JobCreateRequest) -> str:
    """Hash the request fields that an idempotent replay must match.

    The field list is explicit rather than a full model dump so that adding a
    request field cannot silently invalidate previously issued fingerprints.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "expected_documents": body.expected_documents,
                "collection_name": body.collection_name,
                "operation": body.operation,
                "target_document_id": body.target_document_id,
                "metadata": body.metadata,
                "retain_results": body.retain_results,
                "document_manifest": [entry.model_dump(mode="json") for entry in body.document_manifest],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _record_prometheus(
    request: Request,
    endpoint: str,
    status: str,
    *,
    file_size: int = 0,
    is_page: bool = False,
) -> None:
    role = _role(request)
    INGEST_REQUESTS_TOTAL.labels(role=role, endpoint=endpoint, status=status).inc()
    if file_size > 0:
        INGEST_BYTES_TOTAL.labels(role=role, endpoint=endpoint).inc(file_size)
    if is_page:
        INGEST_PAGES_TOTAL.labels(role=role).inc()
    else:
        INGEST_DOCUMENTS_TOTAL.labels(role=role).inc()
    record_ingest_accepted(role=role, endpoint=endpoint, file_size=file_size, is_page=is_page)


def _register_document_under_job(
    *,
    document_id: str,
    job_id: str,
    filename: str | None = None,
    content_sha256: str | None = None,
    stable_document_id: str | None = None,
    manifest_entry_id: str | None = None,
):
    """Register a per-document tracker entry inside an existing job.

    Maps :class:`JobTrackerError` subclasses to HTTP responses so the
    upload routes can surface 404 (job missing) / 409 (full or
    finalized) without leaking the tracker's internal exception types.
    """
    from nemo_retriever.service.services.job_tracker import (
        JobFinalizedError,
        JobFullError,
        JobNotFoundError,
        JobTrackerError,
    )

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    try:
        return tracker.register_document_idempotent(
            document_id,
            job_id=job_id,
            filename=filename,
            content_sha256=content_sha256,
            stable_document_id=stable_document_id,
            manifest_entry_id=manifest_entry_id,
        )
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobFullError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobFinalizedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobTrackerError as exc:
        raise HTTPException(status_code=getattr(exc, "status_code", 500), detail=str(exc)) from exc


def _resolve_stable_document_id(
    attempt_id: str,
    *,
    collection_name: str | None,
    target_document_id: str | None,
) -> str:
    """Keep legacy IDs pollable while separating collection document identity."""
    if target_document_id:
        return target_document_id
    if collection_name:
        return uuid.uuid4().hex
    return attempt_id


def _validate_manifest_entry(job, manifest_entry_id: str | None, filename: str, content_sha256: str) -> None:
    """Bind an upload to exactly one immutable entry in its job manifest."""
    if not job.document_manifest:
        return
    if not manifest_entry_id:
        raise HTTPException(409, "manifest_entry_id is required for this idempotent job")
    entry = next(
        (item for item in job.document_manifest if item.get("manifest_entry_id") == manifest_entry_id),
        None,
    )
    if not entry or entry.get("filename") != filename or entry.get("content_sha256") != content_sha256:
        raise HTTPException(409, "Uploaded document does not match its job manifest entry")


def _validate_collection_pipeline_spec(job, spec: PipelineSpec | None) -> None:
    """Prevent collection writes from bypassing the configured VDB boundary."""
    if not job.collection_name or spec is None:
        return
    if spec.vdb_upload_params is not None:
        raise HTTPException(
            422,
            "collection-aware ingestion cannot override VectorDB upload configuration",
        )


def _require_job(job_id: str, request: Request | None = None):
    """Look up an existing :class:`JobAggregate` or raise HTTP 404."""
    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    agg = tracker.get_job(job_id)
    if agg is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if request is not None and not _is_worker(request):
        from nemo_retriever.service.auth import authorized_scope

        if agg.scope != authorized_scope(request):
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return agg


async def _enqueue_or_reject(pool_type: PoolType, item: WorkItem) -> None:
    """Submit *item* to the pipeline pool, raising HTTP 429 if full."""
    pool = get_pipeline_pool()
    if pool is None:
        return
    if not item.trace_context:
        item.trace_context = _safe_inject_trace_context()
    if not await pool.submit(pool_type, item):
        raise HTTPException(
            status_code=429,
            detail=f"{pool_type.value} pipeline is at capacity — try again shortly",
            headers={"Retry-After": _RETRY_AFTER_SECONDS},
        )


async def _fetch_result_data_from_workers(
    document_id: str,
) -> list[dict[str, Any]] | None:
    """Read rows already handed off to this gateway's retained store."""
    try:
        rows = await asyncio.to_thread(get_result_data, document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(_RESULT_RETRY_AFTER_SECONDS)},
        ) from exc
    if rows is not None:
        return rows
    raise HTTPException(
        status_code=503,
        detail=f"Retained result data for {document_id!r} is temporarily unavailable",
        headers={"Retry-After": str(_RESULT_RETRY_AFTER_SECONDS)},
    )


def _worker_result_url(
    request: Request,
    document_id: str,
    worker_ip_value: Any,
    callback_worker_ip: Any = None,
) -> str:
    """Build a fixed-path owner URL from a validated worker pod IP."""
    if not isinstance(worker_ip_value, str):
        raise HTTPException(
            status_code=503,
            detail="Completion callback is missing result worker identity",
        )
    try:
        worker_ip = ipaddress.ip_address(worker_ip_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Completion callback has an invalid result worker IP",
        ) from exc
    if worker_ip.is_unspecified or worker_ip.is_multicast or worker_ip.is_loopback or worker_ip.is_link_local:
        raise HTTPException(
            status_code=400,
            detail="Completion callback has an unroutable result worker IP",
        )

    if callback_worker_ip is not None:
        try:
            advertised_ip = ipaddress.ip_address(callback_worker_ip)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Completion callback has an invalid result worker IP",
            ) from exc
        if advertised_ip != worker_ip:
            raise HTTPException(status_code=409, detail="Result worker IP does not match lease owner")

    peer_value = request.client.host if request.client is not None else ""
    try:
        peer_ip = ipaddress.ip_address(peer_value)
    except ValueError:
        peer_ip = None
    if peer_ip is not None and not peer_ip.is_loopback and peer_ip != worker_ip:
        raise HTTPException(status_code=400, detail="Result worker IP does not match callback peer")

    host = f"[{worker_ip}]" if worker_ip.version == 6 else str(worker_ip)
    port = request.app.state.config.server.port
    return f"http://{host}:{port}/v1/internal/document-result/{quote(document_id, safe='')}"


async def _pull_and_store_worker_result(
    request: Request,
    document_id: str,
    worker_ip: Any,
    callback_worker_ip: Any = None,
) -> None:
    """Copy rows from the exact completing worker into the gateway store."""
    url = _worker_result_url(request, document_id, worker_ip, callback_worker_ip)
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_internal_auth_headers(request)) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch retained result for {document_id!r} from its worker",
            headers={"Retry-After": "1"},
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Worker returned HTTP {response.status_code} for retained result {document_id!r}",
            headers={"Retry-After": "1"},
        )
    try:
        payload = response.json()
        rows = payload.get("result_data") if isinstance(payload, dict) else None
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Worker returned invalid result data for {document_id!r}",
        ) from exc
    if not rows or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise HTTPException(
            status_code=503,
            detail=f"Worker returned invalid result data for {document_id!r}",
        )
    try:
        await asyncio.to_thread(store_result_data, document_id, rows)
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to retain result data for {document_id!r} on the gateway",
            headers={"Retry-After": "1"},
        ) from exc


async def _gateway_enqueue(
    request: Request,
    pool_type: PoolType,
    *,
    work_id: str,
    job_id: str,
    payload: bytes,
    filename: str | None,
    pipeline_spec: dict[str, Any] | None = None,
    write: DocumentWriteContext | None = None,
    sidecar: Any | None = None,
) -> None:
    """Admit split-mode work to the gateway broker after atomic spooling.

    The write context is nested under a single ``write`` key so the claiming
    worker rebuilds it as one typed object rather than a splat of loose
    fields that :class:`WorkItem` would silently discard.
    """
    from nemo_retriever.service.services.work_queue import (
        WorkQueueFull,
        get_work_broker,
    )

    broker = get_work_broker()
    if broker is None:
        tracker = get_job_tracker()
        if tracker is not None:
            tracker.unregister_pending(work_id)
        raise HTTPException(status_code=503, detail="Gateway work broker not initialised")
    try:
        await broker.enqueue(
            pool_type,
            work_id=work_id,
            job_id=job_id,
            payload=payload,
            filename=filename,
            retain_results=_job_retain_results(job_id),
            pipeline_spec=pipeline_spec,
            trace_context=_safe_inject_trace_context(),
            extra={"write": write.model_dump(mode="json")} if write is not None else None,
            sidecar=sidecar,
        )
    except WorkQueueFull as exc:
        tracker = get_job_tracker()
        if tracker is not None:
            tracker.unregister_pending(work_id)
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": _RETRY_AFTER_SECONDS},
        ) from exc


def _file_size_from_upload(file: UploadFile) -> int:
    """Best-effort upload size without reading bytes."""
    if file.size is not None:
        return file.size
    return 0


def _check_upload_size(file: UploadFile, request: Request) -> None:
    """Reject uploads exceeding the configured size limit before buffering."""
    config = request.app.state.config
    limit = getattr(getattr(config, "resources", None), "max_upload_bytes", None)
    if limit is None:
        return
    size = file.size
    if size is not None and size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload size {size:,} bytes exceeds limit of {limit:,} bytes",
        )


def _count_pdf_pages(file_bytes: bytes) -> int:
    """Return the number of pages in a PDF, or 1 for non-PDF / errors."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(file_bytes)
        n = len(doc)
        doc.close()
        return n
    except Exception as exc:
        logger.warning("Could not determine PDF page count; defaulting to 1 page: %s", exc)
        return 1


def _route_by_page_count(
    file_bytes: bytes,
    meta: IngestRequest,
    file_category: FileCategory | None = None,
) -> PoolType:
    """Route uploads to realtime or batch based on file type and page count.

    * Audio / video files are always routed to **batch** — they involve
      heavyweight ASR / frame-extraction pipelines.
    * Text, HTML, and image files are always routed to **realtime** — they do
      not need PDF page counting.
    * Documents (PDF, DOCX, PPTX) use the original page-count heuristic:
      small docs (<threshold pages) go to realtime, larger ones to batch.

    When the client requested PDF page-chunking via
    :attr:`PipelineSpec.pdf_split`, we route to **batch** as soon as the
    document has more than one chunk's worth of pages — chunking is
    intrinsically a throughput-oriented operation.
    """
    if file_category in (FileCategory.AUDIO, FileCategory.VIDEO):
        return PoolType.BATCH
    if file_category in (FileCategory.TEXT, FileCategory.HTML, FileCategory.IMAGE):
        return PoolType.REALTIME
    if meta.page_number is not None:
        return PoolType.REALTIME
    pages = _count_pdf_pages(file_bytes)
    if meta.pipeline and meta.pipeline.pdf_split is not None:
        if pages > meta.pipeline.pdf_split.pages_per_chunk:
            return PoolType.BATCH
    return PoolType.REALTIME if pages < _PAGE_THRESHOLD_FOR_BATCH else PoolType.BATCH


def _build_policy(request: Request):  # -> PipelineOverridesPolicy
    """Build a :class:`PipelineOverridesPolicy` from the live app config.

    The ``caption_enabled`` flag is derived here so the trust boundary
    stays in one place — clients can only override caption settings when
    the operator has wired up ``nim_endpoints.caption_invoke_url``.
    """
    cfg = request.app.state.config
    caption_enabled = bool(getattr(cfg.nim_endpoints, "caption_invoke_url", None))
    return cfg.pipeline_overrides.to_policy(caption_enabled=caption_enabled)


def _resolve_pipeline_spec(request: Request, meta: IngestRequest) -> PipelineSpec | None:
    """Validate ``meta.pipeline`` against the service's override policy.

    Returns ``None`` when the spec is missing or empty so the worker
    short-circuits to the legacy startup-baked pipeline. Raises
    :class:`HTTPException` (the FastAPI-native error) for policy denials.
    """
    if meta.pipeline is None:
        return None
    policy = _build_policy(request)
    try:
        return validate_pipeline_spec(meta.pipeline, policy)
    except PolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


async def _prepare_job_work_item(
    request: Request,
    *,
    job_id: str,
    file: UploadFile,
    meta: IngestRequest,
    validated_spec: PipelineSpec | None,
    manifest_entry_id: str | None,
    job: JobAggregate,
    pool_type: PoolType | None = None,
) -> tuple[WorkItem, PoolType, FileClassification]:
    """Build the execution envelope shared by the document upload routes.

    Reads the upload exactly once and resolves every non-payload value the
    gateway and standalone paths need: pool routing, content digest,
    manifest binding, attempt identity and durable storage identity.

    Args:
        request: The inbound upload request.
        job_id: Job aggregate the upload belongs to, taken from the URL path.
        file: The uploaded file.
        meta: Parsed ``IngestRequest`` metadata accompanying the upload.
        validated_spec: Policy-validated per-request pipeline overrides.
        manifest_entry_id: Immutable manifest entry the upload claims, if any.
        job: The job aggregate owning this upload.
        pool_type: Fixed pool for callers that do not auto-route.

    Returns:
        The work item, its target pool, and the file classification.
    """
    classification = FileClassifier.classify(file, filename_override=meta.filename or "")
    enforce_media_dependencies(classification)

    file_bytes = await file.read()
    route = pool_type or _route_by_page_count(file_bytes, meta, file_category=classification.category)
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()

    _validate_manifest_entry(job, manifest_entry_id, file.filename or "", content_sha256)

    attempt_id = uuid.uuid4().hex
    storage_document_id = _resolve_stable_document_id(
        attempt_id,
        collection_name=job.collection_name,
        target_document_id=job.target_document_id,
    )

    pipeline_spec = validated_spec.model_dump(mode="json") if validated_spec is not None else None
    sidecar_entry, sidecar_attachment, pipeline_spec = _bind_sidecar_for_admission(request, pipeline_spec)
    item = WorkItem(
        id=attempt_id,
        payload=file_bytes,
        filename=file.filename,
        callback_headers=_internal_auth_headers(request),
        job_id=job_id,
        pipeline_spec=pipeline_spec,
        sidecar_attachment=(sidecar_entry, sidecar_attachment) if sidecar_entry is not None else None,
        retain_results=_job_retain_results(job_id),
        write=DocumentWriteContext(
            scope=job.scope,
            collection_name=job.collection_name,
            operation=job.operation,
            content_sha256=content_sha256,
            storage_document_id=storage_document_id,
            document_metadata=meta.metadata,
        ),
    )
    return item, route, classification


async def _submit_job_work_item(
    request: Request,
    pool_type: PoolType,
    item: WorkItem,
    *,
    manifest_entry_id: str | None,
) -> DocumentRecord | None:
    """Register *item* under its job and admit it for execution.

    Args:
        request: The inbound upload request.
        pool_type: Pool the item is admitted to.
        item: Envelope produced by :func:`_prepare_job_work_item`.
        manifest_entry_id: Immutable manifest entry the upload claims, if any.

    Returns:
        The already-registered record when an idempotent retry matched an
        earlier attempt, otherwise ``None`` once the item is admitted.
    """
    if item.job_id is None:
        raise RuntimeError("job upload work item is missing job_id")

    record, created = _register_document_under_job(
        document_id=item.id,
        job_id=item.job_id,
        filename=item.filename,
        content_sha256=item.write.content_sha256,
        stable_document_id=item.write.storage_document_id,
        manifest_entry_id=manifest_entry_id,
    )
    if not created:
        # Registration can reject an idempotent duplicate after sidecar admission.
        # Put the consumed entry back so the duplicate has no side effect.
        if item.sidecar_attachment is not None:
            from nemo_retriever.service.services.sidecar_store import get_sidecar_store

            store = get_sidecar_store()
            if store is not None:
                store.restore(item.sidecar_attachment[0])
        return record

    if _is_gateway(request):
        try:
            await _gateway_enqueue(
                request,
                pool_type,
                work_id=item.id,
                job_id=item.job_id,
                payload=item.payload,
                filename=item.filename,
                pipeline_spec=item.pipeline_spec,
                write=item.write,
                sidecar=item.sidecar_attachment[1] if item.sidecar_attachment is not None else None,
            )
        except (HTTPException, OSError, ValueError):
            _rollback_rejected_work_item(item)
            raise
    else:
        try:
            await _enqueue_or_reject(pool_type, item)
        except HTTPException:
            _rollback_rejected_work_item(item)
            raise

    return None


def _rollback_rejected_work_item(item: WorkItem) -> None:
    """Undo state reserved before a work item was rejected by its queue."""
    tracker = get_job_tracker()
    if tracker is not None:
        tracker.unregister_pending(item.id)
    if item.sidecar_attachment is not None:
        from nemo_retriever.service.services.sidecar_store import get_sidecar_store

        store = get_sidecar_store()
        if store is not None:
            store.restore(item.sidecar_attachment[0])


def _parse_backend_json(resp: Response) -> dict:
    """Attempt to decode the backend response body as JSON."""
    try:
        return json.loads(resp.body)
    except Exception:
        return {}


def _safe_inject_trace_context() -> dict[str, str]:
    """Best-effort W3C trace context capture for downstream propagation."""
    from nemo_retriever.service import tracing

    try:
        return dict(tracing.inject_trace_context())
    except Exception as exc:
        logger.warning(
            "Trace context injection failed; continuing without propagated context: %s",
            exc,
        )
        return {}


def _safe_extract_trace_context(carrier: dict[str, str] | None) -> Any | None:
    """Best-effort W3C trace context extraction for accept spans."""
    from nemo_retriever.service import tracing

    try:
        return tracing.extract_trace_context(carrier)
    except Exception as exc:
        logger.warning(
            "Trace context extraction failed; continuing without parent context: %s",
            exc,
        )
        return None


def _trace_context_from_request_or_job(request: Request, job_id: str | None) -> dict[str, str]:
    inbound_traceparent = request.headers.get("traceparent")
    if inbound_traceparent:
        carrier = {"traceparent": inbound_traceparent}
        if tracestate := request.headers.get("tracestate"):
            carrier["tracestate"] = tracestate
        return carrier

    if not job_id:
        return {}
    tracker = get_job_tracker()
    if tracker is None:
        return {}
    agg = tracker.get_job(job_id)
    if agg is None:
        return {}
    return dict(agg.trace_context)


def _start_accept_span(request: Request, job_id: str, name: str):
    from opentelemetry.trace import SpanKind

    from nemo_retriever.service import tracing

    carrier = _trace_context_from_request_or_job(request, job_id)
    return tracing.start_span(
        name,
        kind=SpanKind.SERVER,
        context=_safe_extract_trace_context(carrier),
        attributes={
            "service.role": _role(request),
            "job.id": job_id,
            "route": request.url.path,
        },
    )


# ------------------------------------------------------------------
# POST /v1/ingest/job          — create a new job aggregate
# GET  /v1/ingest/job/{job_id} — fetch the aggregate's current state
# ------------------------------------------------------------------


def _aggregate_to_response(agg, *, documents: list[dict[str, Any]] | None = None) -> JobAggregateResponse:
    """Project a :class:`JobAggregate` to the wire response model."""
    return JobAggregateResponse(
        job_id=agg.job_id,
        expected_documents=agg.expected_documents,
        status=agg.status.value,
        created_at=agg.created_at,
        started_at=agg.started_at,
        finalized_at=agg.finalized_at,
        elapsed_s=agg.elapsed_s,
        label=agg.label,
        trace_id=agg.trace_id,
        counts=dict(agg.counts),
        document_ids=list(agg.document_ids),
        documents=documents,
        collection_name=agg.collection_name,
        operation=agg.operation,
    )


@router.post(
    "/ingest/job",
    response_model=JobCreatedResponse,
    status_code=201,
    summary="Create a new ingestion job aggregate",
)
async def create_job(request: Request, response: Response, body: JobCreateRequest) -> JobCreatedResponse | Response:
    """Open a job that will receive ``expected_documents`` uploads.

    The server returns an opaque ``job_id`` the client uses for every
    subsequent ``POST /v1/ingest/job/{job_id}/document`` (or ``/page``)
    call. The job is in-memory only; gateway pod restarts erase it
    (this is intentional — see the J1 design notes).
    """
    from opentelemetry.trace import SpanKind

    from nemo_retriever.service import tracing
    from nemo_retriever.service.services.job_tracker import JobTrackerError

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    from nemo_retriever.service.auth import authorized_scope

    scope = authorized_scope(request)
    if body.collection_name:
        config = request.app.state.config
        if not config.vectordb.enabled:
            raise HTTPException(404, "VectorDB is not enabled in the service configuration")
        target = f"{config.vectordb.vectordb_url.rstrip('/')}/v1/collections/{body.collection_name}"
        collection_response = await _vectordb_get(
            request,
            target,
            scope=scope,
            failure_detail="Failed to validate collection with VectorDB service",
        )
        if collection_response.status_code != 200:
            return _proxied_response(collection_response)
        if collection_response.json().get("status") != "active":
            raise HTTPException(409, "Collection is not active")
        if body.operation is IngestOperation.REPLACE and body.target_document_id:
            document_response = await _vectordb_get(
                request,
                f"{target}/documents/{body.target_document_id}",
                scope=scope,
                failure_detail="Failed to validate replacement document",
            )
            if document_response.status_code != 200:
                return _proxied_response(document_response)
    fingerprint = _job_idempotency_fingerprint(body)
    job_id = uuid.uuid4().hex
    trace_id: str | None = None
    inbound_trace_context = _trace_context_from_request_or_job(request, None)
    try:
        with tracing.start_span(
            "ingest.job",
            kind=SpanKind.SERVER,
            context=_safe_extract_trace_context(inbound_trace_context),
            attributes={
                "service.role": _role(request),
                "job.expected_documents": body.expected_documents,
            },
        ):
            trace_context = _safe_inject_trace_context()
            trace_id = tracing.current_trace_id_hex() if trace_context else None
            agg = tracker.register_job(
                job_id,
                expected_documents=body.expected_documents,
                label=body.label,
                metadata=body.metadata,
                retain_results=body.retain_results,
                trace_id=trace_id,
                trace_context=trace_context,
                collection_name=body.collection_name,
                scope=scope,
                operation=body.operation,
                target_document_id=body.target_document_id,
                idempotency_key=body.idempotency_key,
                idempotency_fingerprint=fingerprint,
                document_manifest=[entry.model_dump(mode="json") for entry in body.document_manifest],
            )
    except JobTrackerError as exc:
        raise HTTPException(status_code=getattr(exc, "status_code", 500), detail=str(exc)) from exc

    created = agg.job_id == job_id
    if not created:
        response.status_code = 200
    if created and trace_id:
        response.headers[tracing.TRACE_ID_HEADER] = trace_id
    if created and (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/job")
        m.record_job_created(job_id)
    return JobCreatedResponse(
        job_id=agg.job_id,
        expected_documents=agg.expected_documents,
        status=agg.status.value,
        created_at=agg.created_at,
        label=agg.label,
        trace_id=agg.trace_id,
        collection_name=agg.collection_name,
        operation=agg.operation,
    )


@router.get(
    "/ingest/job/{job_id}",
    response_model=JobAggregateResponse,
    summary="Fetch the current state of a job aggregate",
)
async def get_job(
    request: Request,
    job_id: str,
    include_documents: bool = False,
) -> JobAggregateResponse:
    """Return the live aggregate snapshot.

    Pass ``?include_documents=true`` to also return the per-document
    records (capped to the first 10k entries to keep payloads bounded).
    """
    agg = _require_job(job_id, request)
    documents: list[dict[str, Any]] | None = None
    if include_documents:
        tracker = get_job_tracker()
        docs = tracker.job_documents(job_id) if tracker is not None else []
        cap = 10_000
        documents = [d.model_dump() for d in docs[:cap]]
    return _aggregate_to_response(agg, documents=documents)


# ------------------------------------------------------------------
# GET /v1/ingest/job/{job_id}/documents
#   J6 — paginated per-document status under a job. Supports filtering
#   by ``status`` and slicing with ``offset`` / ``limit``. Preferred
#   over ``?include_documents=true`` for big jobs.
# GET /v1/ingest/job/{job_id}/document/{document_id}
#   J6 — single-document detail (replaces the legacy ``/ingest/status``
#   shape with a JSON body that exposes ``job_id`` + ``filename``).
# ------------------------------------------------------------------


def _document_to_response(rec, *, result_data=None) -> DocumentStatusResponse:
    """Project a :class:`DocumentRecord` to the wire response shape."""
    return DocumentStatusResponse(
        document_id=rec.stable_document_id,
        attempt_id=rec.id,
        job_id=rec.job_id,
        status=rec.status.value,
        submitted_at=rec.submitted_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        elapsed_s=rec.elapsed_s,
        filename=rec.filename,
        result_rows=rec.result_rows,
        result_data=result_data,
        error=rec.error,
        collection_name=rec.collection_name,
        content_sha256=rec.content_sha256,
    )


@router.get(
    "/ingest/job/{job_id}/documents",
    response_model=JobDocumentsPage,
    summary="Paginated per-document status under a job",
)
async def get_job_documents(
    request: Request,
    job_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> JobDocumentsPage:
    """Return per-document records for *job_id*, paginated.

    Parameters
    ----------
    status:
        Optional filter — one of ``pending``, ``processing``,
        ``completed``, ``failed``. The default returns every document.
    offset:
        Zero-based page start (must be ``>= 0``).
    limit:
        Page size; capped at 1000.

    The aggregate ``total`` is the absolute document count, and
    ``total_filtered`` is what the status filter selected; ``items``
    is the slice for the requested page.
    """
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be in [1, 1000]")

    agg = _require_job(job_id, request)
    tracker = get_job_tracker()
    docs = tracker.job_documents(job_id) if tracker is not None else []

    if status is not None:
        status_lower = status.lower()
        valid = {"pending", "processing", "completed", "failed"}
        if status_lower not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of {sorted(valid)}, got {status!r}",
            )
        filtered = [d for d in docs if d.status.value == status_lower]
    else:
        filtered = docs

    page = filtered[offset : offset + limit]
    return JobDocumentsPage(
        job_id=agg.job_id,
        total=len(docs),
        total_filtered=len(filtered),
        offset=offset,
        limit=limit,
        items=[_document_to_response(d) for d in page],
    )


@router.get(
    "/ingest/job/{job_id}/document/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Fetch the current status of one document in a job",
    responses={
        200: {"model": DocumentStatusResponse},
        202: {"model": DocumentStatusResponse},
        404: {"description": "Job or document not found"},
    },
)
async def get_job_document(
    request: Request,
    job_id: str,
    document_id: str,
) -> Response:
    """Single-document detail nested under the owning job.

    Returns HTTP 200 when the document is terminal (including retained
    ``result_data`` when requested for the job) and HTTP 202
    while still pending/processing. A 404 is returned if either the
    job is unknown or the document does not belong to this job — the
    latter prevents leaking document existence across tenants.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    _require_job(job_id, request)
    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker is not available.")
    rec = tracker.get_document(document_id)
    if rec is None or rec.job_id != job_id:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id!r} not found in job {job_id!r}",
        )
    is_terminal = rec.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
    result_data = tracker.get_result_data(document_id) if is_terminal else None
    if (
        is_terminal
        and result_data is None
        and rec.result_rows
        and _job_retain_results(rec.job_id)
        and _is_gateway(request)
    ):
        result_data = await _fetch_result_data_from_workers(document_id)
    body = _document_to_response(rec, result_data=result_data).model_dump()
    return JSONResponse(content=body, status_code=200 if is_terminal else 202)


# ------------------------------------------------------------------
# POST /v1/ingest/job/{job_id}/document   — auto-routed by page count
# POST /v1/ingest/job/{job_id}/page       — explicit single page
# POST /v1/ingest/job/{job_id}/whole      — explicit whole document
# ------------------------------------------------------------------


@router.post(
    "/ingest/job/{job_id}/document",
    response_model=IngestAccepted,
    status_code=202,
    summary="Submit a document to an existing job (auto-routed by page count)",
)
async def submit_document_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="The file to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
    manifest_entry_id: str | None = Form(default=None, description="Immutable entry ID from the job manifest"),
) -> IngestAccepted | Response:
    """General-purpose upload into a job.

    Routes to the realtime or batch pool based on PDF page count
    (matches the legacy ``POST /v1/ingest`` behaviour). The ``job_id``
    comes from the URL path; any ``job_id`` in the metadata body is
    ignored.
    """
    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    job = _require_job(job_id, request)
    _check_upload_size(file, request)
    validated_spec = _resolve_pipeline_spec(request, meta)
    _validate_collection_pipeline_spec(job, validated_spec)

    with _start_accept_span(request, job_id, "ingest.document.accept"):
        item, route, classification = await _prepare_job_work_item(
            request,
            job_id=job_id,
            file=file,
            meta=meta,
            validated_spec=validated_spec,
            manifest_entry_id=manifest_entry_id,
            job=job,
        )
        now = datetime.now(timezone.utc).isoformat()

        record = await _submit_job_work_item(request, route, item, manifest_entry_id=manifest_entry_id)
        if record is not None:
            return IngestAccepted(
                document_id=record.stable_document_id,
                attempt_id=record.id,
                job_id=item.job_id,
                content_sha256=record.content_sha256 or item.write.content_sha256,
                status="accepted",
                created_at=record.submitted_at,
            )

        file_size = _file_size_from_upload(file) if _is_gateway(request) else len(item.payload)
        _record_prometheus(request, "/v1/ingest/job/document", "2xx", file_size=file_size)
        if (m := get_metrics()) is not None:
            m.record_request("/v1/ingest/job/document")
            m.record_document_accepted(
                document_id=item.id,
                job_id=item.job_id,
                filename=classification.filename,
                file_category=classification.category.value,
                content_type=classification.content_type,
                file_size_bytes=file_size,
                endpoint="/v1/ingest/job/document",
            )

        return IngestAccepted(
            document_id=item.write.storage_document_id,
            attempt_id=item.id,
            job_id=item.job_id,
            content_sha256=item.write.content_sha256,
            status="accepted",
            created_at=now,
        )


@router.post(
    "/ingest/job/{job_id}/page",
    response_model=PageIngestAccepted,
    status_code=202,
    summary="Upload a single page belonging to a pre-split document inside a job",
)
async def submit_page_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="A single-page PDF or image"),
    document_id: str = Form(
        ...,
        description="Client-assigned ID grouping pages from the same source document",
    ),
    page_number: int = Form(..., description="1-based page number within the source document"),
    filename: str = Form(default="", description="Original source document filename"),
) -> PageIngestAccepted | Response:
    if _require_job(job_id, request).collection_name:
        raise HTTPException(
            422,
            "collection-aware ingestion does not support /page; use /document or /whole",
        )
    _check_upload_size(file, request)

    with _start_accept_span(request, job_id, "ingest.page.accept"):
        if _is_gateway(request):
            dry_run = _is_dry_run(request)
            classification = FileClassifier.classify(file, filename_override=filename)
            enforce_media_dependencies(classification)
            file_size = _file_size_from_upload(file)

            page_id = uuid.uuid4().hex
            file_bytes = await file.read()
            content_sha256 = hashlib.sha256(file_bytes).hexdigest()
            now = datetime.now(timezone.utc).isoformat()

            if not dry_run:
                _register_document_under_job(
                    document_id=page_id,
                    job_id=job_id,
                    filename=filename or file.filename,
                )
                await _gateway_enqueue(
                    request,
                    PoolType.REALTIME,
                    work_id=page_id,
                    job_id=job_id,
                    payload=file_bytes,
                    filename=file.filename,
                )

            if not dry_run:
                _record_prometheus(
                    request,
                    "/v1/ingest/job/page",
                    "2xx",
                    file_size=file_size,
                    is_page=True,
                )
                if (m := get_metrics()) is not None:
                    m.record_request("/v1/ingest/job/page")
                    m.record_page_accepted(
                        page_id=page_id,
                        document_id=document_id,
                        job_id=job_id,
                        endpoint="/v1/ingest/job/page",
                        page_number=page_number,
                        file_size_bytes=file_size,
                        file_category=classification.category.value,
                        content_type=classification.content_type,
                    )

            return PageIngestAccepted(
                page_id=page_id,
                document_id=document_id,
                page_number=page_number,
                content_sha256=content_sha256,
                status="accepted",
                created_at=now,
            )

        # ── worker / standalone ──────────────────────────────────────
        dry_run = _is_dry_run(request)
        classification = FileClassifier.classify(file, filename_override=filename)
        enforce_media_dependencies(classification)

        file_bytes = await file.read()
        content_sha256 = hashlib.sha256(file_bytes).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        page_id = uuid.uuid4().hex

        if not dry_run:
            _register_document_under_job(
                document_id=page_id,
                job_id=job_id,
                filename=filename or file.filename,
            )
            await _enqueue_or_reject(
                PoolType.REALTIME,
                WorkItem(
                    id=page_id,
                    payload=file_bytes,
                    filename=file.filename,
                    callback_headers=_internal_auth_headers(request),
                    job_id=job_id,
                    retain_results=_job_retain_results(job_id),
                ),
            )

        if not dry_run:
            _record_prometheus(
                request,
                "/v1/ingest/job/page",
                "2xx",
                file_size=len(file_bytes),
                is_page=True,
            )

            if (m := get_metrics()) is not None:
                m.record_request("/v1/ingest/job/page")
                m.record_page_accepted(
                    page_id=page_id,
                    document_id=document_id,
                    job_id=job_id,
                    endpoint="/v1/ingest/job/page",
                    page_number=page_number,
                    file_size_bytes=len(file_bytes),
                    file_category=classification.category.value,
                    content_type=classification.content_type,
                )

        return PageIngestAccepted(
            page_id=page_id,
            document_id=document_id,
            page_number=page_number,
            content_sha256=content_sha256,
            status="accepted",
            created_at=now,
        )


@router.post(
    "/ingest/job/{job_id}/whole",
    response_model=DocumentIngestAccepted,
    status_code=202,
    summary="Upload a complete document into a job (server handles page splitting)",
)
async def submit_whole_document_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="The full document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
    manifest_entry_id: str | None = Form(default=None, description="Immutable entry ID from the job manifest"),
) -> DocumentIngestAccepted | Response:
    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    job = _require_job(job_id, request)
    _check_upload_size(file, request)
    validated_spec = _resolve_pipeline_spec(request, meta)
    _validate_collection_pipeline_spec(job, validated_spec)

    with _start_accept_span(request, job_id, "ingest.whole.accept"):
        item, route, classification = await _prepare_job_work_item(
            request,
            job_id=job_id,
            file=file,
            meta=meta,
            validated_spec=validated_spec,
            manifest_entry_id=manifest_entry_id,
            job=job,
            pool_type=PoolType.BATCH,
        )
        now = datetime.now(timezone.utc).isoformat()

        dry_run = _is_dry_run(request)
        if not dry_run:
            record = await _submit_job_work_item(request, route, item, manifest_entry_id=manifest_entry_id)
            if record is not None:
                return DocumentIngestAccepted(
                    document_id=record.stable_document_id,
                    attempt_id=record.id,
                    filename=classification.filename,
                    file_size_bytes=len(item.payload),
                    content_sha256=record.content_sha256 or item.write.content_sha256,
                    status="accepted",
                    created_at=record.submitted_at,
                )

        file_size = _file_size_from_upload(file) if _is_gateway(request) else len(item.payload)
        if not dry_run:
            _record_prometheus(request, "/v1/ingest/job/whole", "2xx", file_size=file_size)
            if (m := get_metrics()) is not None:
                m.record_request("/v1/ingest/job/whole")
                m.record_document_accepted(
                    document_id=item.id,
                    job_id=item.job_id,
                    filename=classification.filename,
                    file_category=classification.category.value,
                    content_type=classification.content_type,
                    file_size_bytes=file_size,
                    endpoint="/v1/ingest/job/whole",
                )
        return DocumentIngestAccepted(
            document_id=item.write.storage_document_id,
            attempt_id=item.id,
            filename=classification.filename,
            file_size_bytes=len(item.payload),
            content_sha256=item.write.content_sha256,
            status="accepted",
            created_at=now,
        )


# ------------------------------------------------------------------
# GET /v1/ingest/status/{item_id}  — status for general ingest items
# GET /v1/ingest/page/status/{page_id}  — status for page items
# GET /v1/ingest/document/status/{document_id}  — status for document items
# ------------------------------------------------------------------


async def _status_response(request: Request, item_id: str) -> JSONResponse:
    """Look up document status and return the appropriate HTTP code.

    Returns 200 for completed/failed, 202 for pending/processing, 404 if unknown.
    Terminal result data is read idempotently from the tracker or, in gateway
    mode, from the shared worker result store.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Job tracker is not available on this pod.",
        )
    rec = tracker.get_document(item_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No tracked document with id={item_id!r}")
    _require_job(rec.job_id, request)

    is_terminal = rec.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
    result_data = tracker.get_result_data(item_id) if is_terminal else None
    if (
        is_terminal
        and result_data is None
        and rec.result_rows
        and _job_retain_results(rec.job_id)
        and _is_gateway(request)
    ):
        result_data = await _fetch_result_data_from_workers(item_id)

    body = JobStatusResponse(
        id=rec.id,
        status=rec.status.value,
        submitted_at=rec.submitted_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        elapsed_s=rec.elapsed_s,
        result_rows=rec.result_rows,
        result_data=result_data,
        error=rec.error,
    ).model_dump()

    if is_terminal:
        return JSONResponse(content=body, status_code=200)
    return JSONResponse(content=body, status_code=202)


@router.post(
    "/ingest/sidecar",
    response_model=SidecarUploadResponse,
    status_code=201,
    summary="Upload a sidecar metadata file for use with vdb_upload",
)
async def ingest_sidecar(
    request: Request,
    file: UploadFile = File(..., description="Sidecar metadata payload (csv / json / parquet)."),
    ttl_s: float = Form(default=3600.0, description="Time-to-live in seconds."),
    consume_on_read: bool = Form(default=True, description="Remove after the first admitted ingest."),
) -> SidecarUploadResponse | Response:
    """Store one opaque sidecar ID at the gateway admission boundary."""
    from datetime import datetime, timezone
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    _check_upload_size(file, request)
    config = request.app.state.config
    if config.mode in ("realtime", "batch"):
        raise HTTPException(
            status_code=404, detail="Sidecar operations are available only on the gateway or standalone service"
        )
    store = get_sidecar_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Sidecar store not initialised")

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(min(64 * 1024, config.sidecar_store.max_payload_bytes - total + 1)):
        total += len(chunk)
        if total > config.sidecar_store.max_payload_bytes:
            raise HTTPException(status_code=413, detail="Sidecar upload exceeds the configured payload limit")
        chunks.append(chunk)
    payload = b"".join(chunks)
    if not payload:
        raise HTTPException(status_code=400, detail="Sidecar upload is empty")
    try:
        entry = store.put(
            filename=file.filename or "sidecar",
            content_type=file.content_type or "application/octet-stream",
            payload=payload,
            owner_token=_sidecar_owner_fingerprint(request),
            ttl_s=ttl_s,
            consume_on_read=consume_on_read,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return SidecarUploadResponse(
        sidecar_id=entry.sidecar_id,
        filename=entry.filename,
        content_type=entry.content_type,
        size_bytes=len(entry.payload),
        expires_at=datetime.fromtimestamp(entry.expires_at, tz=timezone.utc).isoformat(),
    )


@router.delete(
    "/ingest/sidecar/{sidecar_id}",
    status_code=204,
    summary="Delete a previously uploaded sidecar",
)
async def delete_sidecar(request: Request, sidecar_id: str) -> Response:
    """Delete an owner-authorized sidecar before it is bound to work."""
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    if request.app.state.config.mode in ("realtime", "batch"):
        raise HTTPException(
            status_code=404, detail="Sidecar operations are available only on the gateway or standalone service"
        )
    store = get_sidecar_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Sidecar store not initialised")
    deleted = store.delete(sidecar_id, owner_token=_sidecar_owner_fingerprint(request))
    if not deleted:
        raise HTTPException(status_code=404, detail="Sidecar was not found or is not owned by this caller")
    return Response(status_code=204)


@router.get(
    "/ingest/status/{item_id}",
    summary="Check processing status of a general ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_status(request: Request, item_id: str) -> JSONResponse:
    return await _status_response(request, item_id)


@router.get(
    "/ingest/page/status/{page_id}",
    summary="Check processing status of a page ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_page_status(request: Request, page_id: str) -> JSONResponse:
    return await _status_response(request, page_id)


@router.get(
    "/ingest/document/status/{document_id}",
    summary="Check processing status of a document ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_document_status(request: Request, document_id: str) -> JSONResponse:
    return await _status_response(request, document_id)


# ------------------------------------------------------------------
# GET /v1/ingest/pipeline-config  — introspect live pipeline setup
# ------------------------------------------------------------------


@router.get(
    "/ingest/pipeline-config",
    summary="Return the live pipeline configuration for this pod (or aggregated from backends via the gateway)",
)
async def pipeline_config(request: Request):
    """Return redacted pipeline configuration.

    * **worker / standalone** — returns the local pipeline configs directly.
    * **gateway** — fans out GET requests to one realtime and one batch
      backend pod, aggregates the responses, and returns them keyed by role.
    """
    mode = _mode(request)

    if _is_gateway(request):
        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")

        aggregated: dict[str, object] = {"source": "gateway", "mode": mode}
        for pool_type in (PoolType.REALTIME, PoolType.BATCH):
            label = pool_type.value
            try:
                resp = await proxy.forward_get(request, pool_type, "/v1/ingest/pipeline-config")
                if resp.status_code == 200:
                    aggregated[label] = json.loads(resp.body)
                else:
                    aggregated[label] = {
                        "error": f"HTTP {resp.status_code}",
                        "body": resp.body.decode(errors="replace")[:500],
                    }
            except Exception as exc:
                aggregated[label] = {"error": f"{type(exc).__name__}: {exc}"}

        return JSONResponse(content=aggregated)

    from nemo_retriever.service.services.pipeline_executor import get_pipeline_configs

    pool = get_pipeline_pool()
    pool_stats = pool.stats() if pool is not None else {}

    policy = _build_policy(request)
    return JSONResponse(
        content={
            "source": mode,
            "mode": mode,
            "pipelines": get_pipeline_configs(),
            "pool_stats": pool_stats,
            "allowed_overrides": policy.describe(),
        }
    )


# ------------------------------------------------------------------
# POST /v1/answer  -- vector search + configured LLM generation
# ------------------------------------------------------------------


def _text_from_hit(hit: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "page_content"):
        value = hit.get(key)
        if value is not None:
            return str(value)
    return ""


def _metadata_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in hit.items() if k not in {"text", "content", "chunk", "page_content", "vector"}}


@router.post(
    "/answer",
    response_model=AnswerResult | AgenticAnswerResponse,
    summary="Search ingested documents and generate an answer",
)
async def answer(req: ServiceAnswerRequest, request: Request) -> Response | AnswerResult | AgenticAnswerResponse:
    """Retrieve context from VectorDB and answer with the configured LLM."""
    import httpx

    config = request.app.state.config

    if not config.vectordb.enabled:
        raise HTTPException(
            status_code=404,
            detail="VectorDB is not enabled in the service configuration.",
        )

    if req.mode == "classic" and not config.llm.enabled:
        raise HTTPException(
            status_code=404,
            detail="LLM answer generation is not enabled in the service configuration.",
        )
    if req.mode == "agentic" and not config.agentic.enabled:
        raise HTTPException(
            status_code=404,
            detail="Agentic answer generation is not enabled in the service configuration.",
        )

    mode = _mode(request)
    if mode in ("realtime", "batch"):
        raise HTTPException(
            status_code=404,
            detail="Answer endpoint is not available on worker pods. Use the gateway.",
        )

    vectordb_url = config.vectordb.vectordb_url.rstrip("/")
    target = f"{vectordb_url}/v1/query"
    query_body: dict[str, Any] = {"query": req.query, "top_k": req.top_k}
    if req.mode == "agentic":
        query_body.update({"agentic": True, "agentic_mode": "answer"})

    try:
        from nemo_retriever.service.auth import authorized_scope, internal_auth_headers

        timeout = config.agentic.request_timeout_s if req.mode == "agentic" else 60.0
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                target,
                json=query_body,
                headers={
                    "X-NRL-Scope": authorized_scope(request),
                    **internal_auth_headers(config.vectordb.internal_api_token),
                },
            )
    except httpx.HTTPError:
        logger.exception("Failed to query vectordb at %s for answer generation", target)
        raise HTTPException(
            status_code=502,
            detail="VectorDB service is unavailable.",
        )

    if resp.status_code != 200:
        return _proxied_response(resp)
    if req.mode == "agentic":
        try:
            return AgenticAnswerResponse.model_validate(resp.json())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.error("VectorDB returned an invalid agentic answer response: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="VectorDB returned an invalid agentic answer response.",
            ) from exc

    answer_req = CoreAnswerRequest(
        query=req.query,
        top_k=req.top_k,
        reasoning_enabled=req.reasoning_enabled,
        reference=req.reference,
        judge_enabled=req.judge,
    )

    payload = resp.json()
    result_sets = payload.get("results") or []
    hits = result_sets[0].get("hits", []) if result_sets else []
    retrieval = RetrievalResult(
        chunks=[_text_from_hit(hit) for hit in hits],
        metadata=[_metadata_from_hit(hit) for hit in hits],
    )

    from nemo_retriever.models.llm.clients import LLMJudge, LiteLLMClient

    llm_cfg = config.llm
    llm = getattr(request.app.state, "answer_llm_client", None)
    if llm is None:
        llm = LiteLLMClient.from_kwargs(
            model=llm_cfg.model,
            api_base=llm_cfg.api_base,
            api_key=llm_cfg.api_key,
            temperature=llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            max_tokens=llm_cfg.max_tokens,
            extra_params=dict(llm_cfg.extra_params),
            num_retries=llm_cfg.num_retries,
            timeout=llm_cfg.timeout,
            rag_system_prompt=llm_cfg.rag_system_prompt,
            rag_system_prompt_prefix=llm_cfg.rag_system_prompt_prefix,
            reasoning_enabled=llm_cfg.reasoning_enabled,
        )
        request.app.state.answer_llm_client = llm

    generate_kwargs: dict[str, Any] = {}
    if answer_req.reasoning_enabled is not None:
        generate_kwargs["reasoning_enabled"] = answer_req.reasoning_enabled
    gen = await asyncio.to_thread(
        llm.generate,
        answer_req.query,
        retrieval.chunks,
        **generate_kwargs,
    )
    if gen.error:
        logger.error("LLM answer generation failed for model %s: %s", gen.model, gen.error)
        raise HTTPException(
            status_code=502,
            detail=f"LLM answer generation failed: {gen.error}",
        )

    judge = None
    if answer_req.judge_enabled:
        judge = getattr(request.app.state, "answer_judge_client", None)
        if judge is None:
            judge = LLMJudge.from_kwargs(
                model=llm_cfg.model,
                api_base=llm_cfg.api_base,
                api_key=llm_cfg.api_key,
                extra_params=dict(llm_cfg.extra_params),
                num_retries=llm_cfg.num_retries,
                timeout=llm_cfg.timeout,
            )
            request.app.state.answer_judge_client = judge

    result = await asyncio.to_thread(
        build_answer_result,
        query=answer_req.query,
        retrieval=retrieval,
        generation=gen,
        reference=answer_req.reference,
        judge=judge,
    )

    return result.model_copy(
        update={
            "chunks": result.chunks if req.include_chunks else None,
            "metadata": result.metadata if req.include_metadata else None,
        }
    )


# ------------------------------------------------------------------
# POST /v1/query  — vector search / agentic retrieval (proxied to vectordb)
# ------------------------------------------------------------------


@router.post(
    "/query",
    summary="Search ingested documents by semantic similarity, hybrid, agentic, or reranked retrieval",
)
async def query(request: Request) -> Response:
    """Run the public query API through VectorDB and optional reranking.

    ``vectordb_app`` remains responsible only for nearest-neighbor retrieval.
    With ``rerank=true``, the main service obtains a larger candidate set from
    VectorDB, then uses the server-configured remote endpoint or local model
    in the main service process, and returns the requested ``top_k`` hits.
    """
    import httpx

    config = request.app.state.config
    if not config.vectordb.enabled:
        raise HTTPException(
            status_code=404,
            detail="VectorDB is not enabled in the service configuration.",
        )
    if _mode(request) in ("realtime", "batch"):
        raise HTTPException(
            status_code=404,
            detail="Query endpoint is not available on worker pods. Use the gateway.",
        )

    body = await request.body()
    agentic = False
    rerank_request: QueryRequest | None = None
    query_body = body
    try:
        parsed = json.loads(body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body)
        agentic = bool(parsed.get("agentic")) if isinstance(parsed, dict) else False
        if isinstance(parsed, dict) and "rerank" in parsed:
            try:
                validated_rerank_request = QueryRequest.model_validate(parsed)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if validated_rerank_request.rerank:
                rerank_request = validated_rerank_request
                has_remote_reranker = bool((config.nim_endpoints.rerank_invoke_url or "").strip())
                has_local_reranker = config.local_models.rerank.enabled
                if not (has_remote_reranker or has_local_reranker):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Reranking is not configured. Set "
                            "nim_endpoints.rerank_invoke_url or enable "
                            "local_models.rerank.enabled on the main service."
                        ),
                    )
                candidate_k = rerank_request.rerank_top_k or max(rerank_request.top_k, 50)
                forwarded = dict(parsed)
                forwarded.pop("rerank", None)
                forwarded.pop("rerank_top_k", None)
                forwarded["top_k"] = candidate_k
                query_body = json.dumps(forwarded).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        agentic = False

    if agentic and not config.agentic.enabled:
        raise HTTPException(
            status_code=400,
            detail="Agentic retrieval is not enabled in the service configuration. "
            "Set agentic.enabled with llm_model and invoke_url.",
        )

    vectordb_url = config.vectordb.vectordb_url.rstrip("/")
    timeout = config.agentic.request_timeout_s if agentic else 60.0
    try:
        from nemo_retriever.service.auth import authorized_scope, internal_auth_headers

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{vectordb_url}/v1/query",
                content=query_body,
                headers={
                    "Content-Type": "application/json",
                    "X-NRL-Scope": authorized_scope(request),
                    **internal_auth_headers(config.vectordb.internal_api_token),
                },
            )
    except httpx.HTTPError as exc:
        logger.exception("Failed to proxy query to vectordb at %s", vectordb_url)
        raise HTTPException(status_code=502, detail="VectorDB service is unavailable.") from exc

    if rerank_request is None or resp.status_code >= 400:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    try:
        response = QueryResponse.model_validate_json(resp.content)
        queries = [rerank_request.query] if isinstance(rerank_request.query, str) else rerank_request.query
        if len(response.results) != len(queries):
            raise ValueError("VectorDB response count did not match the query count")

        from nemo_retriever.operators.rerank import rerank_hits

        local_rerank = config.local_models.rerank
        use_remote_reranker = bool((config.nim_endpoints.rerank_invoke_url or "").strip())
        rerank_lock: asyncio.Lock | None = None
        local_reranker = None
        if not use_remote_reranker:
            # The local model is service-owned and lazily loaded once. Serialize
            # scoring too: HF/vLLM model instances are not safe to invoke from
            # simultaneous request threads, and this avoids duplicate GPU work.
            rerank_lock = getattr(request.app.state, "local_reranker_lock", None)
            if rerank_lock is None:
                rerank_lock = asyncio.Lock()
                request.app.state.local_reranker_lock = rerank_lock

            async with rerank_lock:
                local_reranker = getattr(request.app.state, "local_reranker", None)
                if local_reranker is None:
                    from nemo_retriever.models import create_local_reranker

                    local_reranker = await asyncio.to_thread(
                        create_local_reranker,
                        local_rerank.model_name,
                        backend=local_rerank.backend,
                        device=config.local_models.device,
                        hf_cache_dir=config.local_models.hf_cache_dir,
                        gpu_memory_utilization=local_rerank.gpu_memory_utilization,
                    )
                    request.app.state.local_reranker = local_reranker

        async def _rerank_results() -> None:
            rerank_kwargs: dict[str, Any] = {"top_n": rerank_request.top_k}
            if use_remote_reranker:
                rerank_kwargs.update(
                    rerank_invoke_url=config.nim_endpoints.rerank_invoke_url,
                    model_name=config.nim_endpoints.rerank_model_name or "nvidia/llama-nemotron-rerank-vl-1b-v2",
                    api_key=config.nim_endpoints.api_key or "",
                )
            else:
                rerank_kwargs.update(
                    model=local_reranker,
                    model_name=local_rerank.model_name,
                    max_length=local_rerank.max_length,
                    batch_size=local_rerank.batch_size,
                )
            for query_text, result in zip(queries, response.results):
                result.hits = await asyncio.to_thread(
                    rerank_hits,
                    query_text,
                    result.hits,
                    **rerank_kwargs,
                )

        if rerank_lock is not None:
            async with rerank_lock:
                await _rerank_results()
        else:
            await _rerank_results()
    except Exception as exc:
        logger.exception("Failed to rerank VectorDB query results")
        raise HTTPException(status_code=502, detail="Reranker service is unavailable.") from exc

    return Response(
        content=response.model_dump_json(),
        status_code=resp.status_code,
        media_type="application/json",
    )


# ------------------------------------------------------------------
# GET /v1/internal/document-result/{id}  — gateway ← worker row cache
# POST /v1/internal/job-callback  — worker → gateway completion hook
# ------------------------------------------------------------------


@router.get(
    "/internal/document-result/{document_id}",
    summary="Fetch cached pipeline rows from a worker pod (split topology)",
    include_in_schema=False,
)
async def worker_document_result(document_id: str) -> JSONResponse:
    """Return rows stored by the worker pool after pipeline completion."""
    try:
        rows = get_result_data(document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(_RESULT_RETRY_AFTER_SECONDS)},
        ) from exc
    if rows is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached result rows for document {document_id!r}",
        )
    return JSONResponse({"id": document_id, "result_data": rows})


@router.post(
    "/internal/job-callback",
    summary="Internal callback from worker pods to report job completion",
    include_in_schema=False,
)
async def job_callback(request: Request) -> JSONResponse:
    """Accept a completion notification from a worker pod.

    The gateway's ``JobTracker`` is updated and an SSE event is published
    so connected clients are notified instantly.

    The log line emitted here is the primary diagnostic signal for
    "client hang" reports: it carries the ``job_id`` looked up from the
    tracker, the actual transition outcome (``transitioned`` /
    ``idempotent`` / ``unknown_document``), and the per-job subscriber
    count so operators can correlate worker-pod completion with
    client-side SSE delivery without grepping multiple files.
    """
    body = await request.json()
    item_id = body.get("id")
    if not item_id:
        raise HTTPException(status_code=400, detail="Missing 'id' field")

    broker = None
    lease_record = None
    if _is_gateway(request):
        from nemo_retriever.service.services.work_queue import (
            StaleLease,
            get_work_broker,
        )

        broker = get_work_broker()
        lease_id = body.get("lease_id")
        lease_generation = body.get("lease_generation")
        if lease_id is not None or lease_generation is not None:
            try:
                lease_record = broker.validate_callback(item_id, lease_id, int(lease_generation)) if broker else None
            except (StaleLease, TypeError, ValueError) as exc:
                raise HTTPException(status_code=409, detail="Work lease has been superseded") from exc
        elif broker is not None and broker.has_record(item_id):
            raise HTTPException(status_code=409, detail="Missing work lease identity")

    if body.get("result_data") is not None:
        logger.warning(
            "Ignoring inline result_data on internal callback for %s " "(%d row(s)); workers must store rows locally.",
            item_id,
            len(body.get("result_data") or []),
        )

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")

    # Capture the doc's job_id BEFORE the state transition so we still
    # log a useful job_id even if the transition turns out to be a
    # no-op (idempotent or unknown_document).
    pre_rec = tracker.get_document(item_id)
    job_id_for_log = pre_rec.job_id if pre_rec is not None else None

    status = body.get("status", "completed")
    if status == "failed":
        outcome = tracker.mark_failed(
            item_id,
            body.get("error", "unknown error"),
            elapsed_s=body.get("elapsed_s"),
        )
    else:
        result_rows = body.get("result_rows", 0)
        if pre_rec is None and result_rows and body.get("result_worker_ip"):
            logger.warning(
                "Permanently rejecting retained result handoff for unknown document %s",
                item_id,
            )
            raise HTTPException(
                status_code=410,
                detail=f"Gateway has no tracked document {item_id!r} for retained result handoff",
            )
        if pre_rec is not None and result_rows and tracker.should_retain_results(pre_rec.job_id):
            try:
                retained_rows = await asyncio.to_thread(get_result_data, item_id)
            except ResultStoreTemporarilyUnavailable as exc:
                raise HTTPException(
                    status_code=503,
                    detail=str(exc),
                    headers={"Retry-After": "1"},
                ) from exc
            if retained_rows is None:
                owner_ip = (
                    lease_record.lease.worker_ip
                    if lease_record is not None and lease_record.lease
                    else body.get("result_worker_ip")
                )
                await _pull_and_store_worker_result(
                    request,
                    item_id,
                    owner_ip,
                    body.get("result_worker_ip") if lease_record is not None else None,
                )
        outcome = tracker.mark_completed(
            item_id,
            result_rows=result_rows,
            elapsed_s=body.get("elapsed_s"),
        )

    bus = get_event_bus()
    if bus is not None and job_id_for_log is not None:
        sub_count = bus.subscribers_for(job_id_for_log)
    elif bus is not None:
        sub_count = bus.subscriber_count
    else:
        sub_count = 0

    log_fn = logger.warning if outcome == MarkOutcome.UNKNOWN_DOCUMENT else logger.info
    log_fn(
        "Gateway callback: id=%s job_id=%s status=%s outcome=%s rows=%s subscribers=%d",
        item_id,
        job_id_for_log or "?",
        status,
        outcome.value,
        body.get("result_rows", 0),
        sub_count,
    )
    if broker is not None and lease_record is not None:
        try:
            await broker.acknowledge(item_id, body["lease_id"], int(body["lease_generation"]))
        except StaleLease:
            logger.warning(
                "Work lease for %s was already superseded at acknowledge; tracker already updated",
                item_id,
            )
    return JSONResponse(content={"ok": True})


# ------------------------------------------------------------------
# Legacy / removed route stubs
#
# The Retriever Service v2 refactor (multi-pod architecture) removed
# two legacy routes that older SDK builds may still call:
#
#   * ``POST /v1/ingest``        — the old "single-shot" upload route,
#     replaced by the job-scoped pair
#     ``POST /v1/ingest/job`` + ``POST /v1/ingest/job/{job_id}/document``.
#   * ``GET  /v1/ingest/events`` — the old firehose SSE stream, replaced
#     by per-job ``GET /v1/ingest/job/{job_id}/events``.
#
# When a customer ships a *new* service image with an *older* Retriever
# SDK wheel, the SDK calls these legacy paths and the server otherwise
# falls through to FastAPI's default 404 with an empty body. The client
# sees an opaque "no documents completed" outcome.
#
# We register the legacy paths explicitly so the server can return an
# actionable ``410 Gone`` body that names the replacement route and
# tells the operator to align SDK and service versions. The stubs are
# hidden from the OpenAPI schema (``include_in_schema=False``) so they
# do not advertise themselves as supported endpoints.
# ------------------------------------------------------------------


_LEGACY_REMOVED_VERSION = "26.08.1"

_LEGACY_INGEST_DETAIL = (
    "POST /v1/ingest was removed in retriever-service "
    f"{_LEGACY_REMOVED_VERSION} (multi-pod refactor). Open a job with "
    "POST /v1/ingest/job and then upload each document via "
    "POST /v1/ingest/job/{job_id}/document. This 410 typically means "
    "the Python SDK is older than the deployed nrl-service image — "
    "upgrade the SDK (or downgrade the chart/image) so the two match."
)

_LEGACY_FIREHOSE_DETAIL = (
    "GET /v1/ingest/events (firehose SSE) was removed in "
    f"retriever-service {_LEGACY_REMOVED_VERSION}. Subscribe to "
    "GET /v1/ingest/job/{job_id}/events with the job_id returned by "
    "POST /v1/ingest/job. This 410 typically means the Python SDK is "
    "older than the deployed nrl-service image — upgrade the SDK (or "
    "downgrade the chart/image) so the two match."
)


@router.post(
    "/ingest",
    include_in_schema=False,
)
async def _legacy_ingest_upload_removed() -> None:
    """Return ``410 Gone`` with a migration hint for the removed route.

    Older SDK builds (pre-v2 client) upload through ``POST /v1/ingest``.
    Without this stub FastAPI returns a body-less 404 and the SDK
    surfaces "no documents completed" with no indication of why — the
    customer-visible regression captured in the 26.08.1 release notes.
    """
    raise HTTPException(status_code=410, detail=_LEGACY_INGEST_DETAIL)


@router.get(
    "/ingest/events",
    include_in_schema=False,
)
async def _legacy_ingest_firehose_removed() -> None:
    """Return ``410 Gone`` for the removed firehose SSE endpoint.

    The per-job SSE route (``/v1/ingest/job/{job_id}/events``) replaced
    this in J4. We surface the migration message instead of the default
    404 so old clients fail with a clear, actionable error.
    """
    raise HTTPException(status_code=410, detail=_LEGACY_FIREHOSE_DETAIL)


# ------------------------------------------------------------------
# GET /v1/ingest/job/{job_id}/events
#   SSE stream filtered to a single job aggregate.
#
# The legacy firehose ``GET /v1/ingest/events`` was removed in J4 to
# preserve job isolation between tenants — every public SSE consumer
# must declare which job it is observing. Dashboard internals (which
# are served from a separate router) still use a firehose subscription
# for the operator overview view, but that endpoint is privileged and
# lives under ``/dashboard``. See the legacy stub above for the 410
# Gone behavior that surfaces a clear error to old SDK builds that
# still call this firehose path.
# ------------------------------------------------------------------


@router.get(
    "/ingest/job/{job_id}/events",
    summary="SSE stream of completion events for one job",
)
async def ingest_job_events(request: Request, job_id: str) -> StreamingResponse:
    """Stream Server-Sent Events for a single :class:`JobAggregate`.

    Emits:

    * a catch-up batch of already-terminal ``completed`` / ``failed``
      document events (so clients that connect after some documents
      already finished still see them),
    * live per-document ``processing`` / ``completed`` / ``failed``
      events,
    * lifecycle events (``job_progress``, ``job_finalized``,
      ``job_failed``) — these arrive once J5 wires them through the
      tracker; the endpoint already routes them.

    A keepalive comment is emitted every 30 seconds.
    """
    bus = get_event_bus()
    if bus is None:
        raise HTTPException(status_code=503, detail="Event bus not available")

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    if tracker.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    async def event_generator():
        sub_id, queue = bus.subscribe(job_id=job_id)
        snapshot_count = 0
        live_count = 0
        logger.info("SSE subscriber %d connected (job_id=%s)", sub_id, job_id)
        try:
            snapshot = _snapshot_terminal_jobs(tracker, job_id=job_id)
            snapshot_count = len(snapshot)
            for rec in snapshot:
                yield f"event: {rec['status']}\ndata: {json.dumps(rec)}\n\n"
            logger.info(
                "SSE subscriber %d (job_id=%s): sent %d snapshot events",
                sub_id,
                job_id,
                snapshot_count,
            )

            while True:
                if await request.is_disconnected():
                    logger.info(
                        "SSE subscriber %d (job_id=%s): client disconnected",
                        sub_id,
                        job_id,
                    )
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_TIMEOUT_S)
                    live_count += 1
                    yield f"event: {event.get('type', 'status')}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(sub_id)
            logger.info(
                "SSE subscriber %d (job_id=%s) closed (snapshot=%d live=%d)",
                sub_id,
                job_id,
                snapshot_count,
                live_count,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _snapshot_terminal_jobs(tracker: Any, *, job_id: str | None = None) -> list[dict[str, Any]]:
    """Return already-terminal documents so SSE clients get caught up.

    If *job_id* is supplied, only documents belonging to that job are
    returned; otherwise every terminal document across every job is
    yielded. The latter mode is only used by privileged dashboard
    endpoints — public per-job SSE always passes a *job_id*.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    terminal = {DocumentStatus.COMPLETED, DocumentStatus.FAILED}
    if job_id is not None:
        recs = tracker.job_documents(job_id)
    else:
        recs = tracker.all_documents()
    return [
        {
            "type": rec.status.value,
            "id": rec.id,
            "job_id": rec.job_id,
            "status": rec.status.value,
            "result_rows": rec.result_rows,
            "elapsed_s": rec.elapsed_s,
            "error": rec.error,
        }
        for rec in recs
        if rec.status in terminal
    ]


# ------------------------------------------------------------------
# POST /v1/ingest/status/batch  — bulk status query
# ------------------------------------------------------------------


@router.post(
    "/ingest/status/batch",
    summary="Bulk status check for multiple items",
)
async def ingest_status_batch(request: Request) -> JSONResponse:
    """Return the status of many items in a single request.

    Accepts ``{"ids": ["id1", "id2", ...]}`` and returns a dict keyed by
    item id.  Works on the gateway's local tracker — no backend proxying.
    """
    _MAX_BATCH_IDS = 1000

    body = await request.json()
    ids = body.get("ids", [])
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="'ids' must be a list")
    if len(ids) > _MAX_BATCH_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many IDs ({len(ids)}); maximum is {_MAX_BATCH_IDS}",
        )

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")

    from nemo_retriever.service.services.job_tracker import DocumentStatus

    results: dict[str, dict[str, Any]] = {}
    for item_id in ids:
        rec = tracker.get_document(item_id)
        if rec is None:
            results[item_id] = {"status": "unknown"}
        else:
            results[item_id] = {
                "status": rec.status.value,
                "job_id": rec.job_id,
                "result_rows": rec.result_rows,
                "elapsed_s": rec.elapsed_s,
                "error": rec.error,
            }

    terminal_count = sum(
        1 for r in results.values() if r["status"] in (DocumentStatus.COMPLETED.value, DocumentStatus.FAILED.value)
    )
    return JSONResponse(
        content={
            "total": len(ids),
            "terminal": terminal_count,
            "pending": len(ids) - terminal_count,
            "items": results,
        }
    )
