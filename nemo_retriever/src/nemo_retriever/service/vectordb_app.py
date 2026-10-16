# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VectorDB service built on the backend-neutral VDB contract.

The HTTP layer owns transport, internal authentication, query embedding, and
error mapping. Concrete VDB implementations own persistence, collection
lifecycle behavior, and native retrieval semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Union, cast

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentId,
    DocumentInfo,
    DocumentPage,
    IngestOperation,
)
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    UnsupportedVDBOperation,
    VDB,
    VDBInvalidRequest,
    VDBResourceConflict,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.factory import get_vdb_op_cls
from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY
from nemo_retriever.common.vdb.records import RetrievalContractError
from nemo_retriever.ingest.index_mode import (
    inspect_existing_lancedb_mode,
    resolve_ingest_index_mode,
    validate_requested_index_mode,
)
from nemo_retriever.models import resolve_embed_model
from nemo_retriever.operators.vdb import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.query.evidence import build_evidence_result
from nemo_retriever.service.agentic_query import run_agentic_answer, run_agentic_query
from nemo_retriever.service.auth import tokens_match
from nemo_retriever.service.config import AgenticConfig
from nemo_retriever.service.query_schema import (
    AgenticAnswerResponse,
    AgenticQueryResponse,
    EvidenceQueryResponse,
    EvidenceResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
)

logger = logging.getLogger(__name__)

ServiceIndexMode = Literal["auto", "dense", "hybrid"]


def _validate_service_index_mode(index_mode: str) -> ServiceIndexMode:
    normalized = validate_requested_index_mode(index_mode)
    if normalized == "sparse":
        raise ValueError(f"index_mode must be one of auto, dense, hybrid; got {index_mode!r}.")
    return cast(ServiceIndexMode, normalized)


MAX_CONCURRENT_QUERIES = 4
MAX_CONCURRENT_AGENTIC_QUERIES = 100


class WriteRequest(BaseModel):
    """Canonical internal payload emitted by an ingest worker."""

    records: list[list[dict[str, Any]]]
    scope: str | None = None
    collection_name: str | None = None
    document_id: DocumentId | None = None
    job_id: str | None = None
    filename: str | None = None
    content_sha256: str | None = None
    document_version: str | None = None
    operation: IngestOperation = IngestOperation.APPEND


class WriteResponse(BaseModel):
    """Counts returned after an internal fixed-table or collection write."""

    written: int
    total_rows: int


def _scope(value: str | None) -> str:
    return (value or "default").strip() or "default"


def _tensor_to_embedding_rows(tensor: Any) -> list[list[float]]:
    """Convert a local embedder tensor output to JSON-serializable vectors."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        rows = tensor.tolist()
        if rows and isinstance(rows[0], (int, float)):
            return [rows]
        return rows
    return list(tensor)


def _embed_queries_remote(
    texts: list[str],
    *,
    embed_model: str,
    embed_endpoint: str,
    embed_api_key: str,
    embed_model_provider_prefix: str | None = None,
) -> list[list[float]]:
    from nemo_retriever.models.nim.util import infer_microservice

    return infer_microservice(
        texts,
        model_name=embed_model,
        model_provider_prefix=embed_model_provider_prefix,
        embedding_endpoint=embed_endpoint,
        nvidia_api_key=embed_api_key or None,
        input_type="query",
        grpc=False,
    )


class VectorDBState:
    """Small service-owned wrapper for embedding and VDB operator dispatch."""

    def __init__(
        self,
        *,
        vdb: VDB,
        embed_endpoint: str,
        embed_model: str,
        embed_api_key: str,
        embed_model_provider_prefix: str | None = None,
        local_embed: bool = False,
        local_embed_backend: str = "vllm",
        hf_cache_dir: str | None = None,
        device: str | None = None,
        gpu_memory_utilization: float = 0.45,
        max_concurrent_queries: int = MAX_CONCURRENT_QUERIES,
    ) -> None:
        if max_concurrent_queries <= 0:
            raise ValueError("max_concurrent_queries must be positive")
        self.vdb = vdb
        self.ingest_operator = IngestVdbOperator(vdb=vdb)
        self.retrieve_operator = RetrieveVdbOperator(vdb=vdb)
        self.embed_endpoint = embed_endpoint
        self.embed_model = embed_model
        self.embed_model_provider_prefix = embed_model_provider_prefix
        self.embed_api_key = embed_api_key
        self.local_embed = local_embed
        self.local_embed_backend = local_embed_backend
        self.hf_cache_dir = hf_cache_dir
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.query_semaphore = asyncio.Semaphore(max_concurrent_queries)
        self._embed_lock = threading.Lock()
        self._local_embedder: Any | None = None

    @property
    def embed_mode(self) -> str:
        if self.embed_endpoint:
            return "remote"
        if self.local_embed:
            return "local"
        return "none"

    @property
    def table_exists(self) -> bool:
        """Return whether the configured legacy table is available for queries."""
        return self.vdb.health().get("table_exists") is True

    def _get_local_embedder(self) -> Any:
        if self._local_embedder is None:
            from nemo_retriever.models import create_local_embedder

            self._local_embedder = create_local_embedder(
                self.embed_model,
                backend=self.local_embed_backend,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            logger.info(
                "Loaded local query embedder model=%s backend=%s",
                self.embed_model,
                self.local_embed_backend,
            )
        return self._local_embedder

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts via a remote endpoint or local model."""
        if self.embed_endpoint:
            return _embed_queries_remote(
                texts,
                embed_model=self.embed_model,
                embed_model_provider_prefix=self.embed_model_provider_prefix,
                embed_endpoint=self.embed_endpoint,
                embed_api_key=self.embed_api_key,
            )
        if self.local_embed:
            with self._embed_lock:
                tensor = self._get_local_embedder().embed_queries(texts)
            return _tensor_to_embedding_rows(tensor)
        raise RuntimeError("No embedding backend configured")


def _production_vdb(
    *,
    lancedb_uri: str,
    table_name: str,
    expiration_cleanup_enabled: bool,
    embed_model: str,
    index_mode: ServiceIndexMode = "auto",
) -> VDB:
    """Construct the sole production VDB implementation for this service."""
    existing_mode = inspect_existing_lancedb_mode(lancedb_uri, table_name)
    resolved_embed_model = resolve_embed_model(embed_model)
    effective_mode = resolve_ingest_index_mode(
        index_mode,
        overwrite=False,
        existing_mode=existing_mode,
    )
    if effective_mode == "sparse":
        raise ValueError("The VectorDB service requires a dense vector column; sparse-only tables are unsupported.")
    vdb_cls = get_vdb_op_cls("lancedb")
    backend = vdb_cls(
        uri=lancedb_uri,
        table_name=table_name,
        vector_dim=None,
        overwrite=False,
        build_index=False,
        hybrid=effective_mode == "hybrid",
        _service_table_schema=True,
        _service_index_mode=index_mode,
        embedding_model_name=resolved_embed_model,
        expiration_cleanup_enabled=expiration_cleanup_enabled,
    )
    if existing_mode is None:
        return backend

    stored_embed_model = backend.get_index_metadata("embedding_model_name")
    if not stored_embed_model:
        raise ValueError(
            f"Existing LanceDB table {table_name!r} at {lancedb_uri!r} does not record its embedding "
            "model, so query compatibility cannot be verified. Rebuild the table with the configured "
            "embedding model before starting the VectorDB service."
        )
    resolved_stored_model = resolve_embed_model(stored_embed_model)
    if resolved_stored_model != resolved_embed_model:
        raise ValueError(
            f"Existing LanceDB table {table_name!r} at {lancedb_uri!r} uses embedding model "
            f"{resolved_stored_model!r}, but the VectorDB service is configured for "
            f"{resolved_embed_model!r}. Use the index model or rebuild the table with the configured model."
        )
    return backend


def _safe_backend_health(state: VectorDBState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    try:
        health = state.vdb.health()
    except Exception:
        logger.exception("VectorDB backend health inspection failed")
        return None
    return dict(health) if isinstance(health, dict) else {}


def _legacy_strategies(health: dict[str, Any]) -> list[str]:
    strategies = health.get("retrieval_strategies")
    if isinstance(strategies, list) and all(isinstance(item, str) for item in strategies):
        return list(strategies)
    mode = health.get("effective_retrieval_mode")
    return ["hybrid" if mode == "hybrid" else "dense"]


def create_vectordb_app(
    lancedb_uri: str = "/data/vectordb",
    table_name: str = "nemo_retriever",
    embed_endpoint: str = "",
    embed_model: str = "nvidia/nemotron-3-embed-1b",
    embed_model_provider_prefix: str | None = None,
    embed_api_key: str = "",
    *,
    local_embed: bool = False,
    local_embed_backend: str = "vllm",
    hf_cache_dir: str | None = None,
    device: str | None = None,
    gpu_memory_utilization: float = 0.45,
    index_mode: ServiceIndexMode = "auto",
    internal_api_token: str | None = None,
    max_concurrent_queries: int = MAX_CONCURRENT_QUERIES,
    reconciliation_interval_seconds: int = 60,
    expiration_cleanup_enabled: bool = True,
    vdb: VDB | None = None,
    agentic_config: AgenticConfig | None = None,
) -> FastAPI:
    """Build the VectorDB FastAPI application around an injected VDB contract."""
    if reconciliation_interval_seconds < 0:
        raise ValueError("reconciliation_interval_seconds must be non-negative")
    index_mode = _validate_service_index_mode(index_mode)

    if max_concurrent_queries <= 0:
        raise ValueError("max_concurrent_queries must be positive")
    agentic_config = agentic_config or AgenticConfig()
    state: VectorDBState | None = None
    agentic_executor: ThreadPoolExecutor | None = None
    agentic_slots: threading.BoundedSemaphore | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal state, agentic_executor, agentic_slots
        backend = vdb or _production_vdb(
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            expiration_cleanup_enabled=expiration_cleanup_enabled,
            embed_model=embed_model,
            index_mode=index_mode,
        )
        state = VectorDBState(
            vdb=backend,
            embed_endpoint=embed_endpoint,
            embed_model=embed_model,
            embed_model_provider_prefix=embed_model_provider_prefix,
            embed_api_key=embed_api_key,
            local_embed=local_embed,
            local_embed_backend=local_embed_backend,
            hf_cache_dir=hf_cache_dir,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_concurrent_queries=max_concurrent_queries,
        )
        app.state.vectordb_state = state
        if agentic_config.enabled:
            agentic_executor = ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_AGENTIC_QUERIES,
                thread_name_prefix="agentic-query",
            )
            agentic_slots = threading.BoundedSemaphore(MAX_CONCURRENT_AGENTIC_QUERIES)
        app.state.agentic_slots = agentic_slots
        logger.info(
            "VectorDB service started: embed_mode=%s max_concurrent_queries=%d",
            state.embed_mode,
            max_concurrent_queries,
        )
        if state.embed_mode == "none":
            logger.error(
                "VectorDB started without an embedding backend; /v1/query will "
                "return HTTP 501 until --embed-endpoint or --local-embed is configured."
            )

        async def reconciliation_loop() -> None:
            while True:
                try:
                    await asyncio.to_thread(backend.reconcile_collections)
                except Exception:
                    logger.exception("VectorDB reconciliation iteration failed")
                await asyncio.sleep(reconciliation_interval_seconds)

        reconciliation_task = (
            asyncio.create_task(reconciliation_loop()) if reconciliation_interval_seconds > 0 else None
        )
        try:
            yield
        finally:
            if reconciliation_task is not None:
                reconciliation_task.cancel()
                try:
                    await reconciliation_task
                except asyncio.CancelledError:
                    pass
            if agentic_executor is not None:
                # Agentic LLM calls cannot be interrupted, so detach rather than
                # blocking service shutdown on in-flight work.
                agentic_executor.shutdown(wait=False, cancel_futures=True)
                agentic_executor = None
            agentic_slots = None
            state = None
            app.state.agentic_slots = None
            app.state.vectordb_state = None
            logger.info("VectorDB service stopped")

    app = FastAPI(
        title="NeMo Retriever VectorDB",
        description="Vector storage and retrieval through the VDB contract",
        version="1.0.0",
        lifespan=lifespan,
    )

    def require_state() -> VectorDBState:
        if state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return state

    @app.exception_handler(UnsupportedVDBOperation)
    async def unsupported_operation(_request: Request, exc: UnsupportedVDBOperation) -> JSONResponse:
        logger.info("Unsupported VDB operation: %s", exc)
        return JSONResponse(
            status_code=501,
            content={"detail": "The configured VectorDB backend does not support this operation."},
        )

    @app.exception_handler(VDBResourceNotFound)
    async def resource_not_found(_request: Request, exc: VDBResourceNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(VDBResourceConflict)
    async def resource_conflict(_request: Request, exc: VDBResourceConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(VDBInvalidRequest)
    async def invalid_request(_request: Request, exc: VDBInvalidRequest) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RetrievalContractError)
    async def retrieval_contract_failure(_request: Request, exc: RetrievalContractError) -> JSONResponse:
        logger.exception("VectorDB retrieval contract violation", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "VectorDB retrieval contract violation."},
        )

    @app.exception_handler(Exception)
    async def unexpected_backend_failure(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected VectorDB service failure", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "VectorDB backend operation failed."},
        )

    required_internal_token = (internal_api_token or "").strip()

    @app.middleware("http")
    async def require_internal_credential(request: Request, call_next):
        if request.url.path == "/v1/health" or not required_internal_token:
            return await call_next(request)
        supplied = request.headers.get("X-NRL-Internal-Token", "")
        if not supplied or not tokens_match(supplied, required_internal_token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid internal credential."},
            )
        return await call_next(request)

    @app.get("/v1/health", tags=["system"])
    async def health() -> dict[str, Any]:
        current = state
        backend_health = _safe_backend_health(current)
        if backend_health is None:
            raise HTTPException(503, "VectorDB backend is unavailable")
        return {
            "status": "ok",
            "total_rows": backend_health.pop("total_rows", 0),
            "table_exists": backend_health.pop("table_exists", False),
            "embed_mode": current.embed_mode if current else "none",
            "effective_retrieval_mode": backend_health.pop("effective_retrieval_mode", None),
            **backend_health,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        from prometheus_client import CollectorRegistry, Gauge, generate_latest

        registry = CollectorRegistry()
        backend_health = _safe_backend_health(state) or {}
        collections = backend_health.get("collections") or {}
        cleanup = backend_health.get("cleanup") or {}
        reconciliation = backend_health.get("reconciliation") or {}
        collection_gauge = Gauge(
            "nrl_vectordb_collections",
            "Collection count by lifecycle status",
            ["status"],
            registry=registry,
        )
        for status in ("active", "deleting", "expired"):
            collection_gauge.labels(status=status).set(collections.get(status, 0))
        Gauge(
            "nrl_vectordb_cleanup_pending",
            "Pending lifecycle cleanup",
            registry=registry,
        ).set(cleanup.get("pending", 0))
        Gauge(
            "nrl_vectordb_cleanup_oldest_age_seconds",
            "Oldest pending cleanup age",
            registry=registry,
        ).set(cleanup.get("oldest_age_seconds", 0))
        Gauge(
            "nrl_vectordb_reconciliation_successes_total",
            "Successful reconciliations",
            registry=registry,
        ).set(reconciliation.get("successes", 0))
        Gauge(
            "nrl_vectordb_reconciliation_failures_total",
            "Failed reconciliations",
            registry=registry,
        ).set(reconciliation.get("failures", 0))
        Gauge(
            "nrl_vectordb_open_table_cache",
            "Open collection-table cache size",
            registry=registry,
        ).set(backend_health.get("open_table_cache_count", 0))
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        current = require_state()
        context: CollectionWriteContext | None = None
        if req.collection_name is not None:
            missing = [
                name
                for name, value in (
                    ("document_id", req.document_id),
                    ("job_id", req.job_id),
                    ("filename", req.filename),
                    ("content_sha256", req.content_sha256),
                    ("document_version", req.document_version),
                )
                if not value
            ]
            if missing:
                raise VDBInvalidRequest("Collection writes require: " + ", ".join(missing))
            if not req.records or not any(req.records):
                raise VDBInvalidRequest("Collection writes require at least one record")
            context = CollectionWriteContext(
                scope=_scope(req.scope),
                collection_name=req.collection_name,
                document_id=str(req.document_id),
                document_version=str(req.document_version),
                content_sha256=str(req.content_sha256),
                filename=str(req.filename),
                job_id=req.job_id,
                operation=req.operation,
            )

        result = await asyncio.to_thread(
            current.ingest_operator.run,
            req.records,
            collection_context=context,
        )
        if isinstance(result, CollectionWriteResult):
            return WriteResponse(written=result.written, total_rows=result.total_rows)
        backend_health = current.vdb.health()
        return WriteResponse(
            written=sum(len(batch) for batch in req.records),
            total_rows=int(backend_health.get("total_rows", 0)),
        )

    @app.post(
        "/v1/collections",
        response_model=CollectionInfo,
        status_code=201,
        tags=["collections"],
    )
    async def create_collection(
        req: CollectionCreateRequest,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.create_collection,
            scope=_scope(x_nrl_scope),
            request=req,
        )

    @app.get("/v1/collections", response_model=CollectionPage, tags=["collections"])
    async def list_collections(
        limit: int = Query(100, ge=1, le=200),
        continuation_token: str | None = None,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionPage:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.list_collections,
            scope=_scope(x_nrl_scope),
            limit=limit,
            continuation_token=continuation_token,
        )

    @app.get("/v1/collections/{name}", response_model=CollectionInfo, tags=["collections"])
    async def get_collection(
        name: str,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.get_collection,
            scope=_scope(x_nrl_scope),
            collection_name=name,
        )

    @app.patch("/v1/collections/{name}", response_model=CollectionInfo, tags=["collections"])
    async def update_collection(
        name: str,
        req: CollectionUpdateRequest,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.update_collection,
            scope=_scope(x_nrl_scope),
            collection_name=name,
            request=req,
        )

    @app.delete(
        "/v1/collections/{name}",
        response_model=CollectionDeleteResult,
        tags=["collections"],
    )
    async def delete_collection(
        response: Response,
        name: str,
        if_exists: bool = False,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionDeleteResult:
        backend = require_state().vdb
        result = await asyncio.to_thread(
            backend.delete_collection,
            scope=_scope(x_nrl_scope),
            collection_name=name,
            if_exists=if_exists,
        )
        response.status_code = 202 if result.cleanup_pending else 200
        return result

    @app.get(
        "/v1/collections/{name}/documents",
        response_model=DocumentPage,
        tags=["collections"],
    )
    async def list_documents(
        name: str,
        limit: int = Query(100, ge=1, le=200),
        continuation_token: str | None = None,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentPage:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.list_documents,
            scope=_scope(x_nrl_scope),
            collection_name=name,
            limit=limit,
            continuation_token=continuation_token,
        )

    @app.get(
        "/v1/collections/{name}/documents/{document_id}",
        response_model=DocumentInfo,
        tags=["collections"],
    )
    async def get_document(
        name: str,
        document_id: DocumentId,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentInfo:
        backend = require_state().vdb
        return await asyncio.to_thread(
            backend.get_document,
            scope=_scope(x_nrl_scope),
            collection_name=name,
            document_id=document_id,
        )

    @app.delete(
        "/v1/collections/{name}/documents/{document_id}",
        response_model=DocumentDeleteResult,
        tags=["collections"],
    )
    async def delete_document(
        response: Response,
        name: str,
        document_id: DocumentId,
        if_exists: bool = False,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentDeleteResult:
        backend = require_state().vdb
        result = await asyncio.to_thread(
            backend.delete_document,
            scope=_scope(x_nrl_scope),
            collection_name=name,
            document_id=document_id,
            if_exists=if_exists,
        )
        response.status_code = 202 if result.cleanup_pending else 200
        return result

    @app.post(
        "/v1/query",
        response_model=Union[AgenticAnswerResponse, AgenticQueryResponse, QueryResponse, EvidenceQueryResponse],
        tags=["query"],
    )
    async def query(
        req: QueryRequest,
        x_nrl_scope: str | None = Header(None),
    ) -> AgenticAnswerResponse | AgenticQueryResponse | QueryResponse | EvidenceQueryResponse:
        current = require_state()
        if req.agentic:
            if req.collection_name is not None:
                raise UnsupportedVDBOperation("agentic collection retrieval")
            return await _run_agentic_query(req)

        if current.embed_mode == "none":
            raise HTTPException(
                501,
                "No embedding backend configured. Set --embed-endpoint for a remote "
                "NIM or --local-embed for in-pod Hugging Face query embedding.",
            )

        backend_health: dict[str, Any] = {}
        if req.collection_name is None:
            backend_health = current.vdb.health()
            if backend_health.get("table_exists") is False:
                raise VDBInvalidRequest("No data has been ingested yet. Ingest documents first, then query.")

        queries = req.query if isinstance(req.query, list) else [req.query]
        if not queries:
            if req.format == "evidence":
                return EvidenceQueryResponse(results=[])
            return QueryResponse(results=[])

        async with current.query_semaphore:
            vectors = await asyncio.to_thread(current.embed_queries, queries)
            if req.collection_name is not None:
                result = await asyncio.to_thread(
                    current.retrieve_operator.run,
                    vectors,
                    scope=_scope(x_nrl_scope),
                    collection_name=req.collection_name,
                    query_texts=queries,
                    top_k=req.top_k,
                )
                if not isinstance(result, tuple):
                    raise RetrievalContractError("Collection retrieval did not return strategies")
                hits_per_query, strategies = result
            else:
                hybrid = backend_health.get("effective_retrieval_mode") == "hybrid"
                retrieval_kwargs: dict[str, Any] = {
                    "query_texts": queries,
                    "top_k": req.top_k,
                    "hybrid": hybrid,
                }
                if hybrid:
                    retrieval_kwargs["hybrid_fusion"] = DEFAULT_HYBRID_FUSION_POLICY
                hits_per_query = await asyncio.to_thread(
                    current.retrieve_operator.run,
                    vectors,
                    **retrieval_kwargs,
                )
                if not isinstance(hits_per_query, list):
                    raise RetrievalContractError("Legacy retrieval returned an invalid shape")
                strategies = _legacy_strategies(backend_health)

        if req.format == "evidence":
            return EvidenceQueryResponse(
                results=[EvidenceResult(**build_evidence_result(hits, strategies)) for hits in hits_per_query]
            )
        return QueryResponse(results=[QueryResult(hits=hits) for hits in hits_per_query])

    async def _run_agentic_query(req: QueryRequest) -> AgenticAnswerResponse | AgenticQueryResponse:
        """Run the blocking agentic workflow without consuming plain-query workers."""
        current = require_state()
        if not agentic_config.enabled:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Agentic retrieval is not enabled in the VectorDB configuration. "
                    "Start with --agentic and a remote LLM invoke URL/model, or set "
                    "agentic.enabled in the service config."
                ),
            )
        if current.embed_mode == "none":
            raise HTTPException(
                501,
                "No embedding backend configured. Set --embed-endpoint for a remote "
                "NIM or --local-embed for in-pod Hugging Face query embedding.",
            )
        if current.embed_mode != "remote":
            raise HTTPException(501, "Agentic service queries require a remote embedding endpoint.")
        if not current.table_exists:
            raise VDBInvalidRequest("No data has been ingested yet. Ingest documents first, then query.")
        if req.top_k > agentic_config.backend_top_k:
            raise VDBInvalidRequest(
                f"top_k ({req.top_k}) cannot exceed the configured agentic "
                f"backend_top_k ({agentic_config.backend_top_k})."
            )

        executor, slots = agentic_executor, agentic_slots
        if executor is None or slots is None:
            raise HTTPException(503, "Agentic retrieval workers are not running")
        if not slots.acquire(blocking=False):
            logger.warning(
                "Rejecting agentic query: all %d agentic workers are busy",
                MAX_CONCURRENT_AGENTIC_QUERIES,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"All {MAX_CONCURRENT_AGENTIC_QUERIES} agentic retrieval workers are busy. "
                    "Retry once an in-flight query finishes."
                ),
                headers={"Retry-After": "30"},
            )

        assert isinstance(req.query, str)
        try:
            future = executor.submit(
                run_agentic_answer if req.agentic_mode == "answer" else run_agentic_query,
                query=req.query,
                top_k=req.top_k,
                config=agentic_config,
                lancedb_uri=lancedb_uri,
                table_name=table_name,
                embed_endpoint=current.embed_endpoint,
                embed_model=current.embed_model,
                embed_model_provider_prefix=current.embed_model_provider_prefix,
                embed_api_key=current.embed_api_key,
            )
        except RuntimeError as exc:
            slots.release()
            raise HTTPException(503, "VectorDB is shutting down") from exc

        future.add_done_callback(lambda _future: slots.release())
        try:
            return await asyncio.wrap_future(future)
        except ValueError as exc:
            raise VDBInvalidRequest(str(exc)) from exc

    return app


def main() -> None:
    internal_token = os.environ.get("NRL_INTERNAL_VDB_TOKEN", "")
    if not internal_token and (token_file := os.environ.get("NRL_INTERNAL_VDB_TOKEN_FILE")):
        internal_token = Path(token_file).read_text(encoding="utf-8").strip()

    parser = argparse.ArgumentParser(description="NeMo Retriever VectorDB service")
    parser.add_argument("--lancedb-uri", default="/data/vectordb", help="LanceDB directory")
    parser.add_argument("--table-name", default="nemo_retriever", help="Vector table name")
    parser.add_argument(
        "--index-mode",
        default="auto",
        choices=("auto", "dense", "hybrid"),
        help="Fresh-table index mode; auto creates hybrid and preserves existing table capabilities.",
    )
    parser.add_argument("--embed-endpoint", default="", help="Remote NIM/OpenAI-compatible embed URL")
    parser.add_argument("--embed-model", default="nvidia/nemotron-3-embed-1b")
    parser.add_argument(
        "--embed-model-provider-prefix",
        default="",
        help="Optional LiteLLM provider prefix",
    )
    parser.add_argument(
        "--embed-api-key",
        default="",
        help="Remote embedding API key (defaults to NVIDIA_API_KEY, then NGC_API_KEY).",
    )
    parser.add_argument(
        "--internal-api-token",
        default=internal_token,
        help="Dedicated internal credential (prefer NRL_INTERNAL_VDB_TOKEN from a Secret).",
    )
    parser.add_argument(
        "--reconciliation-interval-seconds",
        type=int,
        default=int(os.environ.get("NRL_RECONCILIATION_INTERVAL_SECONDS", "60")),
        help="Lifecycle reconciliation interval; zero disables the local loop.",
    )
    parser.add_argument(
        "--max-concurrent-queries",
        type=int,
        default=MAX_CONCURRENT_QUERIES,
        help="Maximum number of concurrent non-agentic queries.",
    )
    parser.add_argument(
        "--disable-expiration-cleanup",
        action="store_true",
        help="Disable automatic collection expiration cleanup.",
    )
    parser.add_argument(
        "--local-embed",
        action="store_true",
        help="Load a local embedder for /v1/query (requires local extras and a GPU).",
    )
    parser.add_argument(
        "--local-embed-backend",
        default="vllm",
        choices=("hf", "vllm"),
        help="Backend for --local-embed (default: vllm).",
    )
    parser.add_argument("--hf-cache-dir", default="", help="Hugging Face model cache directory")
    parser.add_argument(
        "--device",
        default="",
        help="Torch device for --local-embed (for example cuda:0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory fraction when --local-embed-backend=vllm.",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Enable agentic=true on POST /v1/query using the agentic retrieval workflow.",
    )
    parser.add_argument("--agentic-llm-model", default="", help="Agentic retrieval chat model.")
    parser.add_argument(
        "--agentic-invoke-url",
        default="",
        help="OpenAI-compatible chat completions endpoint for agentic retrieval.",
    )
    parser.add_argument("--agentic-reasoning-effort", default="high")
    parser.add_argument("--agentic-backend-top-k", type=int, default=20)
    parser.add_argument("--agentic-react-max-steps", type=int, default=50)
    parser.add_argument("--agentic-text-truncation", type=int, default=0)
    parser.add_argument("--agentic-temperature", type=float, default=0.0)
    parser.add_argument("--agentic-request-timeout", type=float, default=1800.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7671)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.embed_endpoint and args.local_embed:
        parser.error("Use either --embed-endpoint or --local-embed, not both.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_vectordb_app(
        lancedb_uri=args.lancedb_uri,
        table_name=args.table_name,
        embed_endpoint=args.embed_endpoint,
        embed_model=args.embed_model,
        embed_model_provider_prefix=args.embed_model_provider_prefix or None,
        embed_api_key=resolve_remote_api_key(args.embed_api_key) or "",
        local_embed=args.local_embed,
        local_embed_backend=args.local_embed_backend,
        hf_cache_dir=args.hf_cache_dir or None,
        device=args.device or None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        index_mode=args.index_mode,
        internal_api_token=args.internal_api_token or None,
        reconciliation_interval_seconds=args.reconciliation_interval_seconds,
        expiration_cleanup_enabled=not args.disable_expiration_cleanup,
        max_concurrent_queries=args.max_concurrent_queries,
        agentic_config=AgenticConfig(
            enabled=args.agentic,
            llm_model=args.agentic_llm_model or None,
            invoke_url=args.agentic_invoke_url or None,
            reasoning_effort=args.agentic_reasoning_effort or None,
            backend_top_k=args.agentic_backend_top_k,
            react_max_steps=args.agentic_react_max_steps,
            text_truncation=args.agentic_text_truncation,
            temperature=args.agentic_temperature,
            request_timeout_s=args.agentic_request_timeout,
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
