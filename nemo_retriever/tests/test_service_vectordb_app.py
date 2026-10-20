# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import nemo_retriever.service.vectordb_app as vectordb_module
from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
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
from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import RetrievalContractError
from nemo_retriever.service.vectordb_app import (
    VectorDBState,
    _embed_queries_remote,
    _tensor_to_embedding_rows,
    create_vectordb_app,
)

_NOW = "2026-07-27T00:00:00+00:00"


class FakeVDB(VDB):
    """Backend-neutral in-memory fake for HTTP contract tests."""

    def __init__(self, *, table_exists: bool = False) -> None:
        self.collections: dict[tuple[str, str], CollectionInfo] = {}
        self.documents: dict[tuple[str, str, str], DocumentInfo] = {}
        self.table_exists = table_exists
        self.legacy_rows = 1 if table_exists else 0
        self.last_write_context: CollectionWriteContext | None = None
        self.last_retrieval: dict[str, Any] | None = None
        self.last_legacy_retrieval: dict[str, Any] | None = None
        self.health_calls = 0

    def create_index(self, **kwargs):
        return None

    def write_to_index(self, records: list, **kwargs):
        self.legacy_rows += sum(len(batch) for batch in records)
        self.table_exists = self.legacy_rows > 0

    def retrieval(self, queries: list, **kwargs):
        self.last_legacy_retrieval = dict(kwargs)
        return [
            [
                {
                    "text": "legacy hit",
                    "source": "legacy.pdf",
                    "metadata": {},
                    "_distance": 0.2,
                }
            ]
            for _ in queries
        ]

    def run(self, records):
        self.write_to_index(records)

    def create_collection(self, *, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        key = (scope, request.name)
        if key in self.collections:
            raise VDBResourceConflict("Collection already exists")
        info = CollectionInfo(
            name=request.name,
            scope=scope,
            description=request.description,
            metadata=request.metadata,
            created_at=_NOW,
            updated_at=_NOW,
            expires_at=request.expires_at,
        )
        self.collections[key] = info
        return info

    def get_collection(self, *, scope: str, collection_name: str) -> CollectionInfo:
        try:
            return self.collections[(scope, collection_name)]
        except KeyError as exc:
            raise VDBResourceNotFound("Collection not found") from exc

    def list_collections(self, *, scope: str, limit: int, continuation_token: str | None) -> CollectionPage:
        if continuation_token == "invalid":
            raise VDBInvalidRequest("Invalid continuation token")
        items = [item for (item_scope, _), item in self.collections.items() if item_scope == scope]
        return CollectionPage(items=items[:limit])

    def update_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        current = self.get_collection(scope=scope, collection_name=collection_name)
        updated = current.model_copy(
            update={
                "description": request.description,
                "metadata": (request.metadata if request.metadata is not None else current.metadata),
                "expires_at": request.expires_at,
                "updated_at": _NOW,
            }
        )
        self.collections[(scope, collection_name)] = updated
        return updated

    def delete_collection(self, *, scope: str, collection_name: str, if_exists: bool) -> CollectionDeleteResult:
        existed = self.collections.pop((scope, collection_name), None) is not None
        if not existed and not if_exists:
            raise VDBResourceNotFound("Collection not found")
        return CollectionDeleteResult(
            name=collection_name,
            scope=scope,
            existed=existed,
            deleted=True,
            status="deleted",
        )

    def get_document(self, *, scope: str, collection_name: str, document_id: str) -> DocumentInfo:
        try:
            return self.documents[(scope, collection_name, document_id)]
        except KeyError as exc:
            raise VDBResourceNotFound("Document not found") from exc

    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        self.get_collection(scope=scope, collection_name=collection_name)
        items = [
            item
            for (item_scope, item_collection, _), item in self.documents.items()
            if (item_scope, item_collection) == (scope, collection_name)
        ]
        return DocumentPage(items=items[:limit])

    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        existed = self.documents.pop((scope, collection_name, document_id), None) is not None
        if not existed and not if_exists:
            raise VDBResourceNotFound("Document not found")
        return DocumentDeleteResult(
            document_id=document_id,
            collection_name=collection_name,
            scope=scope,
            existed=existed,
            deleted=True,
            status="deleted",
        )

    def write_collection(self, records: list, *, context: CollectionWriteContext) -> CollectionWriteResult:
        self.get_collection(scope=context.scope, collection_name=context.collection_name)
        self.last_write_context = context
        written = sum(len(batch) for batch in records)
        self.documents[(context.scope, context.collection_name, context.document_id)] = DocumentInfo(
            document_id=context.document_id,
            collection_name=context.collection_name,
            scope=context.scope,
            filename=context.filename,
            content_sha256=context.content_sha256,
            document_version=context.document_version,
            status="completed",
            chunk_count=written,
            job_id=context.job_id,
            created_at=_NOW,
            updated_at=_NOW,
        )
        return CollectionWriteResult(written=written, total_rows=written)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        self.get_collection(scope=scope, collection_name=collection_name)
        self.last_retrieval = {
            "scope": scope,
            "collection_name": collection_name,
            "query_texts": query_texts,
            "top_k": top_k,
        }
        hit = {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "text": "collection hit",
            "distance": 0.2,
            "filename": "report.pdf",
            "page_number": 1,
            "content_type": "text",
            "source": "report.pdf",
            "source_id": "report.pdf",
            "metadata": {},
            "physical_table": "private-table",
            "_distance": 0.2,
        }
        return ([[hit] for _ in vectors], ["dense"])

    def health(self) -> dict[str, Any]:
        self.health_calls += 1
        return {
            "total_rows": self.legacy_rows,
            "table_exists": self.table_exists,
            "effective_retrieval_mode": "dense" if self.table_exists else None,
            "retrieval_strategies": ["dense"] if self.table_exists else [],
            "collections": {
                "active": len(self.collections),
                "deleting": 0,
                "expired": 0,
            },
            "cleanup": {"pending": 0, "oldest_age_seconds": 0},
            "reconciliation": {"successes": 0, "failures": 0},
            "open_table_cache_count": 0,
        }


class ContractFailureVDB(FakeVDB):
    def retrieve_collection(self, *args: Any, **kwargs: Any):
        raise RetrievalContractError("physical table secret must not be returned")


class ValueFailureVDB(FakeVDB):
    def retrieve_collection(self, *args: Any, **kwargs: Any):
        raise ValueError("backend parsing failed")


class HealthFailureVDB(FakeVDB):
    def health(self) -> dict[str, Any]:
        raise RuntimeError("backend unavailable")


class DefaultHealthVDB(FakeVDB):
    health = VDB.health


def _app(vdb: VDB, **kwargs: Any):
    return create_vectordb_app(
        vdb=vdb,
        embed_endpoint="http://embed.example/v1/embeddings",
        reconciliation_interval_seconds=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("extra_args", "expected_key"),
    [([], "env-key"), (["--embed-api-key", "explicit-key"], "explicit-key")],
)
def test_main_resolves_remote_embed_api_key(monkeypatch, extra_args, expected_key) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setattr(sys, "argv", ["vectordb_app", *extra_args])
    create_app = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(vectordb_module, "create_vectordb_app", create_app)
    monkeypatch.setattr(vectordb_module.uvicorn, "run", MagicMock())

    vectordb_module.main()

    assert create_app.call_args.kwargs["embed_api_key"] == expected_key
    assert create_app.call_args.kwargs["index_mode"] == "auto"


def test_main_passes_max_concurrent_queries(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vectordb_app", "--max-concurrent-queries", "7"])
    create_app = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(vectordb_module, "create_vectordb_app", create_app)
    monkeypatch.setattr(vectordb_module.uvicorn, "run", MagicMock())

    vectordb_module.main()

    assert create_app.call_args.kwargs["max_concurrent_queries"] == 7


def test_create_vectordb_app_uses_default_query_concurrency() -> None:
    app = _app(FakeVDB())

    with TestClient(app):
        semaphore = app.state.vectordb_state.query_semaphore
        assert semaphore._value == vectordb_module.MAX_CONCURRENT_QUERIES


def test_create_vectordb_app_uses_configured_query_concurrency() -> None:
    app = _app(FakeVDB(), max_concurrent_queries=2)

    with TestClient(app):
        semaphore = app.state.vectordb_state.query_semaphore
        assert semaphore._value == 2


@pytest.mark.parametrize("max_concurrent_queries", [0, -1])
def test_create_vectordb_app_rejects_non_positive_query_concurrency(
    max_concurrent_queries: int,
) -> None:
    with pytest.raises(ValueError, match="max_concurrent_queries must be positive"):
        _app(FakeVDB(), max_concurrent_queries=max_concurrent_queries)


def test_fake_vdb_completes_collection_http_flow_without_backend_details() -> None:
    backend = FakeVDB()
    app = _app(backend)
    record = {
        "document_type": "text",
        "embedding": [1.0, 0.0],
        "metadata": {
            "content": "hello",
            "content_metadata": {"page_number": 1},
            "source_metadata": {"source_id": "report.pdf"},
        },
    }

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(app) as client:
            created = client.post(
                "/v1/collections",
                headers={"X-NRL-Scope": "tenant-a"},
                json={"name": "research", "description": "docs"},
            )
            listed = client.get("/v1/collections", headers={"X-NRL-Scope": "tenant-a"})
            updated = client.patch(
                "/v1/collections/research",
                headers={"X-NRL-Scope": "tenant-a"},
                json={"description": "updated"},
            )
            written = client.post(
                "/internal/vectordb/write",
                json={
                    "records": [[record]],
                    "scope": "tenant-a",
                    "collection_name": "research",
                    "document_id": "document-1",
                    "job_id": "job-1",
                    "filename": "report.pdf",
                    "content_sha256": "sha256",
                    "document_version": "v1",
                },
            )
            documents = client.get(
                "/v1/collections/research/documents",
                headers={"X-NRL-Scope": "tenant-a"},
            )
            queried = client.post(
                "/v1/query",
                headers={"X-NRL-Scope": "tenant-a"},
                json={"query": "hello", "collection_name": "research"},
            )
            deleted_document = client.delete(
                "/v1/collections/research/documents/document-1",
                headers={"X-NRL-Scope": "tenant-a"},
            )
            deleted_collection = client.delete(
                "/v1/collections/research",
                headers={"X-NRL-Scope": "tenant-a"},
            )

    assert created.status_code == 201
    assert listed.json()["items"][0]["name"] == "research"
    assert updated.json()["description"] == "updated"
    assert written.json() == {"written": 1, "total_rows": 1}
    assert documents.json()["items"][0]["document_id"] == "document-1"
    assert queried.status_code == 200
    hit = queried.json()["results"][0]["hits"][0]
    assert backend.health_calls == 0
    assert hit["text"] == "collection hit"
    assert "physical_table" not in hit
    assert "_distance" not in hit
    assert backend.last_write_context is not None
    assert backend.last_write_context.scope == "tenant-a"
    assert backend.last_retrieval == {
        "scope": "tenant-a",
        "collection_name": "research",
        "query_texts": ["hello"],
        "top_k": 10,
    }
    assert deleted_document.status_code == 200
    assert deleted_collection.status_code == 200


def test_unsupported_collection_retrieval_returns_501_without_legacy_fallback() -> None:
    class UnsupportedCollectionRetrievalVDB(FakeVDB):
        def __init__(self) -> None:
            super().__init__()
            self.legacy_retrieval_calls = 0

        def retrieval(self, queries: list, **kwargs):
            self.legacy_retrieval_calls += 1
            return super().retrieval(queries, **kwargs)

        def retrieve_collection(self, *args: Any, **kwargs: Any):
            raise UnsupportedVDBOperation("Collection retrieval mode is unsupported")

    backend = UnsupportedCollectionRetrievalVDB()
    backend.create_collection(
        scope="tenant-a",
        request=CollectionCreateRequest(name="research"),
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(_app(backend)) as client:
            response = client.post(
                "/v1/query",
                headers={"X-NRL-Scope": "tenant-a"},
                json={"query": "hello", "collection_name": "research"},
            )

    assert response.status_code == 501
    assert response.json()["detail"] == ("The configured VectorDB backend does not support this operation.")
    assert backend.legacy_retrieval_calls == 0


def _embedding_records() -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "document_type": "text",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "content": "hello",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_name": "doc.pdf"},
                },
            }
        ]
    ]


def test_production_vdb_records_embedding_model_on_new_table(tmp_path) -> None:
    model = "nvidia/nemotron-3-embed-1b"
    backend = vectordb_module._production_vdb(
        lancedb_uri=str(tmp_path),
        table_name="fresh",
        expiration_cleanup_enabled=True,
        embed_model=model,
    )

    backend.run(_embedding_records())

    assert backend.get_index_metadata("embedding_model_name") == model


def test_production_vdb_rejects_untagged_existing_table(tmp_path) -> None:
    LanceDB(
        uri=str(tmp_path),
        table_name="untagged",
        vector_dim=2,
        build_index=False,
    ).run(_embedding_records())

    with pytest.raises(ValueError, match="does not record its embedding model"):
        vectordb_module._production_vdb(
            lancedb_uri=str(tmp_path),
            table_name="untagged",
            expiration_cleanup_enabled=True,
            embed_model="nvidia/nemotron-3-embed-1b",
        )


def test_production_vdb_rejects_existing_table_model_mismatch(tmp_path) -> None:
    LanceDB(
        uri=str(tmp_path),
        table_name="old-model",
        vector_dim=2,
        build_index=False,
        embedding_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    ).run(_embedding_records())

    with pytest.raises(ValueError, match="uses embedding model.*llama-nemotron.*configured for.*nemotron-3"):
        vectordb_module._production_vdb(
            lancedb_uri=str(tmp_path),
            table_name="old-model",
            expiration_cleanup_enabled=True,
            embed_model="nvidia/nemotron-3-embed-1b",
        )


@pytest.mark.parametrize(
    "stored_model",
    [
        "nvidia/nemotron-3-embed-1b",
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "nvidia/Nemotron-3-Embed-1B-NVFP4",
    ],
)
def test_production_vdb_accepts_existing_table_model_match(stored_model: str, tmp_path) -> None:
    configured_model = "nvidia/nemotron-3-embed-1b"
    LanceDB(
        uri=str(tmp_path),
        table_name="matching-model",
        vector_dim=2,
        build_index=False,
        embedding_model_name=stored_model,
    ).run(_embedding_records())

    backend = vectordb_module._production_vdb(
        lancedb_uri=str(tmp_path),
        table_name="matching-model",
        expiration_cleanup_enabled=True,
        embed_model=configured_model,
    )

    assert isinstance(backend, LanceDB)


def test_production_vdb_defaults_to_hybrid_without_vector_index_build(tmp_path) -> None:
    backend = vectordb_module._production_vdb(
        lancedb_uri=str(tmp_path),
        table_name="hybrid",
        expiration_cleanup_enabled=True,
        embed_model="nvidia/nemotron-3-embed-1b",
    )
    assert isinstance(backend, LanceDB)
    assert backend.build_index is False
    assert backend.hybrid is True


def test_production_vdb_dense_mode_preserves_legacy_service_write_without_index_rebuild(
    tmp_path,
    monkeypatch,
) -> None:
    backend = vectordb_module._production_vdb(
        lancedb_uri=str(tmp_path),
        table_name="legacy",
        expiration_cleanup_enabled=True,
        embed_model="nvidia/nemotron-3-embed-1b",
        index_mode="dense",
    )
    assert isinstance(backend, LanceDB)
    assert backend.build_index is False
    assert backend.hybrid is False

    index_writes = []
    monkeypatch.setattr(
        backend,
        "create_index",
        lambda records, table_name: object(),
    )
    monkeypatch.setattr(
        backend,
        "write_to_index",
        lambda *args, **kwargs: index_writes.append((args, kwargs)),
    )

    assert backend.run([]) == []
    assert index_writes == []


def test_legacy_hybrid_query_uses_default_weighted_fusion() -> None:
    backend = FakeVDB(table_exists=True)
    hybrid_health = backend.health()
    hybrid_health.update(
        effective_retrieval_mode="hybrid",
        retrieval_strategies=["hybrid"],
    )

    with patch.object(backend, "health", return_value=hybrid_health), patch.object(
        VectorDBState,
        "embed_queries",
        return_value=[[1.0, 0.0]],
    ):
        with TestClient(_app(backend)) as client:
            response = client.post("/v1/query", json={"query": "legacy"})

    assert response.status_code == 200
    assert backend.last_legacy_retrieval is not None
    assert backend.last_legacy_retrieval["hybrid"] is True
    assert backend.last_legacy_retrieval["hybrid_fusion"] == DEFAULT_HYBRID_FUSION_POLICY


def test_legacy_write_and_query_keep_existing_vdb_path() -> None:
    backend = FakeVDB()
    app = _app(backend)
    record = {
        "document_type": "text",
        "embedding": [1.0, 0.0],
        "metadata": {
            "content": "legacy",
            "content_metadata": {},
            "source_metadata": {},
        },
    }

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(app) as client:
            written = client.post(
                "/internal/vectordb/write",
                json={
                    "records": [[record]],
                    "scope": "default",
                    "filename": "legacy.pdf",
                    "document_version": "1",
                },
            )
            queried = client.post("/v1/query", json={"query": "legacy"})

    assert written.json() == {"written": 1, "total_rows": 1}
    hit = queried.json()["results"][0]["hits"][0]
    assert hit["text"] == "legacy hit"
    assert hit["_distance"] == 0.2


def test_service_managed_lancedb_preserves_multimodal_fields(tmp_path) -> None:
    backend = LanceDB(
        uri=str(tmp_path),
        table_name="legacy",
        vector_dim=None,
        overwrite=False,
        build_index=False,
        _service_table_schema=True,
    )
    app = _app(backend)
    record = {
        "document_type": "text",
        "embedding": [1.0, 0.0, 0.0, 0.0],
        "metadata": {
            "content": "table content",
            "content_metadata": {
                "page_number": 7,
                "type": "table_caption",
                "stored_image_uri": "s3://artifacts/table.png",
                "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            },
            "source_metadata": {
                "source_id": "/documents/report.pdf",
                "source_name": "report.pdf",
            },
        },
    }

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            written = client.post(
                "/internal/vectordb/write",
                json={
                    "records": [[record]],
                    "filename": "report.pdf",
                    "document_version": "1",
                },
            )
            queried = client.post("/v1/query", json={"query": "table", "top_k": 1})

    assert written.status_code == 200
    assert queried.status_code == 200
    hit = queried.json()["results"][0]["hits"][0]
    assert hit["content_type"] == "table"
    assert hit["stored_image_uri"] == "s3://artifacts/table.png"
    assert hit["bbox_xyxy_norm"] == "[0.1, 0.2, 0.8, 0.9]"
    assert hit["page_number"] == 7
    assert hit["source_id"] == "/documents/report.pdf"


def test_typed_collection_errors_map_to_http_contract() -> None:
    backend = FakeVDB()
    with TestClient(_app(backend)) as client:
        missing = client.get("/v1/collections/missing")
        assert missing.status_code == 404

        assert client.post("/v1/collections", json={"name": "duplicate"}).status_code == 201
        conflict = client.post("/v1/collections", json={"name": "duplicate"})
        invalid = client.get("/v1/collections?continuation_token=invalid")

    assert conflict.status_code == 409
    assert invalid.status_code == 422


@pytest.mark.parametrize("backend_cls", [ContractFailureVDB, ValueFailureVDB])
def test_backend_query_failures_return_safe_500(backend_cls) -> None:
    backend = backend_cls()
    backend.create_collection(scope="default", request=CollectionCreateRequest(name="research"))
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(_app(backend), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/query",
                json={"query": "hello", "collection_name": "research"},
            )

    assert response.status_code == 500
    assert "physical table secret" not in response.text
    if backend_cls is ContractFailureVDB:
        assert response.json()["detail"] == "VectorDB retrieval contract violation."
    else:
        assert response.json()["detail"] == "VectorDB backend operation failed."


def test_query_empty_legacy_index_returns_422() -> None:
    with TestClient(_app(FakeVDB())) as client:
        response = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert response.status_code == 422
    assert "No data has been ingested yet" in response.json()["detail"]


def test_legacy_query_allows_backend_with_default_empty_health() -> None:
    app = _app(DefaultHealthVDB())
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(app) as client:
            response = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["hits"][0]["text"] == "legacy hit"


def test_query_without_embed_backend_returns_501() -> None:
    app = create_vectordb_app(vdb=FakeVDB(), reconciliation_interval_seconds=0)
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert response.status_code == 501
    assert "No embedding backend configured" in response.json()["detail"]


def test_internal_auth_is_optional_and_can_be_enabled() -> None:
    app = _app(FakeVDB(), internal_api_token="internal-secret")
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/collections").status_code == 401
        assert (
            client.get(
                "/v1/collections",
                headers={"X-NRL-Internal-Token": "internal-secret"},
            ).status_code
            == 200
        )


def test_health_and_metrics_use_backend_neutral_health() -> None:
    backend = FakeVDB(table_exists=True)
    app = _app(backend)
    with TestClient(app) as client:
        health = client.get("/v1/health")
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["table_exists"] is True
    assert health.json()["effective_retrieval_mode"] == "dense"
    assert metrics.status_code == 200
    assert "nrl_vectordb_collections" in metrics.text


def test_health_returns_503_when_backend_is_unavailable() -> None:
    app = _app(HealthFailureVDB())
    with TestClient(app) as client:
        health = client.get("/v1/health")

    assert health.status_code == 503
    assert health.json() == {"detail": "VectorDB backend is unavailable"}


def test_tensor_to_embedding_rows_handles_batch() -> None:
    tensor = MagicMock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    assert _tensor_to_embedding_rows(tensor) == [[0.1, 0.2], [0.3, 0.4]]


def test_vector_db_state_local_embed_queries() -> None:
    mock_embedder = MagicMock()
    tensor = MagicMock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.tolist.return_value = [[1.0, 2.0]]
    mock_embedder.embed_queries.return_value = tensor
    state = VectorDBState(
        vdb=FakeVDB(),
        embed_endpoint="",
        embed_model="nvidia/llama-nemotron-embed-1b-v2",
        embed_api_key="",
        local_embed=True,
        local_embed_backend="hf",
    )

    with patch("nemo_retriever.models.create_local_embedder", return_value=mock_embedder):
        vectors = state.embed_queries(["hello"])

    assert vectors == [[1.0, 2.0]]
    mock_embedder.embed_queries.assert_called_once_with(["hello"])


def test_remote_embed_queries_delegates_model_prefix(monkeypatch) -> None:
    calls = {}

    def fake_infer_microservice(data, **kwargs):
        calls["data"] = data
        calls.update(kwargs)
        return [[0.1, 0.2]]

    monkeypatch.setattr("nemo_retriever.models.nim.util.infer_microservice", fake_infer_microservice)
    vectors = _embed_queries_remote(
        ["hello"],
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_endpoint="https://litellm.example.com/v1/embeddings",
        embed_api_key="k",
        embed_model_provider_prefix="nvidia",
    )

    assert vectors == [[0.1, 0.2]]
    assert calls["data"] == ["hello"]
    assert calls["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert calls["model_provider_prefix"] == "nvidia"
    assert calls["embedding_endpoint"] == "https://litellm.example.com/v1/embeddings"
