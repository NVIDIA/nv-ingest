# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

import httpx
import lancedb
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from nemo_retriever import RetrieverServiceClient
from nemo_retriever.common.schemas.collections import CollectionCreateRequest
from nemo_retriever.common.schemas.requests import JobCreateRequest
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    VDBInvalidRequest,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.service.auth import ScopeAuthorizer
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import AuthConfig, LLMConfig, ServiceConfig, VectorDbConfig
from nemo_retriever.service.query_schema import QueryRequest
from nemo_retriever.service.errors import RetrieverServiceError
import nemo_retriever.service.client as client_module
from nemo_retriever.service.services.job_tracker import JobFullError, JobTracker
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _record(_chunk_id: str, _document_id: str, text: str, version: str = "v1") -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": [1.0, 0.0],
            "content": text,
            "content_metadata": {"page_number": 1, "type": "text"},
            "source_metadata": {
                "source_id": "report.pdf",
                "source_name": "report.pdf",
            },
        },
    }


def _context(
    document_id: str,
    *,
    version: str = "v1",
    collection_name: str = "research",
    operation: str = "append",
) -> CollectionWriteContext:
    return CollectionWriteContext(
        scope="scope",
        collection_name=collection_name,
        document_id=document_id,
        document_version=version,
        content_sha256=version,
        filename="report.pdf",
        job_id=f"job-{version}",
        operation=operation,
    )


def _vdb(tmp_path, *, table_name: str = "legacy") -> LanceDB:
    return LanceDB(
        uri=str(tmp_path),
        table_name=table_name,
        overwrite=False,
        build_index=False,
    )


def test_collection_crud_scope_pagination_and_injection_rejection(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path), embed_endpoint="http://embed")
    with TestClient(app) as client:
        headers = {"X-NRL-Scope": "workspace-a"}
        assert client.post("/v1/collections", json={"name": "one"}, headers=headers).status_code == 201
        assert client.post("/v1/collections", json={"name": "two"}, headers=headers).status_code == 201
        page = client.get("/v1/collections?limit=1", headers=headers).json()
        assert len(page["items"]) == 1 and page["next_token"]
        other_scope = client.get("/v1/collections/one", headers={"X-NRL-Scope": "workspace-b"})
        assert other_scope.status_code == 404
        injected = client.post(
            "/v1/query",
            json={"query": "x", "collection_name": "one", "table_name": "secret"},
            headers=headers,
        )
        assert injected.status_code == 422
        health = client.get("/v1/health").json()
        assert "table" not in health and "workspace-a" not in str(health)
        metrics = client.get("/metrics").text
        assert "workspace-a" not in metrics and "nrl_vectordb_cleanup_pending" in metrics
        assert client.delete("/v1/collections/one", headers=headers).status_code == 200
        repeated = client.delete("/v1/collections/one?if_exists=true", headers=headers).json()
        assert repeated == {
            "name": "one",
            "scope": "workspace-a",
            "existed": False,
            "deleted": False,
            "status": "deleted",
            "cleanup_pending": False,
        }


def test_append_replace_and_document_delete_are_collection_scoped(tmp_path) -> None:
    backend = _vdb(tmp_path)
    backend.create_collection(
        scope="scope",
        request=CollectionCreateRequest(name="research"),
    )
    appended = backend.write_collection(
        [[_record("a", "doc", "old"), _record("b", "doc", "obsolete")]],
        context=_context("doc"),
    )
    assert appended.total_rows == 2
    replaced = backend.write_collection(
        [[_record("c", "doc", "new", "v2")]],
        context=_context("doc", version="v2", operation="replace"),
    )
    assert replaced.total_rows == 1
    assert (
        backend.get_document(
            scope="scope",
            collection_name="research",
            document_id="doc",
        ).document_version
        == "v2"
    )
    assert backend.delete_document(
        scope="scope",
        collection_name="research",
        document_id="doc",
        if_exists=False,
    ).deleted
    assert not backend.delete_document(
        scope="scope",
        collection_name="research",
        document_id="doc",
        if_exists=True,
    ).deleted


def test_public_sdk_and_citation_ready_query(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path), embed_endpoint="http://embed")
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(app) as service:

            class InProcessClient(RetrieverServiceClient):
                async def _arequest(self, method: str, path: str, **kwargs):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://test",
                        headers=self._auth_headers,
                    ) as client:
                        response = await client.request(method, path, **kwargs)
                    self._raise_for_response(response, f"{method} {path}")
                    return response.json() if response.content else None

            sdk = InProcessClient(scope="workspace")
            collection = sdk.create_collection("research")
            assert collection.name == "research"
            write = service.post(
                "/internal/vectordb/write",
                json={
                    "records": [[_record("chunk", "doc", "finding")]],
                    "scope": "workspace",
                    "collection_name": "research",
                    "document_id": "doc",
                    "job_id": "job-v1",
                    "filename": "report.pdf",
                    "content_sha256": "v1",
                    "document_version": "v1",
                },
            )
            assert write.status_code == 200, write.text
            sync_hits = sdk.query("finding", top_k=10, collection_name="research")
            async_hits = asyncio.run(sdk.aquery("finding", top_k=10, collection_name="research"))
            assert sync_hits[0].model_dump() == async_hits[0].model_dump()
            assert sync_hits[0].chunk_id == hashlib.sha256(f"doc\0v1\0{0}".encode()).hexdigest()
            assert sync_hits[0].text == "finding"
            assert sync_hits[0].distance >= 0.0
            assert sync_hits[0].page_number == 1
            assert sync_hits[0].filename == "report.pdf"
            assert sdk.list_documents("research").items[0].document_id == "doc"


def test_idempotency_replay_and_conflict() -> None:
    tracker = JobTracker()
    original = tracker.register_job(
        "job",
        expected_documents=1,
        scope="workspace",
        idempotency_key="request",
        idempotency_fingerprint="same",
    )
    replay = tracker.register_job(
        "replay",
        expected_documents=1,
        scope="workspace",
        idempotency_key="request",
        idempotency_fingerprint="same",
    )
    assert replay.job_id == original.job_id

    with pytest.raises(JobFullError):
        tracker.register_job(
            "conflict",
            expected_documents=1,
            scope="workspace",
            idempotency_key="request",
            idempotency_fingerprint="different",
        )


def test_idempotent_job_registration_is_atomic() -> None:
    tracker = JobTracker()

    def register(index: int) -> str:
        return tracker.register_job(
            f"job-{index}",
            expected_documents=1,
            scope="workspace",
            idempotency_key="request",
            idempotency_fingerprint="same",
        ).job_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        job_ids = list(executor.map(register, range(8)))

    assert len(set(job_ids)) == 1
    assert len(tracker.all_jobs()) == 1


def test_target_document_id_rejects_query_metacharacters() -> None:
    with pytest.raises(ValueError, match="target_document_id"):
        JobCreateRequest(
            expected_documents=1,
            operation="replace",
            target_document_id="doc' OR 1=1 --",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"expected_documents": 1, "operation": "replace", "target_document_id": "doc"}, "collection_name"),
        (
            {
                "expected_documents": 2,
                "collection_name": "research",
                "operation": "replace",
                "target_document_id": "doc",
            },
            "exactly one document",
        ),
        ({"expected_documents": 1, "collection_name": "research", "operation": "replace"}, "target_document_id"),
        ({"expected_documents": 1, "operation": "append", "target_document_id": "doc"}, "append"),
    ],
)
def test_job_create_request_enforces_collection_operation_invariants(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        JobCreateRequest(**payload)


def test_job_create_request_accepts_valid_replacement() -> None:
    request = JobCreateRequest(
        expected_documents=1,
        collection_name="research",
        operation="replace",
        target_document_id="doc",
    )

    assert request.operation == "replace"
    assert request.target_document_id == "doc"


def test_manifest_entry_replay_is_capacity_neutral_and_identity_stable() -> None:
    tracker = JobTracker()
    tracker.register_job(
        "job",
        expected_documents=1,
        document_manifest=[
            {
                "manifest_entry_id": "a" * 64,
                "filename": "report.pdf",
                "content_sha256": "b" * 64,
            }
        ],
    )
    accepted, created = tracker.register_document_idempotent(
        "attempt-1",
        job_id="job",
        stable_document_id="document-1",
        filename="report.pdf",
        content_sha256="b" * 64,
        manifest_entry_id="a" * 64,
    )
    assert created is True
    tracker.mark_completed("attempt-1")
    replay, created = tracker.register_document_idempotent(
        "attempt-2",
        job_id="job",
        stable_document_id="document-2",
        filename="report.pdf",
        content_sha256="b" * 64,
        manifest_entry_id="a" * 64,
    )
    assert created is False
    assert replay.id == accepted.id == "attempt-1"
    assert replay.stable_document_id == accepted.stable_document_id == "document-1"
    assert len(tracker.job_documents("job")) == 1

    try:
        tracker.register_document_idempotent(
            "attempt-3",
            job_id="job",
            filename="report.pdf",
            content_sha256="c" * 64,
            manifest_entry_id="a" * 64,
        )
    except JobFullError:
        pass
    else:
        raise AssertionError("conflicting manifest replay must fail")


def test_raw_storage_selection_is_rejected_for_query_requests() -> None:
    for key in ("table_name", "lancedb_uri", "uri", "physical_table"):
        try:
            QueryRequest.model_validate({"query": "x", key: "untrusted"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"query storage key {key} must be rejected")


def test_scope_authorizer_secret_mapping_and_internal_vectordb_token(tmp_path) -> None:
    secret = tmp_path / "scope-tokens.json"
    secret.write_text('{"tokens":[{"token":"alpha-token","scopes":["alpha"]}]}', encoding="utf-8")
    authorizer = ScopeAuthorizer(AuthConfig(enabled=True, scope_token_file=str(secret), allow_unscoped_dev=False))
    assert authorizer.authorize("alpha-token", "alpha") == ("alpha", None)
    assert authorizer.authorize("alpha-token", "beta") == (None, 401)
    assert authorizer.authorize("invalid", "alpha") == (None, 401)

    app = create_vectordb_app(
        lancedb_uri=str(tmp_path / "db"),
        embed_endpoint="http://embed",
        internal_api_token="internal-secret",
    )
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/collections").status_code == 401
        assert client.get("/v1/collections", headers={"X-NRL-Internal-Token": "wrong"}).status_code == 401
        assert client.get("/v1/collections", headers={"X-NRL-Internal-Token": "internal-secret"}).status_code == 200


def test_non_ascii_credentials_are_rejected_with_401(tmp_path) -> None:
    # Starlette decodes header bytes as latin-1, so a non-ASCII byte reaches the
    # comparison as non-ASCII text.
    secret = tmp_path / "scope-tokens.json"
    secret.write_text('{"tokens":[{"token":"alpha-token","scopes":["alpha"]}]}', encoding="utf-8")
    authorizer = ScopeAuthorizer(AuthConfig(enabled=True, scope_token_file=str(secret)))
    assert authorizer.authorize("alpha-tok\xe9n", "alpha") == (None, 401)

    vectordb = create_vectordb_app(
        lancedb_uri=str(tmp_path / "db"),
        embed_endpoint="http://embed",
        internal_api_token="internal-secret",
    )
    with TestClient(vectordb) as client:
        assert client.get("/v1/collections", headers={"X-NRL-Internal-Token": b"internal-secr\xe9t"}).status_code == 401

    gateway = create_app(
        ServiceConfig(
            mode="gateway",
            auth=AuthConfig(enabled=True, api_token="alpha-token", default_scope="alpha"),
            vectordb=VectorDbConfig(internal_api_token="internal-secret"),
        )
    )
    with TestClient(gateway) as client:
        assert client.get("/v1/collections", headers={"Authorization": b"Bearer alpha-tok\xe9n"}).status_code == 401
        assert (
            client.get(
                "/v1/internal/document-result/missing",
                headers={"X-NRL-Internal-Token": b"internal-secr\xe9t"},
            ).status_code
            == 401
        )


def test_scope_authorizer_defaults_to_unprotected_and_fails_closed_when_enabled() -> None:
    assert ScopeAuthorizer(AuthConfig()).authorize("", "workspace-a") == ("workspace-a", None)
    assert ScopeAuthorizer(AuthConfig(enabled=True)).authorize("", "workspace-a") == (None, 401)
    assert ScopeAuthorizer(AuthConfig(enabled=True, allow_unscoped_dev=True)).authorize("", "workspace-a") == (
        "workspace-a",
        None,
    )


def test_public_routes_accept_requests_when_auth_is_disabled() -> None:
    app = create_app(ServiceConfig(mode="gateway", auth=AuthConfig()))
    with TestClient(app) as client:
        response = client.get("/v1/collections")

    # The route is reached and then rejects the disabled VectorDB, rather than
    # being rejected by bearer authentication.
    assert response.status_code == 404


def test_service_routes_use_authorized_scope_not_raw_header() -> None:
    app = create_app(
        ServiceConfig(
            mode="gateway",
            auth=AuthConfig(
                enabled=True,
                api_token="alpha-token",
                default_scope="alpha",
                allow_unscoped_dev=False,
            ),
            vectordb=VectorDbConfig(internal_api_token="internal-secret"),
        )
    )
    with TestClient(app) as client:
        invalid_token = client.post("/v1/ingest/job", json={"expected_documents": 1})
        invalid_scope = client.post(
            "/v1/ingest/job",
            json={"expected_documents": 1},
            headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "beta"},
        )
        assert invalid_token.status_code == invalid_scope.status_code == 401
        assert invalid_token.json() == invalid_scope.json() == {"detail": "Missing or invalid bearer token."}
        created = client.post(
            "/v1/ingest/job",
            json={"expected_documents": 1},
            headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "alpha"},
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]
        assert (
            client.get(
                f"/v1/ingest/job/{job_id}",
                headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "alpha"},
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/v1/internal/document-result/missing",
                headers={"Authorization": "Bearer alpha-token"},
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/internal/document-result/missing",
                headers={"X-NRL-Internal-Token": "internal-secret"},
            ).status_code
            == 404
        )


def test_vectordb_proxy_failures_do_not_expose_internal_details(monkeypatch) -> None:
    class FailingAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aclose(self):
            return None

        async def request(self, *_args, **_kwargs):
            raise httpx.ConnectError("private-vectordb:7671?token=sensitive")

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("private-vectordb:7671?token=sensitive")

    monkeypatch.setattr(httpx, "AsyncClient", FailingAsyncClient)
    app = create_app(
        ServiceConfig(
            mode="gateway",
            auth=AuthConfig(allow_unscoped_dev=True),
            vectordb=VectorDbConfig(enabled=True, vectordb_url="http://private-vectordb:7671"),
            llm=LLMConfig(enabled=True),
        )
    )
    with TestClient(app) as client:
        responses = (
            client.get("/v1/collections"),
            client.post("/v1/query", json={"query": "test"}),
            client.post("/v1/answer", json={"query": "test"}),
        )

    for response in responses:
        assert response.status_code == 502
        assert response.json() == {"detail": "VectorDB service is unavailable."}
        assert "private-vectordb" not in response.text
        assert "sensitive" not in response.text


def test_openapi_operation_ids_are_unique() -> None:
    app = create_app(ServiceConfig())
    operation_ids = [
        operation["operationId"]
        for path in app.openapi()["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))


def test_sdk_replays_every_manifest_entry_after_idempotent_job_replay(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    class FakeResponse:
        status_code = 200
        content = b"{}"
        text = ""

        def json(self):
            return {
                "job_id": "job",
                "expected_documents": 2,
                "status": "completed",
                "created_at": "now",
                "counts": {"completed": 2},
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    sdk = RetrieverServiceClient()
    sdk._create_job = AsyncMock(return_value=client_module._CreatedJob("job"))
    sdk._upload_one = AsyncMock(return_value={"status": "accepted"})

    result = asyncio.run(sdk.asubmit_documents("research", [first, second], idempotency_key="key"))
    assert result.job_id == "job"
    assert sdk._upload_one.await_count == 2
    entry_ids = [call.kwargs["manifest_entry_id"] for call in sdk._upload_one.await_args_list]
    expected = []
    for position, path in enumerate((first, second)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected.append(hashlib.sha256(f"{position}\0{path.name}\0{digest}".encode("utf-8")).hexdigest())
    assert entry_ids == expected


def test_sdk_wraps_malformed_sync_and_async_lifecycle_responses() -> None:
    class MalformedClient(RetrieverServiceClient):
        async def _arequest(self, *_args, **_kwargs):
            return {}

    sdk = MalformedClient()
    with pytest.raises(RetrieverServiceError, match="invalid response"):
        sdk.create_collection("research")
    with pytest.raises(RetrieverServiceError, match="invalid response"):
        asyncio.run(sdk.acreate_collection("research"))


def test_expiration_is_timezone_aware_and_normalized() -> None:
    with pytest.raises(ValueError, match="timezone"):
        CollectionCreateRequest(name="bad", expires_at="2030-01-01T00:00:00")
    with pytest.raises(ValueError, match="ISO-8601"):
        CollectionCreateRequest(name="bad-type", expires_at=123)
    request = CollectionCreateRequest(name="good", expires_at="2030-01-01T01:00:00+01:00")
    assert request.expires_at == "2030-01-01T00:00:00+00:00"


def test_keyset_cursors_are_stable_and_context_bound(tmp_path) -> None:
    backend = _vdb(tmp_path)
    for name in ("a", "c"):
        backend.create_collection(scope="scope", request=CollectionCreateRequest(name=name))

    first = backend.list_collections(
        scope="scope",
        limit=1,
        continuation_token=None,
    )
    assert [item.name for item in first.items] == ["a"]
    backend.create_collection(scope="scope", request=CollectionCreateRequest(name="b"))
    second = backend.list_collections(
        scope="scope",
        limit=2,
        continuation_token=first.next_token,
    )
    assert [item.name for item in second.items] == ["b", "c"]
    with pytest.raises(VDBInvalidRequest, match="context"):
        backend.list_collections(
            scope="other",
            limit=1,
            continuation_token=first.next_token,
        )

    backend.write_collection(
        [[_record("one", "doc-1", "one")]],
        context=_context("doc-1", collection_name="a"),
    )
    backend.write_collection(
        [[_record("two", "doc-2", "two")]],
        context=_context("doc-2", collection_name="a"),
    )
    documents = backend.list_documents(
        scope="scope",
        collection_name="a",
        limit=1,
        continuation_token=None,
    )
    assert documents.next_token
    with pytest.raises(VDBInvalidRequest, match="context"):
        backend.list_documents(
            scope="scope",
            collection_name="b",
            limit=1,
            continuation_token=documents.next_token,
        )


def test_search_and_collection_delete_share_lifecycle_lock(tmp_path, monkeypatch) -> None:
    backend = _vdb(tmp_path)
    backend.create_collection(
        scope="scope",
        request=CollectionCreateRequest(name="research"),
    )
    backend.write_collection(
        [[_record("chunk", "doc", "searchable text")]],
        context=_context("doc"),
    )

    search_entered = threading.Event()
    release_search = threading.Event()
    delete_entered = threading.Event()
    original_retrieval = LanceDB.retrieval

    def blocked_retrieval(vdb, *args, **kwargs):
        search_entered.set()
        assert release_search.wait(5)
        return original_retrieval(vdb, *args, **kwargs)

    def observed_drop_table(*args, **kwargs):
        delete_entered.set()

    monkeypatch.setattr(LanceDB, "retrieval", blocked_retrieval)
    monkeypatch.setattr(backend._get_collection_store()._db, "drop_table", observed_drop_table)

    with ThreadPoolExecutor(max_workers=2) as pool:
        search = pool.submit(
            backend.retrieve_collection,
            [[1.0, 0.0]],
            scope="scope",
            collection_name="research",
            query_texts=["searchable text"],
            top_k=1,
        )
        assert search_entered.wait(5)
        delete = pool.submit(
            backend.delete_collection,
            scope="scope",
            collection_name="research",
            if_exists=False,
        )

        assert not delete_entered.wait(0.25)
        release_search.set()

        results, strategies = search.result(timeout=5)
        deleted = delete.result(timeout=30)

    assert results[0][0]["document_id"] == "doc"
    assert strategies == ["dense"]
    assert delete_entered.is_set()
    assert deleted.deleted


def test_collection_searches_run_concurrently(tmp_path, monkeypatch) -> None:
    backend = _vdb(tmp_path)
    backend.create_collection(
        scope="scope",
        request=CollectionCreateRequest(name="research"),
    )
    backend.write_collection(
        [[_record("chunk", "doc", "searchable text")]],
        context=_context("doc"),
    )
    searches_entered = threading.Barrier(2)

    def synchronized_retrieval(_vdb, vectors, **_kwargs):
        searches_entered.wait(timeout=5)
        return [[] for _ in vectors]

    monkeypatch.setattr(LanceDB, "retrieval", synchronized_retrieval)
    with ThreadPoolExecutor(max_workers=2) as pool:
        searches = [
            pool.submit(
                backend.retrieve_collection,
                [[1.0, 0.0]],
                scope="scope",
                collection_name="research",
                query_texts=["searchable text"],
                top_k=1,
            )
            for _ in range(2)
        ]
        assert [future.result(timeout=5)[0] for future in searches] == [[[]], [[]]]


def test_reconcile_catalog_scan_does_not_hold_the_write_lock(tmp_path, monkeypatch) -> None:
    backend = _vdb(tmp_path)
    store = backend._get_collection_store()
    scan_entered = threading.Event()
    release_scan = threading.Event()
    original_rows = store._rows

    def blocked_rows(table_name, where=None, columns=None):
        if table_name == "_nrl_documents":
            assert where == "recovery_state != ''"
            scan_entered.set()
            assert release_scan.wait(5)
        return original_rows(table_name, where, columns)

    write_lock_acquired = threading.Event()

    def acquire_write_lock() -> None:
        with store._write_lock:
            write_lock_acquired.set()

    monkeypatch.setattr(store, "_rows", blocked_rows)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reconciliation = pool.submit(backend.reconcile_collections)
        assert scan_entered.wait(5)
        write_lock_probe = pool.submit(acquire_write_lock)
        try:
            assert write_lock_acquired.wait(2)
        finally:
            release_scan.set()
        write_lock_probe.result(timeout=5)
        assert reconciliation.result(timeout=5) == {"successes": 0, "failures": 0}


def test_replacement_marker_recovers_after_catalog_finalize_failure(tmp_path, monkeypatch) -> None:
    backend = _vdb(tmp_path)
    backend.create_collection(
        scope="scope",
        request=CollectionCreateRequest(name="research"),
    )
    backend.write_collection(
        [[_record("old", "doc", "old", "v1")]],
        context=_context("doc"),
    )
    store = backend._get_collection_store()
    original_persist = store._persist_document_row

    def fail_finalize(row):
        if row.get("current_document_version") == "v2":
            raise RuntimeError("injected finalize failure")
        return original_persist(row)

    monkeypatch.setattr(store, "_persist_document_row", fail_finalize)
    with pytest.raises(RuntimeError, match="injected"):
        backend.write_collection(
            [
                [
                    _record("new-1", "doc", "new first", "v2"),
                    _record("new-2", "doc", "new second", "v2"),
                ]
            ],
            context=_context("doc", version="v2", operation="replace"),
        )
    monkeypatch.setattr(store, "_persist_document_row", original_persist)
    document = backend.get_document(scope="scope", collection_name="research", document_id="doc")
    assert document.status == "replacing"
    result = backend.reconcile_collections()
    assert result["successes"] == 1
    document = backend.get_document(scope="scope", collection_name="research", document_id="doc")
    assert document.document_version == "v2"
    assert document.content_sha256 == "v2"
    assert document.filename == "report.pdf"
    assert document.chunk_count == 2
    # No pending job ID is persisted in the catalog schema, so recovery keeps
    # the last successfully finalized job rather than fabricating an ID.
    assert document.job_id == "job-v1"
    assert document.status == "completed"
    table = store._open_table(store._resolved_table("scope", "research"))
    rows = table.search().to_list()
    versions = {row["document_version"] for row in rows}
    assert versions == {"v2"}
    assert len(rows) == 2


def test_collection_deletion_does_not_delete_external_artifacts(tmp_path) -> None:
    artifact = tmp_path / "artifacts" / "external" / "image.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"image")

    backend = _vdb(tmp_path / "db")
    backend.create_collection(
        scope="scope",
        request=CollectionCreateRequest(name="research"),
    )
    backend.write_collection(
        [[_record("chunk", "doc", "owned")]],
        context=_context("doc"),
    )
    deleted = backend.delete_collection(scope="scope", collection_name="research", if_exists=False)

    assert deleted.deleted
    assert artifact.read_bytes() == b"image"


def test_expired_collection_uses_retryable_deletion_and_health_is_aggregate_only(
    tmp_path,
) -> None:
    backend = _vdb(tmp_path, table_name="secret-legacy-table")
    backend.create_collection(
        scope="tenant-secret",
        request=CollectionCreateRequest(
            name="expired",
            expires_at="2000-01-01T00:00:00Z",
        ),
    )
    assert backend.reconcile_collections()["successes"] == 1
    with pytest.raises(VDBResourceNotFound):
        backend.get_collection(
            scope="tenant-secret",
            collection_name="expired",
        )
    health = backend.health()
    assert health["catalog"]["schema_version"] == 2
    assert "tenant-secret" not in str(health)
    assert "secret-legacy-table" not in str(health)


def test_catalog_startup_fails_fast_on_missing_required_columns(tmp_path) -> None:
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "_nrl_collections",
        schema=pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="missing required columns"):
        _vdb(tmp_path).list_collections(
            scope="scope",
            limit=1,
            continuation_token=None,
        )
    backend = _vdb(tmp_path)
    with pytest.raises(RuntimeError, match="missing required columns"):
        backend.list_collections(scope="scope", limit=1, continuation_token=None)
    with pytest.raises(RuntimeError, match="Collection catalog initialization failed"):
        backend.health()

    db.drop_table("_nrl_collections")
    assert backend.list_collections(scope="scope", limit=1, continuation_token=None).items == []
    assert backend.health()["catalog"]["initialized"] is True


def test_catalog_startup_fails_fast_on_incompatible_schema(tmp_path) -> None:
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "_nrl_collections",
        schema=pa.schema(
            [
                pa.field("scope", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="Incompatible"):
        _vdb(tmp_path).list_collections(
            scope="scope",
            limit=1,
            continuation_token=None,
        )


def test_catalog_startup_does_not_recreate_an_unreadable_existing_table(
    tmp_path,
) -> None:
    backend = _vdb(tmp_path)
    backend.list_collections(scope="scope", limit=1, continuation_token=None)
    store = backend._get_collection_store()
    db = Mock(wraps=store._db)
    db.list_tables.return_value = store._db.list_tables()
    db.open_table.side_effect = RuntimeError("catalog unreadable")
    store._db = db

    with pytest.raises(RuntimeError, match="catalog unreadable"):
        store._ensure_catalogs()

    db.create_table.assert_not_called()
