# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for LanceDB's optional collection capabilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import hashlib
import json
import math
import threading
from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock

import lancedb
import pytest

import nemo_retriever.common.vdb.lancedb_collections as collections_module
from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionUpdateRequest,
)
from nemo_retriever.common.schemas.embedding import embedding_split_metadata
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    UnsupportedVDBOperation,
    VDBInvalidRequest,
    VDBResourceConflict,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.lancedb_capabilities import LanceTableCapabilities
from nemo_retriever.common.vdb.lancedb import (
    LanceDB,
    _create_lancedb_results,
    _to_service_lancedb_rows,
)
from nemo_retriever.common.vdb.lancedb_collections import (
    LanceDBCollectionStore,
    _collection_rows,
    _encode_cursor,
    _normalize_collection_results,
    _public_collection_hit,
)
from nemo_retriever.common.vdb.records import RetrievalContractError, to_client_vdb_records


def test_catalog_scans_reuse_open_table_handle() -> None:
    table = MagicMock()
    table.search.return_value.limit.return_value.to_list.return_value = []
    database = MagicMock()
    database.open_table.return_value = table
    store = object.__new__(LanceDBCollectionStore)
    store._db = database
    store._opened_tables = {}

    assert store._rows("_nrl_collections") == []
    assert store._rows("_nrl_collections") == []

    database.open_table.assert_called_once_with("_nrl_collections")


def _context(
    *,
    version: str = "v1",
    operation: str = "append",
    document_id: str = "document-a",
) -> CollectionWriteContext:
    return CollectionWriteContext(
        scope="workspace-a",
        collection_name="collection-a",
        document_id=document_id,
        document_version=version,
        content_sha256=f"sha-{version}",
        filename="source.pdf",
        job_id="job-a",
        operation=operation,
    )


def _records(
    *,
    text: str = "first chunk",
    vector: list[float] | None = None,
    page_number: int = 2,
) -> list[list[dict]]:
    return [
        [
            {
                "document_type": "text",
                "embedding": vector or [1.0, 0.0],
                "metadata": {
                    "content": text,
                    "content_metadata": {
                        "type": "text",
                        "page_number": page_number,
                        "stored_image_uri": "file:///artifacts/page-2.png",
                        "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                    },
                    "source_metadata": {
                        "source_id": "/inputs/source.pdf",
                        "source_name": "source.pdf",
                    },
                },
            }
        ]
    ]


def _service_records(
    content_type: str,
    *,
    text: str,
    vector: list[float],
) -> list[list[dict]]:
    records = _records(text=text, vector=vector, page_number=4)
    metadata = records[0][0]["metadata"]
    metadata["content_metadata"].update(
        {
            "type": content_type,
            "stored_image_uri": "s3://bucket/figure.png",
            "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
        }
    )
    metadata["source_metadata"] = {
        "source_id": "/inputs/report.pdf",
        "source_name": "report.pdf",
        "custom_source_field": "preserved",
    }
    return records


def _backend_with_collection(tmp_path) -> LanceDB:
    backend = LanceDB(
        uri=str(tmp_path / "lancedb"),
        table_name="legacy",
        vector_dim=2,
        build_index=False,
    )
    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="collection-a"),
    )
    return backend


def _fail_document_finalize(monkeypatch, store):
    original_persist = store._persist_document_row

    def fail_completed(row):
        if row.get("status") == "completed":
            raise RuntimeError("injected catalog finalize failure")
        return original_persist(row)

    monkeypatch.setattr(store, "_persist_document_row", fail_completed)
    return original_persist


@pytest.mark.parametrize("writer", ["ordinary", "service", "collection"])
def test_split_children_survive_adapter_write_and_retrieval(tmp_path, writer):
    pieces = ["alpha", " \n ", "omega"]
    graph_rows = []
    start = 0
    for index, text in enumerate(pieces):
        end = start + len(text)
        graph_rows.append(
            {
                "text": text,
                "path": "/inputs/source.pdf",
                "page_number": 2,
                "embedding": [1.0, float(index)],
                "embedding_error": None,
                "metadata": {
                    **embedding_split_metadata(
                        content=text,
                        parent_id="parent",
                        chunk_id=f"child-{index}",
                        chunk_index=index,
                        chunk_count=len(pieces),
                        start_token=start,
                        end_token=end,
                    ),
                },
            }
        )
        start = end
    records = to_client_vdb_records(graph_rows)
    # Ordinary blank records must still be filtered; only valid split children are exempt.
    records[0].extend(_records(text=" ")[0])
    original = deepcopy(records)
    if writer == "collection":
        backend = _backend_with_collection(tmp_path)
        written = backend.write_collection(records, context=_context())
        assert written.written == len(pieces)
        hits, _ = backend.retrieve_collection(
            [[1.0, 0.0]],
            scope="workspace-a",
            collection_name="collection-a",
            query_texts=[""],
            top_k=10,
        )
        assert len(hits[0]) == len(pieces)
        for hit in hits[0]:
            split = hit["metadata"]["embedding_split"]
            assert hit["text"] == pieces[split["chunk_index"]]
            assert hit["source_id"] == "/inputs/source.pdf"
        table_name = backend._get_collection_store()._resolved_table("workspace-a", "collection-a")
    else:
        backend = LanceDB(
            uri=str(tmp_path / "lancedb"),
            table_name="children",
            vector_dim=2,
            build_index=False,
            _service_table_schema=writer == "service",
        )
        backend.run(records)
        table_name = "children"
    stored = lancedb.connect(backend.uri).open_table(table_name).search([1.0, 0.0]).limit(10).to_list()
    for row in stored:
        row["metadata"] = json.loads(row["metadata"])
        row["source"] = json.loads(row["source"])
    assert len(stored) == len(pieces)
    stored.sort(key=lambda row: row["metadata"]["embedding_split"]["chunk_index"])
    assert [row["text"] for row in stored] == pieces
    assert "".join(row["text"] for row in stored) == "alpha \n omega"
    for index, row in enumerate(stored):
        metadata = row["metadata"]
        assert metadata["embedding_split"] == records[0][index]["metadata"]["embedding_split"]
        assert metadata["page_number"] == 2
        assert row["source"]["source_id"] == "/inputs/source.pdf"
        assert row["vector"] == [1.0, float(index)]
        if writer == "collection":
            assert row["chunk_id"] == hashlib.sha256(f"document-a\0v1\0{index}".encode()).hexdigest()
    assert records == original


def test_collection_row_conversion_preserves_identity_and_provenance():
    records = _records()
    records[0][0]["metadata"]["content_metadata"]["type"] = "table_caption"
    rows = _collection_rows(records, context=_context())

    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "first chunk"
    assert row["filename"] == "source.pdf"
    assert row["page_number"] == 2
    assert row["pdf_page"] == "source_2"
    assert row["source_id"] == "/inputs/source.pdf"
    assert json.loads(row["source"])["source_name"] == "source.pdf"
    assert row["content_type"] == "table"
    assert json.loads(row["metadata"])["type"] == "table"
    assert json.loads(row["bbox_xyxy_norm"]) == [0.1, 0.2, 0.8, 0.9]
    assert row["stored_image_uri"] == "file:///artifacts/page-2.png"
    assert row["document_id"] == "document-a"
    assert row["document_version"] == "v1"
    assert row["content_sha256"] == "sha-v1"
    assert row["chunk_id"] == hashlib.sha256(b"document-a\x00v1\x000").hexdigest()


@pytest.mark.parametrize(
    ("raw_type", "expected_type"),
    [("table_caption", "table"), ("chart_caption", "chart")],
)
def test_service_row_adapter_preserves_multimodal_provenance(raw_type, expected_type):
    narrow_rows, counts = _create_lancedb_results(
        _service_records(raw_type, text="caption", vector=[1.0, 0.0, 0.0]),
        expected_dim=None,
    )

    rows = _to_service_lancedb_rows(narrow_rows)

    assert counts["accepted"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["content_type"] == expected_type
    assert row["filename"] == "report.pdf"
    assert row["page_number"] == 4
    assert row["pdf_page"] == "report_4"
    assert row["source_id"] == "/inputs/report.pdf"
    assert json.loads(row["source"])["custom_source_field"] == "preserved"
    assert json.loads(row["metadata"])["type"] == expected_type
    assert row["stored_image_uri"] == "s3://bucket/figure.png"
    assert json.loads(row["bbox_xyxy_norm"]) == [0.1, 0.2, 0.8, 0.9]


def test_collection_hit_preserves_native_dense_distance():
    hit = {
        "text": "chunk",
        "_score": 42.0,
        "_distance": 0.125,
    }

    public = _public_collection_hit(hit)

    assert public["distance"] == 0.125
    assert public["text"] == "chunk"
    assert not {"_score", "_distance"} & public.keys()


@pytest.mark.parametrize(
    ("content_type", "page_number"),
    [("audio", 3), ("video", 3), ("video_frame", 3), ("text", -1)],
)
def test_collection_hit_does_not_expose_non_document_pages(content_type: str, page_number: int):
    public = _public_collection_hit(
        {
            "text": "chunk",
            "content_type": content_type,
            "page_number": page_number,
            "pdf_page": "document_3",
            "_distance": 0.125,
        }
    )

    assert public["page_number"] is None
    assert public["pdf_page"] == ""


@pytest.mark.parametrize("bad_value", [None, True, math.nan, math.inf, -math.inf, "not-a-number"])
def test_collection_hit_rejects_missing_or_invalid_native_distance(bad_value):
    with pytest.raises(RetrievalContractError):
        _public_collection_hit({"text": "chunk", "_distance": bad_value})


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("hybrid", UnsupportedVDBOperation),
        ("sparse", UnsupportedVDBOperation),
        ("unknown", RetrievalContractError),
    ],
)
def test_collection_retrieval_mode_error_classification(mode, expected_error):
    store = object.__new__(collections_module.LanceDBCollectionStore)
    capabilities = LanceTableCapabilities(
        has_vector=mode in {"dense", "hybrid"},
        has_fts=mode in {"hybrid", "sparse"},
        retrieval_mode=mode,
        vector_column="vector" if mode in {"dense", "hybrid"} else None,
        text_column="text" if mode in {"hybrid", "sparse"} else None,
    )

    with pytest.raises(expected_error):
        store._resolve_effective_retrieval_mode("collection-table", capabilities)


def test_collection_write_enforces_append_and_replace_invariants(tmp_path):
    backend = _backend_with_collection(tmp_path)

    with pytest.raises(VDBResourceNotFound, match="Document not found"):
        backend.write_collection(_records(), context=_context(operation="replace"))

    backend.write_collection(_records(), context=_context())
    with pytest.raises(VDBResourceConflict, match="use replace"):
        backend.write_collection(_records(text="new version"), context=_context(version="v2"))
    with pytest.raises(VDBResourceConflict, match="content does not match"):
        backend.write_collection(
            _records(text="different content"),
            context=replace(_context(), content_sha256="different-sha"),
        )

    document = backend.get_document(
        scope="workspace-a",
        collection_name="collection-a",
        document_id="document-a",
    )
    assert document.document_version == "v1"
    store = backend._get_collection_store()
    table_name = store._resolved_table("workspace-a", "collection-a")
    assert store._open_table(table_name).count_rows() == 1


def test_collection_writes_refresh_sliding_expiration(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    collection = store._collection_row("workspace-a", "collection-a")
    assert collection is not None
    collection.update(
        {
            "updated_at": "2030-01-01T00:00:00+00:00",
            "expires_at": "2030-01-02T00:00:00+00:00",
        }
    )
    store._persist_collection_row(collection)

    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T12:00:00+00:00")
    backend.write_collection(_records(), context=_context())
    appended = backend.get_collection(scope="workspace-a", collection_name="collection-a")
    assert appended.updated_at == "2030-01-01T12:00:00+00:00"
    assert appended.expires_at == "2030-01-02T12:00:00+00:00"

    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T18:00:00+00:00")
    backend.write_collection(
        _records(text="replacement"),
        context=_context(version="v2", operation="replace"),
    )
    replaced = backend.get_collection(scope="workspace-a", collection_name="collection-a")
    assert replaced.updated_at == "2030-01-01T18:00:00+00:00"
    assert replaced.expires_at == "2030-01-02T18:00:00+00:00"


def test_collection_activity_without_expiration_only_updates_timestamp(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T12:00:00+00:00")

    backend.write_collection(_records(), context=_context())

    collection = backend.get_collection(scope="workspace-a", collection_name="collection-a")
    assert collection.updated_at == "2030-01-01T12:00:00+00:00"
    assert collection.expires_at is None


def test_failed_or_empty_collection_write_does_not_refresh_expiration(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    collection = store._collection_row("workspace-a", "collection-a")
    assert collection is not None
    collection.update(
        {
            "updated_at": "2030-01-01T00:00:00+00:00",
            "expires_at": "2030-01-02T00:00:00+00:00",
        }
    )
    store._persist_collection_row(collection)
    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T12:00:00+00:00")

    with pytest.raises(VDBInvalidRequest, match="no writable vector rows"):
        backend.write_collection(
            [[{"document_type": "text", "metadata": {"content": "no vector"}}]],
            context=_context(),
        )
    empty = backend.write_collection([], context=_context())
    assert empty.written == 0

    unchanged = backend.get_collection(scope="workspace-a", collection_name="collection-a")
    assert unchanged.updated_at == "2030-01-01T00:00:00+00:00"
    assert unchanged.expires_at == "2030-01-02T00:00:00+00:00"


def test_collection_updates_preserve_replace_or_clear_expiration_window(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    collection = store._collection_row("workspace-a", "collection-a")
    assert collection is not None
    collection.update(
        {
            "updated_at": "2030-01-01T00:00:00+00:00",
            "expires_at": "2030-01-02T00:00:00+00:00",
        }
    )
    store._persist_collection_row(collection)

    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T12:00:00+00:00")
    preserved = backend.update_collection(
        scope="workspace-a",
        collection_name="collection-a",
        request=CollectionUpdateRequest(description="active"),
    )
    assert preserved.updated_at == "2030-01-01T12:00:00+00:00"
    assert preserved.expires_at == "2030-01-02T12:00:00+00:00"

    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T13:00:00+00:00")
    replaced = backend.update_collection(
        scope="workspace-a",
        collection_name="collection-a",
        request=CollectionUpdateRequest(expires_at="2030-01-04T13:00:00+00:00"),
    )
    assert replaced.updated_at == "2030-01-01T13:00:00+00:00"
    assert replaced.expires_at == "2030-01-04T13:00:00+00:00"

    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T14:00:00+00:00")
    cleared = backend.update_collection(
        scope="workspace-a",
        collection_name="collection-a",
        request=CollectionUpdateRequest(expires_at=None),
    )
    assert cleared.updated_at == "2030-01-01T14:00:00+00:00"
    assert cleared.expires_at is None


def test_collection_table_records_and_validates_embedding_model(tmp_path):
    uri = str(tmp_path / "lancedb")
    model = "nvidia/nemotron-3-embed-1b"
    backend = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        build_index=False,
        embedding_model_name=model,
    )
    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="collection-a"),
    )
    backend.write_collection(_records(), context=_context())
    physical_table = backend._get_collection_store()._resolved_table("workspace-a", "collection-a")
    table = lancedb.connect(uri).open_table(physical_table)
    schema = table.schema() if callable(table.schema) else table.schema

    assert schema.metadata[b"nemo_retriever.embedding_model_name"] == model.encode()

    restarted = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        build_index=False,
        embedding_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    with pytest.raises(VDBInvalidRequest, match="uses embedding model.*nemotron-3"):
        restarted.retrieve_collection(
            [[1.0, 0.0]],
            scope="workspace-a",
            collection_name="collection-a",
            query_texts=["query"],
            top_k=1,
        )


def test_collection_query_rejects_untagged_existing_table(tmp_path):
    uri = str(tmp_path / "lancedb")
    backend = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        build_index=False,
    )
    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="collection-a"),
    )
    backend.write_collection(_records(), context=_context())

    restarted = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        build_index=False,
        embedding_model_name="nvidia/nemotron-3-embed-1b",
    )
    with pytest.raises(VDBInvalidRequest, match="does not record its embedding model"):
        restarted.retrieve_collection(
            [[1.0, 0.0]],
            scope="workspace-a",
            collection_name="collection-a",
            query_texts=["query"],
            top_k=1,
        )


def test_collection_lifecycle_is_lazy_and_restart_safe(tmp_path):
    uri = str(tmp_path / "lancedb")
    backend = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        build_index=False,
    )

    assert backend._collection_store is None
    assert lancedb.connect(uri).list_tables().tables == []
    assert backend.health()["catalog"]["initialized"] is False
    assert backend._collection_store is None
    assert lancedb.connect(uri).list_tables().tables == []

    created = backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="collection-a"),
    )
    assert created.name == "collection-a"
    for invalid_last in ([], ["a", "b"]):
        with pytest.raises(VDBInvalidRequest, match="continuation token"):
            backend.list_collections(
                scope="workspace-a",
                limit=10,
                continuation_token=_encode_cursor("collections", "workspace-a", None, invalid_last),
            )
    expected_table = "nrl_" + hashlib.sha256(b"workspace-a\x00collection-a").hexdigest()[:40]
    assert {"_nrl_collections", "_nrl_documents"} <= set(lancedb.connect(uri).list_tables().tables)

    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="state-test"),
    )
    store = backend._get_collection_store()
    state_row = store._collection_row("workspace-a", "state-test")
    assert state_row is not None
    state_row["status"] = "deleting"
    store._persist_collection_row(state_row)
    with pytest.raises(VDBInvalidRequest, match="deleting"):
        backend.update_collection(
            scope="workspace-a",
            collection_name="state-test",
            request=CollectionUpdateRequest(description="blocked"),
        )
    state_row["status"] = "active"
    state_row["expires_at"] = "2000-01-01T00:00:00+00:00"
    store._persist_collection_row(state_row)
    with pytest.raises(VDBInvalidRequest, match="expired"):
        backend.update_collection(
            scope="workspace-a",
            collection_name="state-test",
            request=CollectionUpdateRequest(description="blocked"),
        )
    state_row["expires_at"] = ""
    store._persist_collection_row(state_row)
    backend.delete_collection(
        scope="workspace-a",
        collection_name="state-test",
        if_exists=False,
    )

    with pytest.raises(VDBInvalidRequest):
        backend.write_collection(
            [[{"document_type": "text", "metadata": {"content": "no vector"}}]],
            context=_context(),
        )

    result = backend.write_collection(_records(), context=_context())
    assert result.written == 1
    assert result.total_rows == 1

    document_row = store._document_rows("workspace-a", "collection-a", "document-a")[0]
    document_row.update(
        {
            "content_sha256": "stale-hash",
            "pending_document_version": "v1",
            "recovery_state": "replacing",
        }
    )
    store._persist_document_row(document_row)
    with store._write_lock:
        assert store._reconcile_document_row_locked(document_row, expected_table)
    assert (
        backend.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        ).content_sha256
        == "sha-v1"
    )
    assert expected_table in lancedb.connect(uri).list_tables().tables

    hits, strategies = backend.retrieve_collection(
        [[1.0, 0.0]],
        scope="workspace-a",
        collection_name="collection-a",
        query_texts=["first"],
        top_k=1,
    )
    assert strategies == ["dense"]
    assert hits[0][0]["document_id"] == "document-a"
    assert hits[0][0]["distance"] >= 0.0
    assert "_distance" not in hits[0][0]

    restarted = LanceDB(uri=uri, table_name="legacy", vector_dim=2, build_index=False)
    assert restarted._collection_store is None
    assert (
        restarted.get_collection(
            scope="workspace-a",
            collection_name="collection-a",
        ).name
        == "collection-a"
    )
    assert (
        restarted.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        ).chunk_count
        == 1
    )

    replacement = restarted.write_collection(
        _records(text="replacement", vector=[0.0, 1.0], page_number=3),
        context=_context(version="v2", operation="replace"),
    )
    assert replacement.written == 1
    assert replacement.total_rows == 1
    assert (
        restarted.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        ).document_version
        == "v2"
    )

    deleted_document = restarted.delete_document(
        scope="workspace-a",
        collection_name="collection-a",
        document_id="document-a",
        if_exists=False,
    )
    assert deleted_document.deleted is True

    deleted_collection = restarted.delete_collection(
        scope="workspace-a",
        collection_name="collection-a",
        if_exists=False,
    )
    assert deleted_collection.deleted is True
    assert expected_table not in lancedb.connect(uri).list_tables().tables


def test_initial_append_reconciles_after_catalog_finalize_failure(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    collection = store._collection_row("workspace-a", "collection-a")
    assert collection is not None
    collection.update(
        {
            "updated_at": "2030-01-01T00:00:00+00:00",
            "expires_at": "2030-01-02T00:00:00+00:00",
        }
    )
    store._persist_collection_row(collection)
    monkeypatch.setattr(collections_module, "_now", lambda: "2030-01-01T12:00:00+00:00")
    _fail_document_finalize(monkeypatch, store)

    with pytest.raises(RuntimeError, match="injected catalog finalize failure"):
        backend.write_collection(_records(), context=_context())

    marker = store._document_rows("workspace-a", "collection-a", "document-a")[0]
    assert marker["recovery_state"] == "appending"
    assert marker["pending_document_version"] == "v1"
    with pytest.raises(VDBResourceNotFound):
        backend.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        )
    assert (
        backend.list_documents(
            scope="workspace-a",
            collection_name="collection-a",
            limit=10,
            continuation_token=None,
        ).items
        == []
    )

    pending_hits, strategies = backend.retrieve_collection(
        [[1.0, 0.0]],
        scope="workspace-a",
        collection_name="collection-a",
        query_texts=["first"],
        top_k=1,
    )
    assert strategies == ["dense"]
    assert pending_hits == [[]]
    table_name = store._resolved_table("workspace-a", "collection-a")
    assert store._open_table(table_name).count_rows() == 1

    restarted = LanceDB(
        uri=str(tmp_path / "lancedb"),
        table_name="legacy",
        vector_dim=2,
        build_index=False,
    )
    assert restarted.reconcile_collections() == {"successes": 1, "failures": 0}
    document = restarted.get_document(
        scope="workspace-a",
        collection_name="collection-a",
        document_id="document-a",
    )
    assert document.status == "completed"
    assert document.document_version == "v1"
    assert document.chunk_count == 1
    collection = restarted.get_collection(scope="workspace-a", collection_name="collection-a")
    assert collection.updated_at == "2030-01-01T12:00:00+00:00"
    assert collection.expires_at == "2030-01-02T12:00:00+00:00"
    assert restarted._get_collection_store()._open_table(table_name).count_rows() == 1
    visible_hits, strategies = restarted.retrieve_collection(
        [[1.0, 0.0]],
        scope="workspace-a",
        collection_name="collection-a",
        query_texts=["first"],
        top_k=1,
    )
    assert strategies == ["dense"]
    assert visible_hits[0][0]["document_id"] == "document-a"


def test_recovery_retries_activity_refresh_without_reextending_expiration(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    backend.write_collection(_records(), context=_context())
    store = backend._get_collection_store()
    collection = store._collection_row("workspace-a", "collection-a")
    assert collection is not None
    collection.update(
        {
            "updated_at": "2030-01-01T00:00:00+00:00",
            "expires_at": "2030-01-02T00:00:00+00:00",
        }
    )
    store._persist_collection_row(collection)
    document = store._document_rows("workspace-a", "collection-a", "document-a")[0]
    document.update(
        {
            "pending_document_version": "v1",
            "recovery_state": "appending",
            "status": "appending",
        }
    )
    store._persist_document_row(document)
    activity_at = "2030-01-01T12:00:00+00:00"
    monkeypatch.setattr(collections_module, "_now", lambda: activity_at)

    refreshes = 0
    original_persist_collection = store._persist_collection_row

    def count_activity_refresh(row):
        nonlocal refreshes
        refreshes += 1
        return original_persist_collection(row)

    monkeypatch.setattr(store, "_persist_collection_row", count_activity_refresh)
    original_persist_document = store._persist_document_row
    marker_clear_failed = False

    def fail_first_marker_clear(row):
        nonlocal marker_clear_failed
        if not marker_clear_failed and row.get("recovery_state") == "" and row.get("updated_at") == activity_at:
            marker_clear_failed = True
            raise RuntimeError("injected marker clear failure")
        return original_persist_document(row)

    monkeypatch.setattr(store, "_persist_document_row", fail_first_marker_clear)

    assert backend.reconcile_collections() == {"successes": 0, "failures": 1}
    marker = store._document_rows("workspace-a", "collection-a", "document-a")[0]
    assert marker["recovery_state"] == "refreshing_collection_activity"
    refreshed = backend.get_collection(scope="workspace-a", collection_name="collection-a")
    assert refreshed.updated_at == activity_at
    assert refreshed.expires_at == "2030-01-02T12:00:00+00:00"

    assert backend.reconcile_collections() == {"successes": 1, "failures": 0}
    assert refreshes == 1
    assert store._document_rows("workspace-a", "collection-a", "document-a")[0]["recovery_state"] == ""


def test_reconciliation_filters_recoverable_documents_before_scan_limit(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    backend.write_collection(
        _records(text="completed", vector=[1.0, 0.0]),
        context=_context(document_id="document-completed"),
    )
    store = backend._get_collection_store()
    original_persist = _fail_document_finalize(monkeypatch, store)
    with pytest.raises(RuntimeError, match="injected catalog finalize failure"):
        backend.write_collection(
            _records(text="pending", vector=[0.0, 1.0]),
            context=_context(document_id="document-pending"),
        )
    monkeypatch.setattr(store, "_persist_document_row", original_persist)
    monkeypatch.setattr(collections_module, "_CATALOG_SCAN_LIMIT", 1)

    assert backend.reconcile_collections() == {"successes": 1, "failures": 0}
    assert (
        backend.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-pending",
        ).status
        == "completed"
    )


def test_reconciliation_filters_expired_collections_before_scan_limit(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(
            name="collection-expired",
            expires_at="2000-01-01T00:00:00Z",
        ),
    )
    monkeypatch.setattr(collections_module, "_CATALOG_SCAN_LIMIT", 1)

    assert backend.reconcile_collections() == {"successes": 1, "failures": 0}
    with pytest.raises(VDBResourceNotFound):
        backend.get_collection(scope="workspace-a", collection_name="collection-expired")


def test_pending_initial_append_does_not_hide_completed_documents(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    backend.write_collection(
        _records(text="completed", vector=[1.0, 0.0]),
        context=_context(document_id="document-completed"),
    )
    store = backend._get_collection_store()
    _fail_document_finalize(monkeypatch, store)

    with pytest.raises(RuntimeError, match="injected catalog finalize failure"):
        backend.write_collection(
            _records(text="pending", vector=[0.0, 1.0]),
            context=_context(document_id="document-pending"),
        )

    hits, strategies = backend.retrieve_collection(
        [[1.0, 0.0]],
        scope="workspace-a",
        collection_name="collection-a",
        query_texts=["completed"],
        top_k=10,
    )

    assert strategies == ["dense"]
    assert [hit["document_id"] for hit in hits[0]] == ["document-completed"]


def test_initial_append_retry_does_not_duplicate_committed_chunks(tmp_path, monkeypatch):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    original_persist = _fail_document_finalize(monkeypatch, store)

    with pytest.raises(RuntimeError, match="injected catalog finalize failure"):
        backend.write_collection(_records(), context=_context())

    table_name = store._resolved_table("workspace-a", "collection-a")
    assert store._open_table(table_name).count_rows() == 1
    monkeypatch.setattr(store, "_persist_document_row", original_persist)

    result = backend.write_collection(_records(), context=_context())
    assert result.written == 1
    assert result.total_rows == 1
    assert store._open_table(table_name).count_rows() == 1
    assert (
        backend.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        ).chunk_count
        == 1
    )


def test_initial_append_marker_without_chunks_is_removed_by_reconciliation(
    tmp_path,
    monkeypatch,
):
    backend = _backend_with_collection(tmp_path)
    store = backend._get_collection_store()
    original_write = collections_module.create_or_append_lancedb_table

    def fail_before_chunk_write(*args, **kwargs):
        raise RuntimeError("injected chunk write failure")

    monkeypatch.setattr(
        collections_module,
        "create_or_append_lancedb_table",
        fail_before_chunk_write,
    )
    with pytest.raises(RuntimeError, match="injected chunk write failure"):
        backend.write_collection(_records(), context=_context())

    marker = store._document_rows("workspace-a", "collection-a", "document-a")[0]
    assert marker["recovery_state"] == "appending"
    with pytest.raises(VDBResourceNotFound):
        backend.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        )
    monkeypatch.setattr(
        collections_module,
        "create_or_append_lancedb_table",
        original_write,
    )

    assert backend.reconcile_collections() == {"successes": 1, "failures": 0}
    assert store._document_rows("workspace-a", "collection-a", "document-a") == []


@pytest.mark.parametrize(
    "raw_results",
    [None, {}, [[], []], [{}], [[object()]]],
)
def test_collection_results_reject_invalid_cardinality_or_hit_shapes(raw_results):
    with pytest.raises(RetrievalContractError):
        _normalize_collection_results(raw_results, expected_queries=1)


def test_legacy_table_can_explicitly_infer_vector_dimension(tmp_path):
    uri = str(tmp_path / "inferred-lancedb")
    backend = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=None,
        overwrite=False,
        build_index=False,
    )

    backend.run(_records(vector=[1.0, 0.0, 0.0]))

    table = lancedb.connect(uri).open_table("legacy")
    assert table.schema.field("vector").type.list_size == 3
    assert table.schema.names == ["vector", "text", "metadata", "source", "id"]
    assert table.count_rows() == 1
    assert LanceDB(uri=str(tmp_path / "default")).vector_dim == 2048


def test_service_table_schema_survives_append_restart_and_query(tmp_path):
    uri = str(tmp_path / "service-lancedb")
    common = {
        "uri": uri,
        "table_name": "legacy",
        "vector_dim": None,
        "overwrite": False,
        "build_index": False,
        "_service_table_schema": True,
    }
    backend = LanceDB(**common)
    backend.run(_service_records("table_caption", text="table caption", vector=[1.0, 0.0, 0.0]))

    restarted = LanceDB(**common)
    restarted.run(_service_records("chart_caption", text="chart caption", vector=[0.0, 1.0, 0.0]))

    table = lancedb.connect(uri).open_table("legacy")
    assert table.schema.field("vector").type.list_size == 3
    assert {"content_type", "stored_image_uri", "bbox_xyxy_norm"} <= set(table.schema.names)
    assert table.count_rows() == 2

    results = restarted.retrieval([[1.0, 0.0, 0.0]], top_k=2)
    hits = {hit["text"]: hit for hit in results[0]}
    for text, content_type in (("table caption", "table"), ("chart caption", "chart")):
        hit = hits[text]
        assert hit["content_type"] == content_type
        assert hit["filename"] == "report.pdf"
        assert hit["page_number"] == 4
        assert hit["pdf_page"] == "report_4"
        assert hit["source_id"] == "/inputs/report.pdf"
        assert json.loads(hit["source"])["custom_source_field"] == "preserved"
        assert json.loads(hit["metadata"])["type"] == content_type
        assert hit["stored_image_uri"] == "s3://bucket/figure.png"
        assert json.loads(hit["bbox_xyxy_norm"]) == [0.1, 0.2, 0.8, 0.9]


def test_collection_query_is_not_blocked_by_unrelated_write(tmp_path, monkeypatch):
    backend = LanceDB(
        uri=str(tmp_path / "query-during-write"),
        table_name="legacy",
        vector_dim=2,
        overwrite=False,
        build_index=False,
    )
    for name in ("collection-a", "collection-b"):
        backend.create_collection(scope="workspace-a", request=CollectionCreateRequest(name=name))
    backend.write_collection(
        _records(text="collection b"),
        context=replace(_context(), collection_name="collection-b", document_id="document-b"),
    )

    write_entered = threading.Event()
    allow_write_to_finish = threading.Event()
    query_entered = threading.Event()
    original_create = collections_module.create_or_append_lancedb_table

    def blocking_create(*args, **kwargs):
        write_entered.set()
        if not allow_write_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release blocked collection write")
        return original_create(*args, **kwargs)

    def immediate_retrieval(vectors, **kwargs):
        query_entered.set()
        return [[] for _ in vectors]

    monkeypatch.setattr(collections_module, "create_or_append_lancedb_table", blocking_create)
    monkeypatch.setattr(backend, "retrieval", immediate_retrieval)

    with ThreadPoolExecutor(max_workers=2) as pool:
        write_future = pool.submit(backend.write_collection, _records(), context=_context())
        assert write_entered.wait(timeout=5)
        query_future = pool.submit(
            backend.retrieve_collection,
            [[1.0, 0.0]],
            scope="workspace-a",
            collection_name="collection-b",
            query_texts=["query"],
            top_k=1,
        )
        try:
            assert query_entered.wait(timeout=1)
            query_future.result(timeout=1)
            assert not write_future.done()
        finally:
            allow_write_to_finish.set()
        write_future.result(timeout=5)


def test_collection_delete_waits_for_active_write(tmp_path, monkeypatch):
    backend = LanceDB(
        uri=str(tmp_path / "delete-during-write"),
        table_name="legacy",
        vector_dim=2,
        overwrite=False,
        build_index=False,
    )
    backend.create_collection(scope="workspace-a", request=CollectionCreateRequest(name="collection-a"))

    write_entered = threading.Event()
    allow_write_to_finish = threading.Event()
    original_create = collections_module.create_or_append_lancedb_table

    def blocking_create(*args, **kwargs):
        write_entered.set()
        if not allow_write_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release blocked collection write")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(collections_module, "create_or_append_lancedb_table", blocking_create)

    with ThreadPoolExecutor(max_workers=2) as pool:
        write_future = pool.submit(backend.write_collection, _records(), context=_context())
        assert write_entered.wait(timeout=5)
        delete_future = pool.submit(
            backend.delete_collection,
            scope="workspace-a",
            collection_name="collection-a",
            if_exists=False,
        )
        try:
            with pytest.raises(TimeoutError):
                delete_future.result(timeout=0.2)
        finally:
            allow_write_to_finish.set()
        write_future.result(timeout=5)
        result = delete_future.result(timeout=5)

    assert result.deleted is True


def test_collection_reconciliation_waits_for_active_write(tmp_path, monkeypatch):
    backend = LanceDB(
        uri=str(tmp_path / "reconcile-during-write"),
        table_name="legacy",
        vector_dim=2,
        overwrite=False,
        build_index=False,
    )
    backend.create_collection(scope="workspace-a", request=CollectionCreateRequest(name="collection-a"))

    write_entered = threading.Event()
    allow_write_to_finish = threading.Event()
    original_create = collections_module.create_or_append_lancedb_table

    def blocking_create(*args, **kwargs):
        write_entered.set()
        if not allow_write_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release blocked collection write")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(collections_module, "create_or_append_lancedb_table", blocking_create)

    with ThreadPoolExecutor(max_workers=2) as pool:
        write_future = pool.submit(backend.write_collection, _records(), context=_context())
        assert write_entered.wait(timeout=5)
        reconcile_future = pool.submit(backend.reconcile_collections)
        try:
            with pytest.raises(TimeoutError):
                reconcile_future.result(timeout=0.2)
        finally:
            allow_write_to_finish.set()
        write_future.result(timeout=5)
        result = reconcile_future.result(timeout=5)

    assert result == {"successes": 0, "failures": 0}


def test_collection_writes_remain_serialized(tmp_path, monkeypatch):
    backend = LanceDB(
        uri=str(tmp_path / "serialized-writes"),
        table_name="legacy",
        vector_dim=2,
        overwrite=False,
        build_index=False,
    )
    for name in ("collection-a", "collection-b"):
        backend.create_collection(scope="workspace-a", request=CollectionCreateRequest(name=name))

    first_write_entered = threading.Event()
    second_write_entered = threading.Event()
    allow_first_write_to_finish = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    original_create = collections_module.create_or_append_lancedb_table

    def blocking_first_create(*args, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_write_entered.set()
            if not allow_first_write_to_finish.wait(timeout=5):
                raise TimeoutError("test did not release first collection write")
        else:
            second_write_entered.set()
        return original_create(*args, **kwargs)

    monkeypatch.setattr(collections_module, "create_or_append_lancedb_table", blocking_first_create)

    second_context = replace(_context(), collection_name="collection-b", document_id="document-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(backend.write_collection, _records(), context=_context())
        assert first_write_entered.wait(timeout=5)
        second_future = pool.submit(backend.write_collection, _records(), context=second_context)
        try:
            assert not second_write_entered.wait(timeout=0.2)
            assert not second_future.done()
        finally:
            allow_first_write_to_finish.set()
        first_future.result(timeout=5)
        second_future.result(timeout=5)

    assert second_write_entered.is_set()


def test_document_delete_waits_for_active_collection_query(tmp_path, monkeypatch):
    uri = str(tmp_path / "concurrent-lancedb")
    backend = LanceDB(
        uri=uri,
        table_name="legacy",
        vector_dim=2,
        overwrite=False,
        build_index=False,
    )
    backend.create_collection(
        scope="workspace-a",
        request=CollectionCreateRequest(name="collection-a"),
    )
    backend.write_collection(_records(), context=_context())

    query_entered = threading.Event()
    allow_query_to_finish = threading.Event()
    query_finished = threading.Event()
    delete_finished = threading.Event()
    query_errors: list[BaseException] = []
    delete_errors: list[BaseException] = []

    def blocking_retrieval(vectors, **kwargs):
        query_entered.set()
        if not allow_query_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release blocked collection query")
        return [[] for _ in vectors]

    monkeypatch.setattr(backend, "retrieval", blocking_retrieval)

    def query_target():
        try:
            backend.retrieve_collection(
                [[1.0, 0.0]],
                scope="workspace-a",
                collection_name="collection-a",
                query_texts=["query"],
                top_k=1,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            query_errors.append(exc)
        finally:
            query_finished.set()

    def delete_target():
        try:
            backend.delete_document(
                scope="workspace-a",
                collection_name="collection-a",
                document_id="document-a",
                if_exists=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            delete_errors.append(exc)
        finally:
            delete_finished.set()

    query_thread = threading.Thread(target=query_target)
    delete_thread = threading.Thread(target=delete_target)
    query_thread.start()
    assert query_entered.wait(timeout=5)
    delete_thread.start()
    try:
        assert not delete_finished.wait(timeout=0.2)
    finally:
        allow_query_to_finish.set()
        query_thread.join(timeout=5)
        delete_thread.join(timeout=5)

    assert query_finished.is_set()
    assert delete_finished.is_set()
    assert query_errors == []
    assert delete_errors == []
