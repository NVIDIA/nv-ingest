# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lancedb
import pytest
from fastapi.testclient import TestClient

import nemo_retriever.common.vdb.lancedb as lancedb_module
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table_object
from nemo_retriever.service.vectordb_app import (
    VectorDBState,
    _production_vdb,
    create_vectordb_app,
)


def _record(
    *,
    vector: list[float] | None = None,
    text: str = "Revenue grew 12% year over year.",
) -> dict:
    return {
        "document_type": "text",
        "embedding": vector or [1.0, 0.0, 0.0, 0.0],
        "metadata": {
            "content": text,
            "content_metadata": {"page_number": 12, "type": "text"},
            "source_metadata": {
                "source_id": "/data/10k_2023.pdf",
                "source_name": "10k_2023.pdf",
            },
        },
    }


_RECORD = _record()


def _backend(tmp_path, *, index_mode: str = "auto") -> LanceDB:
    backend = _production_vdb(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        expiration_cleanup_enabled=True,
        embed_model="nvidia/nemotron-3-embed-1b",
        index_mode=index_mode,
    )
    assert isinstance(backend, LanceDB)
    return backend


def _write(backend: LanceDB, *records: dict) -> int:
    backend.run([list(records)])
    table = lancedb.connect(backend.uri).open_table(backend.table_name)
    return int(table.count_rows())


def _capabilities(backend: LanceDB):
    table = lancedb.connect(backend.uri).open_table(backend.table_name)
    return inspect_lancedb_table_object(table)


def _app(backend: LanceDB):
    return create_vectordb_app(
        vdb=backend,
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        reconciliation_interval_seconds=0,
    )


@pytest.mark.integration
def test_fresh_auto_write_builds_fts_and_query_uses_hybrid(tmp_path) -> None:
    backend = _backend(tmp_path)
    app = _app(backend)

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"records": [[_RECORD]]})
            response = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )
            health = client.get("/v1/health")

    assert write.status_code == 200, write.text
    assert response.status_code == 200, response.text
    assert _capabilities(backend).has_fts
    assert response.json()["results"][0]["coverage"]["strategies_used"] == ["hybrid"]
    assert health.json()["configured_index_mode"] == "auto"
    assert health.json()["effective_index_mode"] == "hybrid"
    assert health.json()["fts_present"] is True


def test_quoted_query_supports_fts_index_without_positions() -> None:
    attempted_texts: list[str] = []

    class FakeQuery:
        def __init__(self) -> None:
            self.text_value = ""

        def vector(self, _vector):
            return self

        def text(self, value: str):
            self.text_value = value
            attempted_texts.append(value)
            return self

        def limit(self, _value):
            return self

        def refine_factor(self, _value):
            return self

        def nprobes(self, _value):
            return self

        def to_list(self):
            if '"' in self.text_value:
                raise RuntimeError(
                    "position is not found but required for phrase queries, try recreating the index with position"
                )
            return [{"text": "Revenue grew 12% year over year."}]

    class FakeTable:
        tags = SimpleNamespace(list=lambda: {})
        schema = SimpleNamespace(metadata=None)

        def search(self, **_kwargs):
            return FakeQuery()

    backend = LanceDB(uri="unused", table_name="nemo_retriever", hybrid=True)
    with patch.object(backend, "_open_table", return_value=FakeTable()):
        results = backend.retrieval(
            [[1.0, 0.0, 0.0, 0.0]],
            query_texts=['What was the "year over year" growth?'],
            top_k=5,
            hybrid=True,
        )

    assert results == [[{"text": "Revenue grew 12% year over year."}]]
    assert attempted_texts == [
        'What was the "year over year" growth?',
        "What was the year over year growth?",
    ]


def test_backend_reuses_connection_and_table_handles() -> None:
    table = object()
    database = MagicMock()
    database.open_table.return_value = table

    backend = LanceDB(uri="memory://cache-test", table_name="nemo_retriever")
    with patch("nemo_retriever.common.vdb.lancedb.lancedb.connect", return_value=database) as connect:
        first_connection = backend._connect()
        second_connection = backend._connect()
        first_table = backend._open_table(backend.table_name)
        second_table = backend._open_table(backend.table_name)

    assert first_connection is second_connection is database
    assert connect.call_count == 1
    assert database.open_table.call_count == 1
    assert first_table is second_table
    assert len(backend._connections) == 1
    assert len(backend._opened_tables) == 1


@pytest.mark.integration
def test_auto_preserves_existing_dense_table(tmp_path) -> None:
    dense = _backend(tmp_path, index_mode="dense")
    assert _write(dense, _RECORD) == 1

    reopened = _backend(tmp_path)
    caps = _capabilities(reopened)

    assert reopened.hybrid is False
    assert caps.has_vector
    assert not caps.has_fts
    assert reopened.health()["effective_retrieval_mode"] == "dense"


@pytest.mark.integration
def test_explicit_hybrid_upgrades_existing_dense_table(tmp_path) -> None:
    dense = _backend(tmp_path, index_mode="dense")
    _write(dense, _RECORD)

    upgraded = _backend(tmp_path, index_mode="hybrid")

    assert upgraded.hybrid is True
    assert _capabilities(upgraded).has_fts
    assert upgraded.health()["effective_retrieval_mode"] == "hybrid"


@pytest.mark.integration
def test_explicit_dense_rejects_existing_hybrid_table(tmp_path) -> None:
    hybrid = _backend(tmp_path)
    _write(hybrid, _RECORD)

    with pytest.raises(ValueError, match="Cannot append"):
        _backend(tmp_path, index_mode="dense")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("write_threshold", "row_threshold"),
    [(1, 100_000), (20, 1)],
    ids=("write-count", "unindexed-rows"),
)
def test_incremental_maintenance_thresholds_optimize_and_update_health(
    tmp_path,
    monkeypatch,
    write_threshold: int,
    row_threshold: int,
) -> None:
    backend = _backend(tmp_path)
    _write(backend, _RECORD)
    table = lancedb.connect(backend.uri).open_table(backend.table_name)
    monkeypatch.setattr(lancedb_module, "_SERVICE_OPTIMIZE_WRITE_THRESHOLD", write_threshold)
    monkeypatch.setattr(lancedb_module, "_SERVICE_OPTIMIZE_ROW_THRESHOLD", row_threshold)

    with patch.object(type(table), "optimize", autospec=True) as optimize:
        _write(backend, _record(text="second row"))

    optimize.assert_called_once()
    assert backend.health()["last_optimization"]["status"] == "ok"


@pytest.mark.integration
def test_explicit_dense_query_remains_dense(tmp_path) -> None:
    backend = _backend(tmp_path, index_mode="dense")
    app = _app(backend)

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"records": [[_RECORD]]})
            response = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert write.status_code == 200, write.text
    assert response.status_code == 200, response.text
    assert not _capabilities(backend).has_fts
    assert response.json()["results"][0]["coverage"]["strategies_used"] == ["dense"]
