# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

import nemo_retriever.graph.retriever as retriever_module  # noqa: E402
from nemo_retriever.common.vdb.hybrid_fusion import (
    DEFAULT_HYBRID_FUSION_POLICY,  # noqa: E402
)
from nemo_retriever.common.vdb.lancedb import LanceDB  # noqa: E402
from nemo_retriever.common.vdb.lancedb_capabilities import (  # noqa: E402
    LanceTableCapabilities,
    inspect_lancedb_table,
    inspect_lancedb_table_object,
)
from nemo_retriever.graph.pipeline_graph import Graph  # noqa: E402
from nemo_retriever.graph.retriever import Retriever  # noqa: E402
from nemo_retriever.operators.vdb import RetrieveVdbOperator  # noqa: E402


def _create_vector_table(
    uri: str,
    table_name: str,
    *,
    fts: bool = False,
    embedding_model_name: str | None = "nvidia/nemotron-3-embed-1b",
) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ],
        metadata=(
            {b"nemo_retriever.embedding_model_name": embedding_model_name.encode("utf-8")}
            if embedding_model_name
            else None
        ),
    )
    rows = [
        {
            "vector": [1.0, 0.0],
            "text": "alpha safety manual",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "alpha.pdf"}),
            "id": "alpha",
        }
    ]
    table = lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")
    if fts:
        table.create_fts_index("text", replace=True)


def _create_sparse_table(uri: str, table_name: str) -> None:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    rows = [
        {
            "text": "alpha safety manual",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "alpha.pdf"}),
            "id": "alpha",
        }
    ]
    table = lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")
    table.create_fts_index("text", replace=True)


def _fail_embed_graph(*_args: Any, **_kwargs: Any) -> list[list[dict[str, Any]]]:
    raise AssertionError("sparse query should not build or execute the embedding graph")


def _patch_graph_hits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_execute(
        _self: Retriever,
        _query_texts: list[str],
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        calls.append(kwargs["vdb_call_kwargs"])
        return [[{"text": "alpha safety manual", "metadata": {"type": "text"}, "source": "alpha.pdf"}]]

    monkeypatch.setattr(Retriever, "_execute_queries_graph", fake_execute)
    return calls


def test_detector_returns_dense_for_vector_only_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "dense")

    caps = inspect_lancedb_table(uri, "dense")

    assert caps.has_vector is True
    assert caps.has_fts is False
    assert caps.vector_column == "vector"
    assert caps.text_column == "text"
    assert caps.retrieval_mode == "dense"


def test_lancedb_metadata_round_trip_drives_query_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    records = [
        [
            {
                "document_type": "text",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "content": "alpha safety manual",
                    "content_metadata": {"id": "alpha"},
                    "source_metadata": {"source_id": "alpha.pdf"},
                },
            }
        ]
    ]
    LanceDB(
        uri=uri,
        table_name="docs",
        vector_dim=2,
        build_index=False,
        embedding_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
    ).run(records)

    operator = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": uri, "table_name": "docs"})

    assert operator.get_index_metadata("embedding_model_name") == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert operator.get_index_metadata("retrieval_mode") == "dense"

    captured_embed_kwargs: dict[str, Any] = {}

    def capture_query_model(
        _self: Retriever,
        _query_texts: list[str],
        *,
        embed_extra: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        captured_embed_kwargs.update(embed_extra or {})
        return [[{"text": "alpha safety manual", "source": "alpha.pdf"}]]

    monkeypatch.setattr(Retriever, "_execute_queries_graph", capture_query_model)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "docs"}).query("alpha", top_k=1)

    assert captured_embed_kwargs["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_detector_returns_hybrid_for_vector_plus_fts_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)

    caps = inspect_lancedb_table(uri, "hybrid")

    assert caps.has_vector is True
    assert caps.has_fts is True
    assert caps.retrieval_mode == "hybrid"


def test_physical_fts_overrides_stale_dense_schema_metadata(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "upgraded", fts=True)
    table = lancedb.connect(uri).open_table("upgraded")
    stale_table = type(
        "StaleSchemaTable",
        (),
        {"schema": table.schema.with_metadata({b"retrieval_mode": b"dense"}), "list_indices": table.list_indices},
    )()

    caps = inspect_lancedb_table_object(stale_table)

    assert caps.retrieval_mode == "hybrid"


def test_physical_vector_only_table_overrides_stale_hybrid_metadata(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "legacy")
    table = lancedb.connect(uri).open_table("legacy")
    stale_table = type(
        "StaleSchemaTable",
        (),
        {"schema": table.schema.with_metadata({b"retrieval_mode": b"hybrid"}), "list_indices": table.list_indices},
    )()

    caps = inspect_lancedb_table_object(stale_table)

    assert caps.retrieval_mode == "dense"


def test_detector_returns_sparse_for_fts_only_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_sparse_table(uri, "sparse")

    caps = inspect_lancedb_table(uri, "sparse")

    assert caps.has_vector is False
    assert caps.has_fts is True
    assert caps.vector_column is None
    assert caps.text_column == "text"
    assert caps.retrieval_mode == "sparse"


def test_fts_index_telemetry_failure_is_nonfatal() -> None:
    class BrokenIndexTable:
        def list_indices(self):
            raise RuntimeError("transient index metadata failure")

    backend = LanceDB.__new__(LanceDB)

    assert backend._fts_unindexed_rows(BrokenIndexTable()) is None


def test_sparse_query_does_not_call_embedding_graph(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_sparse_table(uri, "sparse")

    monkeypatch.setattr(Retriever, "_execute_queries_graph", _fail_embed_graph)

    hits = Retriever(vdb_kwargs={"uri": uri, "table_name": "sparse"}).query("alpha", top_k=1)

    assert hits
    assert hits[0]["text"] == "alpha safety manual"


@pytest.mark.parametrize(
    "query_path",
    [
        "injected",
        "custom_graph",
    ],
)
@pytest.mark.parametrize("hybrid", [False, True])
def test_lancedb_query_paths_reject_untagged_vector_table(
    query_path: str,
    hybrid: bool,
    tmp_path,
) -> None:
    uri = str(tmp_path / "db")
    table_name = "hybrid" if hybrid else "dense"
    _create_vector_table(uri, table_name, fts=hybrid, embedding_model_name=None)
    backend = LanceDB(
        uri=uri,
        table_name=table_name,
        vector_dim=2,
        build_index=False,
        hybrid=hybrid,
    )

    if query_path == "custom_graph":
        graph = Graph()
        graph.add_root(RetrieveVdbOperator(vdb=backend))
        retriever = Retriever(graph=graph)
    else:
        retriever = Retriever(vdb_kwargs={"vdb": backend})

    with pytest.raises(ValueError, match="does not record its embedding model"):
        retriever.query("alpha", top_k=1)


def test_hybrid_table_query_automatically_enables_hybrid(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)
    calls = _patch_graph_hits(monkeypatch)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "hybrid"}).query("alpha", top_k=1)

    assert calls == [{"hybrid": True, "hybrid_fusion": DEFAULT_HYBRID_FUSION_POLICY}]


def test_existing_dense_query_behavior_is_unchanged(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "dense")
    calls = _patch_graph_hits(monkeypatch)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "dense"}).query("alpha", top_k=1)

    assert calls == [{}]


def test_explicit_dense_override_on_hybrid_table(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)
    calls = _patch_graph_hits(monkeypatch)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "hybrid", "retrieval_mode": "dense"}).query("alpha", top_k=1)

    assert calls == [{"hybrid": False}]


def test_explicit_hybrid_override_on_hybrid_table(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)
    calls = _patch_graph_hits(monkeypatch)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "hybrid", "retrieval_mode": "hybrid"}).query("alpha", top_k=1)

    assert calls == [{"hybrid": True, "hybrid_fusion": DEFAULT_HYBRID_FUSION_POLICY}]


def test_explicit_sparse_override_on_hybrid_table_uses_sparse_retrieval(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)

    monkeypatch.setattr(Retriever, "_execute_queries_graph", _fail_embed_graph)

    hits = Retriever(vdb_kwargs={"uri": uri, "table_name": "hybrid", "retrieval_mode": "sparse"}).query(
        "alpha",
        top_k=1,
    )

    assert hits
    assert hits[0]["text"] == "alpha safety manual"


def test_retriever_caches_lancedb_capability_inspection(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    caps = LanceTableCapabilities(
        has_vector=True,
        has_fts=False,
        retrieval_mode="dense",
        vector_column="vector",
        text_column="text",
    )

    def fake_inspect(uri: str, table_name: str) -> LanceTableCapabilities:
        calls.append((uri, table_name))
        return caps

    monkeypatch.setattr(retriever_module, "inspect_lancedb_table", fake_inspect)

    retriever = Retriever(vdb_kwargs={"uri": "memory://db", "table_name": "docs"})

    assert retriever._resolve_lancedb_query_mode({}) == ("dense", caps, "memory://db", "docs", False)
    assert retriever._resolve_lancedb_query_mode({}) == ("dense", caps, "memory://db", "docs", False)
    assert calls == [("memory://db", "docs")]
