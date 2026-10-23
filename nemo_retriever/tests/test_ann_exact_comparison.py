# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake CPU contracts: one embedding, one pinned table, two scored hit legs."""

from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
import yaml

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.lancedb_comparison import compare_dense_search
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import RetrieveVdbOperator
from nemo_retriever.query.comparison import _vector_hashes, _yaml_digest


def _hit(page):
    return {
        "text": "private passage",
        "metadata": {"type": "text", "page_number": page},
        "pdf_basename": "doc",
        "page_number": page,
        "_distance": float(page),
    }


class _Builder:
    def __init__(self, table, vector):
        self.table, self.vector = table, vector
        self.options = {}

    def where(self, value):
        self.options["where"] = value
        return self

    def limit(self, value):
        self.options["limit"] = value
        return self

    def refine_factor(self, value):
        self.options["refine"] = value
        return self

    def nprobes(self, value):
        self.options["nprobes"] = value
        return self

    def ef(self, value):
        self.options["ef"] = value
        return self

    def select(self, value):
        self.options["fields"] = value
        return self

    def bypass_vector_index(self):
        self.options["bypass"] = True
        return self

    def to_list(self):
        self.table.calls.append((self.vector, dict(self.options)))
        if self.table.mutate_vector and not self.options.get("bypass"):
            self.vector[0][0] += 0.25
        if self.table.mutate_version and self.options.get("bypass"):
            self.table.version += 1
        return [_hit(2 if self.options.get("bypass") else 1)]


class _Table:
    def __init__(self):
        self.version, self.calls = 7, []
        self.mutate_vector = self.mutate_version = False
        self.metric = "l2"
        self.schema = pa.schema(
            [pa.field("vector", pa.list_(pa.float32(), 2))],
            metadata={
                b"nemo_retriever.embedding_model_name": b"embedder",
                b"nemo_retriever.embedding_model_revision": b"revision",
            },
        )

    def list_indices(self):
        return [SimpleNamespace(name="vector_idx", columns=["vector"])]

    def index_stats(self, name):
        return SimpleNamespace(
            distance_type=self.metric, index_type="IVF_HNSW_SQ", num_indexed_rows=3, num_unindexed_rows=0
        )

    def count_rows(self):
        return 3

    def search(self, vector, **kwargs):
        assert kwargs == {"vector_column_name": "vector"}
        return _Builder(self, vector)


@pytest.fixture
def backend(monkeypatch):
    import nemo_retriever.common.vdb.lancedb as module

    monkeypatch.setattr(module, "_assert_lancedb_table_ready", lambda table: None)
    table = _Table()
    opened = []

    def open_table(name, **kwargs):
        opened.append((name, kwargs))
        return table

    connection = SimpleNamespace(open_table=open_table)
    value = SimpleNamespace(
        uri="/private/index", table_name="docs", hybrid=False, metric="l2", _connect=lambda uri: connection
    )
    return value, table, opened


def _compare(backend, vectors=None, **kwargs):
    return compare_dense_search(
        backend, vectors or [[1.0, 0.0]], expected_model_name="embedder", expected_model_revision="revision", **kwargs
    )


def test_paired_backend_uses_same_vector_and_version_with_explicit_exact_bypass(backend):
    value, table, opened = backend
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    indexed, exact, evidence = _compare(value, vectors, top_k=50, where="metadata IS NOT NULL", result_fields=["text"])
    assert opened == [("docs", {}), ("docs", {"version": 7})]
    assert len(indexed) == len(exact) == 2
    assert evidence["table_version"] == 7
    assert evidence["indexed_vector_fingerprint"] == evidence["exact_vector_fingerprint"]
    for ordinal in range(2):
        assert table.calls[ordinal][0][0] is table.calls[ordinal + 2][0][0] is vectors[ordinal]
        assert table.calls[ordinal][1] == dict(
            where="metadata IS NOT NULL", limit=50, refine=50, nprobes=64, fields=["text"]
        )
        assert table.calls[ordinal + 2][1] == dict(where="metadata IS NOT NULL", limit=50, bypass=True, fields=["text"])


@pytest.mark.parametrize(
    "option,value",
    [
        ("hybrid", True),
        ("hybrid_fusion", object()),
        ("scope", "private"),
        ("collection_name", "collection"),
        ("search_kwargs", {"query_type": "hybrid"}),
        ("top_k", 0),
        ("nprobes", 0),
    ],
)
def test_unsupported_backend_modes_reject_before_search(backend, option, value):
    instance, table, _ = backend
    with pytest.raises(ValueError):
        _compare(instance, **{option: value})
    assert not table.calls


@pytest.mark.parametrize("vectors", [[[float("nan"), 0.0]], [[1.0]], [[1.0, 0.0]] * 257])
def test_vector_shape_finiteness_and_expansion_limit_are_required(backend, vectors):
    instance, table, _ = backend
    with pytest.raises(ValueError):
        _compare(instance, vectors)
    assert not table.calls


@pytest.mark.parametrize("mutation", ["mutate_vector", "mutate_version"])
def test_mutable_shared_inputs_or_table_fail_closed(backend, mutation):
    instance, table, _ = backend
    setattr(table, mutation, True)
    with pytest.raises(RuntimeError, match="changed"):
        _compare(instance)


def test_pinned_index_metric_and_model_metadata_are_required(backend):
    instance, table, _ = backend
    table.metric = "cosine"
    with pytest.raises(ValueError, match="verified l2"):
        _compare(instance)
    table.metric = "l2"
    with pytest.raises(ValueError, match="model differs"):
        compare_dense_search(instance, [[1.0, 0.0]], expected_model_name="other", expected_model_revision=None)
    with pytest.raises(ValueError, match="revision differs"):
        compare_dense_search(instance, [[1.0, 0.0]], expected_model_name="embedder", expected_model_revision="other")
    assert not table.calls


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_or_absent_vector_index_rejects_before_search(backend, count):
    instance, table, _ = backend
    table.list_indices = lambda: [SimpleNamespace(name=f"index{n}", columns=["vector"]) for n in range(count)]
    with pytest.raises(ValueError, match="exactly one index"):
        _compare(instance)
    assert not table.calls


def test_hashes_preserve_order_and_values_without_changing_vectors():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    original = [list(vector) for vector in vectors]
    assert _vector_hashes(vectors, 2) != _vector_hashes(list(reversed(vectors)), 2)
    assert vectors == original
    assert _yaml_digest({"b": [1, 2], "a": 3}) == _yaml_digest({"a": 3, "b": [1, 2]})


def _fake_pair(backend, vectors, **kwargs):
    hashes = _vector_hashes(vectors, 2)
    evidence = dict(
        table_uri="/private/index",
        table_name="docs",
        table_version=7,
        table_rows=3,
        vector_column="vector",
        vector_dimension=2,
        metric="l2",
        index_name="vector_idx",
        index_type="IVF_HNSW_SQ",
        indexed_rows=3,
        unindexed_rows=0,
        embedding_model_name="embedder",
        embedding_model_revision="revision",
        candidate_k=kwargs["top_k"],
        indexed_nprobes=64,
        indexed_refine_factor=50,
        indexed_ef=kwargs.get("ef"),
        predicate_sha256=None,
        vector_hashes=hashes,
        indexed_vector_fingerprint=_yaml_digest(list(hashes)),
        exact_vector_fingerprint=_yaml_digest(list(hashes)),
    )
    return [[_hit(1)] for _ in vectors], [[_hit(2)] for _ in vectors], evidence


@pytest.fixture
def fake_query(monkeypatch):
    import nemo_retriever.operators.embed.operators as embed_module
    import nemo_retriever.operators.search_comparison as comparison_module

    calls = dict(embeddings=0, pair=0, ordinary=0)

    class FakeEmbed(AbstractOperator):
        def __init__(self, params):
            self.params = params

        def preprocess(self, data, **kwargs):
            return data

        def process(self, data, **kwargs):
            calls["embeddings"] += 1
            # Two admission rows for the same original query must remain fused.
            row = data.iloc[0].to_dict()
            return pd.DataFrame(
                [{**row, "metadata": {"embedding": [1.0, 0.0]}}, {**row, "metadata": {"embedding": [0.0, 1.0]}}]
            )

        def postprocess(self, data, **kwargs):
            return data

    def pair(*args, **kwargs):
        calls["pair"] += 1
        return _fake_pair(*args, **kwargs)

    def ordinary(self, vectors, **kwargs):
        calls["ordinary"] += 1
        return [[_hit(1)] for _ in vectors]

    monkeypatch.setattr(embed_module, "_BatchEmbedActor", FakeEmbed)
    monkeypatch.setattr(comparison_module, "compare_dense_search", pair)
    monkeypatch.setattr(LanceDB, "retrieval", ordinary)
    monkeypatch.setattr(
        RetrieveVdbOperator,
        "get_index_metadata",
        lambda self, key, **kwargs: {"embedding_model_name": "embedder", "embedding_model_revision": "revision"}.get(
            key
        ),
    )
    monkeypatch.setattr(
        Retriever,
        "_resolve_lancedb_query_mode",
        lambda self, kwargs: ("dense", SimpleNamespace(vector_column="vector"), "/private/index", "docs", True),
    )
    retriever = Retriever(
        top_k=1,
        vdb_kwargs={"uri": "/private/index", "table_name": "docs"},
        embed_kwargs={
            "model_name": "embedder",
            "embed_model_name": "embedder",
            "embed_model_revision": "revision",
            "local_ingest_embed_backend": "hf",
        },
    )
    return retriever, calls


def test_public_pair_embeds_once_and_preserves_normal_split_query_shaping(fake_query):
    retriever, calls = fake_query
    result = retriever.query_compare_exact("private query", candidate_k=5, page_dedup=True, content_types="text")
    assert calls == dict(embeddings=1, pair=1, ordinary=0)
    assert [hit["page_number"] for hit in result.indexed_hits] == [1]
    assert [hit["page_number"] for hit in result.exact_hits] == [2]
    assert result.evidence.complete and not result.evidence.timing_qualification
    assert result.evidence.top_k == 1 and result.evidence.candidate_k == 5
    assert result.evidence.expanded_row_count == 2
    assert [row.original_query_index for row in result.evidence.rows] == [0, 0]
    assert result.evidence.indexed_input_sha256 == result.evidence.exact_input_sha256
    serialized = yaml.safe_dump(asdict(result.evidence))
    assert "private query" not in serialized and "private passage" not in serialized
    assert "vector_sha256" in serialized and "embedding:" not in serialized
    assert retriever._dense_search_capture is None and retriever._cached_graph is None


def test_default_query_after_pair_remains_ordinary_and_has_no_capture(fake_query):
    retriever, calls = fake_query
    retriever.query_compare_exact("first", candidate_k=5)
    assert [hit["page_number"] for hit in retriever.query("second")] == [1]
    assert calls == dict(embeddings=2, pair=1, ordinary=1)
    assert retriever._dense_search_capture is None


@pytest.mark.parametrize("change", [{"rerank": True}, {"run_mode": "service"}, {"graph": object()}])
def test_public_unsupported_scopes_reject_before_embedding(fake_query, change):
    retriever, calls = fake_query
    for name, value in change.items():
        setattr(retriever, name, value)
    with pytest.raises(ValueError, match="local default graph"):
        retriever.query_compare_exact("private")
    assert calls["embeddings"] == 0


def test_missing_capture_cannot_claim_complete_pair(fake_query, monkeypatch):
    retriever, calls = fake_query
    monkeypatch.setattr(Retriever, "query", lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match="missing complete"):
        retriever.query_compare_exact("private")
    assert calls["embeddings"] == 0


@pytest.mark.parametrize(
    "mode,uri", [("hybrid", "/private/index"), ("sparse", "/private/index"), ("dense", "s3://bucket/index")]
)
def test_effective_mode_or_remote_storage_rejects_before_embedding(fake_query, monkeypatch, mode, uri):
    retriever, calls = fake_query
    monkeypatch.setattr(
        Retriever,
        "_resolve_lancedb_query_mode",
        lambda self, kwargs: (mode, SimpleNamespace(vector_column="vector"), uri, "docs", True),
    )
    with pytest.raises(ValueError, match="dense filesystem"):
        retriever.query_compare_exact("private")
    assert calls["embeddings"] == 0


def test_same_query_text_is_not_cached_between_pairs(fake_query):
    retriever, calls = fake_query
    first = retriever.query_compare_exact("repeat")
    helper, capture, _ = retriever.__dict__["_exact_search_session"]
    graph = helper._cached_graph
    second = retriever.query_compare_exact("repeat")
    assert calls == dict(embeddings=2, pair=2, ordinary=0)
    assert first.evidence.shared_input_sha256 == second.evidence.shared_input_sha256
    assert helper._cached_graph is graph
    assert capture.query_text == "" and capture.exact_hits is None and capture.evidence is None


@pytest.mark.parametrize("ef", [None, 1, 4096])
def test_pair_applies_ef_only_to_indexed_leg_and_records_explicit_budget(backend, ef):
    value, table, _ = backend
    _, _, evidence = _compare(value, ef=ef)
    assert evidence["indexed_ef"] == ef
    indexed_options, exact_options = table.calls[0][1], table.calls[1][1]
    if ef is None:
        assert "ef" not in indexed_options
    else:
        assert indexed_options["ef"] == ef
    assert "ef" not in exact_options and exact_options["bypass"]


@pytest.mark.parametrize("ef", [True, False, 0, -1, 1.0, "4096", float("nan")])
def test_pair_rejects_invalid_ef_before_table_access(backend, ef):
    value, table, opened = backend
    with pytest.raises(ValueError, match="ef must be a positive integer"):
        _compare(value, ef=ef)
    assert opened == [] and table.calls == []


def test_public_pair_preserves_expansion_and_ef_through_runtime_filter(fake_query):
    retriever, calls = fake_query
    result = retriever.query_compare_exact("query", candidate_k=5, vdb_kwargs={"ef": 4096})
    assert calls == dict(embeddings=1, pair=1, ordinary=0)
    assert result.evidence.indexed_ef == 4096
    assert result.evidence.expanded_row_count == 2
    assert result.evidence.indexed_input_sha256 == result.evidence.exact_input_sha256


def test_normal_retriever_forwards_explicit_runtime_ef(fake_query, monkeypatch):
    retriever, _ = fake_query
    seen = []

    def retrieval(self, vectors, **kwargs):
        seen.append(kwargs)
        return [[_hit(1)] for _ in vectors]

    monkeypatch.setattr(LanceDB, "retrieval", retrieval)
    assert retriever.query("query", candidate_k=5, vdb_kwargs={"ef": 4096})
    assert len(seen) == 1 and seen[0]["ef"] == 4096 and seen[0]["top_k"] == 5
