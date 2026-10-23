# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit HNSW search budgets preserve the normal dense/hybrid builder path."""

import pytest

from nemo_retriever.common.vdb.lancedb import LanceDB


class _Builder:
    def __init__(self, calls, args, kwargs):
        self.calls = calls
        self.options = {"search_args": args, "search_kwargs": kwargs}

    def where(self, value):
        self.options["where"] = value
        return self

    def vector(self, value):
        self.options["vector"] = value
        return self

    def text(self, value):
        self.options["text"] = value
        return self

    def limit(self, value):
        self.options["limit"] = value
        return self

    def refine_factor(self, value):
        self.options["refine_factor"] = value
        return self

    def nprobes(self, value):
        self.options["nprobes"] = value
        return self

    def ef(self, value):
        self.options["ef"] = value
        return self

    def select(self, value):
        self.options["select"] = value
        return self

    def to_list(self):
        self.calls.append(dict(self.options))
        return [{"text": "kept"}]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("ef", [None, 1, 4096])
def test_dense_and_hybrid_preserve_search_options_with_optional_ef(monkeypatch, hybrid, ef):
    import nemo_retriever.common.vdb.lancedb as module

    calls = []

    class Table:
        def search(self, *args, **kwargs):
            return _Builder(calls, args, kwargs)

    backend = LanceDB(uri="/unused", table_name="docs", hybrid=hybrid)
    monkeypatch.setattr(backend, "_open_table", lambda *_args, **_kwargs: Table())
    monkeypatch.setattr(backend, "_checkout_latest", lambda _table: None)
    monkeypatch.setattr(module, "_assert_lancedb_table_ready", lambda _table: None)
    vector = [1.0, 0.0]
    kwargs = dict(
        top_k=5,
        refine_factor=3,
        nprobes=7,
        where="text IS NOT NULL",
        _filter="ignored predicate",
        result_fields=["text"],
    )
    if ef is not None:
        kwargs["ef"] = ef
    if hybrid:
        kwargs["query_texts"] = ["query"]
    assert backend.retrieval([vector], **kwargs) == [[{"text": "kept"}]]
    assert len(calls) == 1
    expected = dict(where="text IS NOT NULL", limit=5, refine_factor=3, nprobes=7, select=["text"])
    if ef is not None:
        expected["ef"] = ef
    if hybrid:
        expected.update(
            search_args=(),
            search_kwargs={"vector_column_name": "vector", "query_type": "hybrid", "fts_columns": "text"},
            vector=vector,
            text="query",
        )
    else:
        expected.update(search_args=([vector],), search_kwargs={"vector_column_name": "vector"})
    assert calls == [expected]


@pytest.mark.parametrize("ef", [True, False, 0, -1, 1.0, "4096", float("nan")])
def test_invalid_ef_rejects_before_table_open(monkeypatch, ef):
    backend = LanceDB(uri="/unused")

    def unexpected_open(*args, **kwargs):
        raise AssertionError("invalid ef must fail before table access")

    monkeypatch.setattr(backend, "_open_table", unexpected_open)
    with pytest.raises(ValueError, match="ef must be a positive integer"):
        backend.retrieval([[1.0, 0.0]], ef=ef)


def test_sparse_backend_does_not_silently_ignore_explicit_ef(monkeypatch):
    backend = LanceDB(uri="/unused")
    monkeypatch.setattr(backend, "_open_table", lambda *_args, **_kwargs: pytest.fail("unexpected table access"))
    with pytest.raises(ValueError, match="dense or hybrid"):
        backend.sparse_retrieval(["query"], ef=4096)
