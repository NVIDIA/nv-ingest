# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise lossless query overflow through the real graph and VDB operator."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy, prepare_embedding_inputs
from nemo_retriever.models.inference.main_text_embed import assign_embedding_columns
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import RetrieveVdbOperator
from nemo_retriever.query.expansion import QUERY_INDEX_COLUMN, regroup_query_hits


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        tokens = [ord(char) for char in text]
        return [1, *tokens, 2] if add_special_tokens else tokens

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


class _SplitEmbed(AbstractOperator):
    def preprocess(self, data, **kwargs):
        return data

    def process(self, data, **kwargs):
        policy = EmbeddingInputPolicy(_CharacterTokenizer(), 128, "q:")
        out = prepare_embedding_inputs(data, policy=policy).frame
        return assign_embedding_columns(
            out, [[float(i)] for i in range(len(out))], [None] * len(out), output_column="embedding"
        )

    def postprocess(self, data, **kwargs):
        return data


class _HybridVdb:
    hybrid = True

    def __init__(self):
        self.calls = []

    def retrieval(self, vectors, **kwargs):
        texts = kwargs["query_texts"]
        if len(texts) != len(vectors):
            raise ValueError(
                "LanceDB hybrid retrieval requires query_texts length to match vectors length; "
                f"got query_texts={len(texts)} vectors={len(vectors)}."
            )
        self.calls.append((vectors, kwargs))
        return [
            [
                {"text": "shared", "source": "doc.pdf", "page_number": 1, "_distance": vector[0]},
                {"text": text, "source": "doc.pdf", "page_number": 2, "_distance": vector[0] + 1},
            ]
            for vector, text in zip(vectors, texts)
        ]


@pytest.mark.parametrize("run_mode", ["local", "service"])
def test_long_query_preserves_hybrid_alignment_and_original_groups(monkeypatch, run_mode):
    vdb = _HybridVdb()
    monkeypatch.setattr("nemo_retriever.operators.embed.operators._BatchEmbedActor", _SplitEmbed)
    monkeypatch.setattr("nemo_retriever.operators.embed.cpu_operator._BatchEmbedCPUActor", _SplitEmbed)
    embed_kwargs = {"model_name": "test", "local_ingest_embed_backend": "hf"}
    if run_mode == "service":
        embed_kwargs["embedding_endpoint"] = "http://test.invalid/embed"
    retriever = Retriever(run_mode=run_mode, embed_kwargs=embed_kwargs, vdb_kwargs={"vdb": vdb})
    queries = ["a" * 200, "short", "a" * 200]
    hits = retriever._execute_queries_graph(
        queries, effective_top_k=2, retrieval_top_k=2, vdb_call_kwargs={"where": "tenant = 'one'"}, embed_extra=None
    )
    assert len(hits) == len(queries)
    assert all(len(group) == 2 for group in hits)
    assert hits[0][0]["text"] == hits[2][0]["text"] == "shared"
    vectors, kwargs = vdb.calls[0]
    assert len(vectors) == 5
    assert "".join(kwargs["query_texts"][:2]) == queries[0]
    assert kwargs["query_texts"][2] == "short"
    assert "".join(kwargs["query_texts"][3:]) == queries[2]
    assert kwargs["where"] == "tenant = 'one'"
    assert hits[1][0]["_distance"] == 2.0
    assert "_relevance_score" not in hits[1][0]
    assert hits[0][0]["_relevance_score"] == pytest.approx(2 / 61)


@pytest.mark.parametrize("query", ["short", "a" * 200], ids=["unsplit", "split"])
def test_rerank_uses_original_full_queries_and_keeps_duplicate_inputs_separate(query):
    from nemo_retriever.graph.retriever_utils import rerank_long_dataframe_to_hits

    vdb = _HybridVdb()
    op = RetrieveVdbOperator(vdb=vdb, explode_for_rerank=True, _query_text_column="text")
    expanded = _SplitEmbed().run(pd.DataFrame({"text": [query, query], QUERY_INDEX_COLUMN: [0, 1]}))
    long = op.run(expanded, query_texts=[query, query], top_k=2)
    assert long["query"].tolist() == [query] * 4
    assert long[QUERY_INDEX_COLUMN].tolist() == [0, 0, 1, 1]
    # Distinct scores/candidates for duplicate text must not cross input groups.
    long["rerank_score"] = [0.9, 0.1, 0.1, 0.9]
    hits = rerank_long_dataframe_to_hits(long, query_texts=[query, query], top_k=1)
    assert hits[0][0]["text"] == "shared"
    assert hits[1][0]["text"] != "shared"


@pytest.mark.parametrize("include_query_index", [False, True])
@pytest.mark.parametrize("has_hits", [False, True])
def test_rerank_dataframe_query_indices_are_opt_in(include_query_index, has_hits):
    from nemo_retriever.graph.retriever_utils import hits_lists_to_rerank_dataframe

    hit = {"text": "second query hit"}
    hits_per_query = [[], [hit] if has_hits else []]
    options = {"include_query_index": True} if include_query_index else {}
    frame = hits_lists_to_rerank_dataframe(["same", "same"], hits_per_query, **options)

    expected_columns = ["query", "text", "_hit"]
    if include_query_index:
        expected_columns.append(QUERY_INDEX_COLUMN)
    assert list(frame.columns) == expected_columns
    if has_hits:
        assert frame["query"].tolist() == ["same"]
        assert frame["text"].tolist() == [hit["text"]]
        assert frame["_hit"].tolist() == [hit]
        if include_query_index:
            assert frame[QUERY_INDEX_COLUMN].tolist() == [1]
    else:
        assert frame.empty


def test_chunk_fusion_uses_ranks_deduplicates_once_per_chunk_and_is_stable():
    first_hit = {"text": "first", "_distance": 0.1}
    second_hit = {"text": "second", "_distance": 0.2}
    hits = regroup_query_hits(
        [[first_hit, first_hit, second_hit], [{**second_hit, "_distance": 9.0}]],
        [0, 0],
        query_count=1,
        top_k=2,
    )[0]
    assert [hit["text"] for hit in hits] == ["second", "first"]
    assert hits[0]["_relevance_score"] == pytest.approx(1 / 63 + 1 / 61)
    tied = regroup_query_hits([[first_hit], [second_hit]], [0, 0], query_count=1, top_k=2)[0]
    assert [hit["text"] for hit in tied] == ["first", "second"]


def test_reordered_rows_preserve_original_order_and_empty_results():
    hits = regroup_query_hits([[{"text": "second"}], []], [1, 0], query_count=2, top_k=3)
    assert hits == [[], [{"text": "second"}]]


@pytest.mark.parametrize("indices,count", [([0], 2), ([2], 2), ([-1], 1), ([True], 1), ([0.0], 1)])
def test_invalid_query_identity_fails_clearly(indices, count):
    with pytest.raises(ValueError, match="query"):
        regroup_query_hits([[]], indices, query_count=count, top_k=2)


def test_dropped_result_group_fails_clearly():
    with pytest.raises(ValueError, match="number of result groups"):
        regroup_query_hits([[]], [0, 0], query_count=1, top_k=2)
