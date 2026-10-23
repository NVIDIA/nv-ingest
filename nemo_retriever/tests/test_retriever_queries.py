# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the graph-based :class:`~nemo_retriever.graph.retriever.Retriever` query surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.query.shaping import shape_query_hits


def _make_hits(n: int, base_score: float = 0.5) -> list[dict[str, Any]]:
    return [
        {
            "text": f"passage {i}",
            "metadata": "{}",
            "source": "doc.pdf",
            "page_number": i,
            "_distance": base_score + i * 0.01,
        }
        for i in range(n)
    ]


def _make_retriever(**overrides: Any) -> Retriever:
    defaults: dict[str, Any] = {
        "rerank": False,
        "top_k": 5,
        "vdb_kwargs": {"vdb_op": "lancedb", "vdb_kwargs": {"uri": "/tmp/r", "table_name": "t"}},
        "embed_kwargs": {"model_name": "embedder", "embed_model_name": "embedder"},
    }
    defaults.update(overrides)
    embed_kwargs = dict(defaults["embed_kwargs"])
    embed_kwargs.setdefault("local_ingest_embed_backend", "hf")
    defaults["embed_kwargs"] = embed_kwargs
    return Retriever(**defaults)


def _install_mock_graph(monkeypatch: pytest.MonkeyPatch, hits: list[list[dict[str, Any]]]) -> MagicMock:
    """Avoid constructing real LanceDB / embed operators."""
    # Resolved graph execution returns one entry per graph leaf; retrieval output
    # is ``list[list[dict]]``.
    graph = MagicMock()
    graph.execute_in_place.return_value = [hits]

    # bypass instance cache from other tests
    def fresh_get(self: Retriever, *, embed_extra: Any = None) -> MagicMock:
        return graph

    monkeypatch.setattr(Retriever, "_get_graph", fresh_get)
    monkeypatch.setattr(Retriever, "_resolve_lancedb_query_mode", lambda self, runtime_vdb_kwargs: None)
    return graph


class TestQueriesGraphExecution:
    @pytest.fixture(autouse=True)
    def _mock_index_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        metadata_reader = MagicMock()
        metadata_reader.get_index_metadata.return_value = None
        monkeypatch.setattr(
            "nemo_retriever.graph.retriever.RetrieveVdbOperator", MagicMock(return_value=metadata_reader)
        )

    def test_empty_queries_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_get = MagicMock()
        monkeypatch.setattr(Retriever, "_get_graph", mock_get)
        assert _make_retriever().queries([]) == []
        mock_get.assert_not_called()

    def test_queries_thread_top_k_and_vdb_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]
        graph = _install_mock_graph(monkeypatch, hit)
        retriever = _make_retriever(top_k=11)
        out = retriever.queries(["q"], vdb_kwargs={"where": "x"})
        assert out == hit
        graph.execute_in_place.assert_called_once()
        _args, kw = graph.execute_in_place.call_args
        assert kw["top_k"] == 11
        assert kw["query_texts"] == ["q"]
        assert kw["where"] == "x"
        df = _args[0]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["text", "_nrl_query_index"]
        assert df["_nrl_query_index"].tolist() == [0]
        assert df["text"].tolist() == ["q"]

    @pytest.mark.parametrize("embed_kwargs", [{}, {"local_ingest_embed_backend": None}])
    def test_default_embed_params_defer_model_to_index_metadata(self, embed_kwargs: dict[str, Any]) -> None:
        params = Retriever(embed_kwargs=embed_kwargs)._merge_embed_params()

        assert params.model_name is None
        assert params.embed_model_name is None
        assert params.local_ingest_embed_backend == "vllm"

    def test_merge_embed_params_per_call_overrides(self) -> None:
        r = _make_retriever(embed_kwargs={"model_name": "base", "embed_model_name": "base"})
        p = r._merge_embed_params({"model_name": "call"})
        assert p.model_name == "call"

    @pytest.mark.parametrize(
        ("explicit_embed_kwargs", "query_embed_kwargs"),
        [
            ({"model_name": "acme/model-b", "embed_model_name": "acme/model-b"}, None),
            ({}, {"model_name": "acme/model-b", "embed_model_name": "acme/model-b"}),
        ],
    )
    def test_explicit_model_mismatch_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        explicit_embed_kwargs: dict[str, str],
        query_embed_kwargs: dict[str, str] | None,
    ) -> None:
        graph = _install_mock_graph(monkeypatch, [[{"text": "retrieved"}]])
        metadata_reader = MagicMock()
        metadata_reader.get_index_metadata.side_effect = lambda key, **_kwargs: {
            "embedding_model_name": "acme/model-a",
            "embedding_model_revision": "a" * 40,
        }.get(key)
        reader_factory = MagicMock(return_value=metadata_reader)
        monkeypatch.setattr("nemo_retriever.graph.retriever.RetrieveVdbOperator", reader_factory)
        retriever = _make_retriever(embed_kwargs=explicit_embed_kwargs)

        with pytest.raises(ValueError, match="does not match model .acme/model-a. recorded on the index"):
            retriever.queries(["q"], embed_kwargs=query_embed_kwargs)

        reader_factory.assert_called_once()
        graph.execute_in_place.assert_not_called()

    def test_dense_lancedb_without_embedding_model_metadata_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        graph = _install_mock_graph(monkeypatch, [[{"text": "retrieved"}]])
        monkeypatch.setattr(
            Retriever,
            "_resolve_lancedb_query_mode",
            lambda self, runtime_vdb_kwargs: ("dense", MagicMock(), "lancedb", "nemo-retriever", False),
        )
        retriever = _make_retriever()

        with pytest.raises(ValueError, match="does not record its embedding model"):
            retriever.queries(["q"])

        graph.execute_in_place.assert_not_called()

    def test_explicit_model_matching_index_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _install_mock_graph(monkeypatch, [[{"text": "retrieved"}]])
        metadata_reader = MagicMock()
        metadata_reader.get_index_metadata.side_effect = lambda key, **_kwargs: {
            "embedding_model_name": "acme/model-a",
        }.get(key)
        monkeypatch.setattr(
            "nemo_retriever.graph.retriever.RetrieveVdbOperator", MagicMock(return_value=metadata_reader)
        )
        retriever = _make_retriever(embed_kwargs={"model_name": "acme/model-a", "embed_model_name": "acme/model-a"})

        with caplog.at_level("WARNING", logger="nemo_retriever.graph.retriever"):
            retriever.queries(["q"])

        assert not caplog.records

    def test_index_model_keeps_constructor_provider_prefix(self) -> None:
        retriever = _make_retriever(
            embed_kwargs={
                "embed_invoke_url": "https://embed.example.com/v1/embeddings",
                "embed_model_provider_prefix": "nvidia",
            }
        )

        resolved = retriever._resolve_embed_kwargs("nvidia/llama-nemotron-embed-vl-1b-v2", None)
        params = retriever._merge_embed_params(resolved)

        assert params.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
        assert params.embed_model_provider_prefix == "nvidia"

    def test_index_model_revision_is_forwarded_to_query_embedder(self) -> None:
        retriever = _make_retriever(embed_kwargs={})

        resolved = retriever._resolve_embed_kwargs(
            "acme/fine-tuned-nemotron",
            None,
            "a" * 40,
        )

        assert resolved["embed_model_name"] == "acme/fine-tuned-nemotron"
        assert resolved["embed_model_revision"] == "a" * 40

    def test_explicit_model_override_does_not_reuse_index_revision(self) -> None:
        retriever = _make_retriever(embed_kwargs={"embed_model_name": "acme/override"})

        resolved = retriever._resolve_embed_kwargs(
            "acme/index-model",
            None,
            "a" * 40,
        )

        assert resolved["embed_model_name"] == "acme/override"
        assert "embed_model_revision" not in resolved

    def test_runtime_model_change_clears_configured_revision(self) -> None:
        retriever = _make_retriever(embed_kwargs={"embed_model_name": "acme/model-a", "embed_model_revision": "a" * 40})

        resolved = retriever._resolve_embed_kwargs(None, {"embed_model_name": "acme/model-b"})
        params = retriever._merge_embed_params(resolved)

        assert params.embed_model_name == "acme/model-b"
        assert params.embed_model_revision is None

    def test_runtime_same_model_keeps_configured_revision(self) -> None:
        retriever = _make_retriever(embed_kwargs={"embed_model_name": "acme/model-a", "embed_model_revision": "a" * 40})

        resolved = retriever._resolve_embed_kwargs(None, {"embed_model_name": "acme/model-a"})
        params = retriever._merge_embed_params(resolved)

        assert params.embed_model_revision == "a" * 40

    def test_explicit_runtime_revision_overrides_index_revision(self) -> None:
        retriever = _make_retriever(embed_kwargs={})

        resolved = retriever._resolve_embed_kwargs(
            "acme/index-model",
            {"embed_model_revision": "b" * 40},
            "a" * 40,
        )
        params = retriever._merge_embed_params(resolved)

        assert params.embed_model_name == "acme/index-model"
        assert params.embed_model_revision == "b" * 40

    @pytest.mark.parametrize(("requires_vllm", "expected_backend"), [(False, "hf"), (True, "vllm")])
    @patch("nemo_retriever.graph.retriever.resolve_embed_model_spec")
    def test_local_query_embedding_selects_compatible_backend(
        self, resolve_spec: MagicMock, requires_vllm: bool, expected_backend: str
    ) -> None:
        resolve_spec.return_value = MagicMock(requires_vllm=requires_vllm, revision="a" * 40)
        retriever = _make_retriever(
            embed_kwargs={
                "model_name": "embedder",
                "embed_model_name": "embedder",
                "local_ingest_embed_backend": None,
            }
        )

        p = retriever._merge_embed_params()

        assert p.local_ingest_embed_backend == expected_backend
        assert p.embed_model_revision == "a" * 40
        resolve_spec.assert_called_once_with("embedder", revision=None)

    @patch("nemo_retriever.graph.retriever.resolve_embed_model_spec")
    def test_index_model_revision_drives_automatic_backend(self, resolve_spec: MagicMock) -> None:
        revision = "b" * 40
        resolve_spec.return_value = MagicMock(requires_vllm=True, revision=revision)
        retriever = _make_retriever(embed_kwargs={"local_ingest_embed_backend": None})
        resolved = retriever._resolve_embed_kwargs("acme/index-model", None, revision)

        p = retriever._merge_embed_params(resolved)

        assert p.embed_model_name == "acme/index-model"
        assert p.embed_model_revision == revision
        assert p.local_ingest_embed_backend == "vllm"
        resolve_spec.assert_called_once_with("acme/index-model", revision=revision)

    def test_local_query_embedding_backend_can_be_overridden(self) -> None:
        r = _make_retriever(embed_kwargs={"local_ingest_embed_backend": "vllm"})
        p = r._merge_embed_params()
        assert p.local_ingest_embed_backend == "vllm"

    @patch("nemo_retriever.graph.retriever.resolve_embed_model_spec")
    def test_remote_embedding_skips_local_model_resolution(self, resolve_spec: MagicMock) -> None:
        retriever = _make_retriever(
            embed_kwargs={
                "embed_invoke_url": "https://embed.example.com/v1/embeddings",
                "local_ingest_embed_backend": None,
            }
        )

        retriever._merge_embed_params()

        resolve_spec.assert_not_called()

    def test_rerank_inflates_retrieval_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        graph = _install_mock_graph(monkeypatch, [[{"text": "x"}]])
        retriever = _make_retriever(top_k=3, rerank=True, rerank_kwargs={"refine_factor": 4})
        retriever._cached_graph = None
        retriever._cache_key = None
        retriever.queries(["q"])
        assert graph.execute_in_place.call_args.kwargs["top_k"] == 12

    def test_rerank_dataframe_output_orders_hits_by_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vector_hits = [
            {"text": "vector retrieval winner", "source": "vector.pdf", "page_number": 1},
            {"text": "reranker winner", "source": "rerank.pdf", "page_number": 2},
        ]
        execute_kwargs: list[dict[str, Any]] = []

        class FakeResolvedGraph:
            def execute(self, _df: pd.DataFrame, **kwargs: Any) -> list[Any]:
                execute_kwargs.append(kwargs)
                return [
                    pd.DataFrame(
                        [
                            {
                                "query": "q",
                                "text": "vector retrieval winner",
                                "_hit": vector_hits[0],
                                "rerank_score": 0.1,
                            },
                            {"query": "q", "text": "reranker winner", "_hit": vector_hits[1], "rerank_score": 0.9},
                        ]
                    )
                ]

        class FakeGraph:
            def resolve_for_local_execution(self) -> FakeResolvedGraph:
                return FakeResolvedGraph()

        out = _make_retriever(top_k=1, rerank=True, graph=FakeGraph()).query("q")

        assert out == [{"text": "reranker winner", "source": "rerank.pdf", "page_number": 2, "_rerank_score": 0.9}]
        assert [call["top_k"] for call in execute_kwargs] == [4]

    def test_query_delegates_to_queries(self) -> None:
        retriever = _make_retriever()
        expected = _make_hits(2)
        with patch.object(retriever, "queries", return_value=[expected]) as mock_q:
            result = retriever.query("find", top_k=4, vdb_kwargs={"uri": "x"})
        mock_q.assert_called_once_with(
            ["find"],
            top_k=4,
            candidate_k=None,
            page_dedup=False,
            content_types=None,
            vdb_kwargs={"uri": "x"},
            embed_kwargs=None,
        )
        assert result is expected

    def test_candidate_k_widens_retrieval_before_final_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [[{"text": f"hit {i}", "page_number": i} for i in range(5)]]
        graph = _install_mock_graph(monkeypatch, hits)
        out = _make_retriever(top_k=2).queries(["q"], candidate_k=5)

        assert [hit["text"] for hit in out[0]] == ["hit 0", "hit 1"]
        assert graph.execute_in_place.call_args.kwargs["top_k"] == 5

    def test_repeated_queries_reuse_cached_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]
        graph = MagicMock()
        graph.execute_in_place.return_value = [hits]
        build = MagicMock(return_value=graph)
        monkeypatch.setattr(Retriever, "_build_default_graph", build)
        monkeypatch.setattr(Retriever, "_resolve_lancedb_query_mode", lambda self, runtime_vdb_kwargs: None)

        retriever = _make_retriever()
        assert retriever.query("first") == hits[0]
        assert retriever.query("second") == hits[0]

        build.assert_called_once()
        assert graph.execute_in_place.call_count == 2

    def test_custom_graph_preserves_resolve_per_query_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]

        class LegacyResolvedGraph:
            def execute(self, _df: pd.DataFrame, **_kwargs: Any) -> list[Any]:
                return [hits]

        graph = MagicMock()
        graph.resolve_for_local_execution.side_effect = (LegacyResolvedGraph(), LegacyResolvedGraph())
        monkeypatch.setattr(Retriever, "_resolve_lancedb_query_mode", lambda self, runtime_vdb_kwargs: None)

        retriever = _make_retriever(graph=graph)
        assert retriever.query("first") == hits[0]
        assert retriever.query("second") == hits[0]

        assert graph.resolve_for_local_execution.call_count == 2

    @patch("nemo_retriever.graph.retriever.resolve_embed_model_spec")
    def test_custom_graph_does_not_resolve_default_embedding_model(self, resolve_spec: MagicMock) -> None:
        hits = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]

        class ResolvedGraph:
            def execute(self, df: pd.DataFrame, **_kwargs: Any) -> list[Any]:
                assert list(df.columns) == ["query_text"]
                assert df["query_text"].tolist() == ["find doc"]
                return [hits]

        class Graph:
            def resolve_for_local_execution(self) -> ResolvedGraph:
                return ResolvedGraph()

        retriever = Retriever(
            graph=Graph(),
            embed_kwargs={
                "text_column": "query_text",
                "embed_model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
                "local_ingest_embed_backend": None,
            },
        )

        assert retriever.query("find doc") == hits[0]
        resolve_spec.assert_not_called()

    def test_embed_override_invalidates_graph_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]
        graph_one = MagicMock()
        graph_one.execute_in_place.return_value = [hits]
        graph_two = MagicMock()
        graph_two.execute_in_place.return_value = [hits]
        build = MagicMock(side_effect=(graph_one, graph_two))
        monkeypatch.setattr(Retriever, "_build_default_graph", build)
        monkeypatch.setattr(Retriever, "_resolve_lancedb_query_mode", lambda self, runtime_vdb_kwargs: None)

        retriever = _make_retriever()
        retriever.query("first")
        retriever.query("same configuration")
        retriever.query("different configuration", embed_kwargs={"model_name": "different"})

        assert graph_one.execute_in_place.call_count == 2
        graph_two.execute_in_place.assert_called_once()
        assert build.call_count == 2

    def test_candidate_k_must_cover_top_k(self) -> None:
        with pytest.raises(ValueError, match=r"candidate_k \(2\).*top_k \(5\)"):
            _make_retriever(top_k=5).queries(["q"], candidate_k=2)


class TestRerankEndpointWiring:
    """A reranker endpoint set through ``rerank_kwargs`` must be the one that serves the request.

    ``NemotronRerankActor`` dispatches on ``rerank_invoke_url``; an endpoint
    stored under any other key leaves the remote variant unselected and loads a
    local reranker instead.
    """

    @staticmethod
    def _build_graph(monkeypatch: pytest.MonkeyPatch, rerank_kwargs: dict[str, Any]) -> Any:
        """Build the default graph with only the rerank stage kept real."""

        class _PassthroughOperator(AbstractOperator):
            def preprocess(self, data: Any, **kwargs: Any) -> Any:
                return data

            def process(self, data: Any, **kwargs: Any) -> Any:
                return data

            def postprocess(self, data: Any, **kwargs: Any) -> Any:
                return data

        monkeypatch.setattr(
            "nemo_retriever.operators.embed.operators._BatchEmbedActor",
            _PassthroughOperator,
        )
        monkeypatch.setattr(
            "nemo_retriever.graph.retriever.RetrieveVdbOperator",
            _PassthroughOperator,
        )
        retriever = _make_retriever(rerank=True, rerank_kwargs=rerank_kwargs)
        return retriever._build_default_graph()

    @staticmethod
    def _rerank_node(graph: Any) -> Any:
        node = graph.roots[0]
        while node.children:
            node = node.children[0]
        return node

    @pytest.mark.parametrize("endpoint_key", ["rerank_invoke_url", "invoke_url"])
    def test_configured_endpoint_selects_remote_reranker(
        self, monkeypatch: pytest.MonkeyPatch, endpoint_key: str
    ) -> None:
        from nemo_retriever.operators.rerank import NemotronRerankActor, NemotronRerankCPUActor

        graph = self._build_graph(monkeypatch, {endpoint_key: "http://localhost:8015", "refine_factor": 4})
        node = self._rerank_node(graph)

        assert isinstance(node.operator, NemotronRerankActor)
        assert NemotronRerankActor.prefers_cpu_variant(node.operator_kwargs) is True

        delegate = node.operator._resolve_delegate()
        assert isinstance(delegate, NemotronRerankCPUActor)
        assert delegate._kwargs["rerank_invoke_url"] == "http://localhost:8015"

    def test_absent_endpoint_still_reranks_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_retriever.operators.rerank import NemotronRerankActor

        graph = self._build_graph(monkeypatch, {"local_reranker_backend": "hf"})
        node = self._rerank_node(graph)

        assert node.operator_kwargs.get("rerank_invoke_url") is None
        assert NemotronRerankActor.prefers_cpu_variant(node.operator_kwargs) is False


class TestRetrieverDefaults:
    def test_default_top_k(self) -> None:
        assert Retriever().top_k == 10

    def test_rerank_disabled_by_default(self) -> None:
        assert Retriever().rerank is False

    def test_retriever_alias_is_class(self) -> None:
        from nemo_retriever.graph.retriever import retriever

        assert retriever is Retriever


class TestQueryHitShaping:
    def test_page_dedup_preserves_first_page_hit(self) -> None:
        hits = [
            {"text": "first p1", "pdf_basename": "handbook", "page_number": 1},
            {"text": "second p1", "pdf_basename": "handbook", "page_number": 1},
            {"text": "first p2", "pdf_basename": "handbook", "page_number": 2},
        ]

        out = shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["first p1", "first p2"]

    def test_page_dedup_does_not_strip_dotted_basename(self) -> None:
        hits = [
            {"text": "v1", "pdf_basename": "report.v1", "page_number": 1},
            {"text": "v2", "pdf_basename": "report.v2", "page_number": 1},
        ]

        out = shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["v1", "v2"]

    def test_page_dedup_fallback_keeps_distinct_source_paths(self) -> None:
        hits = [
            {"text": "dir a", "source_id": "/dir_a/report.pdf", "page_number": 1},
            {"text": "dir b", "source_id": "/dir_b/report.pdf", "page_number": 1},
        ]

        out = shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["dir a", "dir b"]

    def test_filters_content_types_from_dict_and_json_metadata(self) -> None:
        hits = [
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 1},
            {"text": "table row", "metadata": '{"type": "table"}', "page_number": 2},
            {"text": "chart row", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = shape_query_hits(hits, top_k=10, content_types="text,table")

        assert [h["text"] for h in out] == ["text row", "table row"]

    def test_content_type_filter_accepts_emitted_images_alias(self) -> None:
        hits = [
            {"text": "image row", "metadata": {"type": "images"}, "page_number": 1},
            {"text": "table row", "metadata": {"type": "table"}, "page_number": 2},
            {"text": "chart row", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = shape_query_hits(hits, top_k=10, content_types="image")

        assert [h["text"] for h in out] == ["image row"]

    def test_uses_top_level_content_type_fallback(self) -> None:
        hits = [
            {"text": "top-level table", "metadata": "{}", "content_type": "table", "page_number": 1},
            {"text": "top-level chart", "metadata": "{}", "content_type": "chart", "page_number": 2},
        ]

        out = shape_query_hits(hits, top_k=10, content_types="table")

        assert [h["text"] for h in out] == ["top-level table"]

    def test_uses_metadata_content_type_fallback(self) -> None:
        hits = [
            {"text": "metadata table", "metadata": {"_content_type": "table"}, "page_number": 1},
            {"text": "metadata image", "metadata": {"_content_type": "image"}, "page_number": 2},
        ]

        out = shape_query_hits(hits, top_k=10, content_types="table")

        assert [h["text"] for h in out] == ["metadata table"]

    def test_does_not_backfill_excluded_content_types(self) -> None:
        hits = [
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 1},
            {"text": "chart row 1", "metadata": {"type": "chart"}, "page_number": 2},
            {"text": "chart row 2", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = shape_query_hits(hits, top_k=3, content_types="text")

        assert [h["text"] for h in out] == ["text row"]

    def test_content_type_filter_excludes_untyped_hits(self) -> None:
        hits = [
            {"text": "legacy row", "metadata": {}, "page_number": 1},
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 2},
        ]

        out = shape_query_hits(hits, top_k=10, content_types="text")

        assert [h["text"] for h in out] == ["text row"]

    def test_empty_content_type_allowlist_uses_python_api_name(self) -> None:
        with pytest.raises(ValueError, match="content_types must include"):
            shape_query_hits([], top_k=3, content_types=[])


class TestRunModeServiceRequiresHttpEmbed:
    def test_service_mode_errors_without_url(self) -> None:
        with pytest.raises(ValueError, match="run_mode='service'"):
            Retriever(run_mode="service", embed_kwargs={})._merge_embed_params()


def test_retriever_wires_custom_output_column_to_retrieve_operator() -> None:
    from nemo_retriever.operators.vdb import RetrieveVdbOperator

    retriever = _make_retriever(
        embed_kwargs={
            "model_name": "embedder",
            "embed_model_name": "embedder",
            "output_column": "vec",
        },
        vdb_kwargs={"vdb": MagicMock()},
    )
    graph = retriever._build_default_graph()

    retrieve = graph.roots[0].children[0].operator
    assert isinstance(retrieve, RetrieveVdbOperator)
    assert retrieve._embedding_column == "vec"


class TestRetrieveVdbOperatorPreprocess:
    def test_dataframe_to_vectors_reads_embedding_column(self) -> None:
        from nemo_retriever.operators.vdb import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "embedding": [[0.1, 0.2]],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.1, 0.2]]

    def test_dataframe_to_vectors_reads_arrow_embedding_column(self) -> None:
        from nemo_retriever.operators.vdb import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "embedding": pd.Series(
                    [[0.5, 0.25]],
                    dtype=pd.ArrowDtype(pa.list_(pa.float32())),
                ),
                "embedding_error": pd.Series([None], dtype=pd.ArrowDtype(pa.string())),
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.5, 0.25]]

    def test_dataframe_to_vectors_reads_custom_embedding_column(self) -> None:
        from nemo_retriever.operators.vdb import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "tags": [["finance", "annual"]],
                "page_scores": [[9.9, 8.8]],
                "vec": [[0.5, 0.25]],
                "vec_error": [None],
            }
        )
        op = RetrieveVdbOperator(
            vdb_op="lancedb",
            vdb_kwargs={"uri": "/tmp", "table_name": "t"},
            embedding_column="vec",
        )

        assert op.preprocess(df) == [[0.5, 0.25]]

    def test_dataframe_to_vectors_reports_embedding_error(self) -> None:
        from nemo_retriever.operators.vdb import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "embedding": [None],
                "embedding_error": ["429 Client Error: Too Many Requests"],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        with pytest.raises(ValueError, match="429 Client Error: Too Many Requests"):
            op.preprocess(df)

    def test_dataframe_to_vectors_requires_embedding_column(self) -> None:
        from nemo_retriever.operators.vdb import RetrieveVdbOperator

        df = pd.DataFrame({"text": ["a"], "metadata": [{"embedding": [0.7, 0.8]}], "page_scores": [[9.9, 8.8]]})
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})

        with pytest.raises(ValueError, match="Expected query embeddings in the 'embedding' column"):
            op.preprocess(df)


class TestRerankLongDataframe:
    def test_groups_by_query_order(self) -> None:
        from nemo_retriever.graph.retriever_utils import rerank_long_dataframe_to_hits

        df = pd.DataFrame(
            [
                {"query": "q1", "text": "b", "_hit": {"text": "b"}, "rerank_score": 0.5},
                {"query": "q1", "text": "a", "_hit": {"text": "a"}, "rerank_score": 0.9},
                {"query": "q2", "text": "c", "_hit": {"text": "c"}, "rerank_score": 0.3},
            ]
        )
        out = rerank_long_dataframe_to_hits(df, query_texts=["q1", "q2"], top_k=1, score_column="rerank_score")
        assert len(out) == 2
        assert out[0][0]["text"] == "a"
        assert out[0][0]["_rerank_score"] == 0.9
        assert out[1][0]["text"] == "c"
