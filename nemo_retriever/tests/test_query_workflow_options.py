# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

import nemo_retriever.query.service as query_service
import nemo_retriever.query.workflow as query_workflow
from nemo_retriever.query.options import (
    QueryEmbedOptions,
    QueryRerankOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryServiceOptions,
    QueryStorageOptions,
    ServiceQueryRequest,
)


def test_query_request_builds_retriever_kwargs_without_rerank(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(top_k=3),
        storage=QueryStorageOptions(lancedb_uri="/tmp/lancedb", table_name="docs"),
    )

    assert query_workflow.query_documents(request) == []
    assert retriever_calls == [
        {
            "top_k": 3,
            "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs"},
        }
    ]


def test_query_request_builds_retriever_kwargs_with_retrieval_mode(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(top_k=3, retrieval_mode="sparse"),
        storage=QueryStorageOptions(lancedb_uri="/tmp/lancedb", table_name="docs"),
    )

    assert query_workflow.query_documents(request) == []
    assert retriever_calls == [
        {
            "top_k": 3,
            "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs", "retrieval_mode": "sparse"},
        }
    ]


def test_query_request_builds_retriever_kwargs_with_embed_and_remote_rerank(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(
        query="deployment?",
        embed=QueryEmbedOptions(
            embed_invoke_url="http://embed:8000/v1/embeddings",
            embed_model_name="nvidia/llama-nemotron-embed-1b-v2",
        ),
        rerank=QueryRerankOptions(
            enabled=True,
            reranker_invoke_url="http://rerank:8000/v1/ranking",
        ),
    )

    assert query_workflow.query_documents(request) == []
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"},
            "embed_kwargs": {
                "embed_invoke_url": "http://embed:8000/v1/embeddings",
                "embedding_endpoint": "http://embed:8000/v1/embeddings",
                "model_name": "nvidia/llama-nemotron-embed-1b-v2",
                "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
            },
            "rerank": True,
            "rerank_kwargs": {
                "rerank_invoke_url": "http://rerank:8000/v1/ranking",
                "api_key": "nvapi-test",
            },
        }
    ]


def test_query_documents_uses_typed_request(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            query_calls.append((query, kwargs))
            return [{"text": "passage", "source": "doc.pdf", "page_number": 1}]

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)

    request = QueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(
            top_k=1,
            candidate_k=3,
            page_dedup=True,
            content_types="text,table",
        ),
    )

    assert query_workflow.query_documents(request) == [{"text": "passage", "source": "doc.pdf", "page_number": 1}]
    assert retriever_calls == [{"top_k": 1, "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"}}]
    assert query_calls == [
        (
            "deployment?",
            {
                "candidate_k": 3,
                "page_dedup": True,
                "content_types": "text,table",
            },
        )
    ]


@pytest.mark.parametrize(
    ("mode", "strategies"),
    [
        ("dense", ["semantic"]),
        ("hybrid", ["semantic", "lexical"]),
        ("sparse", ["lexical"]),
    ],
)
def test_query_documents_with_metadata_reports_resolved_strategy(monkeypatch, mode, strategies) -> None:
    class FakeRetriever:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def _resolve_lancedb_query_mode(self, _vdb_kwargs: Any) -> tuple[str, object, str, str, bool]:
            return mode, object(), "uri", "table", False

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return [{"text": query, "source": "doc.pdf", "page_number": 1}]

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)

    result = query_workflow.query_documents_with_metadata(QueryRequest(query="deployment?"))

    assert result.hits == [{"text": "deployment?", "source": "doc.pdf", "page_number": 1}]
    assert result.strategies == strategies


def test_service_query_uses_candidate_pool_and_preserves_local_shaping(monkeypatch) -> None:
    client_calls: list[dict[str, Any]] = []

    class FakeServiceClient:
        def __init__(self, *, base_url: str, api_token: str | None = None, **_kwargs: Any) -> None:
            client_calls.append({"base_url": base_url, "api_token": api_token})

        def query(self, query: str, *, top_k: int) -> list[list[dict[str, Any]]]:
            client_calls.append({"query": query, "top_k": top_k})
            return [
                [
                    {"text": "keep", "source": "doc.pdf", "page_number": 1, "metadata": {"type": "text"}},
                    {"text": "duplicate", "source": "doc.pdf", "page_number": 1, "metadata": {"type": "text"}},
                    {"text": "table", "source": "doc.pdf", "page_number": 2, "metadata": {"type": "table"}},
                ]
            ]

    monkeypatch.setattr(query_service, "RetrieverServiceClient", FakeServiceClient)

    request = ServiceQueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(
            top_k=2,
            candidate_k=3,
            page_dedup=True,
            content_types="text",
        ),
        service=QueryServiceOptions(service_url="http://svc:7670", service_api_token="secret"),
    )

    assert query_service.query_documents(request) == [
        {"text": "keep", "source": "doc.pdf", "page_number": 1, "metadata": {"type": "text"}},
    ]
    assert client_calls == [
        {"base_url": "http://svc:7670", "api_token": "secret"},
        {"query": "deployment?", "top_k": 3},
    ]


def test_service_query_validates_candidate_pool_before_calling_service(monkeypatch) -> None:
    class FakeServiceClient:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("service client should not be constructed")

    monkeypatch.setattr(query_service, "RetrieverServiceClient", FakeServiceClient)
    request = ServiceQueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(top_k=3, candidate_k=2),
    )

    with pytest.raises(ValueError, match=r"candidate_k \(2\) must be greater than or equal to top_k \(3\)"):
        query_service.query_documents(request)


@pytest.mark.parametrize("ef", [None, 1, 4096])
def test_optional_ef_is_resolved_and_forwarded_to_normal_query(monkeypatch, ef):
    calls = []

    class FakeRetriever:
        def __init__(self, **kwargs):
            assert "ef" not in kwargs["vdb_kwargs"]

        def query(self, text, **kwargs):
            calls.append((text, kwargs))
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(query="query", retrieval=QueryRetrievalOptions(candidate_k=50, ef=ef))
    plan = query_workflow.resolve_query_plan(request)
    assert plan.ef == ef
    expected = {"candidate_k": 50, "page_dedup": False, "content_types": None}
    if ef is not None:
        expected["vdb_kwargs"] = {"ef": ef}
    assert plan.query_kwargs() == expected
    assert query_workflow.query_documents(request) == []
    assert calls == [("query", expected)]


@pytest.mark.parametrize("ef", [True, False, 0, -1, 1.5, "4096", float("nan")])
def test_query_ef_requires_a_strict_positive_integer(ef):
    with pytest.raises(ValueError, match="ef must be a positive integer"):
        QueryRetrievalOptions(ef=ef)


def test_explicit_ef_rejects_unsupported_typed_query_paths():
    with pytest.raises(ValueError, match="dense or hybrid"):
        QueryRetrievalOptions(retrieval_mode="sparse", ef=4096)
    with pytest.raises(ValueError, match="service query"):
        ServiceQueryRequest(query="query", retrieval=QueryRetrievalOptions(ef=4096))
    with pytest.raises(ValueError, match="agentic query"):
        query_workflow.build_agentic_config(QueryRequest(query="query", retrieval=QueryRetrievalOptions(ef=4096)))
