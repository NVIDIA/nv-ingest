# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

from nemo_retriever.common.params import build_embed_option_kwargs
from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.vdb.lancedb_capabilities import LanceRetrievalMode
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models import VL_RERANK_MODEL
from nemo_retriever.query.options import QueryRequest, QueryRerankOptions

_LOCAL_VL_RERANK_MODEL = VL_RERANK_MODEL


@dataclass(frozen=True)
class QueryDocumentsResult:
    hits: list[RetrievalHit]
    strategies: list[str]


@dataclass(frozen=True)
class AgenticQueryDocumentsResult:
    """Agentic query hits and normalized provider-reported LLM usage.

    Attributes
    ----------
    hits:
        Ranked agentic document hits in the root query output shape.
    usage:
        Normalized usage with ``input_tokens``, observed cache-read
        ``cache_tokens``, ``output_tokens``, ``total_tokens``, and the original
        provider breakdown under ``stages``. Token totals are integers or
        ``None`` when the provider did not report enough information for an
        exact total. The mapping is empty when no provider usage was reported.
    """

    hits: list[dict[str, Any]]
    usage: dict[str, Any]


@dataclass(frozen=True)
class AgenticAnswerDocumentsResult:
    """One integrated agentic answer with validated, rehydrated citations."""

    answer: str | None
    citations: list[str] | None
    citation_hits: list[dict[str, Any]]
    succeeded: bool
    message: str | None
    error: dict[str, Any] | None
    usage: dict[str, Any]


def _strategies_for_retrieval_mode(mode: LanceRetrievalMode | None) -> list[str]:
    if mode == "hybrid":
        return ["semantic", "lexical"]
    if mode == "sparse":
        return ["lexical"]
    return ["semantic"]


def _build_rerank_kwargs(options: QueryRerankOptions) -> dict[str, str]:
    """Build kwargs for the rerank stage using the existing root query behavior."""
    reranker_url = (options.reranker_invoke_url or "").strip()
    if reranker_url:
        rerank_kwargs: dict[str, str] = {"rerank_invoke_url": reranker_url}
        if options.reranker_model_name:
            rerank_kwargs["model_name"] = options.reranker_model_name
        api_key = resolve_remote_api_key(options.reranker_api_key)
        if api_key is not None:
            rerank_kwargs["api_key"] = api_key
        return rerank_kwargs

    local: dict[str, str] = {"model_name": options.reranker_model_name or _LOCAL_VL_RERANK_MODEL}
    if options.reranker_backend:
        local["local_reranker_backend"] = options.reranker_backend
    return local


def _build_retriever_kwargs(request: QueryRequest) -> dict[str, Any]:
    return resolve_query_plan(request).retriever_kwargs()


@dataclass(frozen=True)
class ResolvedQueryPlan:
    """Resolved Retriever query configuration reusable across many queries."""

    top_k: int
    candidate_k: int | None
    page_dedup: bool
    content_types: str | None
    lancedb_uri: str
    table_name: str
    retrieval_mode: str
    embed_kwargs: dict[str, Any]
    rerank: bool
    rerank_kwargs: dict[str, Any]

    def retriever_kwargs(self) -> dict[str, Any]:
        vdb_kwargs: dict[str, Any] = {
            "uri": self.lancedb_uri,
            "table_name": self.table_name,
        }
        if self.retrieval_mode != "auto":
            vdb_kwargs["retrieval_mode"] = self.retrieval_mode

        kwargs: dict[str, Any] = {
            "top_k": self.top_k,
            "vdb_kwargs": vdb_kwargs,
        }
        if self.embed_kwargs:
            kwargs["embed_kwargs"] = dict(self.embed_kwargs)
        if self.rerank:
            kwargs["rerank"] = True
            if self.rerank_kwargs:
                kwargs["rerank_kwargs"] = dict(self.rerank_kwargs)
        return kwargs

    def create_retriever(self) -> Retriever:
        return Retriever(**self.retriever_kwargs())

    def query_kwargs(self) -> dict[str, Any]:
        return {
            "candidate_k": self.candidate_k,
            "page_dedup": self.page_dedup,
            "content_types": self.content_types,
        }


def resolve_query_plan(request: QueryRequest) -> ResolvedQueryPlan:
    """Resolve root query options once so callers can reuse a Retriever."""
    embed_kwargs = build_embed_option_kwargs(
        request.embed.embed_invoke_url,
        request.embed.embed_model_name,
        embed_model_provider_prefix=request.embed.embed_model_provider_prefix,
    )
    rerank_kwargs = _build_rerank_kwargs(request.rerank) if request.rerank.enabled else {}
    content_types = request.retrieval.content_types
    if content_types is not None and not isinstance(content_types, str):
        content_types = ",".join(str(value) for value in content_types)
    return ResolvedQueryPlan(
        top_k=int(request.retrieval.top_k),
        candidate_k=request.retrieval.candidate_k,
        page_dedup=bool(request.retrieval.page_dedup),
        content_types=content_types,
        lancedb_uri=str(request.storage.lancedb_uri),
        table_name=str(request.storage.table_name),
        retrieval_mode=str(request.retrieval.retrieval_mode),
        embed_kwargs=embed_kwargs,
        rerank=bool(request.rerank.enabled),
        rerank_kwargs=rerank_kwargs,
    )


def query_documents_with_metadata(request: QueryRequest) -> QueryDocumentsResult:
    """Run the SDK query path and return hits plus resolved retrieval strategy metadata."""
    plan = resolve_query_plan(request)
    retriever = plan.create_retriever()
    mode: LanceRetrievalMode | None = None
    resolve_mode = getattr(retriever, "_resolve_lancedb_query_mode", None)
    if callable(resolve_mode):
        lancedb_mode = resolve_mode(None)
        if lancedb_mode is not None:
            mode = lancedb_mode[0]
    hits = retriever.query(
        request.query,
        **plan.query_kwargs(),
    )
    return QueryDocumentsResult(hits=hits, strategies=_strategies_for_retrieval_mode(mode))


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run the SDK query path used by the root CLI."""
    return query_documents_with_metadata(request).hits


def build_agentic_config(request: QueryRequest, *, top_k: int | None = None) -> "AgenticRetrievalConfig":
    """Build an :class:`AgenticRetrievalConfig` from a :class:`QueryRequest`.

    Used by the single-query CLI path (:func:`agentic_query_documents`). The
    LanceDB ``uri``/``table_name``, embedding config, and (when ``rerank`` is
    enabled) reranker config are passed straight through to the wrapped
    ``Retriever`` that backs the agent's ``retrieve`` tool. ``top_k`` overrides the
    final document count the agent targets.
    """
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    api_key = resolve_remote_api_key(request.embed.embed_api_key)
    vdb_kwargs: dict[str, Any] = {"uri": request.storage.lancedb_uri, "table_name": request.storage.table_name}
    if request.retrieval.retrieval_mode != "auto":
        vdb_kwargs["retrieval_mode"] = request.retrieval.retrieval_mode
    cfg_kwargs: dict[str, Any] = {
        "vdb_op": "lancedb",
        "vdb_kwargs": vdb_kwargs,
        "top_k": int(top_k if top_k is not None else request.retrieval.top_k),
        "candidate_k": request.retrieval.candidate_k,
        "embedding_endpoint": request.embed.embed_invoke_url,
        "embedding_api_key": api_key or "",
        "llm_model": request.agentic.llm_model,
        "invoke_url": request.agentic.invoke_url,
        "local_llm_backend": request.agentic.local_llm_backend,
        "local_hf_cache_dir": request.agentic.local_hf_cache_dir,
        "local_gpu_memory_utilization": request.agentic.local_gpu_memory_utilization,
        "local_tensor_parallel_size": request.agentic.local_tensor_parallel_size,
        "local_max_model_len": request.agentic.local_max_model_len,
        "local_max_num_seqs": request.agentic.local_max_num_seqs,
        "api_key": api_key,
        "reasoning_effort": request.agentic.reasoning_effort,
        "react_max_steps": int(request.agentic.react_max_steps),
        "text_truncation": int(request.agentic.text_truncation),
        "num_concurrent": int(request.agentic.num_concurrent),
        "temperature": request.agentic.temperature,
        "llm_client": request.agentic.llm_client,
    }
    if request.agentic.llm_backend:
        cfg_kwargs["llm_backend"] = request.agentic.llm_backend
    if request.embed.embed_model_name:
        cfg_kwargs["query_embedder"] = request.embed.embed_model_name
    if request.embed.embed_model_provider_prefix:
        cfg_kwargs["query_embedder_provider_prefix"] = request.embed.embed_model_provider_prefix
    if request.rerank.enabled:
        # `reranker` doubles as the on/off gate (rerank=bool(cfg.reranker)) and the
        # model name, so fall back to the default model when only --rerank is given.
        cfg_kwargs["reranker"] = request.rerank.reranker_model_name or _LOCAL_VL_RERANK_MODEL
        cfg_kwargs["reranker_endpoint"] = request.rerank.reranker_invoke_url
        cfg_kwargs["reranker_api_key"] = resolve_remote_api_key(request.rerank.reranker_api_key) or ""
        if request.rerank.reranker_backend:
            cfg_kwargs["local_reranker_backend"] = request.rerank.reranker_backend

    return AgenticRetrievalConfig(**cfg_kwargs)


def build_agentic_retriever(request: QueryRequest) -> "AgenticRetriever":
    """Construct an :class:`AgenticRetriever` from a :class:`QueryRequest`."""
    from nemo_retriever.query.agentic import AgenticRetriever

    return AgenticRetriever(build_agentic_config(request))


def agentic_query_documents(request: QueryRequest) -> list[dict[str, Any]]:
    """Run agentic (ReAct) retrieval for a single query and return ranked hits.

    The agent selects documents rather than chunks, but each returned hit carries
    the same fields as the dense ``query_documents`` path (``text``, ``source``,
    ``page_number``, ``metadata``, …), rehydrated from the retrieve hop that
    returned the document, plus the agentic annotations ``doc_id``, ``rank``, and
    ``result_source`` (``final_results`` / ``rrf`` / ``selection_agent``). The
    LanceDB ``uri``/``table_name``, embedding config, and (when ``--rerank`` is
    enabled) reranker config are passed straight through to the wrapped
    ``Retriever`` that backs the agent's ``retrieve`` tool. Reranking therefore
    applies per agent retrieval hop.
    """
    retriever = build_agentic_retriever(request)
    try:
        result = retriever.retrieve(["0"], [str(request.query)])
        return _agentic_rows_to_hits(result, top_k=request.retrieval.top_k)
    finally:
        # One-shot CLI/Python entry: tear down cached local vLLM EngineCore so the
        # process can exit instead of hanging on a live child worker.
        retriever.unload()


def agentic_query_documents_with_metadata(request: QueryRequest) -> AgenticQueryDocumentsResult:
    """Run one agentic query and return ranked hits plus exact LLM usage."""
    from nemo_retriever._agentic.nemo_agent.llm.usage import normalize_usage_breakdown

    retriever = build_agentic_retriever(request)
    try:
        result = retriever.retrieve_with_usage(["0"], [str(request.query)])
        return AgenticQueryDocumentsResult(
            hits=_agentic_rows_to_hits(result.documents, top_k=request.retrieval.top_k),
            usage=normalize_usage_breakdown(result.usage.get("0")),
        )
    finally:
        retriever.unload()


def agentic_answer_documents(request: QueryRequest) -> dict[str, Any]:
    """Run integrated ReAct answer mode for one query without usage metadata."""

    result = agentic_answer_documents_with_metadata(request)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "citation_hits": result.citation_hits,
        "succeeded": result.succeeded,
        "message": result.message,
        "error": result.error,
    }


def agentic_answer_documents_with_metadata(request: QueryRequest) -> AgenticAnswerDocumentsResult:
    """Run integrated ReAct answer mode for one query with exact LLM usage."""
    from nemo_retriever._agentic.nemo_agent.llm.usage import normalize_usage_breakdown

    retriever = build_agentic_retriever(request)
    try:
        result = retriever.answer_with_usage(["0"], [str(request.query)])
        if result.answers.empty:
            return AgenticAnswerDocumentsResult(
                answer=None,
                citations=None,
                citation_hits=[],
                succeeded=False,
                message=None,
                error={
                    "category": "unexpected",
                    "message": (
                        "Agentic answer produced no result. Retry the query; if it fails again, "
                        "check the configured agent LLM and embedding endpoints, verify that the "
                        "selected table contains indexed data, and inspect the service logs."
                    ),
                },
                usage=normalize_usage_breakdown(result.usage.get("0")),
            )
        row = result.answers.iloc[0]
        citations = row.get("citations")
        if not isinstance(citations, list):
            citations = None
        citation_hits = row.get("citation_hits")
        if not isinstance(citation_hits, list):
            citation_hits = []
        error_category = row.get("error_category")
        error: dict[str, Any] | None = None
        if isinstance(error_category, str) and error_category:
            error = {
                "category": error_category,
                "message": str(row.get("error_message") or ""),
            }
            exception_class = row.get("error_exception_class")
            if isinstance(exception_class, str) and exception_class:
                error["exception_class"] = exception_class
        answer = row.get("answer")
        message = row.get("message")
        return AgenticAnswerDocumentsResult(
            answer=answer if isinstance(answer, str) else None,
            citations=citations,
            citation_hits=citation_hits,
            succeeded=bool(row.get("succeeded", False)),
            message=message if isinstance(message, str) else None,
            error=error,
            usage=normalize_usage_breakdown(result.usage.get("0")),
        )
    finally:
        retriever.unload()


def _agentic_rows_to_hits(result: Any, *, top_k: int) -> list[dict[str, Any]]:
    """Convert an agentic result DataFrame to ranked, rehydrated hits."""
    from nemo_retriever.query.agentic import rehydrated_agentic_hit

    if "rank" in result.columns:
        result = result.sort_values("rank")
    ranked: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        ranked.append(
            rehydrated_agentic_hit(
                row.get("hit"),
                doc_id=doc_id,
                rank=int(row.get("rank", len(ranked) + 1)),
                result_source=str(row.get("result_source", "")),
            )
        )
        if len(ranked) >= top_k:
            break
    return ranked
