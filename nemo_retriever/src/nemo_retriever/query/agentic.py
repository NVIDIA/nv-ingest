# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-backed agentic retrieval mode.

The implementation is intentionally additive: it composes the existing graph
operators and wraps :class:`nemo_retriever.retriever.Retriever` without changing
the standard retrieval path.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import pandas as pd

from nemo_retriever.common.params import build_embed_option_kwargs
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.models import VL_RERANK_MODEL
from nemo_retriever.query.agentic_options import (
    AGENTIC_DEFAULT_CLIENT,
    agentic_llm_client_error,
    agentic_float_range_value,
    agentic_int_min_error,
    agentic_int_value,
    agentic_temperature_error,
)
from nemo_retriever.tools.recall.beir import (
    BeirDataset,
    VALID_BEIR_DOC_ID_FIELDS,
    build_beir_run_from_ranked_doc_ids,
    compute_beir_metrics,
    load_beir_dataset,
)
from nemo_retriever.tools.recall.beir import _extract_doc_id_from_hit as _beir_doc_id_from_hit
from nemo_retriever.tools.recall.core import (
    _hit_to_audio_segment_key,
    _normalize_pdf_name,
    _normalize_query_df,
    _recall_at_k,
)
from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)

AGENTIC_RETRIEVER_TOP_K = 10
AGENTIC_TARGET_TOP_K = 10
AGENTIC_NUM_CONCURRENT = 1
AGENTIC_TEXT_TRUNCATION = 0
AGENTIC_PARALLEL_TOOL_CALLS = None
AGENTIC_RRF_K = 60
AGENTIC_REACT_MAX_STEPS = 50
AGENTIC_TEMPERATURE = None  # agent LLM sampling temperature; None leaves it unset
AGENTIC_MAX_TOKENS: Optional[int] = None
AGENTIC_LLM_BACKEND = "in_process"
AGENTIC_LLM_BACKENDS = frozenset({"openai_compatible", "in_process"})
AGENTIC_LOCAL_LLM_BACKEND = "vllm"
AGENTIC_LOCAL_LLM_MODEL = "nemotron-8b"
AGENTIC_LOCAL_LLM_BACKENDS = frozenset({"vllm"})

# Raw embeddings are large and are not part of the classic hit envelope, so they
# are dropped from the hits kept for rehydration.
_UNCACHED_HIT_FIELDS = frozenset({"vector", "embedding"})

# Selection stages that can only rank documents a retrieve hop returned. A hit
# missing for one of these means the captured metadata and the selected ids
# disagree, which is an internal inconsistency rather than agent behavior.
_RETRIEVED_ONLY_RESULT_SOURCES = frozenset({"rrf", "selection_agent"})


@dataclass(frozen=True)
class AgenticRetrieveResult:
    """Ranked agentic documents and provider-reported usage by query ID.

    Attributes
    ----------
    documents:
        Ranked rows for all requested queries. See
        :meth:`AgenticRetriever.retrieve` for the column contract.
    usage:
        Mapping from caller query ID to a stage-keyed provider usage breakdown,
        ``{query_id: {stage: provider_usage}}``. Repeated LLM calls within each
        stage are summed. Queries for which the provider reported no usage are
        omitted.
    """

    documents: pd.DataFrame
    usage: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class AgenticAnswerResult:
    """Agentic answers, cited retrieval hits, and provider usage by query ID.

    ``answers`` contains one row per input query with ``answer``, ``citations``,
    ``citation_hits``, success/error fields, and the original query identifiers.
    Citation hits preserve the model's citation order and use the same classic-hit
    hydration contract as :meth:`AgenticRetriever.retrieve`.
    """

    answers: pd.DataFrame
    usage: dict[str, dict[str, Any]]


class AgenticQueryInputOperator(AbstractOperator):
    """Adapt ``Retriever(graph=...)`` input DataFrames to agentic query schema."""

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"AgenticQueryInputOperator expects a pd.DataFrame, got {type(data).__name__}.")
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        out = data.copy()
        if "query_text" not in out.columns:
            if "query" in out.columns:
                out["query_text"] = out["query"].astype(str)
            elif "text" in out.columns:
                out["query_text"] = out["text"].astype(str)
            else:
                raise ValueError("Agentic query graph input requires 'query_text', 'query', or 'text'.")
        if "query_id" not in out.columns:
            out["query_id"] = [str(idx) for idx in range(len(out.index))]
        return out[["query_id", "query_text"]]

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        return data


class AgenticSelectionOutputOperator(AbstractOperator):
    """Convert final agentic selection DataFrame to ``Retriever`` hit-list output."""

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"AgenticSelectionOutputOperator expects a pd.DataFrame, got {type(data).__name__}.")
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> list[list[dict[str, Any]]]:
        _ = kwargs
        if data.empty:
            return []
        required = {"query_id", "doc_id", "rank"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Agentic selection output missing required columns: {sorted(missing)}")

        hits: list[list[dict[str, Any]]] = []
        for _query_id, group in data.groupby("query_id", sort=False):
            query_hits: list[dict[str, Any]] = []
            for _, row in group.sort_values("rank").iterrows():
                hit = row.to_dict()
                doc_id = str(hit.get("doc_id", ""))
                if doc_id and not hit.get("pdf_page"):
                    hit["pdf_page"] = doc_id
                query_hits.append(hit)
            hits.append(query_hits)
        return hits

    def postprocess(self, data: list[list[dict[str, Any]]], **kwargs: Any) -> list[list[dict[str, Any]]]:
        _ = kwargs
        return data


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    """Configuration for graph-backed agentic retrieval."""

    vdb_op: str = "lancedb"
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    query_embedder: Optional[str] = None
    query_embedder_provider_prefix: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    local_hf_batch_size: int = 32
    local_query_embed_backend: Optional[str] = None
    reranker: Optional[str] = None
    reranker_endpoint: Optional[str] = None
    reranker_api_key: str = ""
    local_reranker_backend: str = "vllm"
    embed_modality: str = "text"
    llm_backend: Optional[str] = None
    llm_model: str = ""
    invoke_url: Optional[str] = None
    local_llm_backend: str = AGENTIC_LOCAL_LLM_BACKEND
    local_hf_cache_dir: Optional[str] = None
    local_gpu_memory_utilization: float = 0.8
    local_tensor_parallel_size: int = 1
    local_max_model_len: Optional[int] = None
    local_max_num_seqs: Optional[int] = None
    api_key: Optional[str] = None
    react_max_steps: int = AGENTIC_REACT_MAX_STEPS
    text_truncation: int = AGENTIC_TEXT_TRUNCATION
    num_concurrent: int = AGENTIC_NUM_CONCURRENT
    # Forwarded verbatim as the OpenAI `reasoning_effort` field on every LLM
    # call when explicitly configured.
    reasoning_effort: Optional[str] = None
    # Sampling temperature sent on every agent LLM call. ``None`` leaves it unset
    # so the endpoint/model default applies.
    temperature: Optional[float] = AGENTIC_TEMPERATURE
    # LLM client used to build the ReAct and selection agent LLMs. Optional:
    # when unset it defaults to ``"callable"`` for both in-process (local vLLM)
    # runs and remote (openai_compatible) runs -- the difference is which
    # completion callable the operator injects. A remote-only client (anything
    # other than ``"callable"``) may be named explicitly to override the default;
    # the valid set is the ``nemo_agent`` LLM backend registry.
    llm_client: Optional[str] = None
    # Optional upper bound on tokens in each agent LLM response.
    max_tokens: Optional[int] = AGENTIC_MAX_TOKENS
    # Final number of documents the agent targets/selects and the pipeline returns.
    # Drives the ReAct target, the RRF/selection cut, and the per-hop fetch depth
    # (which is raised to at least this). Defaults to 10.
    top_k: int = AGENTIC_TARGET_TOP_K
    # Wider pre-filter and pre-rerank candidate pool for each retrieval hop.
    candidate_k: Optional[int] = None

    def __post_init__(self) -> None:
        invoke_url = _none_if_empty(self.invoke_url)
        object.__setattr__(self, "invoke_url", invoke_url)

        explicit_llm_backend = str(self.llm_backend or "").strip().lower()
        if explicit_llm_backend and explicit_llm_backend not in AGENTIC_LLM_BACKENDS:
            raise ValueError(f"llm_backend must be one of {sorted(AGENTIC_LLM_BACKENDS)}; got {self.llm_backend!r}")
        inferred_llm_backend = "openai_compatible" if invoke_url else AGENTIC_LLM_BACKEND
        llm_backend = explicit_llm_backend or inferred_llm_backend
        if invoke_url and llm_backend != "openai_compatible":
            raise ValueError(
                "invoke_url selects the openai_compatible agentic LLM backend; "
                "omit invoke_url for in-process local LLMs."
            )
        if not invoke_url and llm_backend == "openai_compatible":
            raise ValueError(
                "llm_backend='openai_compatible' requires invoke_url. Omit llm_backend to use in-process local LLMs."
            )
        object.__setattr__(self, "llm_backend", llm_backend)

        # Two independent axes: ``llm_backend`` is WHERE compute runs (in-process
        # vs a remote endpoint); ``llm_client`` is WHICH nemo_agent adapter drives
        # it. ``callable`` spans both, because it wraps a completion callable that
        # may be either an in-process engine or the shared HTTP client.
        # Validity of an explicit name is delegated to the nemo_agent registry
        # via ``agentic_llm_client_error``.
        explicit_client = str(self.llm_client or "").strip().lower() or None
        if explicit_client is not None:
            client_error = agentic_llm_client_error(explicit_client, field_name="llm_client")
            if client_error:
                raise ValueError(client_error)
        if llm_backend == "in_process":
            if explicit_client is not None and explicit_client != "callable":
                raise ValueError(
                    "in-process agentic runs use the 'callable' LLM client; "
                    "provide invoke_url to use a remote client."
                )
            llm_client = "callable"
        else:
            llm_client = explicit_client or AGENTIC_DEFAULT_CLIENT
        object.__setattr__(self, "llm_client", llm_client)

        local_llm_backend = _normalize_agentic_choice(
            self.local_llm_backend,
            AGENTIC_LOCAL_LLM_BACKENDS,
            field_name="local_llm_backend",
            default=AGENTIC_LOCAL_LLM_BACKEND,
        )
        object.__setattr__(self, "local_llm_backend", local_llm_backend)

        llm_model = str(self.llm_model or "").strip()
        if not llm_model:
            if llm_backend == "in_process":
                llm_model = AGENTIC_LOCAL_LLM_MODEL
            else:
                raise ValueError("Agentic retrieval with invoke_url requires a non-empty llm_model.")

        if llm_backend == "in_process":
            from nemo_retriever.models.local.agent_llm import is_supported_agent_llm_model, supported_agent_llm_names

            if not is_supported_agent_llm_model(llm_model):
                supported = ", ".join(supported_agent_llm_names())
                raise ValueError(
                    f"Unsupported in-process agentic LLM model {llm_model!r}. "
                    "Custom in-process agent LLMs are not supported yet. "
                    "Provide invoke_url for a custom/self-hosted OpenAI-compatible endpoint, "
                    f"or choose one of: {supported}."
                )
        object.__setattr__(self, "llm_model", llm_model)
        for field_name, value, min_value in (
            ("react_max_steps", self.react_max_steps, 1),
            ("text_truncation", self.text_truncation, 0),
            ("top_k", self.top_k, 1),
            ("num_concurrent", self.num_concurrent, 1),
        ):
            integer_error = agentic_int_min_error(value, field_name=field_name, min_value=min_value)
            if integer_error:
                raise ValueError(integer_error)
            object.__setattr__(self, field_name, agentic_int_value(value, field_name=field_name))

        if self.candidate_k is not None:
            candidate_error = agentic_int_min_error(self.candidate_k, field_name="candidate_k", min_value=1)
            if candidate_error:
                raise ValueError(candidate_error)
            candidate_k = agentic_int_value(self.candidate_k, field_name="candidate_k")
            if candidate_k < int(self.top_k):
                raise ValueError(f"candidate_k ({candidate_k}) must be greater than or equal to top_k ({self.top_k}).")
            object.__setattr__(self, "candidate_k", candidate_k)

        local_tp_error = agentic_int_min_error(
            self.local_tensor_parallel_size, field_name="local_tensor_parallel_size", min_value=1
        )
        if local_tp_error:
            raise ValueError(local_tp_error)
        object.__setattr__(
            self,
            "local_tensor_parallel_size",
            agentic_int_value(self.local_tensor_parallel_size, field_name="local_tensor_parallel_size"),
        )

        for field_name in ("local_max_model_len", "local_max_num_seqs", "max_tokens"):
            value = getattr(self, field_name)
            if value is None:
                continue
            integer_error = agentic_int_min_error(value, field_name=field_name, min_value=1)
            if integer_error:
                raise ValueError(integer_error)
            object.__setattr__(self, field_name, agentic_int_value(value, field_name=field_name))

        local_gpu_memory_utilization = agentic_float_range_value(
            self.local_gpu_memory_utilization,
            field_name="local_gpu_memory_utilization",
            min_value=0.0,
            max_value=1.0,
            min_exclusive=True,
        )
        object.__setattr__(self, "local_gpu_memory_utilization", local_gpu_memory_utilization)

        # Temperature: ``None`` stays unset (endpoint/model default). The range
        # check uses a local sentinel so in-process runs (invoke_url=None) get the
        # in-process bound instead of the hosted-endpoint bound.
        if self.temperature is not None:
            temperature_invoke_url = (
                self.invoke_url if self.llm_backend == "openai_compatible" else "local://in-process"
            )
            temperature_error = agentic_temperature_error(
                self.temperature,
                invoke_url=temperature_invoke_url,
                field_name="temperature",
            )
            if temperature_error:
                raise ValueError(temperature_error)
            object.__setattr__(self, "temperature", float(self.temperature))


def _normalize_agentic_choice(value: object, valid: frozenset[str], *, field_name: str, default: str) -> str:
    normalized = str(value or default).strip().lower()
    if normalized not in valid:
        raise ValueError(f"{field_name} must be one of {sorted(valid)}; got {value!r}")
    return normalized


def _build_agent_chat_completion_fn(cfg: AgenticRetrievalConfig) -> Any | None:
    if cfg.llm_backend == "openai_compatible":
        return None

    if cfg.llm_backend == "in_process":
        from nemo_retriever.models import create_local_agent_llm

        return create_local_agent_llm(
            str(cfg.llm_model),
            backend=str(cfg.local_llm_backend),
            hf_cache_dir=_none_if_empty(cfg.local_hf_cache_dir),
            gpu_memory_utilization=float(cfg.local_gpu_memory_utilization),
            tensor_parallel_size=int(cfg.local_tensor_parallel_size),
            max_model_len=cfg.local_max_model_len,
            max_num_seqs=cfg.local_max_num_seqs,
        )

    raise ValueError(f"Unsupported agentic llm_backend {cfg.llm_backend!r}")


class AgenticRetriever:
    """Run graph-backed agentic retrieval over query IDs and query texts."""

    def __init__(
        self,
        cfg: AgenticRetrievalConfig,
        *,
        match_mode: str = "pdf_page",
        doc_id_field: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._match_mode = str(match_mode)
        self._doc_id_field = str(doc_id_field) if doc_id_field else None
        if self._doc_id_field is not None and self._doc_id_field not in VALID_BEIR_DOC_ID_FIELDS:
            raise ValueError(f"Unsupported doc_id_field: {self._doc_id_field}")
        embed_kwargs = build_embed_option_kwargs(
            cfg.embedding_endpoint,
            cfg.query_embedder,
            embed_api_key=cfg.embedding_api_key,
            embed_model_provider_prefix=cfg.query_embedder_provider_prefix,
        )
        if cfg.local_query_embed_backend is not None:
            embed_kwargs["local_ingest_embed_backend"] = str(cfg.local_query_embed_backend)
        embed_kwargs.update(
            {
                "input_type": "query",
                "inference_batch_size": int(cfg.local_hf_batch_size),
                "embed_inference_batch_size": int(cfg.local_hf_batch_size),
            }
        )

        self._retriever = Retriever(
            vdb_kwargs={
                "vdb_op": str(cfg.vdb_op),
                "vdb_kwargs": dict(cfg.vdb_kwargs or {}),
            },
            embed_kwargs=embed_kwargs,
            top_k=AGENTIC_RETRIEVER_TOP_K,
            rerank=bool(cfg.reranker),
            rerank_kwargs={
                "model_name": cfg.reranker or VL_RERANK_MODEL,
                # NemotronRerankActor selects its remote CPU variant on
                # ``rerank_invoke_url``; any other key silently loads a local model.
                "rerank_invoke_url": (cfg.reranker_endpoint or "").strip() or None,
                "api_key": cfg.reranker_api_key,
                "local_reranker_backend": str(cfg.local_reranker_backend),
                "modality": str(cfg.embed_modality),
            },
        )
        self._lock = threading.Lock()
        self._chat_completion_fn: Any | None = None
        # Classic hits keyed by (graph query_id, doc_id), captured on every
        # retrieve hop before the agent boundary reduces them to
        # doc_id/text/score. The query component keeps query-dependent fields
        # (especially scores) isolated when a batch retrieves the same document.
        # Only final selected ids are rehydrated; reset for each retrieve().
        self._hit_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._hit_cache_lock = threading.Lock()

    def _get_chat_completion_fn(self) -> Any | None:
        if self._cfg.llm_backend == "openai_compatible":
            return None
        with self._lock:
            if self._chat_completion_fn is None:
                # Instance-owned, same pattern as Retriever's cached embed/rerank
                # graph: load once per AgenticRetriever and reuse across queries.
                self._chat_completion_fn = _build_agent_chat_completion_fn(self._cfg)
            return self._chat_completion_fn

    def _build_react_operator(
        self,
        *,
        mode: Literal["select", "answer"],
        chat_completion_fn: Any | None,
    ) -> Any:
        """Build the shared ReAct stage for select and answer workflows."""
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        target_top_k = int(self._cfg.top_k)
        return ReActAgentOperator(
            invoke_url=_none_if_empty(self._cfg.invoke_url),
            llm_model=str(self._cfg.llm_model),
            retriever_fn=self._retrieve_for_agent,
            retriever_fn_accepts_query_id=True,
            retriever_top_k=max(AGENTIC_RETRIEVER_TOP_K, target_top_k),
            target_top_k=target_top_k,
            mode=mode,
            max_steps=int(self._cfg.react_max_steps),
            api_key=_none_if_empty(self._cfg.api_key),
            parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
            num_concurrent=int(self._cfg.num_concurrent),
            reasoning_effort=self._cfg.reasoning_effort,
            temperature=self._cfg.temperature,
            backend=self._cfg.llm_client,
            max_tokens=self._cfg.max_tokens,
            chat_completion_fn=chat_completion_fn,
        )

    def unload(self) -> None:
        """Release the in-process agent LLM owned by this retriever.

        OpenAI-compatible endpoint mode is a no-op. Local vLLM mode shuts down
        this instance's EngineCore so CLI jobs can exit cleanly. Embed
        and rerank models stay on ``self._retriever`` and are released with the
        process.
        """

        with self._lock:
            chat_fn = self._chat_completion_fn
            self._chat_completion_fn = None
        if chat_fn is None:
            return
        unload = getattr(chat_fn, "unload", None)
        if callable(unload):
            unload()

    def answer(self, query_ids: Sequence[str], query_texts: Sequence[str]) -> pd.DataFrame:
        """Return one integrated agentic answer per query.

        Answer mode ends inside the ReAct loop and therefore intentionally
        bypasses RRF and the selection agent. Each row includes citation IDs and
        their rehydrated classic retrieval hits in citation order.

        Parameters
        ----------
        query_ids:
            Unique caller-owned IDs for the queries.
        query_texts:
            Query strings aligned positionally with ``query_ids``.

        Returns
        -------
        pd.DataFrame
            One row per query containing the answer, validated citation IDs,
            rehydrated citation hits, status, message, and error fields.

        Raises
        ------
        ValueError
            If the sequences have different lengths or ``query_ids`` contains
            duplicates.
        """

        return self.answer_with_usage(query_ids, query_texts).answers

    def answer_with_usage(
        self,
        query_ids: Sequence[str],
        query_texts: Sequence[str],
    ) -> AgenticAnswerResult:
        """Return integrated agentic answers and exact per-query LLM usage.

        Parameters
        ----------
        query_ids:
            Unique caller-owned IDs used as keys in the returned usage mapping.
        query_texts:
            Query strings aligned positionally with ``query_ids``.

        Returns
        -------
        AgenticAnswerResult
            Answer rows and a stage-keyed provider usage breakdown for each
            query that reported usage.

        Raises
        ------
        ValueError
            If the sequences have different lengths or ``query_ids`` contains
            duplicates.
        """

        if len(query_ids) != len(query_texts):
            raise ValueError("query_ids and query_texts must have the same length.")
        caller_query_ids = [str(query_id) for query_id in query_ids]
        if len(set(caller_query_ids)) != len(caller_query_ids):
            raise ValueError("query_ids must be unique.")

        with self._hit_cache_lock:
            self._hit_cache.clear()

        react_operator = self._build_react_operator(
            mode="answer",
            chat_completion_fn=self._get_chat_completion_fn(),
        )
        input_df = pd.DataFrame(
            {
                "query_id": caller_query_ids,
                "query_text": [str(query_text) for query_text in query_texts],
            }
        )
        usage_by_query: dict[str, dict[str, Any]] = {}
        try:
            raw_answers = react_operator.run(input_df)
        finally:
            for query_id in caller_query_ids:
                breakdown = react_operator.pop_query_usage(query_id)
                if breakdown:
                    usage_by_query[query_id] = breakdown

        with self._hit_cache_lock:
            hit_cache = dict(self._hit_cache)
        answers = _rehydrate_answer_citations(raw_answers, hit_cache)
        return AgenticAnswerResult(answers=answers, usage=usage_by_query)

    def retrieve(self, query_ids: Sequence[str], query_texts: Sequence[str]) -> pd.DataFrame:
        """Return selected ranked documents for each query.

        Columns are ``SelectionAgentOperator``'s (``query_id``, ``doc_id``,
        ``rank``, ``message``, ``result_source``) plus ``hit``: the classic
        ``RetrievalHit`` (``text``, ``source``, ``page_number``, ``metadata``, …)
        captured on the retrieve hop that first returned the document, or ``{}``
        when no hop returned it. Use :func:`rehydrated_agentic_hit` to layer the
        agentic annotations onto that hit.
        """
        return self.retrieve_with_usage(query_ids, query_texts).documents

    def retrieve_with_usage(
        self,
        query_ids: Sequence[str],
        query_texts: Sequence[str],
    ) -> AgenticRetrieveResult:
        """Return ranked documents and exact per-query agent LLM usage.

        Parameters
        ----------
        query_ids:
            Caller-owned IDs used as keys in the returned usage mapping.
        query_texts:
            Query strings aligned positionally with ``query_ids``.

        Returns
        -------
        AgenticRetrieveResult
            Ranked documents and a stage-keyed provider usage breakdown for
            each query that reported usage.

        Raises
        ------
        ValueError
            If ``query_ids`` and ``query_texts`` have different lengths.
        """

        if len(query_ids) != len(query_texts):
            raise ValueError("query_ids and query_texts must have the same length.")
        caller_query_ids = [str(query_id) for query_id in query_ids]

        with self._hit_cache_lock:
            self._hit_cache.clear()

        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        # Honor the caller's requested top_k. The per-hop fetch is raised to at least
        # the default pool depth so the agent always sees a full working set even for
        # small top_k.
        target_top_k = int(self._cfg.top_k)
        chat_completion_fn = self._get_chat_completion_fn()

        react_operator = self._build_react_operator(
            mode="select",
            chat_completion_fn=chat_completion_fn,
        )
        selection_operator = SelectionAgentOperator(
            invoke_url=_none_if_empty(self._cfg.invoke_url),
            llm_model=str(self._cfg.llm_model),
            top_k=target_top_k,
            api_key=_none_if_empty(self._cfg.api_key),
            parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
            text_truncation=int(self._cfg.text_truncation),
            reasoning_effort=self._cfg.reasoning_effort,
            temperature=self._cfg.temperature,
            backend=self._cfg.llm_client,
            max_tokens=self._cfg.max_tokens,
            chat_completion_fn=chat_completion_fn,
        )
        pipeline = (
            AgenticQueryInputOperator()
            >> react_operator
            >> RRFAggregatorOperator(k=AGENTIC_RRF_K)
            >> selection_operator
            >> AgenticSelectionOutputOperator()
        )
        graph_retriever = Retriever(
            graph=pipeline,
            top_k=target_top_k,
            embed_kwargs={"text_column": "query_text"},
        )
        usage_by_query: dict[str, dict[str, Any]] = {}
        try:
            raw_hits = graph_retriever.queries(
                [str(query_text) for query_text in query_texts],
                top_k=target_top_k,
            )
        finally:
            # Retriever.queries assigns positional graph query IDs ("0", "1",
            # ...). Pop both operator-owned backends after all query workers
            # finish, then restore the caller's IDs.
            for position, caller_query_id in enumerate(caller_query_ids):
                # Each backend has already summed repeated calls within its
                # stages. Operator stage names are disjoint
                # (main_agent vs top{K}_agent), so joining the maps is enough.
                breakdown = {
                    **react_operator.pop_query_usage(str(position)),
                    **selection_operator.pop_query_usage(str(position)),
                }
                if breakdown:
                    usage_by_query[caller_query_id] = breakdown

        result = _raw_hits_to_agentic_result(caller_query_ids, raw_hits)
        with self._hit_cache_lock:
            hit_cache = dict(self._hit_cache)
        return AgenticRetrieveResult(
            documents=_rehydrate_selected_hits(result, hit_cache),
            usage=usage_by_query,
        )

    def _retrieve_for_agent(self, query_text: str, top_k: int, *, query_id: str = "") -> list[dict[str, Any]]:
        """Retriever callback used by ``ReActAgentOperator``.

        Retrieval is serialized across concurrent ReAct workers via ``self._lock``
        because the shared ``Retriever`` is not assumed thread-safe. This caps the
        retrieval hop at single-threaded throughput; it is intentional and not the
        bottleneck, since per-query cost is dominated by the multi-step LLM calls,
        which still run concurrently under ``num_concurrent > 1``.
        """

        candidate_k = max(int(self._cfg.candidate_k), int(top_k)) if self._cfg.candidate_k is not None else None
        chunk_pool_k = candidate_k if candidate_k is not None else int(top_k)
        with self._lock:
            hits = self._retriever.query(str(query_text), top_k=chunk_pool_k, candidate_k=candidate_k)

        docs: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        doc_id_field = getattr(self, "_doc_id_field", None)
        for hit in hits:
            hit_dict = dict(hit)
            doc_id = (
                _beir_doc_id_from_hit(hit_dict, doc_id_field=doc_id_field)
                if doc_id_field is not None
                else _doc_id_for_match_mode(hit_dict, match_mode=self._match_mode)
            )
            if not doc_id:
                continue
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            # Keep the untruncated hit for rehydration; truncation below only
            # bounds what the agent LLM sees. First occurrence per query and
            # doc_id wins.
            with self._hit_cache_lock:
                self._hit_cache.setdefault(
                    (str(query_id), doc_id),
                    {key: value for key, value in hit_dict.items() if key not in _UNCACHED_HIT_FIELDS},
                )
            text = str(hit_dict.get("text", ""))
            if int(self._cfg.text_truncation) > 0:
                text = text[: int(self._cfg.text_truncation)]
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "score": _hit_score(hit_dict),
                }
            )
            if len(docs) >= int(top_k):
                break
        return docs


def rehydrated_agentic_hit(hit: Any, *, doc_id: str, rank: int, result_source: str) -> dict[str, Any]:
    """Layer the agentic annotations onto one rehydrated classic hit.

    ``hit`` is a value from the ``hit`` column of
    :meth:`AgenticRetriever.retrieve`. The result carries the classic
    ``RetrievalHit`` fields plus ``doc_id``, ``rank``, and ``result_source``, so
    agentic output matches classic retrieval output while still reporting which
    stage selected the document.

    Args:
        hit: Rehydrated classic hit mapping from ``AgenticRetriever.retrieve``.
            Non-dict values represent unavailable hit metadata and produce an
            annotation-only result.
        doc_id: Non-empty document identifier selected by the agentic pipeline.
        rank: One-based position in the final selected result list.
        result_source: Stage that produced the final selection, normally
            ``final_results``, ``selection_agent``, or ``rrf``.

    Returns:
        A new dictionary containing the classic hit fields, when available,
        plus top-level ``doc_id``, ``rank``, and ``result_source`` annotations.
    """

    rehydrated: dict[str, Any] = dict(hit) if isinstance(hit, dict) else {}
    rehydrated.update({"doc_id": doc_id, "rank": rank, "result_source": result_source})
    return rehydrated


def _rehydrate_answer_citations(
    answers: pd.DataFrame,
    hit_cache: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Attach full retrieval hits to answer citations without changing IDs."""

    result = answers.copy()
    citation_hits_per_answer: list[list[dict[str, Any]]] = []
    for row in result.to_dict("records"):
        query_id = str(row.get("query_id", ""))
        raw_citations = row.get("citations")
        citations = raw_citations if isinstance(raw_citations, list) else []
        citation_hits: list[dict[str, Any]] = []
        for rank, raw_doc_id in enumerate(citations, start=1):
            doc_id = str(raw_doc_id)
            hit = hit_cache.get((query_id, doc_id), {})
            if not hit:
                logger.warning(
                    "Could not rehydrate answer citation %r for query_id=%r; "
                    "the citation ID was not present in the retrieval cache.",
                    doc_id,
                    query_id,
                )
            citation_hits.append(
                rehydrated_agentic_hit(
                    hit,
                    doc_id=doc_id,
                    rank=rank,
                    result_source="citation",
                )
            )
        citation_hits_per_answer.append(citation_hits)
    result["citation_hits"] = pd.Series(citation_hits_per_answer, dtype="object", index=result.index)
    return result


def _rehydrate_selected_hits(
    result: pd.DataFrame,
    hit_cache: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Attach the captured classic hit for each selected doc_id."""

    doc_ids = result["doc_id"].astype(str).tolist() if "doc_id" in result.columns else []
    retrieval_query_ids = (
        result["_retrieval_query_id"].astype(str).tolist()
        if "_retrieval_query_id" in result.columns
        else [""] * len(doc_ids)
    )
    if "result_source" in result.columns:
        result_sources = result["result_source"].astype(str).tolist()
    else:
        result_sources = [""] * len(doc_ids)

    hits: list[dict[str, Any]] = []
    for retrieval_query_id, doc_id, result_source in zip(retrieval_query_ids, doc_ids, result_sources):
        hit = hit_cache.get((retrieval_query_id, doc_id))
        if hit is None:
            _log_missing_rehydration_hit(doc_id, result_source)
            hit = {}
        hits.append(dict(hit))
    result["hit"] = pd.Series(hits, dtype="object", index=result.index)
    result.drop(columns=["_retrieval_query_id"], inplace=True, errors="ignore")
    return result


def _log_missing_rehydration_hit(doc_id: str, result_source: str) -> None:
    if result_source in _RETRIEVED_ONLY_RESULT_SOURCES:
        logger.error(
            "Agentic hit metadata is missing for doc_id=%r selected by result_source=%r, which can only rank "
            "documents a retrieve hop returned; the selected id and the captured hits disagree. "
            "Returning agentic annotations without classic hit fields.",
            doc_id,
            result_source,
        )
        return
    logger.warning(
        "Agentic hit metadata is missing for doc_id=%r selected by result_source=%r: no retrieve hop returned "
        "this document, so the agent named an id it never saw. Returning agentic annotations without classic "
        "hit fields.",
        doc_id,
        result_source,
    )


def _raw_hits_to_agentic_result(query_ids: Sequence[str], raw_hits: Sequence[Sequence[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # The agentic graph (AgenticQueryInputOperator) assigns POSITIONAL query_ids
    # "0".."N-1" to its inputs, independent of the caller's real ids. Each hit
    # therefore carries its positional index; map that back through `query_ids`
    # to recover the caller's real id. This is robust to (a) ThreadPool
    # completion-order reordering at num_concurrent>1, (b) queries that produced
    # no rows (gaps don't shift anything), and (c) sharded offset ranges where the
    # positional index != the caller's real id (e.g. query_ids=["1000".."1049"]).
    # For a full sweep with sequential ids ("0".."N-1") this is a no-op vs the
    # old positional zip.
    n = len(query_ids)
    for pos, hits in enumerate(raw_hits):
        for rank, hit in enumerate(hits, start=1):
            raw_qid = hit.get("query_id")
            retrieval_query_id = str(raw_qid) if raw_qid is not None else str(pos)
            if raw_qid is not None and str(raw_qid).isdigit() and int(raw_qid) < n:
                qid = str(query_ids[int(raw_qid)])
            elif pos < n:
                qid = str(query_ids[pos])
            else:
                qid = str(raw_qid) if raw_qid is not None else ""
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": str(hit.get("doc_id") or hit.get("pdf_page") or ""),
                    "rank": int(hit.get("rank", rank)),
                    "message": str(hit.get("message", "")),
                    "result_source": str(hit.get("result_source", "")),
                    "_retrieval_query_id": retrieval_query_id,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["query_id", "doc_id", "rank", "message", "result_source", "_retrieval_query_id"])
    return pd.DataFrame(rows)


def _doc_id_for_match_mode(hit: dict[str, Any], *, match_mode: str) -> str:
    if match_mode == "audio_segment":
        return _hit_to_audio_segment_key(hit) or ""
    if match_mode == "pdf_only":
        return _doc_id_from_hit(hit)
    return _pdf_page_from_hit(hit)


def _pdf_page_from_hit(hit: dict[str, Any]) -> str:
    pdf_page = hit.get("pdf_page")
    if isinstance(pdf_page, str) and pdf_page.strip():
        return pdf_page.strip()

    source = hit.get("source") or hit.get("source_id") or hit.get("path")
    page_number = hit.get("page_number")
    if source and page_number is not None:
        return f"{Path(str(source)).stem}_{page_number}"
    return _doc_id_from_hit(hit)


def _doc_id_from_hit(hit: dict[str, Any]) -> str:
    for key in ("pdf_basename", "source_id", "path", "source", "doc_id"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_pdf_name(Path(value).stem)
    return ""


def _hit_score(hit: dict[str, Any]) -> float:
    for key in ("_rerank_score", "_score", "score"):
        if key in hit:
            try:
                return float(hit[key])
            except (TypeError, ValueError):
                return 0.0
    if "_distance" in hit:
        try:
            return -float(hit["_distance"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    return stripped


def _query_ids_from_normalized_query_df(df_query: pd.DataFrame) -> list[str]:
    if "query_id" in df_query.columns:
        return df_query["query_id"].astype(str).tolist()
    return [str(idx) for idx in range(len(df_query.index))]


def _agentic_result_to_ranked_doc_ids(query_ids: Sequence[str], result: pd.DataFrame) -> list[list[str]]:
    ranked_by_qid: dict[str, list[str]] = {str(query_id): [] for query_id in query_ids}
    seen_by_qid: dict[str, set[str]] = {str(query_id): set() for query_id in query_ids}
    if result.empty:
        return [ranked_by_qid[str(query_id)] for query_id in query_ids]

    required = {"query_id", "doc_id", "rank"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Agentic result missing required columns: {sorted(missing)}")

    for query_id, group in result.groupby("query_id", sort=False):
        qid = str(query_id)
        if qid not in ranked_by_qid:
            continue
        for _, row in group.sort_values("rank").iterrows():
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id or doc_id in seen_by_qid[qid]:
                continue
            seen_by_qid[qid].add(doc_id)
            ranked_by_qid[qid].append(doc_id)

    return [ranked_by_qid[str(query_id)] for query_id in query_ids]


def run_agentic_audio_recall_evaluation(
    *,
    query_csv: Path,
    cfg: AgenticRetrievalConfig,
    ks: Sequence[int] = (1, 5, 10),
    audio_match_tolerance_secs: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[list[str]], dict[str, float]]:
    """Run agentic retrieval against an audio recall CSV and compute recall metrics."""

    df_query = _normalize_query_df(pd.read_csv(query_csv), match_mode="audio_segment")
    query_ids = _query_ids_from_normalized_query_df(df_query)
    queries = df_query["query"].astype(str).tolist()
    gold_doc_ids = df_query["golden_answer"].astype(str).tolist()

    retriever = AgenticRetriever(cfg, match_mode="audio_segment")
    try:
        result = retriever.retrieve(query_ids, queries)
    finally:
        retriever.unload()
    retrieved_doc_ids = _agentic_result_to_ranked_doc_ids(query_ids, result)
    ks_sorted = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_sorted:
        raise ValueError("ks must contain at least one positive integer")
    metrics = {
        f"recall@{k}": _recall_at_k(
            gold_doc_ids,
            retrieved_doc_ids,
            k,
            match_mode="audio_segment",
            audio_match_tolerance_secs=float(audio_match_tolerance_secs),
        )
        for k in ks_sorted
    }
    return df_query, result, gold_doc_ids, retrieved_doc_ids, metrics


def run_agentic_beir_evaluation(
    *,
    loader: str,
    dataset_name: str,
    cfg: AgenticRetrievalConfig,
    split: str = "test",
    query_language: str | None = None,
    doc_id_field: str = "pdf_basename",
    ks: Sequence[int] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], dict[str, dict[str, float]], dict[str, float]]:
    """Load a BEIR-style dataset, run agentic retrieval, and compute metrics."""

    dataset = load_beir_dataset(
        loader,
        dataset_name=dataset_name,
        split=split,
        query_language=query_language,
        doc_id_field=doc_id_field,
    )
    result, ranked_doc_ids = agentic_beir_retrieve(dataset, cfg, doc_id_field=doc_id_field)
    run = build_beir_run_from_ranked_doc_ids(dataset.query_ids, ranked_doc_ids)
    metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    df_query = pd.DataFrame({"query_id": dataset.query_ids, "query": dataset.queries})
    return df_query, result, dataset.qrels, run, metrics


def agentic_beir_retrieve(
    dataset: BeirDataset,
    cfg: AgenticRetrievalConfig,
    *,
    doc_id_field: str = "pdf_basename",
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Run agentic (ReAct) retrieval over a pre-loaded BEIR dataset.

    Returns the raw agentic result DataFrame (``query_id``/``doc_id``/``rank``/…)
    and the per-query ranked doc-id lists aligned to ``dataset.query_ids``.
    Doc-id matching follows ``doc_id_field`` (BEIR semantics, applied inside the
    agent's retrieve hop), so the ranked ids align with the qrels keys produced by
    ``load_beir_dataset(..., doc_id_field=...)``. Splitting this out lets callers
    that already hold a loaded dataset (e.g. batch evaluation) reuse the agent's
    retrieve+rank step without re-loading or re-implementing it.
    """
    retriever = AgenticRetriever(cfg, match_mode="pdf_page", doc_id_field=doc_id_field)
    try:
        result = retriever.retrieve(
            list(dataset.query_ids),
            list(dataset.queries),
        )
    finally:
        retriever.unload()
    ranked_doc_ids = _agentic_result_to_ranked_doc_ids(list(dataset.query_ids), result)
    return result, ranked_doc_ids
