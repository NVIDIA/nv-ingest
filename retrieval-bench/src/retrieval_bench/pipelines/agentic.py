# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agentic retrieval pipeline: dense retrieval augmented with an LLM agent.

The agent iteratively refines results using tool calls (retrieve, think,
final_results).  The dense backend is selected via the ``backend`` parameter
and initialised through the shared :func:`init_backend` helper.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

_CURRENT_QUERY_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "nemo_agentic_current_query_id", default=None
)

try:
    import torch
except ImportError:
    print("Error: Required GPU dependencies not installed.")
    print("Please install: pip install torch")
    sys.exit(1)

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from retrieval_bench.pipeline_evaluation.tracing import write_query_trace
from retrieval_bench.pipelines.backends import VALID_BACKENDS, infer_bright_task_key, init_backend

from retrieval_bench.nemo_agentic.agent import Agent
from retrieval_bench.nemo_agentic.configs import AgentConfig, LLMConfig
from retrieval_bench.nemo_agentic.llm_handler import LLM, is_error, normalize_messages_for_api
from retrieval_bench.nemo_agentic.tool_helpers import BaseTool, FinalResults, RetrieveToolBase, ThinkTool


# ---------------------------------------------------------------------------
# RetrieveTool adapter
# ---------------------------------------------------------------------------


class RetrieveTool(RetrieveToolBase):
    """
    Adapter tool wrapping a vidore-benchmark retriever singleton for the agent.

    Expects ``retriever.retrieve(query, return_markdown=True, excluded_ids=...)``
    to return ``(scores_dict, markdown_dict)``.
    """

    def __init__(self, retriever: Any, excluded_ids: Optional[List[str]] = None, top_k: int = 20):
        self.retriever = retriever
        self.excluded_ids = excluded_ids or []
        self._default_top_k = int(top_k)

    def _spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Search for documents related to a query using dense retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve.",
                            "default": self._default_top_k,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def _acall(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        effective_top_k = int(kwargs.pop("__art_top_k", top_k or self._default_top_k))

        try:
            scores, markdowns = self.retriever.retrieve(
                str(query),
                return_markdown=True,
                excluded_ids=self.excluded_ids,
            )
        except TypeError:
            scores, markdowns = self.retriever.retrieve(str(query), return_markdown=True)

        results: List[Dict[str, Any]] = []
        for doc_id, score in scores.items():
            results.append(
                {
                    "id": str(doc_id),
                    "score": float(score),
                    "text": str(markdowns.get(doc_id, "")),
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:effective_top_k]


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def extract_final_doc_ids(output_artifacts: Dict[str, Any], *, final_top_k: int = 10) -> Tuple[List[str], str]:
    """
    Extract the agent's final ranked doc ids with structured fallbacks.

    Fallback order:
      1) ``final_results`` tool-call args  (primary)
      2) ``rrf_scores``                    (secondary)
      3) ``top{final_top_k}_selection_result``  (tertiary)
    """
    traj = output_artifacts.get("agent_trajectories", []) or []
    for msg in reversed(traj):
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            if fn.get("name") != "final_results":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if isinstance(args, dict):
                doc_ids = args.get("doc_ids")
                if isinstance(doc_ids, list) and all(isinstance(i, str) for i in doc_ids):
                    return list(doc_ids), "final_results"

    rrf_scores = output_artifacts.get("rrf_scores", None)
    if isinstance(rrf_scores, dict) and rrf_scores:
        try:
            sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
            return [str(i) for i in sorted_ids[: int(final_top_k)]], "rrf"
        except Exception:
            pass

    sel = output_artifacts.get(f"top{int(final_top_k)}_selection_result", None)
    if isinstance(sel, dict):
        doc_ids = sel.get("doc_ids")
        if isinstance(doc_ids, list) and doc_ids and all(isinstance(i, str) for i in doc_ids):
            return list(doc_ids)[: int(final_top_k)], "selection_agent"
    if isinstance(sel, list) and sel and all(isinstance(i, str) for i in sel):
        return list(sel)[: int(final_top_k)], "selection_agent"

    return [], "none"


# ---------------------------------------------------------------------------
# LLM usage-tracking wrapper
# ---------------------------------------------------------------------------


def _wrap_llm_for_usage_tracking(llm: LLM) -> LLM:
    llm._accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0}  # type: ignore[attr-defined]
    llm._per_query_usage = {}  # type: ignore[attr-defined]
    _original_acompletion = llm.acompletion

    async def _tracked_acompletion(*args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = normalize_messages_for_api(kwargs["messages"])
        elif args:
            args_list = list(args)
            if isinstance(args_list[0], list):
                args_list[0] = normalize_messages_for_api(args_list[0])
            args = tuple(args_list)

        resp = await _original_acompletion(*args, **kwargs)

        model_resp = resp
        if isinstance(resp, dict) and "response" in resp:
            model_resp = resp.get("response")

        usage = getattr(model_resp, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            llm._accumulated_usage["prompt_tokens"] += prompt_tokens  # type: ignore[attr-defined]
            llm._accumulated_usage["completion_tokens"] += completion_tokens  # type: ignore[attr-defined]

            qid = _CURRENT_QUERY_ID.get()
            if qid:
                per_q = llm._per_query_usage.get(qid)  # type: ignore[attr-defined]
                if not isinstance(per_q, dict):
                    per_q = {"prompt_tokens": 0, "completion_tokens": 0}
                    llm._per_query_usage[qid] = per_q  # type: ignore[attr-defined]
                per_q["prompt_tokens"] = int(per_q.get("prompt_tokens", 0)) + prompt_tokens
                per_q["completion_tokens"] = int(per_q.get("completion_tokens", 0)) + completion_tokens
        return resp

    llm.acompletion = _tracked_acompletion  # type: ignore[assignment]
    return llm


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AgenticRetrievalPipeline(BasePipeline):
    """
    Dense retrieval augmented with an LLM agent that iteratively refines results.

    The ``backend`` parameter selects which dense retriever runs underneath.
    Additional backend-specific overrides can be passed via ``**backend_kwargs``.
    """

    def __init__(
        self,
        *,
        backend: str,
        retriever_top_k: int = 500,
        # Agent / LLM knobs
        num_concurrent: int = 1,
        llm_model: str,
        api_key: str = "os.environ/OPENAI_API_KEY",
        base_url: Optional[str] = "os.environ/OPENAI_BASE_URL",
        api_version: Optional[str] = None,
        reasoning_effort: str = "high",
        drop_params: bool = True,
        allowed_openai_params: Optional[List[str]] = None,
        raw_log_pardir: str = "nemo_agentic_logs",
        instant_log: bool = False,
        strict_error_handling: bool = False,
        target_top_k: int = 10,
        max_steps: int = 200,
        main_agent_only: bool = False,
        selection_topk_list: Optional[List[int]] = None,
        # All remaining kwargs are forwarded as backend overrides.
        **backend_kwargs: Any,
    ) -> None:
        load_dotenv()

        if backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. " f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
        self.backend = backend
        self.model_id = backend
        self.retriever_top_k = int(retriever_top_k)
        self.num_concurrent = max(1, int(num_concurrent))
        self._backend_kwargs = dict(backend_kwargs)

        # Resolve os.environ/... convention for base_url.
        if base_url and str(base_url).strip().startswith("os.environ/"):
            env_var = str(base_url).strip().removeprefix("os.environ/")
            base_url = os.environ.get(env_var, None)

        if selection_topk_list is None:
            selection_topk_list = [5, 10]

        self._agent_config = AgentConfig(
            system_prompt="02_v1.j2",
            target_top_k=int(target_top_k),
            max_steps=int(max_steps),
            main_agent_only=bool(main_agent_only),
            selection_topk_list=list(selection_topk_list),
        )
        self._llm_config = LLMConfig(
            model=str(llm_model),
            api_key=str(api_key),
            base_url=str(base_url) if base_url else None,
            api_version=str(api_version) if api_version else None,
            reasoning_effort=str(reasoning_effort),
            raw_log_pardir=str(raw_log_pardir),
            instant_log=bool(instant_log),
            strict_error_handling=bool(strict_error_handling),
            drop_params=bool(drop_params),
            allowed_openai_params=list(allowed_openai_params) if allowed_openai_params else None,
        )

        self.llm_model = self._llm_config.model

        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This pipeline requires a GPU.")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Async query loop
    # -----------------------------------------------------------------------

    @staticmethod
    def _summarize_preflight_error(error: Exception, model: str, base_url: Optional[str]) -> str:
        raw = re.sub(r"\s+", " ", str(error)).strip()
        hint_parts: List[str] = []

        low = raw.lower()
        if "notfound" in low or "404" in low:
            hint_parts.append("model not found for this endpoint/provider")
        elif "unauthorized" in low or "401" in low or "invalid api key" in low:
            hint_parts.append("authentication failed (check API key)")
        elif "forbidden" in low or "403" in low:
            hint_parts.append("access denied for this model/key")

        if str(model).startswith("unknown/"):
            hint_parts.append("model id starts with 'unknown/' (provider prefix likely incorrect)")

        hint = ""
        if hint_parts:
            hint = " Hint: " + "; ".join(hint_parts) + "."

        return (
            f"LLM preflight check failed before evaluation. "
            f"model={model!r}, base_url={base_url!r}.{hint} "
            f"Error: {raw}"
        )

    async def _validate_llm_preflight(self, llm: LLM) -> None:
        """
        Run a lightweight LLM call before processing the full dataset.

        This prevents confusing per-query fallback behavior when the endpoint/model
        is misconfigured (e.g. model not found, invalid base URL, bad API key).
        """
        probe_messages = [
            {
                "role": "user",
                "content": "Health check: reply with OK.",
            }
        ]

        try:
            response = await llm.acompletion(
                messages=probe_messages,
                tools=None,
                max_completion_tokens=8,
                num_retries=0,
            )
        except Exception as e:
            raise RuntimeError(
                self._summarize_preflight_error(
                    e, model=str(self._llm_config.model), base_url=self._llm_config.base_url
                )
            ) from e

        if is_error(response):
            # llm_handler returns string errors in non-strict mode.
            err = str(response).replace("LLMError:", "", 1).strip()
            raise RuntimeError(
                self._summarize_preflight_error(
                    RuntimeError(err), model=str(self._llm_config.model), base_url=self._llm_config.base_url
                )
            )

    async def _run_all_queries(
        self,
        *,
        query_ids: Sequence[str],
        queries: Sequence[str],
        llm: LLM,
        excluded_ids_by_query: Dict[str, List[str]],
        tracing_context: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], Dict[str, float]]:
        agent_config = self._agent_config
        final_top_k = int(agent_config.target_top_k or 10)

        results: Dict[str, Dict[str, float]] = {}
        per_query_trace: Dict[str, Dict[str, Any]] = {}
        per_query_ms: Dict[str, float] = {}
        total_queries = len(query_ids)
        completed = 0
        completed_lock = asyncio.Lock()
        sem = asyncio.Semaphore(max(1, int(getattr(self, "num_concurrent", 1) or 1)))

        async def _process_query(q_idx: int, qid: str, query_text: Any) -> None:
            nonlocal completed
            async with sem:
                excluded = excluded_ids_by_query.get(qid, []) or []

                t0 = time.perf_counter()

                trace_entry: Dict[str, Any] = {}
                trace_entry["query_text"] = str(query_text)
                trace_entry["llm_model"] = str(
                    getattr(self, "_llm_config", None).model if hasattr(self, "_llm_config") else ""
                )

                retrieve_tool = RetrieveTool(
                    retriever=self._active_retriever,
                    excluded_ids=list(excluded),
                    top_k=int(agent_config.target_top_k or 20),
                )
                tool_map: Dict[str, BaseTool] = {"retrieve": retrieve_tool}

                tk = (
                    int(agent_config.target_top_k) if agent_config.enforce_top_k and agent_config.target_top_k else None
                )
                tool_map["final_results"] = FinalResults(top_k=tk)

                if not agent_config.disable_think:
                    think = ThinkTool(extended_relevance=agent_config.extended_relevance)
                    tool_map[think.name] = think

                agent = Agent(config=agent_config, llm=llm, tool_map=tool_map, session_id=qid)

                doc_ids: List[str] = []
                source = "none"
                try:
                    if hasattr(llm, "_per_query_usage"):
                        llm._per_query_usage[str(qid)] = {  # type: ignore[attr-defined]
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }

                    token = _CURRENT_QUERY_ID.set(str(qid))
                    try:
                        output = await agent.run_for_input(
                            query=str(query_text),
                            exclude_docids=set(excluded),
                        )
                    finally:
                        _CURRENT_QUERY_ID.reset(token)

                    doc_ids, source = extract_final_doc_ids(output, final_top_k=final_top_k)

                    trajectories = output.get("agent_trajectories", []) or []
                    trace_entry["trajectory_steps"] = len(trajectories)
                    trace_entry["llm_turns"] = sum(1 for m in trajectories if m.get("role") == "assistant")
                    trace_entry["retrieval_calls"] = sum(
                        1
                        for m in trajectories
                        for tc in (m.get("tool_calls") or [])
                        if (tc.get("function") or {}).get("name") == "retrieve"
                    )
                    trace_entry["result_source"] = source
                    trace_entry["rrf_used"] = source == "rrf"
                    trace_entry["selection_agent_ran"] = any(
                        isinstance(k, str) and k.startswith("top") and k.endswith("_selection_result")
                        for k in output.keys()
                    )
                    trace_entry["doc_ids"] = list(doc_ids)
                    retrieval_log = output.get("retrieval_log", []) if isinstance(output, dict) else []
                    trace_entry["num_retrieval_calls"] = (
                        int(len(retrieval_log)) if isinstance(retrieval_log, list) else 0
                    )
                    agent_extra = output.get("agent_extra_data", None) if isinstance(output, dict) else None
                    trace_entry["query_rewriting_used"] = bool(isinstance(agent_extra, dict) and len(agent_extra) > 0)
                    rrf_scores = output.get("rrf_scores", None) if isinstance(output, dict) else None
                    if isinstance(rrf_scores, dict) and rrf_scores:
                        try:
                            top_rrf_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:final_top_k]
                            trace_entry["rrf_scores_summary"] = [str(i) for i in top_rrf_ids]
                        except Exception:
                            trace_entry["rrf_scores_summary"] = []
                    trace_entry["fallback_used"] = False
                except Exception as e:
                    try:
                        scores = self._active_retriever.retrieve(str(query_text), excluded_ids=list(excluded))
                    except TypeError:
                        scores = self._active_retriever.retrieve(str(query_text))
                    doc_ids = [str(i) for i in sorted(scores, key=scores.get, reverse=True)[:final_top_k]]
                    trace_entry["llm_error"] = f"{type(e).__name__}: {e}"
                    trace_entry["result_source"] = "retriever_fallback"
                    trace_entry["doc_ids"] = list(doc_ids)
                    trace_entry["fallback_used"] = True

                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                per_query_ms[qid] = float(elapsed_ms)

                per_q_usage = getattr(llm, "_per_query_usage", {}).get(str(qid), {})  # type: ignore[attr-defined]
                pt = int(per_q_usage.get("prompt_tokens", 0) or 0)
                ct = int(per_q_usage.get("completion_tokens", 0) or 0)
                trace_entry["llm_usage"] = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                }
                trace_entry["elapsed_ms"] = float(elapsed_ms)

                run_for_q = {did: float(len(doc_ids) - rank) for rank, did in enumerate(doc_ids)}
                results[qid] = run_for_q
                per_query_trace[qid] = trace_entry

                if isinstance(tracing_context, dict):
                    try:
                        write_query_trace(
                            traces_dir=str(tracing_context.get("traces_dir", "traces")),
                            trace_run_name=str(tracing_context.get("trace_run_name", "unnamed")),
                            dataset=str(tracing_context.get("dataset", self.dataset_name)),
                            dataset_dir=str(tracing_context.get("dataset_dir", str(self.dataset_name).split("/")[-1])),
                            query_id=str(qid),
                            pipeline_class=str(tracing_context.get("pipeline_class", self.__class__.__name__)),
                            model_id=str(tracing_context.get("model_id", "")),
                            retrieval_time_milliseconds=float(elapsed_ms),
                            run=dict(run_for_q),
                            split=str(tracing_context.get("split", "test")),
                            language=tracing_context.get("language", None),
                            query_ids_selector=tracing_context.get("query_ids_selector", None),
                            pipeline_trace=dict(trace_entry),
                        )
                    except Exception as e:
                        print(f"WARNING: failed to write per-query trace for query_id={qid}: {type(e).__name__}: {e}")

                async with completed_lock:
                    completed += 1
                    if completed == 1 or completed % 10 == 0 or completed == total_queries:
                        print(
                            f"  Agent queries completed: {completed}/{total_queries}"
                            f" (concurrency={self.num_concurrent})"
                        )

        tasks = [
            asyncio.create_task(_process_query(q_idx=i, qid=str(qid), query_text=query_text))
            for i, (qid, query_text) in enumerate(zip(query_ids, queries))
        ]
        await asyncio.gather(*tasks)

        return results, per_query_trace, per_query_ms

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def index(self, corpus_ids: List[str], corpus_images: List[Any], corpus_texts: List[str]) -> None:
        super().index(corpus_ids=corpus_ids, corpus_images=corpus_images, corpus_texts=corpus_texts)

        dataset_name = self.dataset_name
        task_key = infer_bright_task_key(dataset_name)

        corpus = [{"image": img, "markdown": md} for img, md in zip(corpus_images, corpus_texts)]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_init0 = time.perf_counter()

        active_retriever, effective_model_id, init_info = init_backend(
            self.backend,
            dataset_name=dataset_name,
            corpus_ids=corpus_ids,
            corpus=corpus,
            top_k=self.retriever_top_k,
            task_key=task_key,
            overrides=self._backend_kwargs or None,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        retriever_init_ms = (time.perf_counter() - t_init0) * 1000.0

        print(f"Using backend {self.backend} ({effective_model_id})")
        print(f"Retriever init: {retriever_init_ms / 1000.0:.2f}s")

        self._active_retriever = active_retriever
        self._init_info = init_info
        self._retriever_init_ms = retriever_init_ms

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        tracing_context = getattr(self, "tracing_context", None)
        excluded_ids_by_query = getattr(self, "excluded_ids_by_query", None) or {}

        llm = _wrap_llm_for_usage_tracking(LLM(self._llm_config))

        try:
            # Fail fast once if the LLM endpoint/model is invalid.
            asyncio.run(self._validate_llm_preflight(llm))

            results, per_query_trace, per_query_ms = asyncio.run(
                self._run_all_queries(
                    query_ids=query_ids,
                    queries=queries,
                    llm=llm,
                    excluded_ids_by_query=excluded_ids_by_query,
                    tracing_context=tracing_context,
                )
            )
        finally:
            self._active_retriever.unload()

        infos: Dict[str, Any] = {
            **self._init_info,
            "retriever_top_k": self.retriever_top_k,
            "retriever_init_ms": float(self._retriever_init_ms),
            "per_query_retrieval_time_milliseconds": per_query_ms,
            "per_query_trace": per_query_trace,
        }
        return results, infos
