# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agentic-retrieval loop: one Agent class covering select and answer modes.

The generic loop mechanics (step loop, tool dispatch, error policy, raw-IO
logging, result building) live in ``loop.py`` and are shared with
``selection_agent.py``; this module adds everything retrieval-specific —
construction of the standard tool set, the user-message bootstrap, and the
over-fetch/dedup/exclusion bookkeeping around retrieve tools.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from .cache_propagation import PropagationPacer
from .llm import BaseLLMBackend, bind_query_id
from .loop import (
    BaseAgentLoopConfig,
    ToolExecutionError,
    _awrite_json,
    _BaseAgentLoop,
    _RunState,
    build_auto_continue_msg,
)
from .prompts import render_system_prompt
from .results import AgentRunResult
from .tools import (
    BaseEndTool,
    BaseRetrieveTool,
    BaseTool,
    FinalResults,
    LogAnswer,
    RetrieveContext,
    ThinkTool,
)
from .tools.base_tool import ToolError, tool_error_text
from .tools.retrieve import retrieve_output_to_msg_content

logger = logging.getLogger(__name__)


class AgentConfig(BaseAgentLoopConfig):
    """Configuration for the agent.

    Pure data — LLM and tool *instances* are constructor arguments, not
    config. ``mode`` picks what a run produces: ``"select"`` ranks document
    IDs, ``"answer"`` writes a final answer; ``enforce_top_k`` and
    ``target_top_k`` apply to ``"select"`` only. ``system_prompt`` is a
    packaged prompt name or a filesystem path; ``None`` takes the mode's
    default. The loop-policy fields (``max_steps``, ``on_error``, logging and
    pacing knobs) are inherited from
    :class:`~nemo_agent.loop.BaseAgentLoopConfig`.
    """

    mode: Literal["select", "answer"] = "select"
    system_prompt: Optional[str] = None
    enforce_top_k: bool = True
    target_top_k: Optional[int] = 10
    extended_relevance: bool = True
    enable_think: bool = False
    end_tool_with_msg: bool = True
    user_msg_type: Literal["simple", "with_results"] = "with_results"
    ensure_new_docs: bool = True


class Agent(_BaseAgentLoop):
    """LLM agent loop with retrieval tools.

    Construction wires everything the loop needs; per-query state lives in a
    private run state, so one instance safely serves many (including
    concurrent) runs. Standard tools are assembled internally: the think tool
    (``config.enable_think``), the mode's end tool (``select`` →
    ``final_results``, ``answer`` → ``log_answer``; overridable via the
    ``end_tool`` argument), and the required primary ``retrieve_tool``
    (which also powers the ``user_msg_type="with_results"`` bootstrap).
    ``tool_overrides`` adds/replaces tools by name — extra
    :class:`BaseRetrieveTool` instances get the same retrieval bookkeeping as
    the primary one.

    Error behavior is driven by ``config.on_error``; see
    :data:`~nemo_agent.loop.KNOWN_LLM_ERRORS` and
    :class:`~nemo_agent.results.AgentError`.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLMBackend,
        retrieve_tool: BaseRetrieveTool,
        *,
        end_tool: Optional[BaseEndTool] = None,
        tool_overrides: Optional[Dict[str, BaseTool]] = None,
    ) -> None:
        if not isinstance(config, AgentConfig):
            raise TypeError(f"config must be an AgentConfig, got {type(config).__name__}.")
        super().__init__(config=config, llm=llm)
        if not isinstance(retrieve_tool, BaseRetrieveTool):
            raise TypeError(
                "retrieve_tool must be a BaseRetrieveTool (build one with "
                f"create_retrieve_tool or subclass it), got {type(retrieve_tool).__name__}."
            )

        if end_tool is not None and not isinstance(end_tool, BaseEndTool):
            raise TypeError(f"end_tool must be a BaseEndTool, got {type(end_tool).__name__}.")
        is_select = config.mode == "select"
        enforce_top_k = config.enforce_top_k if is_select else False
        target_top_k = config.target_top_k if is_select else None
        if is_select:
            default_prompt = "06_select_lean_v1.j2"
            end_payload_phrase = "with your selected doc_ids"
            if end_tool is None:
                top_k = int(target_top_k) if enforce_top_k and target_top_k else None
                end_tool = FinalResults(top_k=top_k, include_msg=config.end_tool_with_msg)
        else:  # mode == "answer"
            default_prompt = "05_answer_lean_v1.j2"
            end_payload_phrase = "with your final answer"
            if end_tool is None:
                end_tool = LogAnswer(include_msg=config.end_tool_with_msg)
        self._retrieve_tool = retrieve_tool
        self._end_tool = end_tool

        tools: Dict[str, BaseTool] = {}
        # Primary retrieve tool first (it also powers the with_results
        # bootstrap retrieve).
        tools[retrieve_tool.name] = retrieve_tool
        # Extra/override tools next (e.g. a keyword-search retrieve tool) so
        # they sit between the primary retrieve tool and the end tool.
        for key, tool in (tool_overrides or {}).items():
            if not isinstance(tool, BaseTool):
                raise TypeError(f"tool_overrides[{key!r}] must be a BaseTool, got {type(tool).__name__}.")
            if key != tool.name:
                raise ValueError(f"tool_overrides key {key!r} does not match the tool's spec name {tool.name!r}.")
            if key == retrieve_tool.name:
                raise ValueError(
                    f"tool_overrides may not replace the primary retrieve tool {key!r}; "
                    "pass it as the retrieve_tool argument instead."
                )
            if key == end_tool.name:
                raise ValueError(
                    f"tool_overrides may not replace the end tool {key!r}; " "pass it as the end_tool argument instead."
                )
            tools[key] = tool
        # End tool after the retrieve tools.
        if end_tool.name in tools:
            raise ValueError(f"Duplicate name {end_tool.name!r} among the agent's tools.")
        tools[end_tool.name] = end_tool
        # Think tool LAST (optional scratchpad). An override that already
        # supplied a same-named tool keeps its slot.
        if config.enable_think:
            think = ThinkTool(extended_relevance=config.extended_relevance)
            if think.name in tools:
                logger.info("tool_overrides replaces the think tool %r.", think.name)
            else:
                tools[think.name] = think
        self._tool_map = tools
        # Specs are static per instance. Order is deterministic (retrieve,
        # then overrides in insertion order, then the end tool, then the think
        # tool last) — it matters both for model behavior and because
        # Anthropic-style prompt-cache markers land on the last tool spec.
        self._tool_specs = [t.spec for t in tools.values()]

        prompt_name = config.system_prompt or default_prompt
        system_prompt = render_system_prompt(
            prompt_name,
            with_init_docs=config.user_msg_type == "with_results",
            enforce_top_k=enforce_top_k,
            top_k=target_top_k,
            extended_relevance=config.extended_relevance,
        )
        self._system_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        self._auto_user_msg = build_auto_continue_msg(end_tool.name, end_payload_phrase)

    # ------------------------------------------------------------------
    # Entry points.
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        query_id: Optional[str] = None,
        task_instruction: Optional[str] = None,
        task_info: Optional[Any] = None,
        exclude_docids: Optional[Set[str]] = None,
        raw_log_dir: Optional[Union[str, Path]] = None,
    ) -> AgentRunResult:
        """Run the agent for one query.

        Parameters
        ----------
        query:
            The user's question; also the retrieve tools' ``global_query``.
        query_id:
            Optional id for this run. When given, the agent binds it itself,
            so token usage lands under ``llm.get_usage(query_id)`` and
            progress/error logs are labeled — callers then don't need to
            manage ``bind_query_id`` at all. When omitted, an ambient
            ``bind_query_id(...)`` established by the caller still applies.
        task_instruction:
            Optional retrieval instruction, prefixed onto the user message
            (``Instruct: ...``) unless it already starts with one.
        task_info:
            Arbitrary JSON-serializable info (e.g. the query id) written to
            ``extra_info.json`` when ``raw_log_dir`` is set — it associates
            the log directory with the query.
        exclude_docids:
            Document ids retrieve tools must never surface this run.
        raw_log_dir:
            Per-query directory for raw LLM IO artifacts, built by the
            caller. ``None`` discards the artifacts. See
            ``config.write_all_llm_io_logs`` for all-steps vs last-step;
            per-step extras are always written at run end as
            ``api_response_extras.json``.
        """
        binding = bind_query_id(query_id) if query_id is not None else nullcontext()
        with binding:
            state = _RunState(
                query=str(query),
                raw_log_dir=Path(raw_log_dir) if raw_log_dir is not None else None,
                exclude_docs=set(exclude_docids) if exclude_docids is not None else set(),
                pacer=PropagationPacer(target_s=self.config.cache_propagation_target_s),
                tool_map=self._tool_map,
                tool_specs=self._tool_specs,
                auto_user_msg=self._auto_user_msg,
                stage="main_agent",
                message_history=[self._system_msg],
            )
            if state.raw_log_dir is not None and task_info is not None:
                await _awrite_json(task_info, state.raw_log_dir, "extra_info.json")
            return await self._run_state_to_result(
                state, prologue=lambda: self._append_user_message(state, task_instruction)
            )

    def run_sync(self, query: str, **kwargs: Any) -> AgentRunResult:
        """Synchronous facade over :meth:`run` (one ``asyncio.run`` per call).

        For thread-based callers; must not be invoked from inside a running
        event loop — ``await run(...)`` there instead.
        """
        return asyncio.run(self.run(query, **kwargs))

    # ------------------------------------------------------------------
    # Retrieval-specific pieces on top of the shared loop.
    # ------------------------------------------------------------------

    async def _append_user_message(self, state: _RunState, task_instruction: Optional[str]) -> None:
        instruction = (task_instruction or "").strip()
        if instruction and not instruction.lower().startswith("instruct"):
            instruction = f"Instruct: {instruction}"
        if instruction:
            instruction += "\n"
        task_inst_query = f"{instruction}Query:\n{state.query}"

        if self.config.user_msg_type == "simple":
            state.message_history.append({"role": "user", "content": [{"type": "text", "text": task_inst_query}]})
            return
        # "with_results": bootstrap the conversation with an initial retrieve
        # through the primary retrieve tool.
        try:
            content = await self._execute_retrieve(
                state, tool=self._retrieve_tool, llm_kwargs={"query": state.query}, query_type="main"
            )
        except Exception as e:
            raise ToolExecutionError(self._retrieve_tool.name, e) from e
        state.message_history.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_inst_query},
                    {"type": "text", "text": "Retrieved Documents:"},
                ]
                + content,
            }
        )

    async def _dispatch_tool_call(
        self, state: _RunState, fn_name: str, fn_kwargs: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Intercept retrieve tools for the retrieval bookkeeping; defer the rest."""
        tool = state.tool_map.get(fn_name)
        if isinstance(tool, BaseRetrieveTool):
            try:
                content = await self._execute_retrieve(state, tool=tool, llm_kwargs=fn_kwargs, query_type="agent")
            except Exception as e:
                raise ToolExecutionError(fn_name, e) from e
            return content, False
        if isinstance(tool, LogAnswer):
            citations = fn_kwargs.get("citations")
            if isinstance(citations, list):
                cited_ids = {
                    citation.strip() for citation in citations if isinstance(citation, str) and citation.strip()
                }
                unknown = sorted(cited_ids - state.retrieved_docs)
                if unknown:
                    text = tool_error_text(
                        tool.name,
                        ToolError(
                            "Every citation must identify a document retrieved during this run. "
                            f"Unknown citation ID(s): {', '.join(unknown)}."
                        ),
                    )
                    return [{"type": "text", "text": text}], False
        return await super()._dispatch_tool_call(state, fn_name, fn_kwargs)

    async def _execute_retrieve(
        self,
        state: _RunState,
        *,
        tool: BaseRetrieveTool,
        llm_kwargs: Dict[str, Any],
        query_type: str,
    ) -> List[Dict[str, Any]]:
        """Run one retrieval with over-fetch/dedup/exclusion bookkeeping.

        Returns tool-message content blocks; LLM-recoverable problems (bad
        arguments, ToolError from the tool) come back as error-text content.
        """

        def _llm_error(exc: BaseException) -> List[Dict[str, Any]]:
            return [{"type": "text", "text": tool_error_text(tool.name, exc)}]

        unexpected = set(llm_kwargs) - {"query", "top_k"}
        if unexpected:
            return _llm_error(TypeError(f"unexpected argument(s): {', '.join(sorted(map(str, unexpected)))}."))
        if "query" not in llm_kwargs:
            return _llm_error(TypeError("the 'query' argument is required."))
        query = str(llm_kwargs["query"])
        top_k_raw = llm_kwargs.get("top_k", tool.default_top_k)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            return _llm_error(TypeError(f"'top_k' must be an integer, got {top_k_raw!r}."))
        if top_k <= 0:
            return _llm_error(TypeError(f"'top_k' must be a positive integer, got {top_k}."))

        context = RetrieveContext(global_query=state.query, reasoning=state.last_reasoning)
        # Over-fetch so that, after dropping excluded ids and counting only
        # docs new to this run (when ensure_new_docs), top_k new docs remain.
        seen = state.retrieved_docs if self.config.ensure_new_docs else set()
        fetch_k = top_k + len(seen) + len(state.exclude_docs)
        result = await tool.acall(query, top_k=fetch_k, context=context)
        if isinstance(result, str):
            return [{"type": "text", "text": result}]

        result = sorted(result, key=lambda d: d["score"], reverse=True)
        output: List[Dict[str, Any]] = []
        num_new = 0
        call_seen: Set[str] = set()
        for item in result:
            if item["id"] in state.exclude_docs:
                continue
            if item["id"] in call_seen:
                continue
            call_seen.add(item["id"])
            rec = dict(item)
            if rec["id"] not in seen:
                num_new += 1
            output.append(rec)
            if num_new >= top_k:
                break

        # Repeats keep their slot but drop their content (the LLM saw it
        # already); must run BEFORE the ids below join retrieved_docs.
        for rec in output:
            if rec["id"] in state.retrieved_docs:
                rec.pop("image", None)
                rec.pop("text", None)
                rec["note"] = (
                    "This document is retrieved before. See previous retrieval results "
                    f"for the content of this document (id: {rec['id']})."
                )
        for rec in output:
            state.retrieved_docs.add(rec["id"])

        state.retrieval_log.append(
            {
                "input": {"query": query, "top_k": top_k},
                "tool_name": tool.name,
                "query_type": query_type,
                "output": output,
            }
        )
        return retrieve_output_to_msg_content(output)
