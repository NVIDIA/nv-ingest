# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implements an LLM agent with ability to call functions."""

import asyncio
import copy
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, Template

from . import image_explainer, llm_handler, query_rewriter, tool_helpers, utils
from .configs import AgentConfig
from .logging_utils import get_logger_with_config
from .selection_agent.selection_agent import SelectionAgent

logger, _ = get_logger_with_config()


def is_context_exceeds_error(msg: dict) -> bool:
    """Return True if the given message is an agent error because the context window is exceeded."""
    return msg["role"] == "agent_error" and "context" in msg["content"].lower() and "window" in msg["content"].lower()


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        llm: llm_handler.LLM,
        session_id: Optional[str] = None,
        tool_wrapper: Optional[Callable] = None,
        doc_transform_fn: Optional[Callable] = None,
        tool_map: Optional[Dict[str, tool_helpers.BaseTool]] = None,
    ) -> None:
        """Implement an LLM agent loop with access to tools."""

        self.message_history = list()
        self.config = config
        self.doc_transform_fn = doc_transform_fn
        self.tool_wrapper = tool_wrapper
        self.session_id = session_id
        self.llm = llm
        self.tool_map = tool_map

        self._curr_time = datetime.now()
        self._uuid = uuid4().hex

        if self.config.user_msg_type not in ["simple", "with_results"]:
            raise ValueError(f"user_msg_type must be 'simple' or 'with_results', got {self.config.user_msg_type!r}")

        # Load system prompt template
        if Path(config.system_prompt).exists():
            prompt_path = Path(config.system_prompt)
            system_prompt_template = Template(prompt_path.read_text().strip())
        else:
            env = Environment(
                loader=FileSystemLoader(Path(__file__).parent.joinpath("prompts").absolute().resolve().as_posix())
            )
            system_prompt_template = env.get_template(config.system_prompt)

        system_prompt = system_prompt_template.render(
            with_init_docs=config.user_msg_type == "with_results",
            enforce_top_k=config.enforce_top_k,
            top_k=config.target_top_k,
            extended_relevance=config.extended_relevance,
        )

        self._system_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]}
        self.message_history = [self._system_msg]
        self.current_user_msg = None

        self.auto_user_msg = (
            "Please continue on whatever approach you think is suitable.\n"
            "If you think you have solved the task, please finish the interaction.\n"
            "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN RESPONSE.\n"
        )

        self.selection_agent = SelectionAgent(
            llm=self.llm,
            prompt_name=self.config.selection_prompt,
            max_steps=self.config.selection_max_steps,
            extended_relevance=self.config.extended_relevance,
        )

        self.steps = 0
        self.extra_data: Dict[str, Any] = {}
        self.retrieved_docs: Set[str] = set()
        self.retrieval_log: list[dict] = []
        self.exclude_docs: Set[str] = set()
        self.api_response_extras: list = []

    def reset(self) -> None:
        self.message_history = [self._system_msg]
        self.current_user_msg = None
        self.steps = 0
        self._curr_time = datetime.now()
        self._uuid = uuid4().hex
        self.extra_data = {}
        self.retrieved_docs = set()
        self.retrieval_log = []
        self.exclude_docs = set()
        self.api_response_extras = []

    def get_llm_raw_io_subdir(self):
        raw_log_subdir = self._curr_time.strftime("%d-%m-%y_%H-%M-%S-%f") + "_" + self._uuid[:8]
        if self.session_id is not None:
            raw_log_subdir = self.session_id + "_" + raw_log_subdir
        return "main_agent/" + raw_log_subdir

    async def step(self) -> Optional[dict]:
        """Take one step and process LLM outputs."""
        log_exp_name = self._curr_time.strftime("%d-%m_%H-%M") + "_" + self._uuid[:4]
        if self.session_id is not None:
            log_exp_name = self.session_id + "_" + log_exp_name
        trace_prefix = os.environ.get("LOG_TRACE_PREFIX")
        if trace_prefix is not None:
            log_exp_name = str(trace_prefix) + "_" + log_exp_name

        logging_kwargs = {"step": self.steps, "subdir": self.get_llm_raw_io_subdir(), "log_exp_name": log_exp_name}

        params = {
            "messages": copy.deepcopy(self.message_history),
            "tools": [t.spec for t in self.tool_map.values()],
            "logging_kwargs": logging_kwargs,
        }
        response = await self.llm.acompletion(**params, return_metadata=True)
        if llm_handler.is_error(response):
            self.message_history.append(utils.AgentErrorMessage(content=response).model_dump())
            return None

        metadata_kv = response["metadata_kv"]
        io_log_data = response["io_log_kwargs"]
        if "api_response_extras" in response:
            self.api_response_extras.append(response["api_response_extras"])
        response = response["response"]
        self.steps += 1

        metadata_kv = {"S": str(self.steps), **metadata_kv}
        log_str = " ".join([f"{k}: {v}" for k, v in metadata_kv.items()])
        logger.info(f"[{self.session_id}] " + log_str)

        if len(response.choices) != 1:
            raise RuntimeError(f"Expected exactly 1 choice from LLM, got {len(response.choices)}")
        if response.choices[0].finish_reason == "tool_calls":
            conv_msg = {"content": [], "role": "assistant", "tool_calls": []}
            for call_info in response.choices[0].message.tool_calls:
                conv_msg["tool_calls"].append(call_info.model_dump())
            if response.choices[0].message.content is not None:
                if not isinstance(response.choices[0].message.content, str):
                    raise TypeError("Expected string content from LLM, got non-text content")
                conv_msg["content"].append({"type": "text", "text": response.choices[0].message.content})
            self.message_history.append(conv_msg)
        elif response.choices[0].finish_reason == "stop":
            self.message_history.append(
                {"role": "assistant", "content": [{"type": "text", "text": response.choices[0].message.content}]}
            )
        else:
            self.message_history.append(
                utils.AgentErrorMessage(
                    content=f"LLM failed with finish_reason '{response.choices[0].finish_reason}'"
                ).model_dump()
            )
        return io_log_data

    async def call_one_tool(
        self,
        fn_name: str,
        fn_kwargs: Dict,
        store_state: bool = True,
        include_in_retrieval_logs: bool = True,
        query_type: Optional[str] = "agent",
    ):
        """Call one tool and return the results."""
        old_sub_query = None

        if (
            self.config.use_query_rewriting
            and fn_name == "retrieve"
            and isinstance(self.tool_map[fn_name], tool_helpers.RetrieveToolBase)
        ):
            old_sub_query = fn_kwargs["query"]
            new_sub_query = await query_rewriter.rewrite_query(
                llm=self.llm,
                main_query=self.current_user_msg,
                sub_query=old_sub_query,
            )
            fn_kwargs["query"] = new_sub_query
            if store_state:
                if "query_rewriting" not in self.extra_data:
                    self.extra_data["query_rewriting"] = {}
                self.extra_data["query_rewriting"][old_sub_query] = new_sub_query

        async def tool_caller(**kw):
            new_kwargs = {**fn_kwargs, **kw}
            if self.tool_wrapper is not None:
                return await self.tool_wrapper(tool=self.tool_map[fn_name], tool_kwargs=new_kwargs)
            return await self.tool_map[fn_name].acall(**new_kwargs)

        if fn_name == "retrieve" and isinstance(self.tool_map[fn_name], tool_helpers.RetrieveToolBase):
            fn_res = await tool_helpers.retrieve_with_guarantees(
                tool_caller=tool_caller,
                top_k=fn_kwargs.get("top_k", getattr(self.tool_map[fn_name], "_default_top_k", 5)),
                seen_docids=(self.retrieved_docs if self.config.ensure_new_docs else set()),
                exclude_docids=self.exclude_docs,
            )

            if self.doc_transform_fn is not None:
                if isinstance(fn_res, str):
                    if "Error" not in fn_res:
                        raise RuntimeError(f"Unexpected non-list, non-error retrieve result: {fn_res!r}")
                else:
                    fn_res = await asyncio.gather(*(self.doc_transform_fn(d) for d in fn_res))
            if isinstance(fn_res, list):
                for i in range(len(fn_res)):
                    if fn_res[i]["id"] in self.retrieved_docs:
                        fn_res[i].pop("image", None)
                        fn_res[i].pop("text", None)
                        note = (
                            "This document is retrieved before. See previous retrieval results for the content of this "
                            f"document (id: {fn_res[i]['id']})."
                        )
                        fn_res[i]["note"] = note
        else:
            fn_res = await tool_caller()

        if fn_name == "retrieve" and isinstance(self.tool_map[fn_name], tool_helpers.RetrieveToolBase):
            if self.config.use_image_explainer and not isinstance(fn_res, str):
                image_descs = await asyncio.gather(
                    *(
                        image_explainer.explain_image(
                            llm=self.llm,
                            main_query=self.current_user_msg,
                            sub_query=fn_kwargs["query"],
                            image_b64=d.get("image", None),
                            prompt_type=self.config.image_explainer_prompt,
                        )
                        for d in fn_res
                    )
                )
                for i in range(len(fn_res)):
                    if image_descs[i] is not None:
                        fn_res[i]["text"] = image_descs[i]
                        fn_res[i].pop("image", None)

            tmp_output_list: list = []
            if isinstance(fn_res, list):
                for item in fn_res:
                    self.retrieved_docs.add(item["id"])
                    tmp_output_list.append(item)
            elif isinstance(fn_res, str) and not fn_res.startswith("Error"):
                try:
                    res_obj = json.loads(fn_res)
                    for item in res_obj:
                        self.retrieved_docs.add(item["id"])
                        tmp_output_list.append(item)
                except Exception:
                    pass
            if include_in_retrieval_logs:
                self.retrieval_log.append(
                    {
                        "input": fn_kwargs,
                        "query_before_rewriting": old_sub_query,
                        "query_type": query_type,
                        "output": tmp_output_list,
                    }
                )
            content = tool_helpers.retrieve_output_to_msg_content(output=fn_res)
        else:
            if not isinstance(fn_res, str):
                fn_res = json.dumps(fn_res)
            content = [{"type": "text", "text": fn_res}]
        return content

    async def process_tool_calls(self) -> List[Dict[str, Any]]:
        if len(self.message_history[-1].get("tool_calls", [])) < 1:
            raise RuntimeError("There are no tool calls to process")

        tool_messages = []
        for call_info in self.message_history[-1]["tool_calls"]:
            fn_name = call_info["function"]["name"]
            content = None
            try:
                fn_kwargs = json.loads(call_info["function"]["arguments"])
            except Exception:
                content = "Error parsing tool arguments. Tool arguments not correctly formatted."
            if content is None:
                content = await self.call_one_tool(fn_name=fn_name, fn_kwargs=fn_kwargs, store_state=True)
            tool_messages.append({"content": content, "role": "tool", "tool_call_id": call_info["id"], "name": fn_name})
        return tool_messages

    def is_last_msg_error(self) -> bool:
        """Check if the last message in the history is an error."""
        if self.message_history[-1]["role"] == "agent_error":
            err_msg = self.message_history[-1]["content"]
            if self.config.only_warn_on_error:
                warnings.warn(err_msg)
                return True
            raise RuntimeError(err_msg)
        return False

    async def run_for_input(
        self,
        query: str,
        task_instruction: Optional[str] = None,
        task_info: Optional[Any] = None,
        exclude_docids: Optional[Set] = None,
    ) -> Dict[str, Any]:
        """Run the agent for a given user message."""
        self.reset()
        if self.tool_map is None:
            raise RuntimeError("Agent requires tool_map to be provided by the caller.")

        await self.llm.log_extra_data_log_dir(subdir=self.get_llm_raw_io_subdir(), info=task_info)
        self.current_user_msg = query

        self.exclude_docs = set() if exclude_docids is None else set(exclude_docids)

        if task_instruction is None:
            task_instruction = ""
        task_instruction = task_instruction.strip()
        if task_instruction != "" and not task_instruction.lower().startswith("instruct"):
            task_instruction = f"Instruct: {task_instruction}"
        if task_instruction != "":
            task_instruction = task_instruction.strip() + "\n"
        task_inst_query = f"{task_instruction}Query:\n{query}"

        if self.config.user_msg_type == "simple":
            self.message_history.append({"role": "user", "content": [{"type": "text", "text": task_inst_query}]})
        elif self.config.user_msg_type == "with_results":
            res = await self.call_one_tool(
                fn_name="retrieve",
                fn_kwargs={"query": query},
                store_state=False,
                query_type="main",
            )
            user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": task_inst_query}, {"type": "text", "text": "Retrieved Documents:"}]
                + res,
            }
            self.message_history.append(user_msg)
        else:
            raise ValueError(f"`{self.config.user_msg_type}` is not a valid user_msg_type.")

        io_log_data = None
        while True:
            if self.config.max_steps is not None and self.steps >= self.config.max_steps:
                self.message_history.append(
                    utils.AgentErrorMessage(content="Agent reached maximum allowed iterations").model_dump()
                )

            if self.is_last_msg_error():
                break

            new_io_log_data = await self.step()
            if new_io_log_data is not None:
                io_log_data = new_io_log_data

            if not self.is_last_msg_error():
                _tc = self.message_history[-1].get("tool_calls", [])
                if _tc is None or len(_tc) == 0:
                    self.message_history.append(
                        {"role": "user", "content": [{"type": "text", "text": self.auto_user_msg}]}
                    )
                else:
                    tool_calls = [tc["function"]["name"] for tc in self.message_history[-1]["tool_calls"]]
                    tool_messages = await self.process_tool_calls()
                    self.message_history.extend(tool_messages)
                    ended_successfully = False
                    if self.config.end_tool in tool_calls:
                        end_tool = self.tool_map[self.config.end_tool]
                        _correct_val = end_tool.correct_call_return_value  # type: ignore[attr-defined]
                        for tm in tool_messages:
                            if tm["name"] == self.config.end_tool and tm["content"][0]["text"] == _correct_val:
                                ended_successfully = True
                                break
                    if ended_successfully:
                        break
            else:
                break

        await self.llm.log_extra_data_log_dir(
            subdir=self.get_llm_raw_io_subdir(),
            info=self.api_response_extras,
            filename="api_response_extras.json",
        )

        if not self.llm.config.instant_log and io_log_data is not None:
            await llm_handler.awrite_json(**io_log_data["input_json"])
            await llm_handler.awrite_json(**io_log_data["output_json"])

        return await self.conclude_task(query=query, task_info=task_info)

    async def conclude_task(self, query: str, task_info: Optional[Any] = None) -> Dict[str, Any]:
        """Calculate final top_k results if needed and return artifacts."""
        output_artifacts: Dict[str, Any] = {
            "agent_trajectories": self.message_history,
            "retrieval_log": self.retrieval_log,
        }
        if len(self.extra_data):
            output_artifacts["agent_extra_data"] = self.extra_data

        if self.config.main_agent_only:
            return output_artifacts

        selection_input = []
        rrf_input = []

        seen_docids = set()
        for ret_data in self.retrieval_log:
            rrf_input.append(ret_data["output"])
            for doc in ret_data["output"]:
                if doc["id"] not in seen_docids:
                    selection_input.append(doc)
                    seen_docids.add(doc["id"])

        rrf_res: Optional[Dict[str, float]] = None
        if self.config.calculate_rrf or self.is_last_msg_error():
            rrf_res = utils.rrf_from_subquery_results(retrieval_results=rrf_input)
            output_artifacts["rrf_scores"] = rrf_res

        selection_topk_list = self.config.selection_topk_list
        if len(selection_topk_list) == 0 and self.is_last_msg_error():
            selection_topk_list = [5, 10]

        if len(selection_topk_list) != 0:
            for _ in range(2):
                selection_output = await asyncio.gather(
                    *(
                        self.selection_agent.select_topk(
                            query=query,
                            documents=selection_input,
                            top_k=k,
                            session_id=self.session_id,
                            task_info=task_info,
                        )
                        for k in selection_topk_list
                    )
                )
                if not any([a is None and is_context_exceeds_error(h[-1]) for h, a in selection_output]):
                    break
                drop_items = len(selection_input) // 4
                if rrf_res is None:
                    rrf_res = utils.rrf_from_subquery_results(retrieval_results=rrf_input)
                    output_artifacts["rrf_scores"] = rrf_res
                least_rrf_scores = dict(sorted(rrf_res.items(), key=lambda x: x[1])[:drop_items])
                least_rrf_scores_set = set(list(least_rrf_scores.keys()))
                selection_input = [i for i in selection_input if i["id"] not in least_rrf_scores_set]
            for topk, (msg_hist, final_ans) in zip(selection_topk_list, selection_output):
                output_artifacts[f"top{topk}_agent_trajectories"] = msg_hist
                if final_ans is not None:
                    output_artifacts[f"top{topk}_selection_result"] = final_ans
        return output_artifacts
