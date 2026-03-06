# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, Template

from .. import llm_handler, utils
from ..logging_utils import get_logger_with_config
from . import tool_helpers

logger, _ = get_logger_with_config()


async def log_io(io_log_data: dict, llm_instant_log: bool) -> None:
    """Log the LLM's raw IO if needed."""
    if llm_instant_log:
        return
    await llm_handler.awrite_json(**io_log_data["input_json"])
    await llm_handler.awrite_json(**io_log_data["output_json"])


class SelectionAgent:
    def __init__(
        self,
        llm: llm_handler.LLM,
        prompt_name: str,
        max_steps: int,
        extended_relevance: bool = False,
    ) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.extended_relevance = extended_relevance

        if Path(prompt_name).exists():
            self.system_prompt_template = Template(Path(prompt_name).read_text().strip())
        else:
            env = Environment(
                loader=FileSystemLoader(
                    Path(__file__).parent.joinpath("selection_prompts").absolute().resolve().as_posix()
                )
            )
            self.system_prompt_template = env.get_template(prompt_name)

        self.auto_user_msg = (
            "Please continue on whatever approach you think is suitable.\n"
            "If you think you have solved the task, please finish the interaction.\n"
            "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN RESPONSE.\n"
        )

        think_tool = tool_helpers.SelectionThinkTool(extended_relevance=self.extended_relevance)
        self.tool_map = {think_tool.name: think_tool}

    async def select_topk(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        session_id: str,
        task_info: Optional[Any] = None,
    ):
        steps = 0
        message_history: list = []

        seen_docids: list[str] = []
        user_msg: Dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": f"Query:\n{query}"}, {"type": "text", "text": "Candidate Documents:"}],
        }
        for doc in documents:
            if doc["id"] in seen_docids:
                continue
            seen_docids.append(doc["id"])
            user_msg["content"].append({"type": "text", "text": f"Doc ID: {doc['id']}"})
            if doc.get("text", "").strip() != "":
                user_msg["content"].append({"type": "text", "text": f"Doc Text: {doc['text']}"})
            if doc.get("image", None) is not None and str(doc["image"]).strip() != "":
                user_msg["content"].append({"type": "image_url", "image_url": {"url": doc["image"]}})
        feasible_topk = min(int(top_k), len(set(seen_docids)))

        message_history.append(
            {
                "role": "system",
                "content": self.system_prompt_template.render(
                    top_k=feasible_topk, extended_relevance=self.extended_relevance
                ).strip(),
            }
        )
        message_history.append(user_msg)

        finish_tool = tool_helpers.LogSelectedDocs(top_k=feasible_topk, candidate_docids=seen_docids)
        tool_map = {**self.tool_map, finish_tool.name: finish_tool}
        tools = [t.spec for t in tool_map.values()]
        _curr_time = datetime.now()
        _uuid = uuid4().hex

        raw_log_subdir = _curr_time.strftime("%d-%m-%y_%H-%M-%S-%f") + "_" + _uuid[:8]
        if session_id is not None:
            raw_log_subdir = session_id + "_" + raw_log_subdir
        raw_log_subdir = f"select_{top_k}_agent/" + raw_log_subdir

        await self.llm.log_extra_data_log_dir(subdir=raw_log_subdir, info=task_info)
        log_exp_name = _curr_time.strftime("%d-%m_%H-%M") + "_" + _uuid[:4]
        if session_id is not None:
            log_exp_name = session_id + "_" + log_exp_name
        trace_prefix = os.environ.get("LOG_TRACE_PREFIX")
        if trace_prefix is not None:
            log_exp_name = str(trace_prefix) + "_" + log_exp_name
        log_exp_name = f"SL_{top_k}_" + log_exp_name

        api_response_extras: list = []

        while True:
            logging_kwargs = {"step": steps, "subdir": raw_log_subdir, "log_exp_name": log_exp_name}

            response = await self.llm.acompletion(
                messages=message_history,
                tools=tools,
                logging_kwargs=logging_kwargs,
                return_metadata=True,
            )
            if llm_handler.is_error(response):
                message_history.append(utils.AgentErrorMessage(content=response).model_dump())
                return message_history, None

            io_log_data = response["io_log_kwargs"]
            if "api_response_extras" in response:
                api_response_extras.append(response["api_response_extras"])
            response = response["response"]

            steps += 1
            if len(response.choices) != 1:
                raise RuntimeError(f"Expected exactly 1 choice from LLM, got {len(response.choices)}")

            if response.choices[0].finish_reason == "tool_calls":
                conv_msg = {"content": [], "role": "assistant", "tool_calls": []}
                for call_info in response.choices[0].message.tool_calls:
                    conv_msg["tool_calls"].append(call_info.model_dump())
                message_history.append(conv_msg)
            elif response.choices[0].finish_reason == "stop":
                message_history.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response.choices[0].message.content}],
                    }
                )
            else:
                message_history.append(
                    utils.AgentErrorMessage(
                        content=f"LLM failed with finish_reason '{response.choices[0].finish_reason}'"
                    ).model_dump()
                )
                await log_io(io_log_data=io_log_data, llm_instant_log=self.llm.config.instant_log)
                await self.llm.log_extra_data_log_dir(
                    subdir=raw_log_subdir, info=api_response_extras, filename="api_response_extras.json"
                )
                return message_history, None

            if self.max_steps is not None and steps >= self.max_steps:
                message_history.append(
                    utils.AgentErrorMessage(content="Selection Agent reached maximum allowed iterations").model_dump()
                )
                await log_io(io_log_data=io_log_data, llm_instant_log=self.llm.config.instant_log)
                await self.llm.log_extra_data_log_dir(
                    subdir=raw_log_subdir, info=api_response_extras, filename="api_response_extras.json"
                )
                return message_history, None

            if len(message_history[-1]["content"]) != 0:
                message_history.append({"role": "user", "content": [{"type": "text", "text": self.auto_user_msg}]})
            else:
                should_end = False
                tool_messages: list = []
                for call_info in message_history[-1]["tool_calls"]:
                    fn_name = call_info["function"]["name"]
                    err_msg = None
                    try:
                        fn_kwargs = json.loads(call_info["function"]["arguments"])
                    except Exception:
                        err_msg = "Error parsing tool arguments. Tool arguments not correctly formatted."
                    if fn_name not in tool_map or err_msg is not None:
                        if err_msg is None:
                            err_msg = f"Error. Tool '{fn_name}' does not exist."
                        tool_messages.append(
                            {"content": err_msg, "role": "tool", "tool_call_id": call_info["id"], "name": fn_name}
                        )
                    else:
                        res = tool_map[fn_name].call(**fn_kwargs)
                        if fn_name == finish_tool.name and res == finish_tool.correct_call_return_value:
                            should_end = True
                            end_kwargs = fn_kwargs
                        else:
                            tool_messages.append(
                                {"content": res, "role": "tool", "tool_call_id": call_info["id"], "name": fn_name}
                            )
                if should_end:
                    await log_io(io_log_data=io_log_data, llm_instant_log=self.llm.config.instant_log)
                    await self.llm.log_extra_data_log_dir(
                        subdir=raw_log_subdir, info=api_response_extras, filename="api_response_extras.json"
                    )
                    return message_history, end_kwargs
                message_history.extend(tool_messages)
