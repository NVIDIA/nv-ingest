# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool definition and utilities for managing tools for an LLM.

This is an internalized subset of the original external agent code. All MCP and
fastmcp integration has been removed; this module only provides:
- Base tool abstraction (BaseTool)
- Local tools used by the agent (ThinkTool, FinalResults)
- Retrieval output formatting helpers (retrieve_output_to_msg_content)
- Retrieval over-fetch-and-filter helper (retrieve_with_guarantees)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union


class ToolError(Exception):
    """Tool-specific exception raised for invalid tool usage."""


class BaseTool(ABC):
    """Define a tool to be passed to the LLM."""

    _name: Optional[str] = None

    @abstractmethod
    def _spec(self) -> dict:
        raise NotImplementedError

    def _call(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _acall(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def call(self, **kwargs: Any) -> Any:
        try:
            output = self._call(**kwargs)
        except (TypeError, ToolError) as e:
            output = f"Error calling '{self.name}' tool. {type(e).__name__}: {str(e)}"
        return output

    async def acall(self, **kwargs: Any) -> Any:
        try:
            output = await self._acall(**kwargs)
        except (TypeError, ToolError) as e:
            output = f"Error calling '{self.name}' tool. {type(e).__name__}: {str(e)}"
        return output

    def __call__(self, **kwargs: Any) -> Any:
        return self.call(**kwargs)

    @property
    def spec(self) -> dict:
        return self._spec()

    @property
    def name(self) -> str:
        return str(self.spec["function"]["name"])


class RetrieveToolBase(BaseTool):
    """Marker base class for retrieve-like tools.

    The Agent uses `isinstance(tool, RetrieveToolBase)` to apply retrieval-specific
    behaviors (dedup/guarantees/output formatting).
    """

    _default_top_k: int = 20


class ThinkTool(BaseTool):
    """Tool that allows the LLM to think with output tokens."""

    def __init__(self, extended_relevance: bool = False):
        if extended_relevance:
            ext = [
                "- When it is difficult to understand what is the intent of the user and what they are trying to find with this query, use this tool to think about potential definitions of relevance that could be meaningful/useful to the user for this task.",
                "- If the intention of the user is vague especially given the available documents, use this tool to think how you should decide what documents are relevant and what the metric of relevance is.",
            ]
            ext = "\n".join(ext) + "\n"
        else:
            ext = ""
        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "think",
                "description": f"""Use the tool to think about something. It will not obtain new information or make any changes, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
{ext}- When processing a complex query, use this tool to organize your thoughts and think about the sub queries that you need to search for to find the relevant information
- If a query is vague is very difficult to find information for it, you can use this tool to think about clues in the query that you can use to narrow down the search and spot relevant pieces of information.
- When finding related documents that help you create better search queries in the next step, use this tool to think about what pieces of information from these documents are helpful to search for.
- When you fail to find any related information to the query, use this tool to think about other search strategies that you can take to retrieve the related documents

The tool simply logs your thought process for better transparency and does not make any changes.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "The thought to log.",
                        }
                    },
                    "required": ["thought"],
                },
            },
        }

    def _spec(self) -> dict:
        return self.spec_dict

    def _call(self, thought: str) -> str:
        return "Your thought has been logged."

    async def _acall(self, thought: str) -> str:
        return "Your thought has been logged."


class FinalResults(BaseTool):
    """Tool for logging selected document IDs and signaling the end of the interaction."""

    _name: Optional[str] = "final_results"

    def __init__(self, top_k: Optional[int] = None):
        self.correct_call_return_value = "The results have been successfully logged and the interaction ended."
        self.top_k = top_k

        tk_ins = ""
        if top_k is not None:
            tk_ins = f"- You must choose exactly {top_k} document IDs when calling this function."

        desc = f"""Signals the completion of the search process for the current query.

Use this tool when:
- You have found all the relevant documents to the query.
- Despite several attempts, you cannot find good documents for the given query.

The message should include:
- A brief summary of your exploration and the results
- Explanation if the search was unsuccessful

When reporting the selected document IDs, make sure:
- the list of document IDs is sorted in the decreasing level of relevance to the query. I.e., the first document in the list is the most relevant to the query, the second is the second most relevant to the query, and so on.
{tk_ins}

The successful_search field should be set to true if you believed you have found the most relevant documents to the user's query, and false otherwise. And partial if it is in between."""
        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "final_results",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "required": ["doc_ids", "message", "search_successful"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "A message for the user to explain why you think you found all the related documents and there is no related document is missing. Also, include a short description of your exploration process. If your attempts to find related documents were unsuccessful, explain why.",
                        },
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs that are relevant to the user's query sorted descending by their level of relevance to the user's query. I.e., the first document is the most relevant to the query, the second is the second most relevant to the query, and so on.",
                        },
                        "search_successful": {
                            "type": "string",
                            "enum": ["true", "false", "partial"],
                            "description": "Whether you managed to find all the related documents to the query.",
                        },
                    },
                },
            },
        }

    def _spec(self) -> dict:
        return self.spec_dict

    async def _acall(self, doc_ids: List[str], message: str, search_successful: str) -> str:
        return self._call(doc_ids=doc_ids, message=message, search_successful=search_successful)

    def _call(self, doc_ids: List[str], message: str, search_successful: str) -> str:
        if not isinstance(message, str):
            raise TypeError(f"The `message` argument must be a string. Got `{type(message)}` type.")
        if not isinstance(doc_ids, list):
            raise TypeError(f"The `doc_ids` argument must be a list. Got `{type(doc_ids)}` type.")
        if len(doc_ids) == 0:
            raise ToolError("`doc_ids` cannot be empty. You must choose at least one relevant document.")
        if not all(isinstance(i, str) for i in doc_ids):
            raise TypeError("Items in `doc_ids` must be of type string (i.e., python's `str` type).")
        if not isinstance(search_successful, str):
            raise TypeError(f"The `search_successful` argument must be a string. Got `{type(search_successful)}` type.")
        if search_successful not in ["true", "false", "partial"]:
            raise ToolError(
                f"`search_successful` must be one of `true`, `false`, or `partial`. Got `{search_successful}` instead."
            )
        if self.top_k is not None and len(doc_ids) != self.top_k:
            raise ToolError(
                f"`doc_ids` must contain exactly {self.top_k} documents. But got {len(doc_ids)} document IDs instead."
            )
        return self.correct_call_return_value


def retrieve_output_to_msg_content(output: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert retrieve output into LLM message content blocks."""
    if isinstance(output, str):
        if output.startswith("Error"):
            return [{"type": "text", "text": output}]
        raise RuntimeError("Received unexpected value from the retrieve tool.")

    content_list: List[Dict[str, Any]] = []
    for doc_in in output:
        doc = {**doc_in}
        if doc.get("text", "").strip() == "":
            doc.pop("text", None)
        img = doc.pop("image", None)
        content_list.append({"type": "text", "text": json.dumps(doc)})
        if img is not None:
            content_list.append({"type": "image_url", "image_url": {"url": img}})
    return content_list


async def retrieve_with_guarantees(
    tool_caller: Callable[..., Any],
    top_k: int,
    seen_docids: Set[str],
    exclude_docids: Set[str],
) -> Union[str, List[Dict[str, Any]]]:
    """Call retrieve, ensuring `top_k` new docs and excluding `exclude_docids`."""
    seen_docids = set(seen_docids)
    exclude_docids = set(exclude_docids)
    res = await tool_caller(__art_top_k=top_k + len(seen_docids) + len(exclude_docids))
    if isinstance(res, str) and res.startswith("Error"):
        return res

    res_list = list(sorted(res, key=lambda x: x["score"], reverse=True))

    output_list: List[Dict[str, Any]] = []
    num_new = 0
    for item in res_list:
        if item["id"] in exclude_docids:
            continue
        rec = {**item}
        if rec["id"] not in seen_docids:
            num_new += 1
        output_list.append(rec)
        if num_new >= top_k:
            break
    return output_list
