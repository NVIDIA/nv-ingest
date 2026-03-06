# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional

from ..tool_helpers import BaseTool, ToolError


class SelectionThinkTool(BaseTool):
    """Tool for selection agent thinking."""

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
{ext}- When processing a complex query, use this tool to organize your thoughts and think about how each document might be related to the given query.
- If a query is vague or hard to understand, you can use this tool to think about clues in the query that help you identify the connections between a document and the query.
- You can use this tool to think what pieces of information in each document are the most important or relevant for the given query.

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

    def _spec(self):
        return self.spec_dict

    def _call(self, thought: str):
        return "Your thought has been logged."

    async def _acall(self, thought: str):
        return "Your thought has been logged."


class LogSelectedDocs(BaseTool):
    """Tool for reporting selected doc IDs and ending the interaction."""

    _name: Optional[str] = "log_selected_documents"

    def __init__(self, top_k: int, candidate_docids: List[str]):
        self.correct_call_return_value = "The results have been successfully logged and the interaction ended."
        self.top_k = int(top_k)
        self.allowed_doc_ids = set(candidate_docids)

        desc = f"""Records the selected documents and signals the end of the task.

Use this tool when you have carefully considered the candidate documents and have selected exactly the {self.top_k} most relevant documents to the query.

The message argument should explain your reasoning and justification for selecting this specific set of documents as the most relevant to the query.

**Note**: the list of document IDs passed as the `doc_ids` argument must be sorted in the decreasing level of relevance. In other words, the first document in `doc_ids` list is the most relevant to the query, the second document is the second most relevant document, and so on."""

        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "log_selected_documents",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "required": ["message", "doc_ids"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "A message for the user to explain why you think the selected are the most relevant to the query. Also, explain why this specific order of document IDs satisfies the most to least relevant ordering criteria.",
                        },
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"The ID of the {self.top_k} most relevant documents to the given query. The IDs must be sorted in the decreasing level of relevance. I.e., the first document must be the most relevant to the query.",
                        },
                    },
                },
            },
        }

    def _spec(self):
        return self.spec_dict

    async def _acall(self, doc_ids: List[str], message: str):
        return self._call(doc_ids=doc_ids, message=message)

    def _call(self, doc_ids: List[str], message: str):
        if not isinstance(message, str):
            raise TypeError(f"The `message` argument must be a string. Got `{type(message)}` type.")
        if not isinstance(doc_ids, list):
            raise TypeError(f"The `doc_ids` argument must be a list. Got `{type(doc_ids)}` type.")
        if len(doc_ids) != self.top_k:
            raise ToolError(f"You must select at least {self.top_k} documents. Got {len(doc_ids)} documents.")
        if not all(isinstance(i, str) for i in doc_ids):
            raise TypeError("Items in `doc_ids` must be of type string (i.e., python's `str` type).")
        for i in doc_ids:
            if i not in self.allowed_doc_ids:
                raise ToolError(f"Document with ID `{i}` is not among the candidate documents.")
        return self.correct_call_return_value
