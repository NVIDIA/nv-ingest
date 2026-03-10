# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Short instruction prompts for BRIGHT datasets (future integration).

These are the task descriptions that the BRIGHT authors used to train/evaluate
instruction-tuned embedding models.

We keep them here so pipelines can reference them by a stable key (e.g. "theoremqa_theorems").
"""

from typing import Optional


BRIGHT_TASKS_POST: set[str] = {
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
}

BRIGHT_TASKS_QUESTION: set[str] = {
    "pony",
}

# These are derived to match BRIGHT's `configs/*/<task>.json` instruction text.
# Important: we return the task description WITHOUT the "Instruct:" / "Query:" wrappers,
# since our retriever wraps as:
#   Instruct: <task_description>\nQuery: <query_text>
BRIGHT_SHORT_INSTRUCTIONS: dict[str, str] = {
    # StackExchange-style posts
    # Matches BRIGHT configs (no trailing period):
    #   "Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: "
    "biology": "Given a biology post, retrieve relevant passages that help answer the post",
    "earth_science": "Given a earth_science post, retrieve relevant passages that help answer the post",
    "economics": "Given a economics post, retrieve relevant passages that help answer the post",
    "psychology": "Given a psychology post, retrieve relevant passages that help answer the post",
    "robotics": "Given a robotics post, retrieve relevant passages that help answer the post",
    "stackoverflow": "Given a stackoverflow post, retrieve relevant passages that help answer the post",
    "sustainable_living": "Given a sustainable_living post, retrieve relevant passages that help answer the post",
    # Coding
    #   "Instruct: Given a coding problem, retrieve relevant examples that help answer the problem\nQuery: "
    "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem",
    # Pony
    #   "Instruct: Given a {task} question, retrieve relevant passages that help answer the question\nQuery: "
    "pony": "Given a pony question, retrieve relevant passages that help answer the question",
    # Theorem-based
    #   "Instruct: Given a Math problem, retrieve relevant examples/theorems that help answer the problem\nQuery: "
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem",
}


# ---------------------------------------------------------------------------
# Nemo reasoning retriever prompt formatting
# ---------------------------------------------------------------------------
#
# Some instruction-tuned dense retrievers (including our Nemo reasoning checkpoint)
# were trained with a *full* query prefix that already includes the "Instruct:" and
# "Query:" wrappers. For those models, we replicate the training-time formatting at
# inference time by doing:
#   formatted_query = prefix + query_text
#
# We normalize all returned prefixes to end with: "Query: " (colon + space).
BRIGHT_NEMO_QUERY_PREFIXES: dict[str, str] = {
    "biology": "Instruct: Given a Biology post, retrieve relevant passages that help answer the post.\nQuery:",
    "earth_science": "Instruct: Given an Earth Science post, retrieve relevant passages that help answer the post.\nQuery:",
    "economics": "Instruct: Given an Economics post, retrieve relevant passages that help answer the post.\nQuery:",
    "psychology": "Instruct: Given a Psychology post, retrieve relevant passages that help answer the post.\nQuery:",
    "robotics": "Instruct: Given a Robotics post, retrieve relevant passages that help answer the post.\nQuery:",
    "stackoverflow": "Instruct: Given a Stack Overflow post, retrieve relevant passages that help answer the post.\nQuery:",
    "sustainable_living": "Instruct: Given a Sustainable Living post, retrieve relevant passages that help answer the post.\nQuery:",
    "leetcode": "Instruct: Given a Coding problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "pony": "Instruct: Given a Pony question, retrieve relevant passages that help answer the question.\nQuery:",
    "aops": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "theoremqa_questions": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "theoremqa_theorems": "Instruct: Given a Math problem, retrieve relevant theorems that help answer the problem.\nQuery:",
}

# Doc prefix used for Nemo reasoning retrieval over text passages.
NEMO_REASONING_PASSAGE_PREFIX: str = "passage: "


def _ensure_query_colon_space(prefix: str) -> str:
    p = str(prefix or "")
    needle = "Query:"
    idx = p.rfind(needle)
    if idx == -1:
        return (p.rstrip() + " ") if p.strip() else ""
    head = p[: idx + len(needle)]
    return head + " "


def get_bright_query_prefix_nemo(*, task_key: Optional[str], fallback: str) -> str:
    """
    Resolve a full Nemo-style query prefix for a BRIGHT task key, else use fallback.

    Returned value is normalized to end with: "Query: " (colon + space).
    """
    if isinstance(task_key, str):
        v = BRIGHT_NEMO_QUERY_PREFIXES.get(task_key.strip(), None)
        if isinstance(v, str) and v.strip():
            return _ensure_query_colon_space(v)
    return _ensure_query_colon_space(str(fallback or ""))


def get_task_description(*, task_key: Optional[str], fallback: str) -> str:
    """
    Resolve a task description from a BRIGHT short key (if provided), otherwise use fallback.

    This is intentionally small and permissive; callers control the default behavior via `fallback`.
    """
    if isinstance(task_key, str):
        v = BRIGHT_SHORT_INSTRUCTIONS.get(task_key.strip(), None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(fallback or "").strip()
