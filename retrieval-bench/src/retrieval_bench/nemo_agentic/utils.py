# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class AgentErrorMessage(BaseModel):
    # a message type for when the agent hit an error
    model_config = ConfigDict(extra="forbid")
    role: str = "agent_error"
    content: str


def rrf_from_subquery_results(retrieval_results: List[List[Dict[str, Any]]], k: int = 60) -> Dict[str, float]:
    """Calculates the RRF score for retrieval results."""
    sorted_results: List[List[str]] = []
    for ret_rs in retrieval_results:
        sorted_docs = sorted(ret_rs, key=lambda x: x["score"], reverse=True)
        sorted_results.append([i["id"] for i in sorted_docs])
    return rrf(sorted_results=sorted_results, k=k)


def rrf(sorted_results: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Calculates the Reciprocal Rank Fusion score for each document."""
    rrf_scores = defaultdict(float)
    for result_list in sorted_results:
        for i, item in enumerate(result_list):
            rank = i + 1
            rrf_scores[item] += 1 / (rank + k)
    return dict(rrf_scores)
