# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve retained-result options without changing explicit Boolean requests."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.params import IngestExecuteParams
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph


def resolve_return_results_option(params: Any, kwargs: dict[str, Any]) -> bool | None:
    """Read the requested flag, preserving ``None`` as the context default."""
    if "return_results" in kwargs:
        value = kwargs["return_results"]
    elif isinstance(params, IngestExecuteParams):
        value = params.return_results
    elif isinstance(params, dict):
        value = params.get("return_results")
    else:
        value = None
    return None if value is None else bool(value)


def resolve_batch_return_results(graph: Graph, return_results: bool | None) -> bool:
    """Default eligible terminal VDB graphs to summary mode before execution."""
    if return_results is True:
        return True
    nodes = RayDataExecutor._linearize(graph)
    sink_index = RayDataExecutor._stream_ingest_index(nodes, require_terminal=return_results is False)
    if return_results is False:
        return False
    return sink_index is None or sink_index != len(nodes) - 1
