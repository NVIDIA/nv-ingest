# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical graph-execution package for operators, graphs, and executors."""

from __future__ import annotations

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.graph_ops.custom_operator import UDFOperator
from nemo_retriever.graph.executor import AbstractExecutor, InprocessExecutor, RayDataExecutor
from nemo_retriever.operators.graph_ops.file_loader_operator import FileListLoaderOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.graph.graph_pipeline_registry import GraphPipelineRegistry, default_registry
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.operators.graph_ops.store_operator import StoreOperator
from nemo_retriever.operators.graph_ops.webhook_operator import WebhookNotifyOperator

__all__ = [
    "AbstractExecutor",
    "AbstractOperator",
    "ArchetypeOperator",
    "CPUOperator",
    "FileListLoaderOperator",
    "GPUOperator",
    "Graph",
    "GraphPipelineRegistry",
    "InprocessExecutor",
    "MultiTypeExtractOperator",
    "Node",
    "RayDataExecutor",
    "StoreOperator",
    "UDFOperator",
    "WebhookNotifyOperator",
    "default_registry",
]


def __getattr__(name: str):
    if name == "MultiTypeExtractOperator":
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractOperator

        return MultiTypeExtractOperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
