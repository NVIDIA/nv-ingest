# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.common import ray_resource_hueristics as _rrh
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources


def resolve_operator_class(
    operator_class: type[AbstractOperator],
    resources: ClusterResources | Resources,
    operator_kwargs: dict | None = None,
) -> type[AbstractOperator]:
    if issubclass(operator_class, ArchetypeOperator):
        return operator_class.resolve_operator_class(resources, operator_kwargs=operator_kwargs)
    return operator_class


def resolve_operator_kwargs(
    operator_class: type[AbstractOperator],
    resolved_class: type[AbstractOperator],
    operator_kwargs: dict | None = None,
) -> dict:
    if issubclass(operator_class, ArchetypeOperator):
        return operator_class.variant_operator_kwargs(resolved_class, operator_kwargs=operator_kwargs)
    return dict(operator_kwargs or {})


def resolve_graph(
    graph: Graph,
    resources: ClusterResources | Resources,
) -> Graph:
    from nemo_retriever.ingest.index_mode import _embedding_model_kwargs
    from nemo_retriever.models import resolve_local_embed_model
    from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision
    from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor
    from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as BatchEmbedGPUActor

    resolved = Graph()
    visited: dict[int, Node] = {}

    def _clone(node: Node, embedding_stage: tuple[Any, bool] | None = None) -> Node:
        node_id = id(node)
        if node_id in visited:
            return visited[node_id]

        operator = node.operator
        if isinstance(operator, ArchetypeOperator):
            operator = type(operator)(**node.operator_kwargs)

        resolved_class = resolve_operator_class(node.operator_class, resources, operator_kwargs=node.operator_kwargs)
        resolved_kwargs = resolve_operator_kwargs(node.operator_class, resolved_class, node.operator_kwargs)
        if issubclass(resolved_class, (BatchEmbedGPUActor, _BatchEmbedCPUActor)):
            params = resolved_kwargs["params"]
            local = issubclass(resolved_class, BatchEmbedGPUActor)
            if local:
                model_name = resolve_local_embed_model(
                    params.embed_model_name or params.model_name,
                    backend=params.local_ingest_embed_backend,
                    # Ray's counts do not prove the architecture of eligible
                    # workers. Use a worker-safe BF16 default, never driver CUDA.
                    allow_native_nvfp4=not isinstance(resources, ClusterResources),
                )
                revision = resolve_embed_model_revision(model_name, params.embed_model_revision)
                # Resolve once before actor creation. Both the worker and sink
                # use this immutable identity, including unregistered Hub repos.
                params = params.model_copy(
                    update={"model_name": model_name, "embed_model_name": model_name, "embed_model_revision": revision}
                )
                resolved_kwargs["params"] = params
            embedding_stage = (params, local)
        if resolved_kwargs.pop("_embedding_params", None) is not None:
            if embedding_stage is None:
                raise ValueError("Cannot infer index embedding identity without an upstream embedding stage.")
            params, local = embedding_stage
            sink_kwargs = dict(resolved_kwargs["vdb_kwargs"])
            sink_kwargs.pop("embedding_model_name", None)
            sink_kwargs.pop("embedding_model_revision", None)
            resolved_kwargs["vdb_kwargs"] = {**_embedding_model_kwargs(params, local=local), **sink_kwargs}
        cloned = Node(
            operator,
            name=node.name,
            operator_class=resolved_class,
            operator_kwargs=resolved_kwargs,
        )
        visited[node_id] = cloned
        for child in node.children:
            cloned.children.append(_clone(child, embedding_stage))
        return cloned

    for root in graph.roots:
        resolved.roots.append(_clone(root))

    if graph._tail is not None:
        resolved._tail = visited.get(id(graph._tail))
    return resolved


def resolve_graph_for_local_execution(graph: Graph) -> Graph:
    return resolve_graph(graph, _rrh.gather_local_resources())
