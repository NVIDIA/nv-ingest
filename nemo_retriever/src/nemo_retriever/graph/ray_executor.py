from __future__ import annotations

from typing import Any, Dict, List, Optional

import ray
import ray.data as rd

from nemo_retriever.graph.graph import Graph
from nemo_retriever.graph.node import Node
from nemo_retriever.graph.executor import Executor


class RayDataExecutor(Executor):
    """Executor that runs a `Graph` using Ray Data `map_batches`.

    Supports operator instances or operator classes (with constructor kwargs),
    and executes DAGs by topological order. When a node has multiple parents,
    parent datasets are unioned before applying the node's operator.
    """

    def __init__(self, init_ray: bool = True, map_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self._map_kwargs = map_kwargs or {}
        if init_ray:
            ray.init(ignore_reinit_error=True)

    def _topological_order(self, graph: Graph) -> List[Node]:
        # Kahn's algorithm on nodes reachable from root
        if graph.root is None:
            return []
        # Build in-degree map
        nodes = list(graph.nodes.values())
        indeg: Dict[str, int] = {n.id: 0 for n in nodes}
        children_map: Dict[str, List[str]] = {n.id: [c.id for c in n.children] for n in nodes}
        parent_map: Dict[str, List[str]] = {n.id: [] for n in nodes}
        for n in nodes:
            for c in n.children:
                parent_map[c.id].append(n.id)
                indeg[c.id] += 1

        # Start from nodes with indeg 0 that are reachable from root
        # We'll perform BFS from root to collect reachable nodes
        reachable = set()
        stack = [graph.root]
        while stack:
            cur = stack.pop()
            if cur.id in reachable:
                continue
            reachable.add(cur.id)
            for c in cur.children:
                stack.append(c)

        # Kahn over reachable subset
        q: List[Node] = [n for n in nodes if indeg[n.id] == 0 and n.id in reachable]
        order: List[Node] = []
        while q:
            n = q.pop(0)
            order.append(n)
            for cid in children_map.get(n.id, []):
                indeg[cid] -= 1
                if indeg[cid] == 0 and cid in reachable:
                    q.append(graph.nodes[cid])

        # Verify all reachable nodes are included
        if set(n.id for n in order) != reachable:
            raise RuntimeError("Graph contains a cycle or disconnected components")
        return order

    def run(self, graph: Graph, data: Any, **kwargs: Any) -> Dict[str, rd.Dataset]:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a Graph instance")

        # Prepare initial dataset
        if isinstance(data, rd.Dataset):
            base_ds = data
        else:
            base_ds = rd.from_items(data)

        order = self._topological_order(graph)
        node_outputs: Dict[str, rd.Dataset] = {}

        for node in order:
            # Determine input dataset by unioning parent outputs or use base_ds for root
            parents = [p for p in graph.nodes.values() if node.id in [c.id for c in p.children]]
            if not parents:
                inp_ds = base_ds
            else:
                # Union parent datasets
                parent_ds_list = [node_outputs[p.id] for p in parents if p.id in node_outputs]
                if not parent_ds_list:
                    inp_ds = base_ds
                else:
                    inp_ds = parent_ds_list[0]
                    for other in parent_ds_list[1:]:
                        inp_ds = inp_ds.union(other)

            map_opts = dict(self._map_kwargs)
            map_opts.update(kwargs)

            fn, ctor_kwargs = node.get_map_fn_and_constructor_kwargs()
            if ctor_kwargs:
                # pass fn_constructor_kwargs for actor-style classes
                map_opts_local = dict(map_opts)
                map_opts_local["fn_constructor_kwargs"] = ctor_kwargs
                ds_out = inp_ds.map_batches(fn, **map_opts_local)
            else:
                ds_out = inp_ds.map_batches(fn, **map_opts)

            # Materialize intermediate result to free resources as we proceed
            node_outputs[node.id] = ds_out

        # Materialize all outputs
        # for k, v in node_outputs.items():
        #     node_outputs[k] = v.materialize()
        node_output = inp_ds.materialize()

        return node_outputs
