from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from nemo_retriever.graph.node import Node
from nemo_retriever.utils.operator import AbstractOperator


class Graph:
    """Simple graph container holding nodes and a root reference.

    The graph manages Node instances and exposes a `run` method that
    executes from the root node, returning a dict of node outputs.
    """

    def __init__(self, operators: Optional[Iterable[AbstractOperator]] = None, root: Optional[Node] = None) -> None:
        """Construct a Graph.

        If *operators* is provided, a `Node` is created for each operator and
        they are connected sequentially in the order given. The first
        operator becomes the root automatically.
        """
        self.nodes: Dict[str, Node] = {}
        self.root: Optional[Node] = None
        if operators:
            prev: Optional[Node] = None
            for idx, op in enumerate(operators):
                if not isinstance(op, AbstractOperator):
                    raise TypeError("operators must be instances of AbstractOperator")
                node_id = f"{op.__class__.__name__}_{idx}"
                node = Node(node_id, op)
                self.add_node(node)
                if prev is not None:
                    prev.add_child(node)
                else:
                    self.root = node
                prev = node
        # allow explicit root override
        if root is not None:
            self.set_root(root)

    def add_operator(self, operator: AbstractOperator) -> Node:
        """Create a Node from *operator*, add it to the graph, and append it
        sequentially after the current last node. If the graph has no root,
        the new node becomes the root.
        Returns the created Node.
        """
        if not isinstance(operator, AbstractOperator):
            raise TypeError("operator must be an AbstractOperator instance")
        base = operator.__class__.__name__
        # build a unique id
        idx = 0
        node_id = f"{base}_{idx}"
        while node_id in self.nodes:
            idx += 1
            node_id = f"{base}_{idx}"
        node = Node(node_id, operator)
        # attach sequentially to last node if present
        last_node = None
        if self.nodes:
            # insertion-ordered dict -> last value is last added
            last_node = next(reversed(self.nodes.values()))
        self.add_node(node)
        if last_node is not None:
            last_node.add_child(node)
        if self.root is None:
            self.root = node
        return node

    def add(self, item: Union[Node, AbstractOperator]) -> Node:
        """Add either a `Node` or an `AbstractOperator` to the graph.

        If given an operator, it is wrapped in a `Node` via `add_operator`.
        If given a `Node`, it is added directly and becomes the root if the
        graph has no root yet.
        Returns the added `Node`.
        """
        if isinstance(item, Node):
            self.add_node(item)
            if self.root is None:
                self.root = item
            return item
        if isinstance(item, AbstractOperator):
            return self.add_operator(item)
        raise TypeError("add expects a Node or AbstractOperator")

    def add_node(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("node must be a Node")
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id!r} already exists")
        self.nodes[node.id] = node

    def connect(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if parent is None or child is None:
            raise KeyError("parent or child node id not found in graph")
        parent.add_child(child)

    def set_root(self, node: Node) -> None:
        if not isinstance(node, Node):
            raise TypeError("root must be a Node")
        if node.id not in self.nodes:
            self.add_node(node)
        self.root = node

    def run(self, batch: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.root is None:
            raise RuntimeError("Graph root is not set")
        results: Dict[str, Any] = {}
        current = self.root
        inp = batch
        visited: List[str] = []
        # Traverse sequential chain from root following the first child.
        while current is not None:
            if current.id in visited:
                raise RuntimeError("Graph contains a cycle")
            visited.append(current.id)
            res = current.run(inp, **kwargs)
            results.update(res)
            # prepare input for next node
            inp = res.get(current.id)
            # pick the first child if present (graph is sequential by construction)
            current = current.children[0] if current.children else None
        return results
