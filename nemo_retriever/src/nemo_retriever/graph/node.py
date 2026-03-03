from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

from nemo_retriever.utils.operator import AbstractOperator


class Node:
    """Graph node that wraps an `AbstractOperator` and connects to other nodes.

    Each node holds a reference to an `AbstractOperator` instance and a list
    of child `Node`s. Calling `run` executes the node's operator and then
    propagates the result to each child node, collecting outputs.
    """

    def __init__(
        self,
        node_id: str,
        operator: Union[AbstractOperator, Type[AbstractOperator]],
        constructor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")
        self.id = node_id
        # operator can be either a concrete AbstractOperator instance or a class
        if isinstance(operator, AbstractOperator):
            self.operator_instance: Optional[AbstractOperator] = operator
            self.operator_class: Optional[Type[AbstractOperator]] = None
        elif isinstance(operator, type) and issubclass(operator, AbstractOperator):
            self.operator_instance = None
            self.operator_class = operator
        else:
            raise TypeError("operator must be an AbstractOperator instance or subclass")

        self.constructor_kwargs = dict(constructor_kwargs or {})
        self.children: List[Node] = []

    def add_child(self, child: "Node") -> None:
        if not isinstance(child, Node):
            raise TypeError("child must be a Node")
        self.children.append(child)

    def run(self, batch: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute this node and its descendants.

        Returns a mapping from node id to the node's operator result.
        This method only executes this node's operator; propagation to
        children is handled by the Graph container.
        """
        # Prefer instance if present, otherwise construct a temporary instance
        if self.operator_instance is not None:
            op = self.operator_instance
        elif self.operator_class is not None:
            op = self.operator_class(**self.constructor_kwargs)
        else:  # pragma: no cover - defensive
            raise RuntimeError("Node has no operator configured")

        out = op.run(batch, **kwargs)
        return {self.id: out}

    def is_actor_class(self) -> bool:
        return self.operator_class is not None

    def get_map_fn_and_constructor_kwargs(self):
        """Return a tuple (fn, fn_constructor_kwargs) suitable for ray.data.map_batches.

        If node wraps an operator class, return (operator_class, constructor_kwargs).
        If node wraps an instance, return (operator_instance, None).
        """
        if self.operator_instance is not None:
            return self.operator_instance, None
        return self.operator_class, dict(self.constructor_kwargs or {})
