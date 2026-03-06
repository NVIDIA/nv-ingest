from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, Union

from nemo_retriever.utils.operator import AbstractOperator

# Accepted operator types: an AbstractOperator instance, an AbstractOperator
# subclass (actor pattern), or a plain callable (regular function / partial).
OperatorLike = Union[AbstractOperator, Type[AbstractOperator], Callable[..., Any]]


class Node:
    """Graph node that wraps an operator and connects to other nodes.

    The operator can be:
    - An ``AbstractOperator`` *instance* (serialised and called by Ray workers).
    - An ``AbstractOperator`` *class* (actor pattern — Ray constructs one per worker).
    - A plain callable / ``functools.partial`` (used as-is by ``map_batches``).

    Per-node ``map_kwargs`` (e.g. ``batch_size``, ``num_cpus``, ``num_gpus``,
    ``compute``, ``batch_format``) are forwarded to ``ray.data.map_batches``
    by the ``RayDataExecutor``.
    """

    def __init__(
        self,
        node_id: str,
        operator: OperatorLike,
        constructor_kwargs: Optional[Dict[str, Any]] = None,
        map_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")
        self.id = node_id

        # Classify the operator type.
        if isinstance(operator, AbstractOperator):
            self.operator_instance: Optional[AbstractOperator] = operator
            self.operator_class: Optional[Type[AbstractOperator]] = None
            self.operator_fn: Optional[Callable[..., Any]] = None
        elif isinstance(operator, type) and issubclass(operator, AbstractOperator):
            self.operator_instance = None
            self.operator_class = operator
            self.operator_fn = None
        elif callable(operator):
            self.operator_instance = None
            self.operator_class = None
            self.operator_fn = operator
        else:
            raise TypeError(
                "operator must be an AbstractOperator instance, "
                "an AbstractOperator subclass, or a callable"
            )

        self.constructor_kwargs = dict(constructor_kwargs or {})
        self.map_kwargs: Dict[str, Any] = dict(map_kwargs or {})
        self.children: List[Node] = []

    def add_child(self, child: "Node") -> None:
        if not isinstance(child, Node):
            raise TypeError("child must be a Node")
        self.children.append(child)

    def run(self, batch: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute this node's operator on *batch* (used by ``Graph.run``)."""
        if self.operator_instance is not None:
            op = self.operator_instance
        elif self.operator_class is not None:
            op = self.operator_class(**self.constructor_kwargs)
        elif self.operator_fn is not None:
            return {self.id: self.operator_fn(batch, **kwargs)}
        else:  # pragma: no cover
            raise RuntimeError("Node has no operator configured")

        out = op.run(batch, **kwargs)
        return {self.id: out}

    def is_actor_class(self) -> bool:
        return self.operator_class is not None

    def get_map_fn_and_constructor_kwargs(self):
        """Return ``(fn, fn_constructor_kwargs)`` for ``ray.data.map_batches``.

        - Operator class  → ``(cls, constructor_kwargs)``
        - Operator instance → ``(instance, None)``
        - Plain callable  → ``(callable, None)``
        """
        if self.operator_instance is not None:
            return self.operator_instance, None
        if self.operator_class is not None:
            return self.operator_class, dict(self.constructor_kwargs or {})
        return self.operator_fn, None
