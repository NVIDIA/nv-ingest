from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from nemo_retriever.graph.graph import Graph


class Executor(ABC):
    """Abstract executor base class.

    Subclasses implement `run(graph, batch, **kwargs)` which is responsible
    for executing nodes in the provided `Graph` according to the executor's
    strategy (sequential, parallel, etc.).
    """

    @abstractmethod
    def run(self, graph: Graph, batch: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute the given *graph* starting from its root and return a
        mapping of node id to the node's result.

        Parameters
        - graph: Graph to execute
        - batch: initial input passed to the graph root
        - **kwargs: forwarded to node/operator run calls
        """
        raise NotImplementedError()

    def __call__(self, graph: Graph, batch: Any = None, **kwargs: Any) -> Dict[str, Any]:
        return self.run(graph, batch, **kwargs)
