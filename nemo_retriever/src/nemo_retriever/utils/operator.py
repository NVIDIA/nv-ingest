from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class AbstractOperator(ABC):
    """Abstract dataclass for Ray Data operators.

    Subclasses should implement `process`. Optional hooks `pre_process`
    and `post_process` may be overridden. Instances are callable so they
    can be used directly with `ray.data.map_batches`.
    """

    def pre_process(self, batch: Any, **kwargs: Any) -> Any:
        return batch

    @abstractmethod
    def process(self, data: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def post_process(self, data: Any, **kwargs: Any) -> Any:
        return data

    def run(self, batch: Any, **kwargs: Any) -> Any:
        v1 = self.pre_process(batch, **kwargs)
        v2 = self.process(v1, **kwargs)
        v3 = self.post_process(v2, **kwargs)
        return v3

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        return self.run(batch, **kwargs)
