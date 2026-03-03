from __future__ import annotations

from typing import Any

from nemo_retriever.utils.operator import AbstractOperator
from nemo_retriever.graph.graph import Graph
from nemo_retriever.graph.ray_executor import RayDataExecutor


class PdIncOperator(AbstractOperator):
    def process(self, batch: Any, **kwargs: Any) -> Any:
        # Expect a pandas DataFrame with column 'value'
        try:
            df = batch
            df["value"] = df["value"].astype(float) + 1
            return df
        except Exception:
            return batch


class PdMulOperator(AbstractOperator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def process(self, batch: Any, **kwargs: Any) -> Any:
        try:
            df = batch
            df["value"] = df["value"].astype(float) * float(self.factor)
            return df
        except Exception:
            return batch


def example_run():
    # Create simple graph: inc -> mul2 -> mul3
    g = Graph(operators=[PdIncOperator(), PdMulOperator(2), PdMulOperator(3)])

    # Sample data: list of dicts with 'value' field
    data = [{"value": 1}, {"value": 2}, {"value": 3}]

    exec = RayDataExecutor(init_ray=True)
    ds = exec.run(g, data, batch_size=2)

    # Print materialized dataset
    print(ds.show())


if __name__ == "__main__":
    example_run()
