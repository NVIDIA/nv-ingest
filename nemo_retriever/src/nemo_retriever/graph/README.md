# NeMo Retriever Graph

The `nemo_retriever.graph` package contains the graph-based execution model used to build explicit ingestion pipelines from composable operators.

It is useful when you want to:

- Build a pipeline stage-by-stage instead of using a higher-level fluent API
- Reuse operators across multiple workflows
- Run the same graph in-process or with Ray Data
- Add your own custom logic as a first-class pipeline stage

## Core Concepts

The graph package is built around a few small pieces:

- `AbstractOperator`: base class for a pipeline stage
- `Node`: wraps an operator and tracks downstream children
- `Graph`: owns one or more root nodes and executes them
- `InprocessExecutor`: runs a graph locally on pandas DataFrames
- `RayDataExecutor`: runs a graph as a Ray Data pipeline
- `UDFOperator`: wraps a plain Python function as an operator

Most users interact with the graph by chaining operators with `>>`.

## Smallest Example

```python
from nemo_retriever.graph import Graph, UDFOperator


def double(x):
    return x * 2


graph = Graph()
graph.add_root(UDFOperator(double, name="Double"))
result = graph.execute(3)

print(result)  # [6]
```

Because `AbstractOperator.__rshift__` returns a `Graph`, chaining works naturally:

```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda x: x + 1, name="AddOne")
    >> UDFOperator(lambda x: x * 10, name="TimesTen")
)

print(graph.execute(5))  # [60]
```

## Building a Multi-Stage Graph

Here is a more realistic example with a few named stages:

```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda text: text.strip(), name="Trim")
    >> UDFOperator(lambda text: text.lower(), name="Lower")
    >> UDFOperator(lambda text: f"normalized::{text}", name="Prefix")
)

result = graph.execute("  Hello World  ")
print(result)  # ['normalized::hello world']
```

The returned value is a list because a graph can have multiple leaf nodes.

## Writing a Custom Operator

Use a custom operator when you want reusable logic, configuration, or access to the full `preprocess -> process -> postprocess` lifecycle.

```python
from typing import Any

from nemo_retriever.graph import AbstractOperator


class AddSuffixOperator(AbstractOperator):
    def __init__(self, suffix: str = "_done") -> None:
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return str(data).strip()

    def process(self, data: Any, **kwargs: Any) -> Any:
        return f"{data}{self.suffix}"

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

```



```python
from nemo_retriever.graph import UDFOperator


graph = (
    UDFOperator(lambda text: text.upper(), name="Upper")
    >> AddSuffixOperator("_READY")
)

print(graph.execute("hello"))  # ['HELLO_READY']
```

## Operator Lifecycle

Each operator runs in three steps:

1. `preprocess(data, **kwargs)`
2. `process(data, **kwargs)`
3. `postprocess(data, **kwargs)`

`run()` calls those three in order, and both executors use that same lifecycle.

This pattern is helpful when you want to:

- Normalize or validate input in `preprocess`
- Perform the main computation in `process`
- Clean up or reshape output in `postprocess`

## Custom Operator Tips

- Keep `process()` focused on the main transformation
- Use `__init__()` for stable configuration like thresholds, URLs, or model names
- Accept `**kwargs` in lifecycle methods so your operator stays executor-friendly
- Return a value that the next stage can consume directly
- If the operator will run in `RayDataExecutor`, make sure construction is safe on workers

## Using `UDFOperator`

`UDFOperator` is the easiest way to adopt the graph system when you do not need a full class yet.

It lets you wrap a plain Python callable:

```python
from nemo_retriever.graph import Graph, UDFOperator


def uppercase_and_prefix(text: str) -> str:
    return f"PROCESSED: {text.upper()}"


graph = Graph()
graph.add_root(UDFOperator(uppercase_and_prefix, name="UppercasePrefix"))
print(graph.execute("hello world"))  # ['PROCESSED: HELLO WORLD']
```

You can also chain it with custom operators:

```python
graph = (
    UDFOperator(lambda x: x * 4, name="MultiplyByFour")
    >> UDFOperator(lambda x: x + 3, name="AddThree")
    >> UDFOperator(lambda x: f"{x}_done", name="Finalize")
)

print(graph.execute(2))  # ['11_done']
```

## When To Use `UDFOperator` vs Custom Operator

Use `UDFOperator` when:

- You want to prototype quickly
- The logic is small and self-contained
- You only need one transformation step
- You want a low-friction way to join existing graph stages

Use a custom `AbstractOperator` subclass when:

- The stage has configuration or internal state
- You want reusable code with a clear name
- You need `preprocess` or `postprocess`
- The implementation is complex enough that a class is easier to maintain

## Executing a Graph

### Direct execution

`Graph.execute()` is the simplest option and is good for small in-memory workflows:

```python
result = graph.execute("sample input")
```

### In-process execution

Use `InprocessExecutor` for linear graphs that should run locally on pandas DataFrames:

```python
from nemo_retriever.graph import InprocessExecutor


executor = InprocessExecutor(graph)
result_df = executor.ingest(["/path/to/file.pdf"])
```

### Ray Data execution

Use `RayDataExecutor` for linear graphs that should run with Ray Data:

```python
from nemo_retriever.graph import RayDataExecutor


executor = RayDataExecutor(graph, batch_size=8)
result_ds = executor.ingest(["/path/to/*.pdf"])
```

`RayDataExecutor.ingest()` defaults `return_results` to `None`. When the graph
has one streaming-capable VDB upload as its final stage, it releases consumed
batches and returns a summary with `input_rows` and `submitted_records`.
Other graphs return full result rows. Pass `return_results=True` when you need
those rows. Explicit `False` requires the supported terminal VDB configuration
and raises `ValueError` otherwise. The fluent `GraphIngestor` API also checks
its error policy and `return_failures` setting when choosing a default. Refer to
[ingest result defaults](../../../../docs/docs/extraction/nemo-retriever-api-reference.md#return-ingest-results).

## Notes About Graph Shape

The base `Graph` type supports multiple roots and fan-out.

`InprocessExecutor` and `RayDataExecutor` currently support linear graphs only:

- one root
- one child per node

If you use direct `Graph.execute()`, branching graphs are supported and leaf outputs are collected into a list.

## Imports

Recommended imports:

```python
from nemo_retriever.graph import (
    AbstractOperator,
    Graph,
    InprocessExecutor,
    Node,
    RayDataExecutor,
    UDFOperator,
)
```

`nemo_retriever.graph` is the graph API location used by the codebase.

For NimClient and custom NIM integrations in UDFs, refer to [NimClient and Custom NIM Endpoints](../../developer_docs/nimclient.md).
