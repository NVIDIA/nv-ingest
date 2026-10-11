# Contributing to NeMo Retriever

External contributions will be welcome soon, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Table of Contents

1. [Filing Issues](#filing-issues)
2. [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
3. [Cloning the Repository](#cloning-the-repository)
4. [Code Contributions](#code-contributions)
   - [Your First Issue](#your-first-issue)
   - [Seasoned Developers](#seasoned-developers)
   - [Workflow](#workflow)
   - [Common Processing Patterns](#common-processing-patterns)
     - [traceable decorator](#traceable-decorator)
     - [Node failure decorator (try/except)](#node-failure-decorator-tryexcept)
     - [filter_by_task decorator](#filter_by_task-decorator)
   - [Adding a New Stage or Module](#adding-a-new-stage-or-module)
     - [Overview](#overview)
     - [End-to-End Checklist](#end-to-end-checklist)
     - [Task Definition: CLI and Configuration](#task-definition-cli-and-configuration)
     - [Processing Module: Operators and Tasks](#processing-module-operators-and-tasks)
     - [Wiring into the Pipeline](#wiring-into-the-pipeline)
     - [Unit Testing for New Stages](#unit-testing-for-new-stages)
     - [Example: Adding a Summarization Task](#example-adding-a-summarization-task)
   - [Common Practices for Writing Unit Tests](#common-practices-for-writing-unit-tests)
     - [General Guidelines](#general-guidelines)
     - [Mocking External Services](#mocking-external-services)
   - [Submodules, Third Party Libraries, and Models](#submodules-third-party-libraries-and-models)
     - [Submodules](#submodules)
     - [Models](#models)
5. [Architectural Guidelines](#architectural-guidelines)
   - [Single Responsibility Principle (SRP)](#1-single-responsibility-principle-srp)
   - [Interface Segregation Principle (ISP)](#2-interface-segregation-principle-isp)
   - [Dependency Inversion Principle (DIP)](#3-dependency-inversion-principle-dip)
   - [Physical Design Structure Mirroring Logical Design Structure](#4-physical-design-structure-mirroring-logical-design-structure)
   - [Levelization](#5-levelization)
   - [Acyclic Dependencies Principle (ADP)](#6-acyclic-dependencies-principle-adp)
   - [Package Cohesion Principles](#7-package-cohesion-principles)
     - [Common Closure Principle (CCP)](#common-closure-principle-ccp)
     - [Common Reuse Principle (CRP)](#common-reuse-principle-crp)
   - [Encapsulate What Varies](#8-encapsulate-what-varies)
   - [Favor Composition Over Inheritance](#9-favor-composition-over-inheritance)
   - [Clean Separation of Concerns (SoC)](#10-clean-separation-of-concerns-soc)
   - [Principle of Least Knowledge (Law of Demeter)](#11-principle-of-least-knowledge-law-of-demeter)
   - [Document Assumptions and Decisions](#12-document-assumptions-and-decisions)
   - [Continuous Integration and Testing](#13-continuous-integration-and-testing)
6. [Writing Good and Thorough Documentation](#writing-good-and-thorough-documentation)
7. [Licensing](#licensing)
8. [Attribution](#attribution)

## Filing Issues

1. **Bug Reports, Feature Requests, and Documentation Issues:** Please file
   an [issue](https://github.com/NVIDIA/NeMo-Retriever/issues) with a detailed
   description of
   the problem, feature request, or documentation issue. The NeMo Retriever team will review and triage these issues,
   and if appropriate, schedule them for a future release.

## Developer Certificate of Origin (DCO)

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original
work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not signed off are not accepted.

To sign off on a commit, use the `--signoff` (or `-s`) option when you commit your changes:

```bash
git commit --signoff --message "Add cool feature."
```

This appends the following text to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

### Full text of the Developer Certificate of Origin

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## Cloning the repository

```bash
DATASET_ROOT=[path to your dataset root]
MODULE_NAME=[]
NEMO_RETRIEVER_ROOT=[path to your NeMo Retriever clone]
git clone https://github.com/NVIDIA/NeMo-Retriever.git $NEMO_RETRIEVER_ROOT
cd $NEMO_RETRIEVER_ROOT
```

Ensure all submodules are checked out:

```bash
git submodule update --init --recursive
```

## Code Contributions

### Your First Issue

1. **Finding an Issue:** Start with issues
   labeled [good first issue](https://github.com/NVIDIA/NeMo-Retriever/labels/good%20first%20issue).
2. **Claim an Issue:** Comment on the issue you wish to work on.
3. **Implement Your Solution:** Dive into the code! Update or add unit tests as necessary.
4. **Submit Your Pull Request:
   ** [Create a pull request](https://github.com/NVIDIA/NeMo-Retriever/pulls) once your
   code is ready.
5. **Code Review:** Wait for the review by other developers and make necessary updates.
6. **Merge:** After approval, an NVIDIA developer will approve your pull request.

### Seasoned Developers

For those familiar with the codebase, please check
the [project boards](https://github.com/orgs/NVIDIA/projects/48/views/1) for
issues. Look for unassigned issues and follow the steps starting from **Claim an Issue**.

### Workflow

1. **NeMo Retriever foundation**: Built on top
   of [Ray](https://docs.ray.io/en/latest/serve/architecture.html).

2. **Pipeline Structure**: Designed around a pipeline that processes individual jobs within an asynchronous execution
   graph. Each job is processed by a series of stages or task handlers.

3. **Job Composition**: Jobs consist of a data payload, metadata, and task specifications that determine the processing
   steps applied to the data.

4. **Job Submission**:

   - A job is submitted as a JSON specification and converted into
     a [ControlMessage](https://github.com/nv-morpheus/Morpheus/blob/branch-24.06/docs/source/developer_guide/guides/9_control_messages.md),
     with the payload consisting of a cuDF dataframe.
   - For example:
     ```text
         document_type source_id   uuid     metadata
         0             pdf         somefile  1234  { ... }
     ```
   - The `metadata` column contents correspond to
     the [schema-enforced metadata format of returned data](docs/docs/extraction/content-metadata.md).

5. **Pipeline Processing**:

   - The `ControlMessage` is passed through the pipeline, where each stage processes the data and metadata as needed.
   - Subsequent stages may add, transform, or filter data as needed, with all resulting artifacts stored in
     the `ControlMessage`'s payload.
   - For example, after processing, the payload may look like:
     ```text
         document_type   source_id   uuid       metadata
         0               text        somefile   abcd-1234   {'content': "The quick brown fox jumped...", ...}
         1               image       somefile   efgh-5678   {'content': "base64 encoded image", ...}
         2               image       somefile   xyza-5618   {'content': "base64 encoded image", ...}
         3               image       somefile   zxya-5628   {'content': "base64 encoded image", ...}
         4               status      somefile   kvq9-5600   {'content': "", 'status': "filtered", ...}
     ```
   - A single job can result in multiple artifacts, each with its own metadata element definition.

6. **Job Completion**:
   - Upon reaching the end of the pipeline, the `ControlMessage` is converted into a `JobResult` object and pushed to
     the ephemeral output queue for client retrieval.
   - `JobResult` objects consist of a dictionary containing:
     1. **data**: A list of metadata artifacts produced by the job.
     2. **status**: The job status as success or failure.
     3. **description**: A human-readable description of the job status.
     4. **trace**: A list of timing traces generated during the job's processing.
     5. **annotations**: A list of task annotations generated during the job's processing.

### Updating Dependencies

- Dependencies are managed with `uv` and project-local `pyproject.toml` files.
- Dependencies are stored in package definitions:
    1. **Service Dependencies** `src/pyproject.toml`.
    2. **Client Dependencies** `client/pyproject.toml`.

- To update dependencies:
  - Create a clean environment using `uv venv`.
  - Update dependencies in the relevant `pyproject.toml` and validate the changes.
  - Recreate the environment and install via `uv pip`.
    - For example:
      ```bash
      uv venv .venv
      source .venv/bin/activate
      uv pip install -e ./src -e ./client -e ./api
      ```

### Common Processing Patterns

In this repository, decorators are used to enhance the functionality of functions by adding additional processing logic. These
decorators help ensure consistency, traceability, and robust error handling across the pipeline. Below, we introduce
some common decorators used in the ingestion pipeline, explain their usage, and provide examples.

#### traceable decorator

Defined in `nemo_retriever/src/nemo_retriever/api/internal/primitives/tracing/tagging.py`.

The `traceable` decorator adds entry and exit trace timestamps to an `IngestControlMessage`'s metadata when trace tagging is enabled. This helps in
monitoring and debugging by recording the time taken for function execution.

**Usage:**

- To track function execution time (trace name from `self.stage_name` or the function name on methods):
  ```python
  @traceable()
  def process_message(self, message):
      pass
  ```
- To use a custom trace name:
  ```python
  @traceable("CustomTraceName")
  def process_message(message):
      pass
  ```

#### Node failure decorator (try/except)

Defined in `nemo_retriever/src/nemo_retriever/api/util/exception_handlers/decorators.py`.

This module exposes a try/except-style decorator for pipeline stages that process `IngestControlMessage` values. It wraps failures so they are annotated consistently (and optionally re-raised). **Import the decorator from that file in your checkout**—use the definition whose name ends with `_try_except` (source is authoritative; names have changed across releases).

**Usage (pattern):**

```python
# from nemo_retriever.api.util.exception_handlers.decorators import <NODE_FAILURE_TRY_EXCEPT>
# @<NODE_FAILURE_TRY_EXCEPT>(annotation_id="example_task")
def process_message(message):
    pass
```

```python
# from nemo_retriever.api.util.exception_handlers.decorators import <NODE_FAILURE_TRY_EXCEPT>
# @<NODE_FAILURE_TRY_EXCEPT>(annotation_id="example_task", payload_can_be_empty=True)
def process_message(message):
    pass
```

By default, `skip_processing_if_failed=True`. If the `IngestControlMessage` is already marked failed (`cm_failed` in metadata), the wrapped function is not called and the message is returned (or passed to `forward_func` if set)—so downstream stages do not re-process a failed message. Set `skip_processing_if_failed=False` when a stage must run even after a prior failure.

#### filter_by_task decorator

Task-gating decorators live alongside pipeline stage implementations; search the repository for `filter_by_task` if your branch still carries that helper.

The `filter_by_task` decorator checks if the `IngestControlMessage` contains any of the required tasks. Each entry can be a
task name string or a tuple of the task name and required task properties. If the message does not contain any listed task
and/or task properties, the message is returned directly without calling the wrapped function, unless a forwarding
function is provided.

**Usage:**

- To filter messages based on tasks:
  ```python
  @filter_by_task(required_tasks=["task1", "task2"])
  def process_message(message):
      pass
  ```
- To filter messages based on tasks with specific properties:
  ```python
  @filter_by_task(required_tasks=[("task", {"prop": "value"})])
  def process_message(message):
      pass
  ```
- To forward messages to another function when the decorated function does not return the message
  directly, provide `forward_func`:
  ```python
  @filter_by_task(required_tasks=["task1", "task2"], forward_func=other_function)
  def process_message(message):
      pass
  ```

### Adding a New Stage or Module

This section covers the full end-to-end process for adding a new processing task type to NeMo Retriever—from exposing it through the CLI and library, to writing the processing module, wiring it into the pipeline, and setting up unit tests.

#### Overview

NeMo Retriever's ingest pipeline is a directed graph of **operators**. Each operator is a class that extends `AbstractOperator` and processes a pandas DataFrame of document rows. Operators are constructed from typed **Pydantic params objects**; their presence or absence in a plan determines which stages run.

The path from a user-facing flag to actual processing is:

```
CLI flag (options.py)
  → IngestXxxOptions dataclass (ingest/plan.py)
  → resolve_ingest_plan() → XxxParams (common/params/models.py)
  → build_graph() (graph/ingestor_runtime.py) → Graph of operators
  → GraphIngestor.ingest() (ingestor/graph_ingestor.py)
```

#### End-to-End Checklist

Use this checklist when adding a new stage. Each step is expanded in detail below.

1. **Params model** — Add a Pydantic params class in `common/params/models.py`.
2. **Task class** *(optional, for LLM-backed tasks)* — Add a task class in `models/llm/tasks/` that encapsulates prompt building and execution.
3. **Operator class** — Add an operator in `operators/<category>/my_task.py` extending `AbstractOperator`.
4. **Stage registry** *(stateless function stages only)* — Register a handler in `graph/stages/stage_registry.py`.
5. **Graph wiring** — Insert the operator into `graph/ingestor_runtime.py :: build_graph()`.
6. **GraphIngestor fluent API** — Add a `.my_task(params)` method to `ingestor/graph_ingestor.py`.
7. **Plan options dataclass** — Add `IngestMyTaskOptions` in `ingest/plan.py`.
8. **Plan resolution** — Add resolution logic in `resolve_ingest_plan()` in `ingest/plan.py`.
9. **CLI options** — Add `Annotated[...]` type aliases in `cli/ingest/options.py`.
10. **CLI command** — Wire the new options into `_graph_ingest_command` in `cli/ingest/graph_commands.py`.
11. **Tests** — Write unit tests for the operator, graph structure, and CLI wiring.

---

#### Task Definition: CLI and Configuration

##### 1. Define a Params Model

All stage configuration flows as typed Pydantic models. Add your params class to `common/params/models.py`:

```python
# common/params/models.py
from pydantic import BaseModel

class MyNewTaskParams(BaseModel):
    invoke_url: str | None = None
    api_key: str | None = None
    my_option: str = "default"
```

Export it from `common/params/__init__.py` alongside existing params.

##### 2. Define CLI Options

Options are `Annotated` type aliases in `cli/ingest/options.py`. Add one alias per CLI flag:

```python
# cli/ingest/options.py
MyNewTaskOption = Annotated[
    bool,
    typer.Option("--my-new-task", help="Enable the new task stage."),
]
MyNewTaskInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--my-new-task-invoke-url", help="Remote endpoint URL for my new task."),
]
```

##### 3. Add the Options Dataclass

Add a frozen dataclass in `ingest/plan.py` to group the new options:

```python
# ingest/plan.py
@dataclass(frozen=True)
class IngestMyNewTaskOptions:
    enabled: bool = False
    invoke_url: str | None = None
    my_option: str = "default"
```

Then add the field to `IngestPlanRequest`:

```python
@dataclass(frozen=True)
class IngestPlanRequest:
    ...
    my_new_task: IngestMyNewTaskOptions = field(default_factory=IngestMyNewTaskOptions)
```

##### 4. Resolve the Options to Params

In `resolve_ingest_plan()` in `ingest/plan.py`, translate the options dataclass into your Pydantic params:

```python
# inside resolve_ingest_plan()
def _build_my_new_task_params(options: IngestMyNewTaskOptions) -> MyNewTaskParams | None:
    if not options.enabled:
        return None
    kwargs = {}
    if options.invoke_url is not None:
        kwargs["invoke_url"] = options.invoke_url
    return MyNewTaskParams(**kwargs)

my_new_task_params = _build_my_new_task_params(request.my_new_task)
```

Return `my_new_task_params` from `resolve_ingest_plan()` and include it in the `ResolvedIngestPlan` dataclass. Also add it to `_ingest_plan_to_dry_run_data()` in `cli/ingest_workflow.py` so it appears in `--dry-run` output.

##### 5. Wire the CLI Flags into the Command

In `cli/ingest/graph_commands.py`, add the new option types as parameters to `_graph_ingest_command` and build them in `_build_graph_ingest_request`:

```python
def _build_my_new_task_options(values: Mapping[str, Any]) -> IngestMyNewTaskOptions:
    return IngestMyNewTaskOptions(
        enabled=values["my_new_task"],
        invoke_url=values.get("my_new_task_invoke_url"),
    )

def _build_graph_ingest_request(values, *, run_mode) -> IngestPlanRequest:
    return IngestPlanRequest(
        ...
        my_new_task=_build_my_new_task_options(values),
    )
```

---

#### Processing Module: Operators and Tasks

##### Operator Base Classes

All operators extend `AbstractOperator` from `operators/abstract_operator.py`:

```python
class AbstractOperator(ABC):
    def preprocess(self, data: Any, **kwargs) -> Any: ...
    def process(self, data: Any, **kwargs) -> Any: ...
    def postprocess(self, data: Any, **kwargs) -> Any: ...

    def run(self, data, **kwargs):
        data = self.preprocess(data, **kwargs)
        data = self.process(data, **kwargs)
        return self.postprocess(data, **kwargs)
```

Mix in `CPUOperator` for CPU-bound work or `GPUOperator` for GPU-bound work. For stages that need to auto-select between CPU and GPU variants at runtime based on cluster resources, extend `ArchetypeOperator` and provide sibling CPU/GPU subclasses—see `operators/extract/ocr/ocr.py` for an example.

##### Writing an Operator

Create the file at `operators/<category>/my_task.py`. Follow these conventions:

- Accept a params object as the first constructor argument.
- Store all constructor state so `get_constructor_kwargs()` can reconstruct the operator without capturing live clients.
- Use the decorators described in [Common Processing Patterns](#common-processing-patterns) to handle tracing, error annotation, and task gating.

```python
# operators/my_category/my_task.py
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.common.params import MyNewTaskParams

class MyNewTaskOperator(AbstractOperator, CPUOperator):
    def __init__(self, params: MyNewTaskParams) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data, **kwargs):
        # Validate required columns, reject unsupported input shapes.
        return data

    def process(self, data, **kwargs):
        # Core transformation. Input and output are pandas DataFrames.
        results = []
        for _, row in data.iterrows():
            results.append(self._run_one(row))
        return data.assign(my_output=results)

    def postprocess(self, data, **kwargs):
        return data

    def _run_one(self, row):
        # Per-row logic. Keep side effects out of preprocess/postprocess.
        ...
```

##### Control Message Decorators

Apply these decorators to methods that process `IngestControlMessage` objects in service-mode (Morpheus) stages. They are not required for pure DataFrame operators used in the Ray-backed ingest path.

- `@traceable()` — records entry/exit timestamps in the message's metadata; use it on every `process_message` method so timing traces are available without additional instrumentation.
- Node failure decorator — wraps failures so they are consistently annotated and respects `skip_processing_if_failed=True` by default; downstream stages skip already-failed messages automatically.
- `@filter_by_task(required_tasks=["my_task"])` — short-circuits the method and passes the message through unchanged when the control message does not carry the required task. Use this whenever a stage is task-specific rather than running unconditionally.

##### LLM-Backed Tasks (Optional)

If the new stage calls a large-language model, follow the task/operator split used by `SummarizationOperator`:

1. **Task class** in `models/llm/tasks/` — builds the prompt and calls the completion client:

   ```python
   # models/llm/tasks/my_task.py
   from nemo_retriever.models.llm.tasks.base import TextGenerationTask

   class MyNewTask(TextGenerationTask):
       def build_request(self, *, text: str):
           return self._build_messages_request(
               user=f"Do something with:\n\n{text}",
               system="You are a helpful assistant.",
           )
   ```

2. **Operator class** in `operators/generation/` — extends `TextGenerationOperator` and wires the task:

   ```python
   # operators/generation/my_task.py
   from nemo_retriever.operators.generation.base import TextGenerationOperator
   from nemo_retriever.models.llm.tasks import MyNewTask

   class MyNewTaskOperator(TextGenerationOperator):
       def __init__(self, params, input_column="text", output_column="my_output", ...):
           task = MyNewTask(prompt=params.prompt, system_prompt=params.system_prompt)
           super().__init__(
               params, task=task,
               input_columns={"text": input_column},
               output_column=output_column,
               ...
           )
   ```

   `TextGenerationOperator` handles batching, concurrency, error capture, and graph serialization automatically.

##### Stage Registry (Stateless Function Stages)

If the stage is a stateless function rather than a class-based operator, register it in `graph/stages/stage_registry.py`:

```python
# graph/stages/stage_registry.py
from nemo_retriever.operators.my_category.my_task import my_task_handler

STAGE_REGISTRY: Dict[str, StageHandler] = {
    ...
    "my_new_task": my_task_handler,
}
```

The handler contract is `(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]`.

---

#### Wiring into the Pipeline

##### build_graph()

In `graph/ingestor_runtime.py`, `build_graph()` constructs the operator chain based on which params objects are present. Add a branch for the new stage:

```python
# graph/ingestor_runtime.py :: build_graph()

def build_graph(
    ...
    my_new_task_params: MyNewTaskParams | None = None,
    ...
) -> Graph:
    ...
    if my_new_task_params is not None:
        my_new_task_node = Node(
            MyNewTaskOperator,
            operator_kwargs={"params": my_new_task_params},
        )
        # Append after the last existing node, before the embed/store stages.
        last_node.children.append(my_new_task_node)
        last_node = my_new_task_node
    ...
```

Choose where in the chain to insert the node based on data dependencies. Extraction stages come before enrichment stages; enrichment before embedding; embedding before VDB upload.

##### GraphIngestor Fluent API

Add a `.my_new_task(params)` method to `GraphIngestor` in `ingestor/graph_ingestor.py`:

```python
# ingestor/graph_ingestor.py :: GraphIngestor

def my_new_task(self, params: MyNewTaskParams) -> "GraphIngestor":
    """Add a my-new-task enrichment stage to the ingest pipeline."""
    self._my_new_task_params = params
    self._record_stage("my_new_task")
    return self
```

Pass `self._my_new_task_params` through to `build_graph()` in the `.ingest()` call.

---

#### Unit Testing for New Stages

##### Operator Tests

Place the test file at the mirrored path under `tests/`:

```
nemo_retriever/src/nemo_retriever/operators/my_category/my_task.py
  → nemo_retriever/tests/operators/my_category/test_my_task.py
```

Test the operator's `run()` method directly with a pandas DataFrame. Inject fake clients via the constructor rather than patching at the class level:

```python
# tests/operators/my_category/test_my_task.py
import pandas as pd
import pytest
from nemo_retriever.operators.my_category.my_task import MyNewTaskOperator
from nemo_retriever.common.params import MyNewTaskParams


def _make_params(**kwargs):
    return MyNewTaskParams(**kwargs)


class TestMyNewTaskOperator:
    def test_happy_path_adds_output_column(self):
        params = _make_params()
        out = MyNewTaskOperator(params).run(pd.DataFrame({"text": ["hello"]}))

        assert "my_output" in out.columns
        assert out["my_output"].tolist() == ["expected result"]

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            MyNewTaskOperator(_make_params()).run(pd.DataFrame({"body": ["hello"]}))

    def test_empty_dataframe_returns_correct_schema(self):
        out = MyNewTaskOperator(_make_params()).run(
            pd.DataFrame({"text": pd.Series(dtype=str)})
        )
        assert list(out.columns) == ["text", "my_output"]
        assert out.empty
```

For LLM-backed operators, use the `FakeCompletionClient` pattern to avoid real network calls:

```python
class FakeCompletionClient:
    def __init__(self, response="generated text", latency=0.1):
        self.calls = []
        self._response = response
        self._latency = latency

    @property
    def model(self):
        return "fake/model"

    def complete(self, messages, max_tokens=None, extra_params=None):
        self.calls.append((messages, max_tokens, extra_params))
        return self._response, self._latency


def test_llm_operator_calls_client(self):
    client = FakeCompletionClient(response="  result  ")
    out = MyNewTaskOperator(_make_params(), client=client).run(
        pd.DataFrame({"text": ["source text"]})
    )

    assert out["my_output"].tolist() == ["result"]
    assert len(client.calls) == 1
```

##### Graph Structure Tests

Verify the operator appears at the correct position in the graph. Add a test in `tests/test_ingest_plans.py` (or a dedicated `tests/test_build_graph_my_task.py`):

```python
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.common.params import EmbedParams, ExtractParams, MyNewTaskParams


def _node_names(graph):
    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            return names
        node = node.children[0]


def test_build_graph_inserts_my_new_task_before_embed():
    graph = build_graph(
        extract_params=ExtractParams(method="pdfium"),
        my_new_task_params=MyNewTaskParams(),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
    )
    names = _node_names(graph)

    assert "MyNewTaskOperator" in names
    assert names.index("MyNewTaskOperator") < names.index("_BatchEmbedActor")


def test_build_graph_omits_my_new_task_when_params_absent():
    graph = build_graph(
        extract_params=ExtractParams(method="pdfium"),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
    )
    assert "MyNewTaskOperator" not in _node_names(graph)
```

##### CLI Tests

Use `typer.testing.CliRunner` and `monkeypatch` to test the CLI flag end-to-end without constructing a real ingestor:

```python
# tests/test_root_cli_workflow.py (or a focused test module)
import importlib
from unittest.mock import create_autospec
from typer.testing import CliRunner
import nemo_retriever.ingest.execution as ingest_execution
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.common.params import MyNewTaskParams

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")


def _make_fake_ingestor():
    fake = create_autospec(GraphIngestor, instance=True, spec_set=True)
    fake.files.return_value = fake
    fake.extract.return_value = fake
    fake.my_new_task.return_value = fake
    fake.embed.return_value = fake
    fake.vdb_upload.return_value = fake
    fake.ingest.return_value = [{"status": "ok"}]
    return fake


def test_my_new_task_flag_passes_params_to_ingestor(monkeypatch, tmp_path):
    fake = _make_fake_ingestor()
    document = tmp_path / "test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "local", "--my-new-task", str(document)],
    )

    assert result.exit_code == 0, result.output
    assert fake.my_new_task.called
    params = fake.my_new_task.call_args.args[0]
    assert isinstance(params, MyNewTaskParams)


def test_my_new_task_disabled_by_default(monkeypatch, tmp_path):
    fake = _make_fake_ingestor()
    document = tmp_path / "test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake)

    RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert not fake.my_new_task.called
```

---

#### Example: Adding a Summarization Task

The `SummarizationOperator` is the canonical reference for the full pattern. Follow this chain when reading the code:

| Layer | File |
|---|---|
| Params model | `common/params/models.py` → `TextGenerationParams` |
| Task class | `models/llm/tasks/summarize.py` → `SummarizeTask` |
| Operator | `operators/generation/summarization.py` → `SummarizationOperator` |
| Graph wiring | `graph/ingestor_runtime.py` → `build_graph()` |
| Fluent API | `ingestor/graph_ingestor.py` → `GraphIngestor` |
| Plan resolution | `ingest/plan.py` → `resolve_ingest_plan()` |
| CLI options | `cli/ingest/options.py` |
| CLI command | `cli/ingest/graph_commands.py` → `_graph_ingest_command` |
| Tests | `tests/test_generation_tasks.py` |

### Common Practices for Writing Unit Tests

Writing unit tests is essential for maintaining code quality and ensuring that changes do not introduce new bugs. In
this project, we use `pytest` for running tests and adopt blackbox testing principles. Below are some common practices
for writing unit tests, which are located in the `[repo_root]/tests` directory.

#### General Guidelines

1. **Test Structure**: Each test module should test a specific module or functionality within the codebase. The test
   module should be named `test_<module_name>.py`, and reside on a mirrored physical path to its corresponding test
   target to be easily discoverable by `pytest`.

   1. Example: `nemo_retriever/src/nemo_retriever/some_path/another_path/my_module.py` should have a corresponding test file:
      `tests/some_path/another_path/test_my_module.py`.

2. **Test Functions**: Each test function should focus on a single aspect of the functionality. Use descriptive names
   that clearly indicate what is being tested. For example, `test_function_returns_correct_value`
   or `test_function_handles_invalid_input`.

3. **Setup and Teardown**: Use `pytest` fixtures to manage setup and teardown operations for your tests. Fixtures help
   in creating a consistent and reusable setup environment.

4. **Assertions**: Use assertions to validate the behavior of the code. Ensure that the tests cover both expected
   outcomes and edge cases.

#### Mocking External Services

When writing tests that depend on external services (e.g., databases, APIs), it is important to mock these dependencies
to ensure that tests are reliable, fast, and do not depend on external factors.

1. **Mocking Libraries**: Use libraries like `unittest.mock` to create mocks for external services. The `pytest-mock`
   plugin can also be used to integrate mocking capabilities directly with `pytest`.

2. **Mock Objects**: Create mock objects to simulate the behavior of external services. Use these mocks to test how your
   code interacts with these services without making actual network calls or database transactions.

3. **Patching**: Use `patch` to replace real objects in your code with mocks. This can be done at the function, method,
   or object level. Ensure that patches are applied in the correct scope to avoid side effects.

#### Example Test Structure

Here is an example of how to structure a test module in the `[repo_root]/tests` directory:

```python
import pytest
from unittest.mock import patch, Mock

# Assuming the module to test is located at [repo_root]/module.py
from module import function_to_test


@pytest.fixture
def mock_external_service():
    with patch('module.ExternalService') as mock_service:
        yield mock_service


def test_function_returns_correct_value(mock_external_service):
    # Arrange
    mock_external_service.return_value.some_method.return_value = 'expected_value'

    # Act
    result = function_to_test()

    # Assert
    assert result == 'expected_value'


def test_function_handles_invalid_input(mock_external_service):
    # Arrange
    mock_external_service.return_value.some_method.side_effect = ValueError("Invalid input")

    # Act and Assert
    with pytest.raises(ValueError, match="Invalid input"):
        function_to_test(invalid_input)
```

## Submodules, Third Party Libraries, and Models

### Submodules

1. Submodules are used to manage third-party libraries and dependencies.
2. Submodules should be created in the `third_party` directory.
3. Ensure that the submodule is updated to the latest commit before making changes.

### Models

1. **Model Integration**: NeMo Retriever is designed to be scalable and flexible, so running models directly in the pipeline
   is discouraged.
2. **Model Export**: Models should be exported to a format compatible with Triton Inference Server or TensorRT.
   - Model acquisition and conversion should be documented in `triton_models/README.md`, including the model name,
     version, pbtxt file, Triton model files, etc., along with an example of how to query the model in Triton.
   - Models should be externally hosted and downloaded during the pipeline execution, or added via LFS.
   - Any additional code, configuration files, or scripts required to run the model should be included in
     the `triton_models/[MODEL_NAME]` directory.
3. **Self-Contained Dependencies**: No assumptions should be made regarding other models or libraries being available in
   the pipeline. All dependencies should be self-contained.
4. **Base Triton Container**: Directions for the creation of the base Triton container are listed in
   the `triton_models/README.md` file. If a new model requires additional base dependencies, please update
   the `Dockerfile` in the `triton_models` directory.

## Architectural Guidelines

To ensure the quality and maintainability of the NeMo Retriever codebase, the following architectural guidelines should be
followed:

### 1. Single Responsibility Principle (SRP)

- Ensure that each module, class, or function has only one reason to change.

### 2. Interface Segregation Principle (ISP)

- Avoid forcing clients to depend on interfaces they do not use.

### 3. Dependency Inversion Principle (DIP)

- High-level modules should not depend on low-level modules, both should depend on abstractions.

### 4. Physical Design Structure Mirroring Logical Design Structure

- The physical layout of the codebase should reflect its logical structure.

### 5. Levelization

- Organize code into levels where higher-level components depend on lower-level components but not vice versa.

### 6. Acyclic Dependencies Principle (ADP)

- Ensure the dependency graph of packages/modules has no cycles.

### 7. Package Cohesion Principles

#### Common Closure Principle (CCP)

- Package classes that change together.

#### Common Reuse Principle (CRP)

- Package classes that are used together.

### 8. Encapsulate What Varies

- Identify aspects of the application that vary and separate them from what stays the same.

### 9. Favor Composition Over Inheritance

- Utilize object composition over class inheritance for behavior reuse where possible.

### 10. Clean Separation of Concerns (SoC)

- Divide the application into distinct features with minimal overlap in functionality.

### 11. Principle of Least Knowledge (Law of Demeter)

- Objects should assume as little as possible about the structure or properties of anything else, including their
  subcomponents.

### 12. Document Assumptions and Decisions

- Assumptions made and reasons behind architectural and design decisions should be clearly documented.

### 13. Continuous Integration and Testing

- Integrate code frequently into a shared repository and ensure comprehensive testing is an integral part of the
  development cycle.

Contributors are encouraged to follow these guidelines to ensure contributions are in line with the project's
architectural consistency and maintainability.


## Writing Good and Thorough Documentation

As a contributor to our codebase, writing high-quality documentation is an essential part of ensuring that others can
understand and work with your code effectively. Good documentation helps to reduce confusion, facilitate collaboration,
and streamline the development process. In this guide, we will outline the principles and best practices for writing
thorough and readable documentation that adheres to the Chicago Manual of Style.

### Chicago Manual of Style

Our documentation follows the Chicago Manual of Style, a widely accepted standard for writing and formatting. This style
guide provides a consistent approach to writing, grammar, and punctuation, making it easier for readers to understand
and navigate our documentation.

### Key Principles

When writing documentation, keep the following principles in mind:

1. **Clarity**: Use clear and concise language to convey your message. Avoid ambiguity and jargon that may confuse readers.
2. **Accuracy**: Ensure that your documentation is accurate and up-to-date. Verify facts, details, and code snippets
    before publishing.
3. **Completeness**: Provide all necessary information to understand the code, including context, syntax, and examples.
4. **Consistency**: Use a consistent tone, voice, and style throughout the documentation.
5. **Accessibility**: Make your documentation easy to read and understand by using headings, bullet points, and short paragraphs.

### Documentation Structure

A well-structured documentation page should include the following elements:

1. **Header**: A brief title that summarizes the content of the page.
2. **Introduction**: A short overview of the topic, including its purpose and relevance.
3. **Syntax and Parameters**: A detailed explanation of the code syntax, including parameters, data types, and return values.
4. **Examples**: Concrete examples that illustrate how to use the code, including input and output.
5. **Tips and Variations**: Additional information, such as best practices, common pitfalls, and alternative approaches.
6. **Related Resources**: Links to relevant documentation, tutorials, and external resources.

### Best Practices

To ensure high-quality documentation, follow these best practices:

1. **Use headings and subheadings**: Organize your content with clear headings and subheadings to facilitate scanning and navigation.
2. **Use bullet points and lists**: Break up complex information into easy-to-read lists and bullet points.
3. **Provide context**: Give readers a clear understanding of the code's purpose, history, and relationships to other components.
4. **Review and edit**: Carefully review and edit your documentation to ensure accuracy, completeness, and consistency.

### Resources

For more information on the Chicago Manual of Style, refer to their
[online published version](https://www.chicagomanualofstyle.org/home.html?_ga=2.188145128.1312333204.1728079521-706076405.1727890116).

By following these guidelines and principles, you will be able to create high-quality documentation that helps others
understand and work with your code effectively. Remember to always prioritize clarity, accuracy, and completeness, and
to use the Chicago Style Guide as your reference for writing and formatting.


## Licensing

NeMo Retriever is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) -- ensure that any contributions are compatible.

The following should be included in the header of any new files:

```text
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

## Attribution

Portions adopted from

- [https://github.com/nv-morpheus/Morpheus/blob/branch-24.06/CONTRIBUTING.md](https://github.com/nv-morpheus/Morpheus/blob/branch-24.06/CONTRIBUTING.md)
- [https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)
- [https://github.com/dask/dask/blob/master/docs/source/develop.rst](https://github.com/dask/dask/blob/master/docs/source/develop.rst)
