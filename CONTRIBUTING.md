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
   - [Adding a New Text-Generation Task](#adding-a-new-text-generation-task)
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

#### TODO(Devin): Add details about adding a new stage or module once we have router node functionality in place.

### Adding a New Text-Generation Task

Text-generation tasks are stateless strategy classes that turn logical inputs
(for example, a question plus retrieved context) into exactly one completion
call and parse the provider's response. They live in
`nemo_retriever/src/nemo_retriever/models/llm/tasks/`.

Existing tasks are good references:

- `generic.py` - `GenericPromptTask`, a validated prompt-template task
- `rag_answer.py` - `RagAnswerTask`, question and context answering
- `summarize.py` - `SummarizeTask`, document summarization

#### 1. Subclass TextGenerationTask

Create `tasks/<your_task>.py` and subclass the ABC in `tasks/base.py`:

```python
from nemo_retriever.models.llm.tasks.base import TextGenerationTask
from nemo_retriever.models.llm.types import GenerationRequest


class MyTask(TextGenerationTask):
    """One-line description of what the task produces."""

    # Logical inputs callers must provide (validated by the lifecycle).
    required_inputs = ("document", "focus")

    # Override sampling defaults only when your task needs different ones.
    _default_sampling = {"temperature": 0.2, "top_p": None, "max_tokens": 1024}

    def build_request(self, **inputs: object) -> GenerationRequest:
        """Build one provider-neutral request from the logical inputs."""
        ...
```

`build_request` is the only method you must implement. Two optional hooks:

- `parse(self, raw_text: str) -> str` - post-process the completion text. The
  default returns `raw_text.strip()`; override it to strip think tags, extract
  structured output, or normalize formatting.
- `_preflight_error(self, **inputs) -> Optional[str]` - return an error code
  when no provider request should be made (for example, empty input). The
  lifecycle raises it as a non-retryable `request`-phase failure.

Callers run tasks through `TextGenerationTask.invoke(client, **inputs)`, which
drives the strict build, complete, and parse lifecycle - your task never calls
the provider directly.

Design rules the lifecycle enforces (see `base.py`):

- **Stateless** - a task instance holds configuration, never per-call state.
- **One completion call** per invocation; no hidden retries inside the task.
- **Declared inputs** - everything `build_request` consumes must be listed in
  `required_inputs`.
- **Strict failures** - errors surface as `GenerationTaskError` with a `phase`
  of `request`, `transport`, `response`, or `parse`, a stable `code`, and a
  sanitized `public_message`.

#### 2. Validate at Construction Time

Follow `generic.py`: validate templates and configuration in `__init__` (or a
module-level helper) and fail fast with a clear `ValueError` or `TypeError`,
not at request time. If your task accepts a prompt template, require a
one-to-one match between template fields and declared inputs.

#### 3. Register the Task

Export the task from `tasks/__init__.py` so it is importable from the package
root:

```python
from nemo_retriever.models.llm.tasks.my_task import MyTask

__all__ = [..., "MyTask"]
```

#### 4. Test It

Add unit tests alongside the existing task tests in
`nemo_retriever/tests/test_generation_tasks.py`, covering:

- request building from valid inputs, including template edge cases
- rejection of missing or extra inputs
- response parsing, including empty-output handling and think-tag stripping
  where applicable
- error mapping: each failure phase raises `GenerationTaskError` with the
  correct `code` and `retryable` flag

Follow the [Workflow](#workflow) steps and sign your commits per the
[DCO](#developer-certificate-of-origin-dco) section before opening a pull
request.

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
