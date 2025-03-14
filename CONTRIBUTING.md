# Contributing to NV-Ingest

External contributions will be welcome soon, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Table of Contents

1. [Filing Issues](#filing-issues)
2. [Cloning the Repository](#cloning-the-repository)
3. [Code Contributions](#code-contributions)
   - [Your First Issue](#your-first-issue)
   - [Seasoned Developers](#seasoned-developers)
   - [Workflow](#workflow)
   - [Common Processing Patterns](#common-processing-patterns)
     - [traceable](#traceable---srcnv_ingestutiltracingtaggingpy)
     - [nv_ingest_node_failure_context_manager](#nv_ingest_node_failure_context_manager---srcnv_ingestutilexception_handlersdecoratorspy)
     - [filter_by_task](#filter_by_task---srcnv_ingestutilflow_controlfilter_by_taskpy)
     - [cm_skip_processing_if_failed](#cm_skip_processing_if_failed---morpheusutilscontrol_message_utilspy)
   - [Adding a New Stage or Module](#adding-a-new-stage-or-module)
   - [Common Practices for Writing Unit Tests](#common-practices-for-writing-unit-tests)
     - [General Guidelines](#general-guidelines)
     - [Mocking External Services](#mocking-external-services)
   - [Submodules, Third Party Libraries, and Models](#submodules-third-party-libraries-and-models)
     - [Submodules](#submodules)
     - [Models](#models)
4. [Architectural Guidelines](#architectural-guidelines)
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
5. [Writing Good and Thorough Documentation](#writing-good-and-thorough-documentation)
6. [Licensing](#licensing)
7. [Attribution](#attribution)

## Filing Issues

1. **Bug Reports, Feature Requests, and Documentation Issues:** Please file
   an [issue](https://github.com/NVIDIA/nv-ingest/issues) with a detailed
   description of
   the problem, feature request, or documentation issue. The NV-Ingest team will review and triage these issues,
   and if appropriate, schedule them for a future release.

## Cloning the repository

```bash
DATASET_ROOT=[path to your dataset root]
MODULE_NAME=[]
MORPHEUS_ROOT=[path to your Morpheus root]
NV_INGEST_ROOT=[path to your NV-Ingest root]
git clone https://github.com/nv-morpheus/Morpheus.git $MORPHEUS_ROOT
git clone https://github.com/NVIDIA/nv-ingest.git $NV_INGEST_ROOT
cd $NV_INGEST_ROOT
```

Ensure all submodules are checked out:

```bash
git submodule update --init --recursive
```

## Code Contributions

### Your First Issue

1. **Finding an Issue:** Start with issues
   labeled [good first issue](https://github.com/NVIDIA/nv-ingest/labels/bug).
2. **Claim an Issue:** Comment on the issue you wish to work on.
3. **Implement Your Solution:** Dive into the code! Update or add unit tests as necessary.
4. **Submit Your Pull Request:
   ** [Create a pull request](https://github.com/NVIDIA/nv-ingest/pulls) once your
   code is ready.
5. **Code Review:** Wait for the review by other developers and make necessary updates.
6. **Merge:** After approval, an NVIDIA developer will approve your pull request.

### Seasoned Developers

For those familiar with the codebase, please check
the [project boards](https://github.com/orgs/NVIDIA/projects/48/views/1) for
issues. Look for unassigned issues and follow the steps starting from **Claim an Issue**.

### Workflow

1. **NV-Ingest Foundation**: Built on top
   of [NVIDIA Morpheus](https://github.com/nv-morpheus/Morpheus/blob/branch-24.10/docs/source/developer_guide/architecture.md).

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

- Dependencies are managed via 'Conda' and 'Pip'.
- Dependencies are stored in .yml files
    1. **Service Dependencies** 'conda/environments/nv_ingest_environment.yml' file.
    2. **Client Dependencies** 'conda/environments/nv_ingest_client_environment.yml' file.

- To update dependencies:
  - Create a clean environment using the relevant .yml file.
  - Update the dependencies using 'Conda' or 'Pip' and validate the changes.
  - Update the .yml file by exporting the updated environment.
    - For example:
      ```bash
      conda env export --name nv_ingest_runtime --no-builds > conda/environment/nv_ingest_environment.yml
      conda env export --name nv_ingest_client --no-builds > conda/environment/nv_ingest_client_environment.yml
      ```

### Common Processing Patterns

In NV-Ingest, decorators are used to enhance the functionality of functions by adding additional processing logic. These
decorators help ensure consistency, traceability, and robust error handling across the pipeline. Below, we introduce
some common decorators used in NV-Ingest, explain their usage, and provide examples.

#### `traceable` -> `src/nv_ingest/util/tracing/tagging.py`

The `traceable` decorator adds entry and exit trace timestamps to a `ControlMessage`'s metadata. This helps in
monitoring and debugging by recording the time taken for function execution.

**Usage:**

- To track function execution time with default trace names:
  ```python
  @traceable()
  def process_message(message):
      pass
  ```
- To use a custom trace name:
  ```python
  @traceable(trace_name="CustomTraceName")
  def process_message(message):
      pass
  ```

#### `nv_ingest_node_failure_context_manager` -> `src/nv_ingest/util/exception_handlers/decorators.py`

This decorator wraps a function with failure handling logic to manage potential failures involving `ControlMessages`. It
ensures that failures are managed consistently, optionally raising exceptions or annotating the `ControlMessage`.

**Usage:**

- To handle failures with default settings:
  ```python
  @nv_ingest_node_failure_context_manager(annotation_id="example_task")
  def process_message(message):
      pass
  ```
- To handle failures and allow empty payloads:
  ```python
  @nv_ingest_node_failure_context_manager(annotation_id="example_task", payload_can_be_empty=True)
  def process_message(message):
      pass
  ```

#### `filter_by_task` -> `src/nv_ingest/util/flow_control/filter_by_task.py`

The `filter_by_task` decorator checks if the `ControlMessage` contains any of the specified tasks. Each task can be a
string of the task name or a tuple of the task name and task properties. If the message does not contain any listed task
and/or task properties, the message is returned directly without calling the wrapped function, unless a forwarding
function is provided.

**Usage:**

- To filter messages based on tasks:
  ```python
  @filter_by_task(["task1", "task2"])
  def process_message(message):
      pass
  ```
- To filter messages based on tasks with specific properties:
  ```python
  @filter_by_task([("task", {"prop": "value"})])
  def process_message(message):
      pass
  ```
- To forward messages to another function. This is necessary when the decorated function does not return the message
  directly, but instead forwards it to another function. In this case, the forwarding function should be provided as an
  argument to the decorator.
  ```python
  @filter_by_task(["task1", "task2"], forward_func=other_function)
  def process_message(message):
      pass
  ```

#### `cm_skip_processing_if_failed` -> `morpheus/utils/control_message_utils.py`

The `cm_skip_processing_if_failed` decorator skips the processing of a `ControlMessage` if it has already failed. This
ensures that no further processing is attempted on a failed message, maintaining the integrity of the pipeline.

**Usage:**

- To skip processing if the message has failed:
  ```python
  @cm_skip_processing_if_failed
  def process_message(message):
      pass
  ```

### Adding a New Stage or Module

#### TODO(Devin): Add details about adding a new stage or module once we have router node functionality in place.

### Common Practices for Writing Unit Tests

Writing unit tests is essential for maintaining code quality and ensuring that changes do not introduce new bugs. In
this project, we use `pytest` for running tests and adopt blackbox testing principles. Below are some common practices
for writing unit tests, which are located in the `[repo_root]/tests` directory.

#### General Guidelines

1. **Test Structure**: Each test module should test a specific module or functionality within the codebase. The test
   module should be named `test_<module_name>.py`, and reside on a mirrored physical path to its corresponding test
   target to be easily discoverable by `pytest`.

   1. Example: `nv_ingest/some_path/another_path/my_module.py` should have a corresponding test file:
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

1. **Model Integration**: NV-Ingest is designed to be scalable and flexible, so running models directly in the pipeline
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

To ensure the quality and maintainability of the NV-Ingest codebase, the following architectural guidelines should be
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

NV-Ingest is licensed under the NVIDIA Proprietary Software License -- ensure that any contributions are compatible.

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
