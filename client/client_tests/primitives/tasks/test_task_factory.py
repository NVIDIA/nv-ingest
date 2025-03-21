# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import Task
from nv_ingest_client.primitives.tasks.task_factory import TaskType
from nv_ingest_client.primitives.tasks.task_factory import task_factory


# Mock implementations for the purpose of testing
class MockTask(Task):
    def __init__(self, **kwargs):
        super().__init__()


def test_task_factory_with_valid_string():
    valid_task_type_str = "EXTRACT"
    # Assuming ExtractTask is properly implemented and accepts kwargs
    task = task_factory(valid_task_type_str, document_type="pdf")
    assert isinstance(
        task, ExtractTask
    ), "task_factory did not correctly convert string to TaskType and instantiate the task"


def test_task_factory_with_invalid_type():
    invalid_task_type = 123  # Not a string or TaskType
    with pytest.raises(ValueError) as exc_info:
        task_factory(invalid_task_type)
    assert "task_type must be a TaskType enum member or a valid task type string" in str(exc_info.value)


# Test successful task creation for implemented tasks
def test_task_factory_success():
    task = task_factory(TaskType.EXTRACT, document_type="test", extract_method="pdfium")
    assert isinstance(task, ExtractTask)


# Test handling of unimplemented tasks
def test_task_factory_unimplemented():
    with pytest.raises(NotImplementedError):
        task_factory(TaskType.TRANSFORM)


# Test invalid task type handling
def test_task_factory_invalid_type():
    with pytest.raises(ValueError) as exc_info:
        task_factory("nonexistent")
    assert "Invalid task type" in str(exc_info.value)


# Test passing unexpected keyword arguments
def test_task_factory_unexpected_kwargs():
    with pytest.raises(ValueError) as exc_info:
        task_factory(TaskType.SPLIT, unexpected_arg="test")
    assert "Unexpected keyword argument" in str(exc_info.value)


# Test proper handling of expected keyword arguments
def test_task_factory_expected_kwargs():
    task = task_factory(TaskType.SPLIT)
    assert isinstance(task, SplitTask)
