# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.task_base import Task
from nv_ingest_client.primitives.tasks.task_base import TaskType
from nv_ingest_client.primitives.tasks.task_base import is_valid_task_type

# TaskType Enum Tests


def test_task_type_enum_valid_values():
    for task_type in TaskType:
        assert isinstance(task_type, TaskType), f"{task_type} should be an instance of TaskType Enum"


def test_task_type_enum_invalid_value():
    invalid_task_type = "INVALID"
    assert not is_valid_task_type(
        invalid_task_type
    ), f"'{invalid_task_type}' should not be recognized as a valid TaskType"


# is_valid_task_type Function Tests


@pytest.mark.parametrize("valid_task_type", [task_type.name for task_type in TaskType])
def test_is_valid_task_type_with_valid_types(valid_task_type):
    assert is_valid_task_type(valid_task_type), f"{valid_task_type} should be recognized as a valid TaskType"


def test_is_valid_task_type_with_invalid_type():
    invalid_task_type = "NON_EXISTENT_TASK"
    assert not is_valid_task_type(
        invalid_task_type
    ), f"{invalid_task_type} should not be recognized as a valid TaskType"


# Task Class Tests


def test_task_str_method():
    task = Task()
    expected_str = f"{task.__class__.__name__}\n"
    assert str(task) == expected_str, "The __str__ method of Task does not return the expected string format"


def test_task_to_dict_method():
    task = Task()
    expected_dict = {}
    assert task.to_dict() == expected_dict, (
        "The to_dict method of Task should return an empty dictionary for a " "generic task"
    )
