# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from nv_ingest.util.flow_control.filter_by_task import _is_subset, filter_by_task


# =============================================================================
# Helper Classes for Testing filter_by_task
# =============================================================================


class DummyTask:
    """A simple dummy task object with a type and properties."""

    def __init__(self, type, properties):
        self.type = type
        self.properties = properties


class DummyMessage:
    """
    A dummy IngestControlMessage that provides a get_tasks() method.
    The get_tasks() method returns a list of DummyTask objects.
    """

    def __init__(self, tasks):
        self._tasks = tasks

    def get_tasks(self):
        return self._tasks


# A dummy Pydantic model for testing tasks with BaseModel properties.
class DummyModel(BaseModel):
    a: int
    b: str
    c: float = 0.0


# -----------------------------------------------------------------------------
# Tests for _is_subset (for reference)
# -----------------------------------------------------------------------------


def test_is_subset_wildcard():
    """Test that the special wildcard '*' matches any value."""
    assert _is_subset("anything", "*")
    assert _is_subset(123, "*")
    assert _is_subset({"a": 1}, "*")


def test_is_subset_dict_true():
    """Test that a dictionary is a subset when all required keys/values are present."""
    superset = {"a": 1, "b": 2, "c": 3}
    subset = {"a": 1, "b": 2}
    assert _is_subset(superset, subset)


# -----------------------------------------------------------------------------
# Tests for filter_by_task Decorator – Complex Task and Properties Requirements
# -----------------------------------------------------------------------------


def test_filter_decorator_complex_nested_match():
    """
    Test that a complex nested property requirement with regex and list matching is satisfied.

    Required task:
      ("taskComplex", {"nested": {"key": "regex:^start"}, "list": [1, 2]})

    Dummy task:
      - type: "taskComplex"
      - properties: a dict with a nested dict whose 'key' starts with "start"
        and a list that includes the elements 1 and 2.
    """
    required_tasks = [("taskComplex", {"nested": {"key": "regex:^start"}, "list": [1, 2]})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    properties = {
        "nested": {"key": "startingValue", "other": "ignored"},
        "list": [0, 1, 2, 3],
        "extra": "data",
    }
    tasks = [DummyTask("taskComplex", properties)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    assert result == "processed"


def test_filter_decorator_complex_nested_no_match():
    """
    Test that a complex nested property requirement fails when nested values do not match.

    Required task:
      ("taskComplex", {"nested": {"key": "regex:^start"}, "list": [1, 2, 3]})

    Dummy task:
      - type: "taskComplex"
      - properties: a dict with a nested dict whose 'key' does not start with "start"
        and a list missing one required element.
    """
    required_tasks = [("taskComplex", {"nested": {"key": "regex:^start"}, "list": [1, 2, 3]})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    properties = {
        "nested": {"key": "notMatching", "other": "ignored"},
        "list": [0, 1, 2],  # Missing the element '3'
    }
    tasks = [DummyTask("taskComplex", properties)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    # Expecting no match so the original message is returned.
    assert result == msg


def test_filter_decorator_multiple_required_conditions_match():
    """
    Test that multiple property conditions within a single required task tuple are all satisfied.

    Required task:
      ("taskMulti", {"a": 1}, {"b": 2})

    Dummy task:
      - type: "taskMulti"
      - properties: a dict that includes 'a': 1 and 'b': 2 (along with extra data).
    """
    required_tasks = [("taskMulti", {"a": 1}, {"b": 2})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    properties = {"a": 1, "b": 2, "c": 3}
    tasks = [DummyTask("taskMulti", properties)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    assert result == "processed"


def test_filter_decorator_multiple_required_conditions_no_match():
    """
    Test that if not all property conditions within a required task tuple are satisfied,
    the function is not executed.

    Required task:
      ("taskMulti", {"a": 1}, {"b": 2})

    Dummy task:
      - type: "taskMulti"
      - properties: a dict that satisfies 'a': 1 but has 'b': 3 (which does not match the required value).
    """
    required_tasks = [("taskMulti", {"a": 1}, {"b": 2})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    properties = {"a": 1, "b": 3}  # 'b' does not equal 2.
    tasks = [DummyTask("taskMulti", properties)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    # Since the conditions are not met, the original message is returned.
    assert result == msg


def test_filter_decorator_pydantic_properties_match():
    """
    Test that the decorator correctly handles Pydantic model instances as task properties.

    Required task:
      ("taskPydantic", {"a": 10, "b": "hello"})

    Dummy task:
      - type: "taskPydantic"
      - properties: a DummyModel instance with values that satisfy the requirement.
    """
    required_tasks = [("taskPydantic", {"a": 10, "b": "hello"})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    model_instance = DummyModel(a=10, b="hello", c=3.14)
    tasks = [DummyTask("taskPydantic", model_instance)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    assert result == "processed"


def test_filter_decorator_pydantic_properties_no_match():
    """
    Test that a Pydantic model instance does not satisfy the required property criteria if its values differ.

    Required task:
      ("taskPydantic", {"a": 10, "b": "world"})

    Dummy task:
      - type: "taskPydantic"
      - properties: a DummyModel instance that does not match the required properties.
    """
    required_tasks = [("taskPydantic", {"a": 10, "b": "world"})]

    @filter_by_task(required_tasks)
    def dummy_func(message):
        return "processed"

    model_instance = DummyModel(a=10, b="hello", c=3.14)
    tasks = [DummyTask("taskPydantic", model_instance)]
    msg = DummyMessage(tasks)
    result = dummy_func(msg)
    # Expect no match so the original message is returned.
    assert result == msg
