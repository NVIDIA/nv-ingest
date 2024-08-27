# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import MagicMock

import pytest

# Assuming the decorator and all necessary components are defined in decorator_module.py
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager


# Mock the ControlMessage class and related functions
class MockControlMessage:
    def __init__(self, dataframe=None, failed=False):
        self.metadata = {"cm_failed": failed}
        self.dataframe = dataframe

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def has_metadata(self, key):
        return key in self.metadata and self.metadata[key]

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def payload(self):
        return self

    def mutable_dataframe(self):
        return self.dataframe


@pytest.fixture
def control_message_without_dataframe():
    return MockControlMessage()


@pytest.fixture
def control_message_with_dataframe():
    return MockControlMessage(dataframe="")


@pytest.fixture
def control_message_failed():
    return MockControlMessage(failed=True)


@pytest.fixture
def function_mock():
    return MagicMock(return_value=MockControlMessage(dataframe="data"))


def test_decorator_success_scenario(control_message_with_dataframe, function_mock):
    decorator = nv_ingest_node_failure_context_manager("test_id", False, False)
    wrapped_function = decorator(function_mock)
    result = wrapped_function(control_message_with_dataframe)
    function_mock.assert_called_once()
    assert isinstance(result, MockControlMessage), "The result should be a ControlMessage instance."


def test_decorator_failure_scenario_raise(control_message_with_dataframe, function_mock):
    function_mock.side_effect = Exception("Forced error")
    decorator = nv_ingest_node_failure_context_manager("test_id", False, True)
    wrapped_function = decorator(function_mock)
    with pytest.raises(Exception, match="Forced error"):
        wrapped_function(control_message_with_dataframe)


def test_decorator_failure_scenario_no_raise(control_message_with_dataframe, function_mock):
    function_mock.side_effect = Exception("Forced error")
    decorator = nv_ingest_node_failure_context_manager("test_id", False, False)
    wrapped_function = decorator(function_mock)
    result = wrapped_function(control_message_with_dataframe)
    assert isinstance(result, MockControlMessage), "The result should be a ControlMessage instance."
    assert result.get_metadata("cm_failed"), "ControlMessage should have failure metadata set."


def test_payload_not_empty_required(control_message_with_dataframe):
    decorator = nv_ingest_node_failure_context_manager("test_id", payload_can_be_empty=False)
    wrapped_function = decorator(lambda cm: cm)
    control_message_with_dataframe.dataframe = "data"

    # Setting up a scenario where payload should not be considered empty
    result = wrapped_function(control_message_with_dataframe)
    assert isinstance(
        result, MockControlMessage
    ), "The result should be a ControlMessage even if payload is required and present."


def test_payload_can_be_empty(control_message_without_dataframe):
    decorator = nv_ingest_node_failure_context_manager("test_id", payload_can_be_empty=True)
    wrapped_function = decorator(lambda cm: cm)
    control_message_without_dataframe.dataframe = None  # Ensuring dataframe simulates an empty payload
    result = wrapped_function(control_message_without_dataframe)
    assert isinstance(result, MockControlMessage), "The result should be a ControlMessage even if payload is empty."


def test_decorator_with_already_failed_message(control_message_failed, function_mock):
    decorator = nv_ingest_node_failure_context_manager("test_id", False, False)
    wrapped_function = decorator(function_mock)
    result = wrapped_function(control_message_failed)
    function_mock.assert_not_called()
    assert isinstance(
        result, MockControlMessage
    ), "The result should be a ControlMessage even if it was already marked as failed."
    assert control_message_failed.get_metadata("cm_failed"), "Failed ControlMessage should retain its failed state."


def test_payload_not_empty_when_dataframe_present(control_message_with_dataframe):
    """Test no ValueError is raised regardless of payload_can_be_empty setting if dataframe is present."""
    decorator = nv_ingest_node_failure_context_manager(annotation_id="test_annotation", payload_can_be_empty=False)
    wrapped_function = decorator(lambda cm: cm)

    try:
        result = wrapped_function(control_message_with_dataframe)
        assert isinstance(result, MockControlMessage), "Function should return a ControlMessage instance."
    except ValueError:
        pytest.fail("ValueError raised unexpectedly when dataframe is present.")


def test_payload_allowed_empty_when_dataframe_absent(control_message_without_dataframe):
    """Test that payload_can_be_empty=True allows processing when mutable_dataframe() is None."""
    decorator = nv_ingest_node_failure_context_manager(annotation_id="test_annotation", payload_can_be_empty=True)
    wrapped_function = decorator(lambda cm: cm)

    result = wrapped_function(control_message_without_dataframe)
    assert isinstance(result, MockControlMessage), "Function should return a ControlMessage instance."
