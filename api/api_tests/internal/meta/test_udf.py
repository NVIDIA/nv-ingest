# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive black box tests for UDF stage functionality.
"""

import pytest

from nv_ingest_api.internal.meta.udf import udf_stage_callable_fn
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.schemas.meta.udf import UDFStageSchema


def test_udf_stage_callable_fn_single_udf():
    """Test UDF stage callable with a single UDF task."""
    # Create control message with UDF task
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def test_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('udf_executed', True)
    return control_message
""",
            "udf_function_name": "test_udf",
        },
    )
    control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result
    assert isinstance(result, IngestControlMessage)
    assert not result.has_task("udf")  # UDF task should be removed

    # Verify UDF was executed
    assert result.get_metadata("udf_executed") is True


def test_udf_stage_callable_fn_multiple_udfs():
    """Test UDF stage callable with multiple UDF tasks."""
    # Create control message with two UDF tasks
    control_message = IngestControlMessage()

    # First UDF task
    udf_task1 = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def first_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('first_udf_executed', True)
    return control_message
""",
            "udf_function_name": "first_udf",
        },
    )
    control_message.add_task(udf_task1)

    # Second UDF task
    udf_task2 = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def second_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('second_udf_executed', True)
    return control_message
""",
            "udf_function_name": "second_udf",
        },
    )
    control_message.add_task(udf_task2)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result
    assert isinstance(result, IngestControlMessage)
    assert not result.has_task("udf")  # All UDF tasks should be removed

    # Verify both UDFs were executed
    assert result.get_metadata("first_udf_executed") is True
    assert result.get_metadata("second_udf_executed") is True


def test_udf_stage_callable_fn_no_udf_ignore_true():
    """Test UDF stage callable with no UDF tasks and ignore_empty_udf=True."""
    # Create control message with no UDF tasks
    control_message = IngestControlMessage()

    # Create stage config with ignore_empty_udf=True
    stage_config = UDFStageSchema(ignore_empty_udf=True)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result - should return unchanged control message
    assert isinstance(result, IngestControlMessage)
    assert result is control_message


def test_udf_stage_callable_fn_no_udf_ignore_false():
    """Test UDF stage callable with no UDF tasks and ignore_empty_udf=False."""
    # Create control message with no UDF tasks
    control_message = IngestControlMessage()

    # Create stage config with ignore_empty_udf=False
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise ValueError
    with pytest.raises(ValueError, match="No UDF tasks found in control message"):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_empty_udf_function_ignore_true():
    """Test UDF stage callable with empty UDF function and ignore_empty_udf=True."""
    # Create control message with empty UDF task
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(type="udf", id="udf", properties={"udf_function": ""})
    control_message.add_task(udf_task)

    # Create stage config with ignore_empty_udf=True
    stage_config = UDFStageSchema(ignore_empty_udf=True)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result - should return control message with task removed
    assert isinstance(result, IngestControlMessage)
    assert not result.has_task("udf")


def test_udf_stage_callable_fn_empty_udf_function_ignore_false():
    """Test UDF stage callable with empty UDF function and ignore_empty_udf=False."""
    # Create control message with empty UDF task
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(type="udf", id="udf", properties={"udf_function": ""})
    control_message.add_task(udf_task)

    # Create stage config with ignore_empty_udf=False
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise ValueError
    with pytest.raises(ValueError, match="UDF task 1 has empty function string"):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_execution_failure():
    """Test UDF stage callable with UDF function that raises an exception."""
    # Create control message with failing UDF task
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def failing_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    raise RuntimeError("UDF execution failed")
""",
            "udf_function_name": "failing_udf",
        },
    )
    control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise RuntimeError
    with pytest.raises(ValueError, match="UDF task 1 execution failed"):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_invalid_return_type():
    """Test UDF stage callable with UDF function that returns wrong type."""
    # Create control message with UDF task that returns wrong type
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def invalid_return_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    return "invalid_return_type"
""",
            "udf_function_name": "invalid_return_udf",
        },
    )
    control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise ValueError
    with pytest.raises(ValueError, match="UDF task 1 must return an IngestControlMessage"):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_invalid_signature():
    """Test UDF stage callable with invalid function signature."""
    control_message = IngestControlMessage()

    # UDF with invalid signature (wrong parameter count)
    udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def invalid_udf(control_message: IngestControlMessage, extra_param: str) -> IngestControlMessage:
    return control_message
""",
            "udf_function_name": "invalid_udf",
        },
    )
    control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise ValueError
    with pytest.raises(ValueError, match="UDF task 1 has invalid function signature"):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_no_callable_function():
    """Test UDF stage callable with UDF string that has no callable function."""
    # Create control message with UDF task that has no callable
    control_message = IngestControlMessage()
    udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
# This is just a comment, no function defined
x = 1 + 1
""",
            "udf_function_name": "nonexistent_function",
        },
    )
    control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage - should raise ValueError
    with pytest.raises(
        ValueError, match="UDF task 1: Specified UDF function 'nonexistent_function' " "not found or not callable"
    ):
        udf_stage_callable_fn(control_message, stage_config)


def test_udf_stage_callable_fn_mixed_valid_invalid():
    """Test UDF stage callable with mix of valid and invalid UDF tasks."""
    # Create control message with mixed UDF tasks
    control_message = IngestControlMessage()

    # Add valid UDF task
    valid_udf_task = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def valid_udf(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('valid_udf_executed', True)
    return control_message
""",
            "udf_function_name": "valid_udf",
        },
    )
    control_message.add_task(valid_udf_task)

    # Add empty UDF task
    empty_udf_task = ControlMessageTask(type="udf", id="udf", properties={"udf_function": ""})
    control_message.add_task(empty_udf_task)

    # Add another valid UDF task
    valid_udf_task2 = ControlMessageTask(
        type="udf",
        id="udf",
        properties={
            "udf_function": """
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def valid_udf2(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('valid_udf2_executed', True)
    return control_message
""",
            "udf_function_name": "valid_udf2",
        },
    )
    control_message.add_task(valid_udf_task2)

    # Create stage config with ignore_empty_udf=True
    stage_config = UDFStageSchema(ignore_empty_udf=True)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result
    assert isinstance(result, IngestControlMessage)
    assert not result.has_task("udf")  # All UDF tasks should be removed

    # Verify valid UDFs were executed, empty one was skipped
    assert result.get_metadata("valid_udf_executed") is True
    assert result.get_metadata("valid_udf2_executed") is True


def test_udf_stage_callable_fn_scalability():
    """Test UDF stage callable with many UDF tasks to verify scalability."""
    # Create control message with 5 UDF tasks
    control_message = IngestControlMessage()

    for i in range(5):
        udf_task = ControlMessageTask(
            type="udf",
            id="udf",
            properties={
                "udf_function": f"""
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

def udf_{i}(control_message: IngestControlMessage) -> IngestControlMessage:
    control_message.set_metadata('udf_{i}_executed', True)
    return control_message
""",
                "udf_function_name": f"udf_{i}",
            },
        )
        control_message.add_task(udf_task)

    # Create stage config
    stage_config = UDFStageSchema(ignore_empty_udf=False)

    # Execute UDF stage
    result = udf_stage_callable_fn(control_message, stage_config)

    # Verify result
    assert isinstance(result, IngestControlMessage)
    assert len(list(result.get_tasks())) == 0  # All UDF tasks should be removed

    # Verify all UDFs were executed
    for i in range(5):
        assert result.get_metadata(f"udf_{i}_executed") is True
