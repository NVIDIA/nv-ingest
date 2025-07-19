# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# noqa
# flake8: noqa

import os
import sys
import pytest
from nv_ingest_client.primitives.tasks.udf import UDFTask
from nv_ingest.pipeline.pipeline_schema import PipelinePhase

# Add path to tests directory for utilities
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tests"))
from utilities_for_test import get_project_root

# Get the project root for proper path resolution
PROJECT_ROOT = get_project_root(__file__)

# Initialization and Property Setting


def test_udf_task_initialization():
    """Test UDFTask initialization with basic parameters."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        phase=PipelinePhase.EXTRACTION,
    )
    assert task._udf_function == udf_function
    assert task._phase == PipelinePhase.EXTRACTION


def test_udf_task_default_params():
    """Test UDFTask with default parameters."""
    task = UDFTask()
    assert task._udf_function is None
    assert task._phase == PipelinePhase.RESPONSE


# String Representation Tests


def test_udf_task_str_representation():
    """Test UDFTask string representation."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        phase=PipelinePhase.EXTRACTION,
    )
    str_repr = str(task)
    assert "User-Defined Function (UDF) Task:" in str_repr
    assert "udf_function:" in str_repr
    assert "phase: EXTRACTION (1)" in str_repr


def test_udf_task_str_representation_long_function():
    """Test UDFTask string representation with long function."""
    udf_function = "def my_very_long_udf_function_name_that_exceeds_one_hundred_characters_and_should_be_truncated(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        phase=PipelinePhase.MUTATION,
    )
    str_repr = str(task)
    assert "User-Defined Function (UDF) Task:" in str_repr
    assert "..." in str_repr  # Should be truncated
    assert "phase: MUTATION (3)" in str_repr


def test_udf_task_str_representation_none_function():
    """Test UDFTask string representation with None function."""
    task = UDFTask(
        udf_function=None,
        phase=PipelinePhase.RESPONSE,
    )
    str_repr = str(task)
    assert "User-Defined Function (UDF) Task:" in str_repr
    assert "udf_function: None" in str_repr
    assert "phase: RESPONSE (5)" in str_repr


# Phase Validation Tests


@pytest.mark.parametrize(
    "phase_input, expected_phase",
    [
        (PipelinePhase.EXTRACTION, PipelinePhase.EXTRACTION),
        (1, PipelinePhase.EXTRACTION),
        (2, PipelinePhase.POST_PROCESSING),
        (3, PipelinePhase.MUTATION),
        (4, PipelinePhase.TRANSFORM),
        (5, PipelinePhase.RESPONSE),
        ("EXTRACTION", PipelinePhase.EXTRACTION),
        ("extraction", PipelinePhase.EXTRACTION),
        ("POST_PROCESSING", PipelinePhase.POST_PROCESSING),
        ("MUTATION", PipelinePhase.MUTATION),
        ("TRANSFORM", PipelinePhase.TRANSFORM),
        ("RESPONSE", PipelinePhase.RESPONSE),
        # Test aliases
        ("EXTRACT", PipelinePhase.EXTRACTION),
        ("POSTPROCESS", PipelinePhase.POST_PROCESSING),
        ("POST_PROCESS", PipelinePhase.POST_PROCESSING),
        ("POSTPROCESSING", PipelinePhase.POST_PROCESSING),
        ("MUTATE", PipelinePhase.MUTATION),
    ],
)
def test_udf_task_phase_validation(phase_input, expected_phase):
    """Test UDFTask phase validation and conversion."""
    task = UDFTask(
        udf_function="def test(control_message): return control_message",
        phase=phase_input,
    )
    assert task._phase == expected_phase


def test_udf_task_invalid_phase_number():
    """Test UDFTask with invalid phase number."""
    with pytest.raises(ValueError, match="Invalid phase number"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=99,
        )


def test_udf_task_invalid_phase_string():
    """Test UDFTask with invalid phase string."""
    with pytest.raises(ValueError, match="Invalid phase name"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase="INVALID_PHASE",
        )


def test_udf_task_invalid_phase_type():
    """Test UDFTask with invalid phase type."""
    with pytest.raises(ValueError, match="Phase must be a PipelinePhase enum, integer, or string"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=3.14,  # Float is not supported
        )


def test_udf_task_api_schema_phase_bounds():
    """Test UDFTask phase bounds validation according to API schema."""
    # Test phase 0 (PRE_PROCESSING) - should fail due to API schema constraint ge=1
    with pytest.raises(Exception):  # Will be a validation error from API schema
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=0,
        )

    # Test phase 6 - should fail due to API schema constraint le=5
    with pytest.raises(Exception):  # Will be a validation error from API schema
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=6,
        )

    # Test PRE_PROCESSING enum directly - should fail due to API schema constraint
    with pytest.raises(Exception):  # Will be a validation error from API schema
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=PipelinePhase.PRE_PROCESSING,
        )


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "udf_function, phase, expected_phase_value",
    [
        ("def test(control_message): return control_message", PipelinePhase.EXTRACTION, 1),
        ("def test(control_message): return control_message", PipelinePhase.POST_PROCESSING, 2),
        ("def test(control_message): return control_message", PipelinePhase.MUTATION, 3),
        ("def test(control_message): return control_message", PipelinePhase.TRANSFORM, 4),
        ("def test(control_message): return control_message", PipelinePhase.RESPONSE, 5),
        ("def test(control_message): return control_message", 1, 1),
        ("def test(control_message): return control_message", "EXTRACTION", 1),
        ("def test(control_message): return control_message", "extract", 1),
    ],
)
def test_udf_task_to_dict(udf_function, phase, expected_phase_value):
    """Test UDFTask to_dict method."""
    task = UDFTask(
        udf_function=udf_function,
        phase=phase,
    )

    expected_dict = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "phase": expected_phase_value,
        },
    }

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


def test_udf_task_to_dict_none_function():
    """Test UDFTask to_dict method with None function."""
    task = UDFTask(
        udf_function=None,
        phase=PipelinePhase.RESPONSE,
    )

    # When udf_function is None, it should not be included in the serialization
    expected_dict = {
        "type": "udf",
        "task_properties": {
            "phase": 5,
        },
    }

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Function Resolution Tests


def test_udf_task_function_resolution_inline():
    """Test UDFTask function resolution with inline function."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(udf_function=udf_function, phase=PipelinePhase.EXTRACTION)

    # Test that the function can be resolved
    resolved_function = task._resolve_udf_function()
    assert resolved_function == udf_function


def test_udf_task_function_resolution_import_path():
    """Test UDFTask function resolution with import path."""
    # Most standard library functions don't have accessible source code
    # So we'll test the error handling instead
    import_path = "os.path.join"
    task = UDFTask(udf_function=import_path, phase=PipelinePhase.EXTRACTION)

    # Test that it handles the error gracefully when source code is not available
    with pytest.raises(ValueError, match="Could not get source code"):
        task._resolve_udf_function()


def test_udf_task_function_resolution_builtin_error():
    """Test UDFTask function resolution with built-in function (should handle error)."""
    # Built-in functions don't have source code available
    import_path = "builtins.len"
    task = UDFTask(udf_function=import_path, phase=PipelinePhase.EXTRACTION)

    # Test that it handles the error gracefully
    with pytest.raises(ValueError, match="Could not get source code"):
        task._resolve_udf_function()


def test_udf_task_function_resolution_file_path():
    """Test UDFTask function resolution with file path."""
    # This test would require a test file, so we'll test the error case
    file_path = "/nonexistent/path.py:my_function"
    task = UDFTask(udf_function=file_path, phase=PipelinePhase.EXTRACTION)

    # Test that it handles the error gracefully
    with pytest.raises(Exception):  # Should raise an error for nonexistent file
        task._resolve_udf_function()


def test_udf_task_function_resolution_git_root():
    """Test UDFTask function resolution with file path using get_git_root."""
    file_path = os.path.join(PROJECT_ROOT, "data", "random_udf.py:add_random_metadata")
    task = UDFTask(udf_function=file_path, phase=PipelinePhase.EXTRACTION)

    # Test that the function can be resolved
    resolved_function = task._resolve_udf_function()
    assert "def add_random_metadata" in resolved_function
    assert "control_message" in resolved_function


# Schema Consolidation Tests


def test_udf_task_schema_consolidation():
    """Test that UDFTask uses API schema for validation."""
    # Test that valid parameters work
    task = UDFTask(
        udf_function="def test_udf(control_message): return control_message",
        phase=PipelinePhase.EXTRACTION,
    )
    assert task._udf_function == "def test_udf(control_message): return control_message"
    assert task._phase == PipelinePhase.EXTRACTION


def test_udf_task_empty_function_handling():
    """Test UDFTask handling of empty function string."""
    # API schema requires non-empty string, but client allows None
    task = UDFTask(udf_function=None, phase=PipelinePhase.RESPONSE)
    assert task._udf_function is None
    assert task._phase == PipelinePhase.RESPONSE


# Edge Cases and Error Handling


def test_udf_task_edge_cases():
    """Test UDFTask edge cases and boundary conditions."""
    # Test with minimal valid function
    task = UDFTask(
        udf_function="def f(x): return x",
        phase=1,
    )
    assert task._udf_function == "def f(x): return x"
    assert task._phase == PipelinePhase.EXTRACTION

    # Test serialization
    result_dict = task.to_dict()
    assert result_dict["task_properties"]["udf_function"] == "def f(x): return x"
    assert result_dict["task_properties"]["phase"] == 1


def test_udf_task_whitespace_handling():
    """Test UDFTask handling of whitespace in phase strings."""
    task = UDFTask(
        udf_function="def test(control_message): return control_message",
        phase="  EXTRACTION  ",  # With whitespace
    )
    assert task._phase == PipelinePhase.EXTRACTION
