# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# noqa
# flake8: noqa

import pytest
from unittest.mock import patch, mock_open

from nv_ingest_api.internal.enums.common import PipelinePhase
from nv_ingest_client.primitives.tasks.udf import UDFTask

# Initialization and Property Setting


def test_udf_task_basic_construction_with_phase():
    """Test basic UDFTask construction with phase parameter."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="my_udf",
        phase=PipelinePhase.TRANSFORM,
    )

    assert task._udf_function == udf_function
    assert task._udf_function_name == "my_udf"
    assert task._phase == PipelinePhase.TRANSFORM
    assert task._target_stage is None
    assert task._run_before == False
    assert task._run_after == False


def test_udf_task_basic_construction_with_target_stage():
    """Test basic UDFTask construction with target_stage parameters."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="my_udf",
        target_stage="text_embedder",
        run_before=True,
        phase=None,  # Must explicitly set to None when using target_stage
    )

    assert task._udf_function == udf_function
    assert task._udf_function_name == "my_udf"
    assert task._phase is None
    assert task._target_stage == "text_embedder"
    assert task._run_before == True
    assert task._run_after == False


def test_udf_task_construction_with_run_after():
    """Test UDFTask construction with run_after parameter."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="my_udf",
        target_stage="pdf_extractor",
        run_after=True,
        phase=None,
    )

    assert task._target_stage == "pdf_extractor"
    assert task._run_before == False
    assert task._run_after == True


def test_udf_task_construction_with_both_timing_flags():
    """Test UDFTask construction with both run_before and run_after."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        target_stage="text_embedder",
        run_before=True,
        run_after=True,
        phase=None,
    )

    assert task._run_before == True
    assert task._run_after == True


# Validation Error Tests


def test_udf_task_validation_error_both_phase_and_target_stage():
    """Test validation error when both phase and target_stage are specified."""
    with pytest.raises(ValueError, match="Cannot specify both 'phase' and 'target_stage'"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=PipelinePhase.TRANSFORM,
            target_stage="text_embedder",
            run_before=True,
        )


def test_udf_task_validation_error_neither_phase_nor_target_stage():
    """Test validation error when neither phase nor target_stage are specified."""
    with pytest.raises(ValueError, match="Phase must be a PipelinePhase enum, integer, or string"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=None,
            target_stage=None,
        )


def test_udf_task_validation_error_run_before_without_target_stage():
    """Test validation error when run_before is used without target_stage."""
    with pytest.raises(ValueError, match="target_stage must be specified when using run_before or run_after"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=PipelinePhase.TRANSFORM,
            run_before=True,
        )


def test_udf_task_validation_error_run_after_without_target_stage():
    """Test validation error when run_after is used without target_stage."""
    with pytest.raises(ValueError, match="target_stage must be specified when using run_before or run_after"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=PipelinePhase.TRANSFORM,
            run_after=True,
        )


def test_udf_task_validation_error_target_stage_without_timing():
    """Test validation error when target_stage is specified without timing flags."""
    with pytest.raises(ValueError, match="At least one of run_before or run_after must be True"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            target_stage="text_embedder",
            phase=None,
        )


# Phase Conversion Tests


@pytest.mark.parametrize(
    "phase_input, expected_phase",
    [
        (PipelinePhase.EXTRACTION, PipelinePhase.EXTRACTION),
        (PipelinePhase.POST_PROCESSING, PipelinePhase.POST_PROCESSING),
        (PipelinePhase.MUTATION, PipelinePhase.MUTATION),
        (PipelinePhase.TRANSFORM, PipelinePhase.TRANSFORM),
        (PipelinePhase.RESPONSE, PipelinePhase.RESPONSE),
        (1, PipelinePhase.EXTRACTION),
        (2, PipelinePhase.POST_PROCESSING),
        (3, PipelinePhase.MUTATION),
        (4, PipelinePhase.TRANSFORM),
        (5, PipelinePhase.RESPONSE),
        ("EXTRACTION", PipelinePhase.EXTRACTION),
        ("POST_PROCESSING", PipelinePhase.POST_PROCESSING),
        ("MUTATION", PipelinePhase.MUTATION),
        ("TRANSFORM", PipelinePhase.TRANSFORM),
        ("RESPONSE", PipelinePhase.RESPONSE),
        ("extract", PipelinePhase.EXTRACTION),
        ("EXTRACT", PipelinePhase.EXTRACTION),
        ("POSTPROCESS", PipelinePhase.POST_PROCESSING),
        ("MUTATE", PipelinePhase.MUTATION),
    ],
)
def test_udf_task_phase_conversion(phase_input, expected_phase):
    """Test UDFTask phase conversion for various input formats."""
    udf_function = "def my_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
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
            phase="invalid_phase",
        )


def test_udf_task_invalid_phase_type():
    """Test UDFTask with invalid phase type."""
    with pytest.raises(ValueError, match="Phase must be a PipelinePhase enum, integer, or string"):
        UDFTask(
            udf_function="def test(control_message): return control_message",
            phase=[1, 2, 3],
        )


# Serialization Tests


def test_udf_task_to_dict_with_phase():
    """Test UDFTask to_dict method with phase parameter."""
    udf_function = "def test_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="test_func",
        phase=PipelinePhase.TRANSFORM,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "udf_function_name": "test_func",
            "phase": 4,  # TRANSFORM phase value
            "run_before": False,
            "run_after": False,
        },
    }

    assert result == expected


def test_udf_task_to_dict_with_target_stage():
    """Test UDFTask to_dict method with target_stage parameters."""
    udf_function = "def test_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="test_func",
        target_stage="text_embedder",
        run_before=True,
        phase=None,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "udf_function_name": "test_func",
            "phase": None,
            "target_stage": "text_embedder",
            "run_before": True,
            "run_after": False,
        },
    }

    assert result == expected


def test_udf_task_to_dict_with_run_after():
    """Test UDFTask to_dict method with run_after parameter."""
    udf_function = "def test_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        target_stage="pdf_extractor",
        run_after=True,
        phase=None,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "phase": None,
            "target_stage": "pdf_extractor",
            "run_before": False,
            "run_after": True,
        },
    }

    assert result == expected


def test_udf_task_to_dict_with_both_timing_flags():
    """Test UDFTask to_dict method with both timing flags."""
    udf_function = "def test_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        target_stage="embedding_storage",
        run_before=True,
        run_after=True,
        phase=None,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "phase": None,
            "target_stage": "embedding_storage",
            "run_before": True,
            "run_after": True,
        },
    }

    assert result == expected


def test_udf_task_to_dict_without_udf_function():
    """Test UDFTask to_dict method when udf_function is None."""
    task = UDFTask(
        udf_function=None,
        udf_function_name="test_func",
        phase=PipelinePhase.RESPONSE,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function_name": "test_func",
            "phase": 5,  # RESPONSE phase value
            "run_before": False,
            "run_after": False,
        },
    }

    assert result == expected


def test_udf_task_to_dict_without_udf_function_name():
    """Test UDFTask to_dict method when udf_function_name is None."""
    udf_function = "def test_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name=None,
        phase=PipelinePhase.TRANSFORM,
    )

    result = task.to_dict()
    expected = {
        "type": "udf",
        "task_properties": {
            "udf_function": udf_function,
            "phase": 4,  # TRANSFORM phase value
            "run_before": False,
            "run_after": False,
        },
    }

    assert result == expected


# Property Tests


def test_udf_task_properties():
    """Test UDFTask property accessors."""
    udf_function = "def my_func(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="my_func",
        phase=PipelinePhase.MUTATION,
    )

    assert task.udf_function == udf_function
    assert task.udf_function_name == "my_func"
    assert task.phase == PipelinePhase.MUTATION


# String Representation Tests


def test_udf_task_string_representation():
    """Test UDFTask string representation."""
    udf_function = "def comprehensive_udf(control_message): return control_message"
    task = UDFTask(
        udf_function=udf_function,
        udf_function_name="comprehensive_udf",
        phase=PipelinePhase.TRANSFORM,
    )

    str_repr = str(task)
    assert "User-Defined Function (UDF) Task:" in str_repr
    assert "udf_function:" in str_repr
    assert "phase: TRANSFORM (4)" in str_repr


def test_udf_task_string_representation_long_function():
    """Test UDFTask string representation with long function (truncation)."""
    # Create a function string longer than 100 characters
    long_function = (
        "def very_long_function_name_that_exceeds_one_hundred_characters(control_message): return control_message"
    )
    task = UDFTask(
        udf_function=long_function,
        phase=PipelinePhase.RESPONSE,
    )

    str_repr = str(task)
    assert "..." in str_repr  # Should be truncated
    assert len(str_repr.split("udf_function: ")[1].split("\n")[0]) <= 103  # 100 chars + "..."


def test_udf_task_string_representation_none_function():
    """Test UDFTask string representation with None function."""
    task = UDFTask(
        udf_function=None,
        udf_function_name="test_func",
        phase=PipelinePhase.EXTRACTION,
    )

    str_repr = str(task)
    assert "udf_function: None" in str_repr


# Function Resolution Tests


def test_udf_task_inline_function_resolution():
    """Test UDFTask function resolution for inline function (no resolution needed)."""
    inline_function = "def inline_func(control_message): return control_message"
    task = UDFTask(
        udf_function=inline_function,
        phase=PipelinePhase.EXTRACTION,
    )

    result = task.to_dict()
    assert result["task_properties"]["udf_function"] == inline_function


def test_udf_task_function_resolution_caching():
    """Test that UDFTask caches resolved functions."""
    inline_function = "def cached_func(control_message): return control_message"
    task = UDFTask(
        udf_function=inline_function,
        phase=PipelinePhase.TRANSFORM,
    )

    # Call _resolve_udf_function multiple times to test caching
    resolved1 = task._resolve_udf_function()
    resolved2 = task._resolve_udf_function()

    # Should return the same result and use caching
    assert resolved1 == resolved2 == inline_function
    assert task._resolved_udf_function == inline_function


def test_udf_task_function_resolution_error_handling():
    """Test UDFTask function resolution error handling for invalid paths."""
    # Test with a non-existent import path
    task = UDFTask(
        udf_function="nonexistent.module.function",
        phase=PipelinePhase.RESPONSE,
    )

    # Should raise an error when trying to resolve
    with pytest.raises(ValueError, match="Failed to import module"):
        task._resolve_udf_function()


def test_udf_task_function_resolution_file_path_error():
    """Test UDFTask function resolution error handling for invalid file paths."""
    # Test with a non-existent file path
    task = UDFTask(
        udf_function="/nonexistent/path.py:my_function",
        phase=PipelinePhase.MUTATION,
    )

    # Should raise an error when trying to resolve
    with pytest.raises(ValueError, match="File not found"):
        task._resolve_udf_function()
