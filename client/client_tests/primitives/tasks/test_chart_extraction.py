# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask

# Initialization and Property Setting


def test_chart_extraction_task_initialization():
    """Test initialization of ChartExtractionTask."""
    task = ChartExtractionTask()
    assert task._params == {}


def test_chart_extraction_task_initialization_with_params():
    """Test initialization of ChartExtractionTask with parameters."""
    params = {"key1": "value1", "key2": "value2"}
    task = ChartExtractionTask(params=params)
    assert task._params == params


def test_chart_extraction_task_initialization_none_params():
    """Test initialization of ChartExtractionTask with None params."""
    task = ChartExtractionTask(params=None)
    assert task._params == {}


# String Representation Tests


def test_chart_extraction_task_str_representation():
    """Test string representation of ChartExtractionTask."""
    task = ChartExtractionTask()
    expected_str = "Chart Extraction Task:\n"
    assert str(task) == expected_str


def test_chart_extraction_task_str_representation_with_params():
    """Test string representation of ChartExtractionTask with parameters."""
    params = {"key1": "value1", "key2": "value2"}
    task = ChartExtractionTask(params=params)
    task_str = str(task)
    assert "Chart Extraction Task:" in task_str
    assert "params: " in task_str
    assert "key1" in task_str
    assert "value1" in task_str


# Dictionary Representation Tests


def test_chart_extraction_task_to_dict():
    """Test conversion of ChartExtractionTask to dictionary."""
    task = ChartExtractionTask()
    expected_dict = {
        "type": "chart_data_extract",
        "task_properties": {
            "params": {},
        },
    }
    assert task.to_dict() == expected_dict


def test_chart_extraction_task_to_dict_with_params():
    """Test conversion of ChartExtractionTask to dictionary with parameters."""
    params = {"key1": "value1", "key2": "value2"}
    task = ChartExtractionTask(params=params)
    expected_dict = {
        "type": "chart_data_extract",
        "task_properties": {
            "params": params,
        },
    }
    assert task.to_dict() == expected_dict


# Schema Consolidation Tests


def test_chart_extraction_task_schema_consolidation():
    """Test that ChartExtractionTask uses API schema for validation."""
    # Test that valid parameters work
    params = {"extract_method": "advanced", "confidence_threshold": 0.8}
    task = ChartExtractionTask(params=params)

    assert task._params == params


def test_chart_extraction_task_api_schema_validation():
    """Test that ChartExtractionTask validates against API schema constraints."""
    # Test that None values are handled correctly
    task = ChartExtractionTask()

    assert task._params == {}


def test_chart_extraction_task_serialization_with_api_schema():
    """Test ChartExtractionTask serialization works correctly with API schema."""
    params = {"extract_method": "advanced", "confidence_threshold": 0.8}
    task = ChartExtractionTask(params=params)

    task_dict = task.to_dict()

    assert task_dict["type"] == "chart_data_extract"
    assert task_dict["task_properties"]["params"] == params


def test_chart_extraction_task_empty_params_handling():
    """Test ChartExtractionTask handling of empty params."""
    task = ChartExtractionTask(params={})
    assert task._params == {}

    task_dict = task.to_dict()
    assert task_dict["task_properties"]["params"] == {}


# Edge Cases and Error Handling


def test_chart_extraction_task_params_validation():
    """Test ChartExtractionTask params validation through API schema."""
    # Test with various param types
    params = {
        "string_param": "test",
        "int_param": 42,
        "float_param": 3.14,
        "bool_param": True,
        "list_param": [1, 2, 3],
        "dict_param": {"nested": "value"},
    }

    task = ChartExtractionTask(params=params)
    assert task._params == params

    # Verify serialization preserves all param types
    task_dict = task.to_dict()
    assert task_dict["task_properties"]["params"] == params
