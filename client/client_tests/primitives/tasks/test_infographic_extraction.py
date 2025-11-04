# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_client.primitives.tasks.infographic_extraction import InfographicExtractionTask


def test_infographic_extraction_task_initialization():
    """Test InfographicExtractionTask initialization with parameters."""
    params = {"key1": "value1", "key2": "value2"}
    task = InfographicExtractionTask(params=params)
    assert task._params == params


def test_infographic_extraction_task_default_initialization():
    """Test InfographicExtractionTask with default parameters."""
    task = InfographicExtractionTask()
    assert task._params == {}


def test_infographic_extraction_task_none_params():
    """Test InfographicExtractionTask with None params."""
    task = InfographicExtractionTask(params=None)
    assert task._params == {}


# String Representation Tests


def test_infographic_extraction_task_str_representation():
    """Test string representation of InfographicExtractionTask."""
    params = {"key1": "value1"}
    task = InfographicExtractionTask(params=params)
    expected_str = "Infographic Extraction Task:\n  params: {'key1': 'value1'}\n"
    assert str(task) == expected_str


def test_infographic_extraction_task_str_representation_empty():
    """Test string representation with empty params."""
    task = InfographicExtractionTask()
    expected_str = "Infographic Extraction Task:\n  params: {}\n"
    assert str(task) == expected_str


# Dictionary Representation Tests


def test_infographic_extraction_task_to_dict():
    """Test conversion of InfographicExtractionTask to dictionary."""
    task = InfographicExtractionTask()
    expected_dict = {
        "type": "infographic_data_extract",
        "task_properties": {"params": {}},
    }
    assert task.to_dict() == expected_dict


def test_infographic_extraction_task_to_dict_with_params():
    """Test conversion with parameters."""
    params = {"key1": "value1", "key2": "value2"}
    task = InfographicExtractionTask(params=params)
    expected_dict = {
        "type": "infographic_data_extract",
        "task_properties": {
            "params": params,
        },
    }
    assert task.to_dict() == expected_dict


# Schema Consolidation Tests


def test_infographic_extraction_task_schema_consolidation():
    """Test that InfographicExtractionTask validates against API schema constraints."""
    params = {"test_param": "test_value"}
    task = InfographicExtractionTask(params=params)

    # Verify the task was created with proper validation
    assert task._params == params

    # Test serialization matches API schema expectations
    task_dict = task.to_dict()
    assert task_dict["type"] == "infographic_data_extract"
    assert task_dict["task_properties"]["params"] == params


def test_infographic_extraction_task_api_schema_validation():
    """Test that the task uses API schema for validation."""
    # Test that empty dict is handled correctly
    task = InfographicExtractionTask(params={})
    assert task._params == {}

    # Test that complex params are handled correctly
    complex_params = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "string": "test",
        "number": 42,
        "boolean": True,
    }
    task = InfographicExtractionTask(params=complex_params)
    assert task._params == complex_params

    task_dict = task.to_dict()
    assert task_dict["task_properties"]["params"] == complex_params


# Edge Cases


def test_infographic_extraction_task_empty_params():
    """Test InfographicExtractionTask with explicitly empty params."""
    task = InfographicExtractionTask(params={})
    assert task._params == {}

    task_dict = task.to_dict()
    expected_dict = {
        "type": "infographic_data_extract",
        "task_properties": {"params": {}},
    }
    assert task_dict == expected_dict


def test_infographic_extraction_task_complex_params():
    """Test InfographicExtractionTask with complex parameter structures."""
    params = {
        "model_config": {
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "extraction_settings": {
            "extract_text": True,
            "extract_images": False,
        },
        "output_format": "json",
    }

    task = InfographicExtractionTask(params=params)
    assert task._params == params

    task_dict = task.to_dict()
    assert task_dict["task_properties"]["params"] == params
