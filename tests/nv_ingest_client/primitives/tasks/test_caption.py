# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
from nv_ingest_client.primitives.tasks.caption import CaptionTaskSchema, CaptionTask


# Testing CaptionTaskSchema

def test_valid_schema_initialization():
    """Test valid initialization of CaptionTaskSchema with all fields."""
    schema = CaptionTaskSchema(api_key="test_key", endpoint_url="http://example.com", prompt="Generate a caption")
    assert schema.api_key == "test_key"
    assert schema.endpoint_url == "http://example.com"
    assert schema.prompt == "Generate a caption"


def test_partial_schema_initialization():
    """Test valid initialization of CaptionTaskSchema with some fields omitted."""
    schema = CaptionTaskSchema(api_key="test_key")
    assert schema.api_key == "test_key"
    assert schema.endpoint_url is None
    assert schema.prompt is None


def test_empty_schema_initialization():
    """Test valid initialization of CaptionTaskSchema with no fields."""
    schema = CaptionTaskSchema()
    assert schema.api_key is None
    assert schema.endpoint_url is None
    assert schema.prompt is None


def test_schema_invalid_extra_field():
    """Test that CaptionTaskSchema raises an error with extra fields."""
    try:
        CaptionTaskSchema(api_key="test_key", extra_field="invalid")
    except ValidationError as e:
        assert "extra_field" in str(e)


# Testing CaptionTask

def test_caption_task_initialization():
    """Test initializing CaptionTask with all fields."""
    task = CaptionTask(api_key="test_key", endpoint_url="http://example.com", prompt="Generate a caption")
    assert task._api_key == "test_key"
    assert task._endpoint_url == "http://example.com"
    assert task._prompt == "Generate a caption"


def test_caption_task_partial_initialization():
    """Test initializing CaptionTask with some fields omitted."""
    task = CaptionTask(api_key="test_key")
    assert task._api_key == "test_key"
    assert task._endpoint_url is None
    assert task._prompt is None


def test_caption_task_empty_initialization():
    """Test initializing CaptionTask with no fields."""
    task = CaptionTask()
    assert task._api_key is None
    assert task._endpoint_url is None
    assert task._prompt is None


def test_caption_task_str_representation_all_fields():
    """Test string representation of CaptionTask with all fields."""
    task = CaptionTask(api_key="test_key", endpoint_url="http://example.com", prompt="Generate a caption")
    task_str = str(task)
    assert "Image Caption Task:" in task_str
    assert "api_key: [redacted]" in task_str
    assert "endpoint_url: http://example.com" in task_str
    assert "prompt: Generate a caption" in task_str


def test_caption_task_str_representation_partial_fields():
    """Test string representation of CaptionTask with partial fields."""
    task = CaptionTask(api_key="test_key")
    task_str = str(task)
    assert "Image Caption Task:" in task_str
    assert "api_key: [redacted]" in task_str
    assert "endpoint_url" not in task_str
    assert "prompt" not in task_str


def test_caption_task_to_dict_all_fields():
    """Test to_dict method of CaptionTask with all fields."""
    task = CaptionTask(api_key="test_key", endpoint_url="http://example.com", prompt="Generate a caption")
    task_dict = task.to_dict()
    assert task_dict == {
        "type": "caption",
        "task_properties": {
            "api_key": "test_key",
            "endpoint_url": "http://example.com",
            "prompt": "Generate a caption"
        }
    }


def test_caption_task_to_dict_partial_fields():
    """Test to_dict method of CaptionTask with partial fields."""
    task = CaptionTask(api_key="test_key")
    task_dict = task.to_dict()
    assert task_dict == {
        "type": "caption",
        "task_properties": {
            "api_key": "test_key"
        }
    }


def test_caption_task_to_dict_empty_fields():
    """Test to_dict method of CaptionTask with no fields."""
    task = CaptionTask()
    task_dict = task.to_dict()
    assert task_dict == {
        "type": "caption",
        "task_properties": {}
    }


# Execute tests
if __name__ == "__main__":
    test_valid_schema_initialization()
    test_partial_schema_initialization()
    test_empty_schema_initialization()
    test_schema_invalid_extra_field()
    test_caption_task_initialization()
    test_caption_task_partial_initialization()
    test_caption_task_empty_initialization()
    test_caption_task_str_representation_all_fields()
    test_caption_task_str_representation_partial_fields()
    test_caption_task_to_dict_all_fields()
    test_caption_task_to_dict_partial_fields()
    test_caption_task_to_dict_empty_fields()
    print("All tests passed.")
