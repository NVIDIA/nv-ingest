# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskCaptionSchema
from nv_ingest_client.primitives.tasks.caption import CaptionTask


def test_caption_task_schema_valid_all_fields():
    """Test valid initialization of IngestTaskCaptionSchema with all fields."""
    schema = IngestTaskCaptionSchema(api_key="test_key", endpoint_url="http://example.com", prompt="Generate a caption")
    assert schema.api_key == "test_key"
    assert schema.endpoint_url == "http://example.com"
    assert schema.prompt == "Generate a caption"
    assert schema.model_name is None


def test_caption_task_schema_valid_partial_fields():
    """Test valid initialization of IngestTaskCaptionSchema with some fields omitted."""
    schema = IngestTaskCaptionSchema(api_key="test_key")
    assert schema.api_key == "test_key"
    assert schema.endpoint_url is None
    assert schema.prompt is None
    assert schema.model_name is None


def test_caption_task_schema_valid_no_fields():
    """Test valid initialization of IngestTaskCaptionSchema with no fields."""
    schema = IngestTaskCaptionSchema()
    assert schema.api_key is None
    assert schema.endpoint_url is None
    assert schema.prompt is None
    assert schema.model_name is None


def test_caption_task_schema_invalid_extra_fields():
    """Test that IngestTaskCaptionSchema raises an error with extra fields."""
    with pytest.raises(ValueError):
        IngestTaskCaptionSchema(api_key="test_key", extra_field="invalid")


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
            "prompt": "Generate a caption",
        },
    }


def test_caption_task_to_dict_partial_fields():
    """Test to_dict method of CaptionTask with partial fields."""
    task = CaptionTask(api_key="test_key")
    task_dict = task.to_dict()
    assert task_dict == {"type": "caption", "task_properties": {"api_key": "test_key"}}


def test_caption_task_to_dict_empty_fields():
    """Test to_dict method of CaptionTask with no fields."""
    task = CaptionTask()
    task_dict = task.to_dict()
    assert task_dict == {"type": "caption", "task_properties": {}}


def test_caption_task_temperature_init():
    """Test initializing CaptionTask with temperature."""
    task = CaptionTask(temperature=0.7)
    assert task._temperature == 0.7


def test_caption_task_temperature_default():
    """Test that temperature defaults to None."""
    task = CaptionTask()
    assert task._temperature is None


def test_caption_task_temperature_to_dict():
    """Test to_dict includes temperature when set."""
    task = CaptionTask(temperature=0.5)
    task_dict = task.to_dict()
    assert task_dict["task_properties"]["temperature"] == 0.5


def test_caption_task_temperature_to_dict_unset():
    """Test to_dict excludes temperature when not set."""
    task = CaptionTask()
    task_dict = task.to_dict()
    assert "temperature" not in task_dict["task_properties"]


def test_caption_task_temperature_str():
    """Test __str__ includes temperature when set."""
    task = CaptionTask(temperature=0.3)
    task_str = str(task)
    assert "temperature: 0.3" in task_str


def test_caption_task_temperature_str_unset():
    """Test __str__ omits temperature when not set."""
    task = CaptionTask()
    task_str = str(task)
    assert "temperature" not in task_str


def test_caption_task_context_text_max_chars_init():
    """Test initializing CaptionTask with context_text_max_chars."""
    task = CaptionTask(context_text_max_chars=512)
    assert task._context_text_max_chars == 512


def test_caption_task_context_text_max_chars_default():
    """Test that context_text_max_chars defaults to None."""
    task = CaptionTask()
    assert task._context_text_max_chars is None


def test_caption_task_context_text_max_chars_to_dict():
    """Test to_dict includes context_text_max_chars when set."""
    task = CaptionTask(context_text_max_chars=256)
    task_dict = task.to_dict()
    assert task_dict["task_properties"]["context_text_max_chars"] == 256


def test_caption_task_context_text_max_chars_to_dict_unset():
    """Test to_dict excludes context_text_max_chars when not set."""
    task = CaptionTask()
    task_dict = task.to_dict()
    assert "context_text_max_chars" not in task_dict["task_properties"]


def test_caption_task_context_text_max_chars_str():
    """Test __str__ includes context_text_max_chars when set."""
    task = CaptionTask(context_text_max_chars=1024)
    task_str = str(task)
    assert "context_text_max_chars: 1024" in task_str


def test_caption_task_context_text_max_chars_str_unset():
    """Test __str__ omits context_text_max_chars when not set."""
    task = CaptionTask()
    task_str = str(task)
    assert "context_text_max_chars" not in task_str


def test_caption_task_schema_context_text_max_chars():
    """Test IngestTaskCaptionSchema accepts context_text_max_chars."""
    schema = IngestTaskCaptionSchema(context_text_max_chars=100)
    assert schema.context_text_max_chars == 100


def test_caption_task_schema_context_text_max_chars_default():
    """Test IngestTaskCaptionSchema context_text_max_chars defaults to None."""
    schema = IngestTaskCaptionSchema()
    assert schema.context_text_max_chars is None


def test_caption_task_schema_temperature():
    """Test IngestTaskCaptionSchema accepts temperature."""
    schema = IngestTaskCaptionSchema(temperature=0.5)
    assert schema.temperature == 0.5


def test_caption_task_schema_temperature_default():
    """Test IngestTaskCaptionSchema temperature defaults to None."""
    schema = IngestTaskCaptionSchema()
    assert schema.temperature is None


# Execute tests
if __name__ == "__main__":
    test_caption_task_schema_valid_all_fields()
    test_caption_task_schema_valid_partial_fields()
    test_caption_task_schema_valid_no_fields()
    test_caption_task_schema_invalid_extra_fields()
    test_caption_task_initialization()
    test_caption_task_partial_initialization()
    test_caption_task_empty_initialization()
    test_caption_task_str_representation_all_fields()
    test_caption_task_str_representation_partial_fields()
    test_caption_task_to_dict_all_fields()
    test_caption_task_to_dict_partial_fields()
    test_caption_task_to_dict_empty_fields()
    test_caption_task_context_text_max_chars_init()
    test_caption_task_context_text_max_chars_default()
    test_caption_task_context_text_max_chars_to_dict()
    test_caption_task_context_text_max_chars_to_dict_unset()
    test_caption_task_context_text_max_chars_str()
    test_caption_task_context_text_max_chars_str_unset()
    test_caption_task_schema_context_text_max_chars()
    test_caption_task_schema_context_text_max_chars_default()
    print("All tests passed.")
