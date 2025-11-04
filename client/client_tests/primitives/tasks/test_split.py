# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.split import SplitTask

# Initialization and Property Setting


def test_split_task_initialization():
    task = SplitTask(
        tokenizer="meta-llama/Llama-3.2-1B",
        chunk_size=1024,
        chunk_overlap=0,
        params={},
    )
    assert task._tokenizer == "meta-llama/Llama-3.2-1B"
    assert task._chunk_size == 1024
    assert task._chunk_overlap == 0
    assert task._params == {}


# String Representation Tests


def test_split_task_str_representation():
    task = SplitTask(tokenizer="intfloat/e5-large-unsupervised", chunk_size=50, chunk_overlap=5)
    expected_str = (
        "Split Task:\n" "  tokenizer: intfloat/e5-large-unsupervised\n" "  chunk_size: 50\n" "  chunk_overlap: 5\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "tokenizer, chunk_size, chunk_overlap, params",
    [
        ("intfloat/e5-large-unsupervised", 100, 10, {}),
        ("microsoft/deberta-large", 50, 5, None),
        ("meta-llama/Llama-3.2-1B", 1024, 0, {"hf_access_token": "TOKEN"}),
    ],
)
def test_split_task_to_dict(
    tokenizer,
    chunk_size,
    chunk_overlap,
    params,
):
    task = SplitTask(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        params=params,
    )

    expected_dict = {"type": "split", "task_properties": {}}

    # Only add properties to expected_dict if they are not None
    if tokenizer is not None:
        expected_dict["task_properties"]["tokenizer"] = tokenizer
    if chunk_size is not None:
        expected_dict["task_properties"]["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        expected_dict["task_properties"]["chunk_overlap"] = chunk_overlap
    # params is always included because API schema converts None to {}
    expected_dict["task_properties"]["params"] = params if params is not None else {}

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_split_task_default_params():
    task = SplitTask()
    expected_str_contains = [
        "chunk_size: 1024",
        "chunk_overlap: 150",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "split",
        "task_properties": {
            "chunk_size": 1024,
            "chunk_overlap": 150,
            "params": {},
        },
    }
    assert task.to_dict() == expected_dict


# Schema Consolidation and Validation Tests


def test_split_task_api_schema_validation():
    """Test that SplitTask uses API schema validation internally."""
    # Valid configuration should work
    task = SplitTask(tokenizer="meta-llama/Llama-3.2-1B", chunk_size=1024, chunk_overlap=150, params={"test": "value"})
    assert task._tokenizer == "meta-llama/Llama-3.2-1B"
    assert task._chunk_size == 1024
    assert task._chunk_overlap == 150
    assert task._params == {"test": "value"}


def test_split_task_chunk_size_validation():
    """Test that chunk_size validation works correctly."""
    # Valid chunk_size should work (with chunk_overlap < chunk_size)
    task = SplitTask(chunk_size=200, chunk_overlap=50)
    assert task._chunk_size == 200
    assert task._chunk_overlap == 50

    # chunk_size must be > 0
    with pytest.raises(ValueError, match="greater than 0"):
        SplitTask(chunk_size=0)

    with pytest.raises(ValueError, match="greater than 0"):
        SplitTask(chunk_size=-1)


def test_split_task_chunk_overlap_validation():
    """Test that chunk_overlap validation works correctly."""
    # Valid chunk_overlap should work
    task = SplitTask(chunk_overlap=50)
    assert task._chunk_overlap == 50

    # chunk_overlap must be >= 0
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        SplitTask(chunk_overlap=-1)

    # chunk_overlap = 0 should be valid
    task = SplitTask(chunk_overlap=0)
    assert task._chunk_overlap == 0


def test_split_task_chunk_overlap_less_than_chunk_size():
    """Test that chunk_overlap must be less than chunk_size."""
    # Valid: chunk_overlap < chunk_size
    task = SplitTask(chunk_size=100, chunk_overlap=50)
    assert task._chunk_size == 100
    assert task._chunk_overlap == 50

    # Invalid: chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        SplitTask(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        SplitTask(chunk_size=100, chunk_overlap=150)


def test_split_task_params_handling():
    """Test that params field is handled correctly."""
    # Empty params dict should work
    task = SplitTask(params={})
    assert task._params == {}

    # Non-empty params dict should work
    params = {"key1": "value1", "key2": 42, "key3": True}
    task = SplitTask(params=params)
    assert task._params == params

    # Default params should be empty dict
    task = SplitTask()
    assert task._params == {}


def test_split_task_tokenizer_handling():
    """Test that tokenizer field is handled correctly."""
    # None tokenizer should work
    task = SplitTask(tokenizer=None)
    assert task._tokenizer is None

    # String tokenizer should work
    task = SplitTask(tokenizer="meta-llama/Llama-3.2-1B")
    assert task._tokenizer == "meta-llama/Llama-3.2-1B"

    # Default tokenizer should be None
    task = SplitTask()
    assert task._tokenizer is None


def test_split_task_comprehensive_validation():
    """Test comprehensive validation scenarios."""
    # All valid parameters
    task = SplitTask(
        tokenizer="meta-llama/Llama-3.2-1B",
        chunk_size=2048,
        chunk_overlap=200,
        params={"temperature": 0.7, "max_tokens": 1000},
    )

    assert task._tokenizer == "meta-llama/Llama-3.2-1B"
    assert task._chunk_size == 2048
    assert task._chunk_overlap == 200
    assert task._params == {"temperature": 0.7, "max_tokens": 1000}

    # Verify serialization works correctly
    task_dict = task.to_dict()
    expected_dict = {
        "type": "split",
        "task_properties": {
            "tokenizer": "meta-llama/Llama-3.2-1B",
            "chunk_size": 2048,
            "chunk_overlap": 200,
            "params": {"temperature": 0.7, "max_tokens": 1000},
        },
    }
    assert task_dict == expected_dict


def test_split_task_edge_cases():
    """Test edge cases for split task validation."""
    # Minimum valid chunk_size
    task = SplitTask(chunk_size=1, chunk_overlap=0)
    assert task._chunk_size == 1
    assert task._chunk_overlap == 0

    # Large chunk_size
    task = SplitTask(chunk_size=10000, chunk_overlap=5000)
    assert task._chunk_size == 10000
    assert task._chunk_overlap == 5000

    # chunk_overlap just under chunk_size
    task = SplitTask(chunk_size=100, chunk_overlap=99)
    assert task._chunk_size == 100
    assert task._chunk_overlap == 99


def test_split_task_serialization_consistency():
    """Test that serialization is consistent with validation."""
    # Create task with various parameters
    task = SplitTask(
        tokenizer="test-tokenizer", chunk_size=512, chunk_overlap=64, params={"custom_param": "custom_value"}
    )

    # Serialize to dict
    task_dict = task.to_dict()

    # Verify all parameters are present and correct
    assert task_dict["type"] == "split"
    assert task_dict["task_properties"]["tokenizer"] == "test-tokenizer"
    assert task_dict["task_properties"]["chunk_size"] == 512
    assert task_dict["task_properties"]["chunk_overlap"] == 64
    assert task_dict["task_properties"]["params"] == {"custom_param": "custom_value"}


def test_split_task_none_values():
    """Test handling of None values in parameters."""
    # Explicit None values
    task = SplitTask(tokenizer=None, params={})
    assert task._tokenizer is None
    assert task._params == {}

    # Verify serialization handles None correctly
    task_dict = task.to_dict()
    # tokenizer should not be in the dict if it's None
    assert "tokenizer" not in task_dict["task_properties"]
    assert task_dict["task_properties"]["params"] == {}


def test_split_task_string_representation_completeness():
    """Test that string representation includes all relevant information."""
    task = SplitTask(
        tokenizer="meta-llama/Llama-3.2-1B",
        chunk_size=1024,
        chunk_overlap=150,
        params={"param1": "value1", "param2": "value2"},
    )

    str_repr = str(task)

    # Check that all important information is in the string representation
    assert "Split Task:" in str_repr
    assert "tokenizer: meta-llama/Llama-3.2-1B" in str_repr
    assert "chunk_size: 1024" in str_repr
    assert "chunk_overlap: 150" in str_repr
    assert "param1: value1" in str_repr
    assert "param2: value2" in str_repr
