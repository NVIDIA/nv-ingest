# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import TextSplitterSchema


def test_text_splitter_schema_defaults():
    """
    Test the TextSplitterSchema with default values.
    """
    schema = TextSplitterSchema()
    assert schema.tokenizer is None
    assert schema.chunk_size == 1024
    assert schema.chunk_overlap == 150
    assert schema.raise_on_failure is False


def test_text_splitter_schema_custom_values():
    """
    Test the TextSplitterSchema with custom values.
    """
    tokenizer = "meta-llama/Llama-3.2-1B"
    chunk_size = 500
    chunk_overlap = 10
    schema = TextSplitterSchema(
        tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, raise_on_failure=True
    )
    assert schema.tokenizer == tokenizer
    assert schema.chunk_size == chunk_size
    assert schema.chunk_overlap == chunk_overlap
    assert schema.raise_on_failure is True


@pytest.mark.parametrize("invalid_value", [50, 5.5])
def test_text_splitter_schema_invalid_tokenizer(invalid_value):
    """
    Test TextSplitterSchema with invalid tokenizer values.
    """
    with pytest.raises(ValidationError):
        TextSplitterSchema(tokenizer=invalid_value)


@pytest.mark.parametrize("invalid_value", [-1, 0])
def test_text_splitter_schema_invalid_chunk_size(invalid_value):
    """
    Test TextSplitterSchema with invalid chunk_size values.
    """
    with pytest.raises(ValidationError):
        TextSplitterSchema(chunk_size=invalid_value)


@pytest.mark.parametrize("invalid_value", [-1, "a"])
def test_text_splitter_schema_invalid_chunk_overlap(invalid_value):
    """
    Test TextSplitterSchema with invalid chunk_overlap values.
    """
    with pytest.raises(ValidationError):
        TextSplitterSchema(chunk_overlap=invalid_value)


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap, is_valid",
    [
        (300, 50, True),
        (150, 0, True),
        (100, 100, False),
        (50, 200, False),
    ],
)
def test_text_splitter_schema_chunk_overlap_validation(chunk_size, chunk_overlap, is_valid):
    """
    Parametrized test for validating the chunk_overlap logic in TextSplitterSchema.
    """
    if is_valid:
        schema = TextSplitterSchema(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert schema.chunk_size == chunk_size
        assert schema.chunk_overlap == chunk_overlap
    else:
        with pytest.raises(ValidationError) as excinfo:
            TextSplitterSchema(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        assert "chunk_overlap must be less than chunk_size" in str(excinfo.value)
