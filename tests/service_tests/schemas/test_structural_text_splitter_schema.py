# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_structural_text_splitter_schema import (
    StructuralTextSplitterSchema
)


def test_structural_text_splitter_schema_defaults():
    """
    Test the StructuralTextSplitterSchema with default values.
    """
    schema = StructuralTextSplitterSchema()
    assert schema.markdown_headers_to_split_on == ["#", "##", "###", "####", "#####", "######"]
    assert schema.max_chunk_size_tokens == 1024
    assert schema.preserve_headers_in_chunks is True
    assert schema.min_chunk_size_chars == 50
    assert schema.raise_on_failure is False


def test_structural_text_splitter_schema_custom_values():
    """
    Test the StructuralTextSplitterSchema with custom values.
    """
    custom_headers = ["#", "##", "###"]
    schema = StructuralTextSplitterSchema(
        markdown_headers_to_split_on=custom_headers,
        max_chunk_size_tokens=512,
        preserve_headers_in_chunks=False,
        min_chunk_size_chars=100,
        raise_on_failure=True
    )
    assert schema.markdown_headers_to_split_on == custom_headers
    assert schema.max_chunk_size_tokens == 512
    assert schema.preserve_headers_in_chunks is False
    assert schema.min_chunk_size_chars == 100
    assert schema.raise_on_failure is True


def test_structural_text_splitter_schema_invalid_max_tokens():
    """
    Test StructuralTextSplitterSchema with invalid max_chunk_size_tokens values.
    """
    invalid_values = [0, -1, -100]
    for invalid_value in invalid_values:
        with pytest.raises(ValidationError):
            StructuralTextSplitterSchema(max_chunk_size_tokens=invalid_value)


def test_structural_text_splitter_schema_invalid_min_chars():
    """
    Test StructuralTextSplitterSchema with invalid min_chunk_size_chars values.
    """
    invalid_values = [-1, -100]
    for invalid_value in invalid_values:
        with pytest.raises(ValidationError):
            StructuralTextSplitterSchema(min_chunk_size_chars=invalid_value)


def test_structural_text_splitter_schema_empty_headers():
    """
    Test StructuralTextSplitterSchema with empty headers list.
    """
    # Empty list should be valid (will fallback to returning original content)
    schema = StructuralTextSplitterSchema(markdown_headers_to_split_on=[])
    assert schema.markdown_headers_to_split_on == []


def test_structural_text_splitter_schema_custom_headers():
    """
    Test StructuralTextSplitterSchema with various header configurations.
    """
    # Test with custom header patterns
    custom_headers = ["#", "##", "===", "---"]
    schema = StructuralTextSplitterSchema(markdown_headers_to_split_on=custom_headers)
    assert schema.markdown_headers_to_split_on == custom_headers
    
    # Test with single header
    single_header = ["#"]
    schema = StructuralTextSplitterSchema(markdown_headers_to_split_on=single_header)
    assert schema.markdown_headers_to_split_on == single_header


def test_structural_text_splitter_schema_extra_fields_forbidden():
    """
    Test that extra fields are forbidden in StructuralTextSplitterSchema.
    """
    with pytest.raises(ValidationError):
        StructuralTextSplitterSchema(
            max_chunk_size_tokens=1024,
            extra_field="this_should_fail"
        ) 