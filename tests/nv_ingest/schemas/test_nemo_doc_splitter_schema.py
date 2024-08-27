# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import DocumentSplitterSchema


def test_document_splitter_schema_defaults():
    """
    Test the DocumentSplitterSchema with default values.
    """
    schema = DocumentSplitterSchema()
    assert schema.split_by == "word"
    assert schema.split_length == 60
    assert schema.split_overlap == 10
    assert schema.max_character_length == 450
    assert schema.sentence_window_size == 0
    assert schema.raise_on_failure is False


@pytest.mark.parametrize("invalid_value", [-1, 0])
def test_document_splitter_schema_invalid_split_length(invalid_value):
    """
    Test DocumentSplitterSchema with invalid split_length values.
    """
    with pytest.raises(ValidationError):
        DocumentSplitterSchema(split_length=invalid_value)


@pytest.mark.parametrize(
    "split_by, sentence_window_size, is_valid",
    [
        ("sentence", 5, True),  # Valid use of sentence_window_size
        (
            "word",
            0,
            True,
        ),  # Valid when split_by is not 'sentence' but sentence_window_size is 0
        (
            "word",
            5,
            False,
        ),  # Invalid because sentence_window_size > 0 requires split_by to be 'sentence'
    ],
)
def test_document_splitter_schema_sentence_window_size_validation(split_by, sentence_window_size, is_valid):
    """
    Parametrized test for validating the sentence_window_size logic in DocumentSplitterSchema.
    """
    if is_valid:
        schema = DocumentSplitterSchema(split_by=split_by, sentence_window_size=sentence_window_size)
        assert schema.sentence_window_size == sentence_window_size
        assert schema.split_by == split_by
    else:
        with pytest.raises(ValidationError) as excinfo:
            DocumentSplitterSchema(split_by=split_by, sentence_window_size=sentence_window_size)
        assert "split_by must be 'sentence'" in str(excinfo.value)


def test_document_splitter_schema_optional_fields_none():
    """
    Test DocumentSplitterSchema with optional fields set to None.
    """
    schema = DocumentSplitterSchema(max_character_length=None, sentence_window_size=None)
    assert schema.max_character_length is None
    assert schema.sentence_window_size is None
