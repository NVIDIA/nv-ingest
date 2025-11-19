# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema


def test_text_splitter_schema_defaults():
    schema = TextSplitterSchema()
    assert schema.chunk_size == 1024
    assert schema.chunk_overlap == 150
    assert schema.raise_on_failure is False


def test_text_splitter_schema_accepts_valid_overlap():
    schema = TextSplitterSchema(chunk_size=200, chunk_overlap=100)
    assert schema.chunk_overlap == 100
    assert schema.chunk_size == 200


def test_text_splitter_schema_rejects_overlap_greater_than_size():
    with pytest.raises(ValidationError) as excinfo:
        TextSplitterSchema(chunk_size=100, chunk_overlap=150)
    assert "chunk_overlap must be less than chunk_size" in str(excinfo.value)


def test_text_splitter_schema_rejects_zero_chunk_size():
    with pytest.raises(ValidationError) as excinfo:
        TextSplitterSchema(chunk_size=0)
    assert "greater than 0" in str(excinfo.value)


def test_text_splitter_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        TextSplitterSchema(chunk_size=200, chunk_overlap=100, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
