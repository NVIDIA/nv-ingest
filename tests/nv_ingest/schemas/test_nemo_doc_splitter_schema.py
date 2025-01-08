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
    assert schema.tokenizer == "intfloat/e5-large-unsupervised"
    assert schema.chunk_size == 300
    assert schema.chunk_overlap == 0
    assert schema.raise_on_failure is False


@pytest.mark.parametrize("invalid_value", [-1, 0])
def test_document_splitter_schema_invalid_split_length(invalid_value):
    """
    Test DocumentSplitterSchema with invalid chunk_size values.
    """
    with pytest.raises(ValidationError):
        DocumentSplitterSchema(chunk_size=invalid_value)
