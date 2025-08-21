# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.util.logging.configuration import LogLevel


def test_text_embedding_schema_defaults():
    schema = TextEmbeddingSchema()
    assert schema.api_key == ""
    assert schema.batch_size == 4
    assert schema.embedding_model.startswith("nvidia/")
    assert schema.embedding_nim_endpoint.startswith("http")
    assert schema.encoding_format == "float"
    assert schema.httpx_log_level == LogLevel.WARNING
    assert schema.input_type == "passage"
    assert schema.raise_on_failure is False
    assert schema.truncate == "END"


def test_text_embedding_schema_accepts_custom_values():
    schema = TextEmbeddingSchema(
        api_key="mykey",
        batch_size=16,
        embedding_model="custom/model",
        embedding_nim_endpoint="http://custom.endpoint",
        encoding_format="bfloat16",
        httpx_log_level=LogLevel.DEBUG,
        input_type="query",
        raise_on_failure=True,
        truncate="START",
    )
    assert schema.api_key == "mykey"
    assert schema.batch_size == 16
    assert schema.embedding_model == "custom/model"
    assert schema.embedding_nim_endpoint == "http://custom.endpoint"
    assert schema.encoding_format == "bfloat16"
    assert schema.httpx_log_level == LogLevel.DEBUG
    assert schema.input_type == "query"
    assert schema.raise_on_failure is True
    assert schema.truncate == "START"


def test_text_embedding_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        TextEmbeddingSchema(extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
