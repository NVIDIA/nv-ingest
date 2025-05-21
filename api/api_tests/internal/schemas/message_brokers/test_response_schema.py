# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema


def test_response_schema_valid_with_string_response():
    schema = ResponseSchema(response_code=200, response="Success")
    assert schema.response_code == 200
    assert schema.response == "Success"
    assert schema.response_reason == "OK"
    assert schema.trace_id is None
    assert schema.transaction_id is None


def test_response_schema_valid_with_dict_response():
    schema = ResponseSchema(response_code=201, response={"key": "value"})
    assert schema.response == {"key": "value"}


def test_response_schema_valid_with_none_response():
    schema = ResponseSchema(response_code=202)
    assert schema.response is None
    assert schema.response_reason == "OK"


def test_response_schema_full_fields():
    schema = ResponseSchema(
        response_code=500,
        response="Error occurred",
        response_reason="Server Error",
        trace_id="abc-123",
        transaction_id="tx-789",
    )
    assert schema.response_code == 500
    assert schema.response == "Error occurred"
    assert schema.response_reason == "Server Error"
    assert schema.trace_id == "abc-123"
    assert schema.transaction_id == "tx-789"


def test_response_schema_missing_required_field():
    with pytest.raises(ValidationError):
        ResponseSchema(response="Missing code")


def test_response_schema_invalid_response_type():
    with pytest.raises(ValidationError):
        ResponseSchema(response_code=200, response=1234)  # int is invalid
