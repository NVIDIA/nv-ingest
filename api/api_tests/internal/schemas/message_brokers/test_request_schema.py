# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.message_brokers.request_schema import (
    PushRequestSchema,
    PopRequestSchema,
    SizeRequestSchema,
)


### Tests for PushRequestSchema ###


def test_push_request_valid():
    schema = PushRequestSchema(command="push", queue_name="my_queue", message="hello")
    assert schema.command == "push"
    assert schema.queue_name == "my_queue"
    assert schema.message == "hello"
    assert schema.timeout == 100


def test_push_request_custom_timeout():
    schema = PushRequestSchema(command="push", queue_name="my_queue", message="hello", timeout=50)
    assert schema.timeout == 50


def test_push_request_missing_required_fields():
    with pytest.raises(ValidationError):
        PushRequestSchema(command="push", message="hello")
    with pytest.raises(ValidationError):
        PushRequestSchema(command="push", queue_name="my_queue")


def test_push_request_min_length_violations():
    with pytest.raises(ValidationError):
        PushRequestSchema(command="push", queue_name="", message="hello")
    with pytest.raises(ValidationError):
        PushRequestSchema(command="push", queue_name="my_queue", message="")


def test_push_request_extra_field_forbidden():
    with pytest.raises(ValidationError):
        PushRequestSchema(command="push", queue_name="my_queue", message="hello", extra_field="oops")


### Tests for PopRequestSchema ###


def test_pop_request_valid():
    schema = PopRequestSchema(command="pop", queue_name="my_queue")
    assert schema.command == "pop"
    assert schema.queue_name == "my_queue"
    assert schema.timeout == 100


def test_pop_request_custom_timeout():
    schema = PopRequestSchema(command="pop", queue_name="my_queue", timeout=30)
    assert schema.timeout == 30


def test_pop_request_missing_required_fields():
    with pytest.raises(ValidationError):
        PopRequestSchema(queue_name="my_queue")
    with pytest.raises(ValidationError):
        PopRequestSchema(command="pop")


def test_pop_request_min_length_violation():
    with pytest.raises(ValidationError):
        PopRequestSchema(command="pop", queue_name="")


def test_pop_request_extra_field_forbidden():
    with pytest.raises(ValidationError):
        PopRequestSchema(command="pop", queue_name="my_queue", extra_field="oops")


### Tests for SizeRequestSchema ###


def test_size_request_valid():
    schema = SizeRequestSchema(command="size", queue_name="my_queue")
    assert schema.command == "size"
    assert schema.queue_name == "my_queue"


def test_size_request_missing_required_fields():
    with pytest.raises(ValidationError):
        SizeRequestSchema(queue_name="my_queue")
    with pytest.raises(ValidationError):
        SizeRequestSchema(command="size")


def test_size_request_min_length_violation():
    with pytest.raises(ValidationError):
        SizeRequestSchema(command="size", queue_name="")


def test_size_request_extra_field_forbidden():
    with pytest.raises(ValidationError):
        SizeRequestSchema(command="size", queue_name="my_queue", extra_field="oops")
