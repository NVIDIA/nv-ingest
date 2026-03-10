# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.message_brokers.message_broker_client_schema import MessageBrokerClientSchema


def test_broker_valid_defaults():
    schema = MessageBrokerClientSchema()
    assert schema.host == "redis"
    assert schema.port == 6379
    assert schema.client_type == "redis"
    assert schema.broker_params == {}
    assert schema.connection_timeout == 300
    assert schema.max_backoff == 300
    assert schema.max_retries == 0


def test_broker_valid_simple_client_type():
    schema = MessageBrokerClientSchema(client_type="simple")
    assert schema.client_type == "simple"


def test_broker_invalid_client_type():
    with pytest.raises(ValidationError) as excinfo:
        MessageBrokerClientSchema(client_type="rabbitmq")
    assert "Input should be 'redis' or 'simple'" in str(excinfo.value)


def test_broker_invalid_port_zero_or_out_of_range():
    with pytest.raises(ValidationError):
        MessageBrokerClientSchema(port=0)
    with pytest.raises(ValidationError):
        MessageBrokerClientSchema(port=70000)


def test_broker_valid_port_range_edges():
    schema = MessageBrokerClientSchema(port=1)
    assert schema.port == 1
    schema = MessageBrokerClientSchema(port=65535)
    assert schema.port == 65535


def test_broker_invalid_connection_timeout_negative():
    with pytest.raises(ValidationError):
        MessageBrokerClientSchema(connection_timeout=-1)


def test_broker_invalid_max_backoff_negative():
    with pytest.raises(ValidationError):
        MessageBrokerClientSchema(max_backoff=-5)


def test_broker_invalid_max_retries_negative():
    with pytest.raises(ValidationError):
        MessageBrokerClientSchema(max_retries=-2)


def test_broker_custom_all_values():
    schema = MessageBrokerClientSchema(
        host="localhost",
        port=1234,
        client_type="redis",
        broker_params={"param1": "value"},
        connection_timeout=100,
        max_backoff=200,
        max_retries=5,
    )
    assert schema.host == "localhost"
    assert schema.port == 1234
    assert schema.client_type == "redis"
    assert schema.broker_params == {"param1": "value"}
    assert schema.connection_timeout == 100
    assert schema.max_backoff == 200
    assert schema.max_retries == 5
