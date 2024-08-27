# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import RedisClientSchema


def test_redis_client_schema_defaults():
    """
    Test RedisClientSchema with default values.
    """
    schema = RedisClientSchema()
    assert schema.host == "redis"
    assert schema.port == 6379
    assert schema.use_ssl is False
    assert schema.connection_timeout == 300
    assert schema.max_backoff == 300
    assert schema.max_retries == 0


@pytest.mark.parametrize("port", [0, -1, 65536, 100000])
def test_redis_client_schema_invalid_port(port):
    """
    Test RedisClientSchema with invalid port values.
    """
    with pytest.raises(ValidationError):
        RedisClientSchema(port=port)


@pytest.mark.parametrize(
    "use_ssl, connection_timeout, max_backoff, max_retries",
    [
        (True, 10, 100, 5),
        (None, None, None, None),  # Test with optional fields set to None
        (False, 0, 0, 0),  # Test with optional fields set to their minimum values
    ],
)
def test_redis_client_schema_optional_fields(use_ssl, connection_timeout, max_backoff, max_retries):
    """
    Parametrized test for RedisClientSchema to check behavior with various combinations of optional fields.
    """
    schema = RedisClientSchema(
        use_ssl=use_ssl,
        connection_timeout=connection_timeout,
        max_backoff=max_backoff,
        max_retries=max_retries,
    )
    # Check each field, assuming None values remain None and provided values are correctly set
    assert schema.use_ssl == use_ssl
    assert schema.connection_timeout == connection_timeout
    assert schema.max_backoff == max_backoff
    assert schema.max_retries == max_retries


def test_redis_client_schema_with_custom_host_and_port():
    """
    Test RedisClientSchema instantiation with custom host and port.
    """
    custom_host = "custom_redis_host"
    custom_port = 1234
    schema = RedisClientSchema(host=custom_host, port=custom_port)
    assert schema.host == custom_host
    assert schema.port == custom_port
