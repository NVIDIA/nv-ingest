# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import RedisClientSchema
from nv_ingest.schemas import RedisTaskSourceSchema


def test_redis_task_source_schema_defaults():
    """
    Test RedisTaskSourceSchema with default values.
    """
    schema = RedisTaskSourceSchema()
    assert isinstance(
        schema.redis_client, RedisClientSchema
    ), "redis_client should be an instance of RedisClientSchema."
    assert schema.task_queue == "morpheus_task_queue", "Default value for task_queue should be 'morpheus_task_queue'."
    assert schema.progress_engines == 6, "Default value for progress_engines should be 6."


def test_redis_task_source_schema_custom_values():
    """
    Test RedisTaskSourceSchema with custom values for its fields.
    """
    custom_redis_client = RedisClientSchema(host="custom_host", port=12345, use_ssl=True)
    schema = RedisTaskSourceSchema(redis_client=custom_redis_client, task_queue="custom_queue", progress_engines=10)

    assert schema.redis_client.host == "custom_host", "Custom host value for redis_client should be respected."
    assert schema.redis_client.port == 12345, "Custom port value for redis_client should be respected."
    assert schema.redis_client.use_ssl is True, "Custom use_ssl value for redis_client should be True."
    assert schema.task_queue == "custom_queue", "Custom value for task_queue should be respected."
    assert schema.progress_engines == 10, "Custom value for progress_engines should be respected."


@pytest.mark.parametrize("progress_engines", [0, -1])
def test_redis_task_source_schema_invalid_progress_engines(progress_engines):
    """
    Test RedisTaskSourceSchema with invalid values for progress_engines to ensure validation catches them.
    """
    with pytest.raises(ValidationError) as excinfo:
        RedisTaskSourceSchema(progress_engines=progress_engines)
    assert "ensure this value is greater than or equal to 1" in str(
        excinfo.value
    ), "Schema should validate progress_engines to be >= 1."
