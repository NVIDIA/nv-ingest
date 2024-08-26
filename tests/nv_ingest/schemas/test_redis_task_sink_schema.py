# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import RedisClientSchema
from nv_ingest.schemas import RedisTaskSinkSchema


def test_redis_task_sink_schema_defaults():
    """
    Test RedisTaskSinkSchema with default values, including the default embedded RedisClientSchema.
    """
    schema = RedisTaskSinkSchema()
    assert isinstance(
        schema.redis_client, RedisClientSchema
    ), "redis_client should be an instance of RedisClientSchema."
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."
    assert schema.progress_engines == 6, "Default value for progress_engines should be 6."


def test_redis_task_sink_schema_custom_values():
    """
    Test RedisTaskSinkSchema with custom values for its fields and embedded RedisClientSchema.
    """
    custom_redis_client = RedisClientSchema(host="custom_host", port=12345, use_ssl=True)
    schema = RedisTaskSinkSchema(redis_client=custom_redis_client, raise_on_failure=True, progress_engines=10)

    assert schema.redis_client.host == "custom_host", "Custom host value for redis_client should be respected."
    assert schema.redis_client.port == 12345, "Custom port value for redis_client should be respected."
    assert schema.redis_client.use_ssl is True, "Custom use_ssl value for redis_client should be True."
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
    assert schema.progress_engines == 10, "Custom value for progress_engines should be respected."


@pytest.mark.parametrize("progress_engines", [-1, 0])
def test_redis_task_sink_schema_invalid_progress_engines(progress_engines):
    """
    Test RedisTaskSinkSchema with invalid values for progress_engines to ensure validation catches them.
    """
    with pytest.raises(ValidationError):
        RedisTaskSinkSchema(progress_engines=progress_engines)
