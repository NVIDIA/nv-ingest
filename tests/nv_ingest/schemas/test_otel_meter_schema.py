# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nv_ingest.schemas import RedisClientSchema
from nv_ingest.schemas.otel_meter_schema import OpenTelemetryMeterSchema


def test_otel_meter_schema_defaults():
    schema = OpenTelemetryMeterSchema()
    assert isinstance(
        schema.redis_client, RedisClientSchema
    ), "redis_client should be an instance of RedisClientSchema."
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_otel_meter_schema_custom_values():
    custom_redis_client = RedisClientSchema(host="custom_host", port=12345, use_ssl=True)
    schema = OpenTelemetryMeterSchema(redis_client=custom_redis_client, raise_on_failure=True)

    assert schema.redis_client.host == "custom_host", "Custom host value for redis_client should be respected."
    assert schema.redis_client.port == 12345, "Custom port value for redis_client should be respected."
    assert schema.redis_client.use_ssl is True, "Custom use_ssl value for redis_client should be True."
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
