# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nv_ingest.schemas import MessageBrokerClientSchema
from nv_ingest.schemas.otel_meter_schema import OpenTelemetryMeterSchema
from nv_ingest.util.message_brokers.client_base import MessageBrokerClientBase


def test_otel_meter_schema_defaults():
    schema = OpenTelemetryMeterSchema()
    assert isinstance(
        schema.broker_client, MessageBrokerClientSchema
    ), "broker_client should be an instance of MessageBrokerClientSchema."
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_otel_meter_schema_custom_values():
    custom_redis_client = MessageBrokerClientSchema(host="custom_host", port=12345, broker_params={"use_ssl": True})
    schema = OpenTelemetryMeterSchema(broker_client=custom_redis_client, raise_on_failure=True)

    assert schema.broker_client.host == "custom_host", "Custom host value for redis_client should be respected."
    assert schema.broker_client.port == 12345, "Custom port value for redis_client should be respected."
    assert (
        schema.broker_client.broker_params["use_ssl"] is True
    ), "Custom use_ssl value for broker_client should be True."
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
