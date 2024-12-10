# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel

from nv_ingest.schemas.message_broker_client_schema import MessageBrokerClientSchema


class OpenTelemetryMeterSchema(BaseModel):
    broker_client: MessageBrokerClientSchema = MessageBrokerClientSchema()

    otel_endpoint: str = "localhost:4317"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
