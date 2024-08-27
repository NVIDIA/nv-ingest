# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel

from nv_ingest.schemas.redis_client_schema import RedisClientSchema


class OpenTelemetryMeterSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
    otel_endpoint: str = "localhost:4317"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
