# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Literal

from pydantic import Field, BaseModel
from typing_extensions import Annotated


class MessageBrokerClientSchema(BaseModel):
    host: str = "redis"
    port: Annotated[int, Field(gt=0, lt=65536)] = 6379

    # Update this for new broker types
    client_type: Literal["redis", "simple"] = "redis"  # Restrict to 'redis' or 'simple'

    broker_params: Optional[dict] = Field(default_factory=dict)

    connection_timeout: Optional[Annotated[int, Field(ge=0)]] = 300
    max_backoff: Optional[Annotated[int, Field(ge=0)]] = 300
    max_retries: Optional[Annotated[int, Field(ge=0)]] = 0
