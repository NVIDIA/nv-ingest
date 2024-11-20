# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Literal

from pydantic import BaseModel
from pydantic import conint


class MessageBrokerClientSchema(BaseModel):
    host: str = "redis"
    port: conint(gt=0, lt=65536) = 6379

    # Update this for new broker types
    client_type: Literal["redis", "simple"] = "redis"  # Restrict to 'redis' or 'simple'

    broker_params: Optional[dict] = {}

    connection_timeout: Optional[conint(ge=0)] = 300
    max_backoff: Optional[conint(ge=0)] = 300
    max_retries: Optional[conint(ge=0)] = 0
