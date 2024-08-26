# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from pydantic import BaseModel
from pydantic import conint


class RedisClientSchema(BaseModel):
    host: str = "redis"
    port: conint(gt=0, lt=65536) = 6379  # Ports must be in the range 1-65535
    use_ssl: Optional[bool] = False

    connection_timeout: Optional[conint(ge=0)] = 300
    max_backoff: Optional[conint(ge=0)] = 300
    max_retries: Optional[conint(ge=0)] = 0
