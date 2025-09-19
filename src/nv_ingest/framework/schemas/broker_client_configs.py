# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class BrokerParamsRedis(BaseModel):
    """Specific parameters for Redis broker_params."""

    db: int = 0
    use_ssl: bool = False


class BaseBrokerClientConfig(BaseModel):
    """Base configuration common to all broker clients."""

    host: str = Field(..., description="Hostname or IP address of the message broker.")
    port: int = Field(..., description="Port number of the message broker.")
    max_retries: int = Field(default=5, ge=0, description="Maximum number of connection retries.")
    max_backoff: float = Field(default=5.0, gt=0, description="Maximum backoff delay in seconds between retries.")
    connection_timeout: float = Field(default=30.0, gt=0, description="Connection timeout in seconds.")


class RedisClientConfig(BaseBrokerClientConfig):
    """Configuration specific to the Redis client."""

    client_type: Literal["redis"] = Field(..., description="Specifies the client type as Redis.")
    broker_params: BrokerParamsRedis = Field(
        default_factory=BrokerParamsRedis, description="Redis-specific parameters like db and ssl."
    )


class SimpleClientConfig(BaseBrokerClientConfig):
    """Configuration specific to the Simple client."""

    client_type: Literal["simple"] = Field(..., description="Specifies the client type as Simple.")
    broker_params: Optional[Dict[str, Any]] = Field(
        default={}, description="Optional parameters for Simple client (currently unused)."
    )
    interface_type: Literal["auto", "direct", "socket"] = Field(
        default="auto", description="Interface policy: 'auto', 'direct', or 'socket'."
    )
