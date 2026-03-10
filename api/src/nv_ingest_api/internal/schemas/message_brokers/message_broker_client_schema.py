# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional, Literal, Annotated


class MessageBrokerClientSchema(BaseModel):
    """
    Configuration schema for message broker client connections.
    Supports Redis or simple in-memory clients.
    """

    host: str = Field(default="redis", description="Hostname of the broker service.")

    port: Annotated[int, Field(gt=0, lt=65536)] = Field(
        default=6379, description="Port to connect to. Must be between 1 and 65535."
    )

    client_type: Literal["redis", "simple"] = Field(
        default="redis", description="Type of broker client. Supported values: 'redis', 'simple'."
    )

    broker_params: Optional[dict] = Field(
        default_factory=dict, description="Optional parameters passed to the broker client."
    )

    connection_timeout: Annotated[int, Field(ge=0)] = Field(
        default=300, description="Connection timeout in seconds. Must be >= 0."
    )

    max_backoff: Annotated[int, Field(ge=0)] = Field(
        default=300, description="Maximum backoff time in seconds. Must be >= 0."
    )

    max_retries: Annotated[int, Field(ge=0)] = Field(default=0, description="Maximum number of retries. Must be >= 0.")
