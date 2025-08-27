# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
from pydantic import BaseModel, Field

from nv_ingest.framework.schemas.broker_client_configs import (
    RedisClientConfig,
    SimpleClientConfig,
)


class RayMessageBrokerTaskSourceConfig(BaseModel):
    """
    Configuration for the Ray MessageBrokerTaskSourceStage.

    Attributes
    ----------
    broker_client : Union[RedisClientConfig, SimpleClientConfig]
        Configuration parameters for connecting to the message broker.
        The specific schema is determined by the 'client_type' field.
    task_queue : str
        The name of the queue to fetch tasks from.
    poll_interval : float, optional
        The polling interval (in seconds) for fetching messages. Defaults to 0.1.
    """

    broker_client: Union[RedisClientConfig, SimpleClientConfig] = Field(..., discriminator="client_type")
    task_queue: str = Field(..., description="The name of the queue to fetch tasks from.")
    poll_interval: float = Field(default=0.1, gt=0, description="Polling interval in seconds.")
