# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from nv_ingest.framework.schemas.broker_client_configs import SimpleClientConfig


def _default_simple_client() -> SimpleClientConfig:
    # Preserve previous defaults used by the Python stage for convenience
    return SimpleClientConfig(host="0.0.0.0", port=7671)


class PythonMessageBrokerTaskSourceConfig(BaseModel):
    """Configuration for the PythonMessageBrokerTaskSource."""

    broker_client: SimpleClientConfig = Field(default_factory=_default_simple_client)
    task_queue: str = Field(..., description="The name of the queue to fetch tasks from.")
    poll_interval: float = Field(default=0.1, gt=0, description="Polling interval in seconds.")
