# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

from pydantic import ConfigDict, BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


# Define schemas for request validation
class PushRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    timeout: Optional[float] = 100  # Optional timeout for blocking push
    model_config = ConfigDict(extra="forbid")


class PopRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)
    timeout: Optional[float] = 100  # Optional timeout for blocking pop
    model_config = ConfigDict(extra="forbid")


class SizeRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")
