# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra
from pydantic import Field

logger = logging.getLogger(__name__)


# Define schemas for request validation
class PushRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    timeout: Optional[float] = 100  # Optional timeout for blocking push

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments


class PopRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)
    timeout: Optional[float] = 100  # Optional timeout for blocking pop

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments


class SizeRequestSchema(BaseModel):
    command: str
    queue_name: str = Field(..., min_length=1)

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments
