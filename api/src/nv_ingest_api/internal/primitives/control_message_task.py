# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, Union


class ControlMessageTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    id: Union[str, UUID]
    properties: Dict[str, Any] = Field(default_factory=dict)
