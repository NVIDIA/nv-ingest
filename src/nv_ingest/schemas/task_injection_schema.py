# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskInjectionSchema(BaseModel):
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
