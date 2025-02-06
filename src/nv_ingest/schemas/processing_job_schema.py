# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict
from enum import Enum


class ConversionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"

    model_config = ConfigDict(extra="forbid")


class ProcessingJob(BaseModel):
    submitted_job_id: str
    filename: str
    raw_result: str = ""
    content: str = ""
    status: ConversionStatus
    error: str | None = None

    model_config = ConfigDict(extra="forbid")
