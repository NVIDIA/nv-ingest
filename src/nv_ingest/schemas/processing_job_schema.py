# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
