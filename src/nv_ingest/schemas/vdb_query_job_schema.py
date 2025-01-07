# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pydantic import BaseModel
from typing import List


# Define request schema
class OpenAIRequest(BaseModel):
    query: str
    k: int
    job_id: str


# OpenAI-compatible response schema
class OpenAIResponse(BaseModel):
    content: List[str]
