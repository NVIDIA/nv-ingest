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
from enum import Enum
from typing import List

# Define request schema
class OpenAIRequest(BaseModel):
    # model: str
    # messages: List[dict]  # [{"role": "user", "content": "question"}]
    # max_tokens: int = 256
    # temperature: float = 0.7
    # top_p: float = 1.0
    query: str
    k: int

# OpenAI-compatible response schema
class OpenAIResponse(BaseModel):
    # id: str
    # object: str
    # created: int
    # model: str
    # choices: List[dict]  # [{"message": {"role": "assistant", "content": "answer"}}]
    content: List[str]
