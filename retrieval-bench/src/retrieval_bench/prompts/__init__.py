# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prompt registries and small prompt-building helpers.
"""

from .bright_instructions import BRIGHT_SHORT_INSTRUCTIONS, get_task_description  # noqa: F401

__all__ = [
    "BRIGHT_SHORT_INSTRUCTIONS",
    "get_task_description",
]
