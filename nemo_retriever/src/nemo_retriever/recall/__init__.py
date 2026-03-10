# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recall evaluation utilities and CLI.
"""

from .__main__ import app
from .core import RecallConfig, evaluate_recall

__all__ = [
    "app",
    "RecallConfig",
    "evaluate_recall",
]
