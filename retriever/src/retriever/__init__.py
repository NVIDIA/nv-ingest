# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retriever application package."""

from .api import create_ingestor
from .version import __version__, get_version, get_version_info

__all__ = ["__version__", "create_ingestor", "get_version", "get_version_info"]
