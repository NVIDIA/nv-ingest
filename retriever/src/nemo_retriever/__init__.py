# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retriever application package."""

from __future__ import annotations

__all__ = ["__version__", "create_ingestor", "get_version", "get_version_info"]


def __getattr__(name: str):
    if name == "create_ingestor":
        from .api import create_ingestor

        return create_ingestor
    if name in {"__version__", "get_version", "get_version_info"}:
        from .version import __version__, get_version, get_version_info

        return {
            "__version__": __version__,
            "get_version": get_version,
            "get_version_info": get_version_info,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
