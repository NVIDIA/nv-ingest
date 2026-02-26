# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple


def endpoints_from_yaml(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Normalize YAML endpoint values into the tuple shape used by nv-ingest-api schemas."""
    if value is None:
        return (None, None)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        a, b = value[0], value[1]
        return (a or None, b or None)
    raise ValueError(f"Expected endpoints as [grpc, http] (len=2), got: {value!r}")
