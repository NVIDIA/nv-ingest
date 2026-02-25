# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .executor import run_mode_ingest


def run_online(**kwargs: Any) -> Any:
    return run_mode_ingest(run_mode="online", **kwargs)
