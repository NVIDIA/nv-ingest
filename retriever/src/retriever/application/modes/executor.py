# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .factory import RunMode, create_runmode_ingestor


def run_mode_ingest(*, run_mode: RunMode, create_kwargs: dict[str, Any] | None = None, **ingest_kwargs: Any) -> Any:
    ingestor = create_runmode_ingestor(run_mode=run_mode, **(create_kwargs or {}))
    return ingestor.ingest(**ingest_kwargs)
