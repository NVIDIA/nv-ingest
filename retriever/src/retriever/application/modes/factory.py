# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

RunMode = Literal["inprocess", "batch", "fused", "online"]


def create_runmode_ingestor(*, run_mode: RunMode = "inprocess", **kwargs: Any):
    if run_mode == "inprocess":
        from retriever.ingest_modes.inprocess import InProcessIngestor

        return InProcessIngestor(**kwargs)
    if run_mode == "batch":
        from retriever.ingest_modes.batch import BatchIngestor

        return BatchIngestor(**kwargs)
    if run_mode == "fused":
        from retriever.ingest_modes.fused import FusedIngestor

        return FusedIngestor(**kwargs)
    if run_mode == "online":
        from retriever.ingest_modes.online import OnlineIngestor

        return OnlineIngestor(**kwargs)
    raise ValueError(f"Unknown run_mode: {run_mode!r}")
