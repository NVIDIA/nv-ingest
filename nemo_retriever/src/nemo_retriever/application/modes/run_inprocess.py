# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams

from .executor import run_mode_ingest


def run_inprocess(
    *,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    return run_mode_ingest(run_mode="inprocess", create_params=create_params, ingest_params=ingest_params)
