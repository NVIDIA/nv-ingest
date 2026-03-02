# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode

from .factory import create_runmode_ingestor


def run_mode_ingest(
    *,
    run_mode: RunMode,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    ingestor = create_runmode_ingestor(run_mode=run_mode, params=create_params)
    return ingestor.ingest(params=ingest_params)
