# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .contracts import StageContract, validate_stage_input

from .stage_registry import STAGE_REGISTRY


def run_stage_plan(
    df: pd.DataFrame, stage_plan: List[Tuple[str, Dict[str, Any]]]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    validate_stage_input(df, StageContract(name="pipeline-entry"))
    current_df = df
    stage_info: Dict[str, Any] = {}

    for stage_name, kwargs in stage_plan:
        handler = STAGE_REGISTRY[stage_name]
        current_df, info = handler(current_df, **kwargs)
        stage_info[stage_name] = info

    return current_df, stage_info
