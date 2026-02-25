# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import pandas as pd


class StageProcessor(Protocol):
    def __call__(self, df_primitives: pd.DataFrame, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]: ...


@dataclass(frozen=True)
class StageContract:
    name: str
    required_input_columns: tuple[str, ...] = ("metadata",)


DEFAULT_STAGE_CONTRACT = StageContract(name="primitives-stage")


def validate_stage_input(df_primitives: pd.DataFrame, contract: StageContract = DEFAULT_STAGE_CONTRACT) -> None:
    missing = [column for column in contract.required_input_columns if column not in df_primitives.columns]
    if missing:
        raise KeyError(f"Stage '{contract.name}' requires missing columns: {missing}")
