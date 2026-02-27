# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .stage_registry import STAGE_REGISTRY

StagePlan = List[Tuple[str, Dict[str, Any]]]


def build_stage_plan(stage_names: Sequence[str], *, stage_kwargs: Dict[str, Dict[str, Any]] | None = None) -> StagePlan:
    stage_kwargs = stage_kwargs or {}
    plan: StagePlan = []
    for stage_name in stage_names:
        if stage_name not in STAGE_REGISTRY:
            raise ValueError(f"Unknown stage '{stage_name}'. Registered stages: {sorted(STAGE_REGISTRY)}")
        plan.append((stage_name, dict(stage_kwargs.get(stage_name, {}))))
    return plan


def stage_names_from_flags(
    *,
    extract_infographics: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    embed_text: bool = False,
) -> Iterable[str]:
    if extract_infographics:
        yield "enrich_infographic"
    if extract_tables:
        yield "enrich_table"
    if extract_charts:
        yield "enrich_chart"
    if embed_text:
        yield "embed_text"
