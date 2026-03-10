# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
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


def validate_table_structure_flags(
    use_table_structure: bool,
    table_output_format: str,
) -> None:
    """Validate the combination of use_table_structure and table_output_format.

    Raises ValueError if table_output_format is 'markdown' but the stage is
    disabled.  Emits a warning when the stage is enabled but the output format
    is 'pseudo_markdown' (the user probably wants 'markdown').
    """
    if not use_table_structure and table_output_format == "markdown":
        raise ValueError(
            "table_output_format='markdown' requires use_table_structure=True. "
            "Either set use_table_structure=True or use table_output_format='pseudo_markdown'."
        )
    if use_table_structure and table_output_format == "pseudo_markdown":
        warnings.warn(
            "use_table_structure is enabled but table_output_format is 'pseudo_markdown'; "
            "consider using table_output_format='markdown' for proper cell/row/column layout.",
            stacklevel=3,
        )


def stage_names_from_flags(
    *,
    extract_infographics: bool = False,
    extract_tables: bool = False,
    use_table_structure: bool = False,
    table_output_format: str = "pseudo_markdown",
    extract_charts: bool = False,
    use_graphic_elements: bool = False,
    embed_text: bool = False,
) -> Iterable[str]:
    validate_table_structure_flags(use_table_structure, table_output_format)
    if extract_infographics:
        yield "enrich_infographic"
    if extract_tables and use_table_structure:
        yield "enrich_table_structure"
    elif extract_tables:
        yield "enrich_table"
    if extract_charts and use_graphic_elements:
        yield "enrich_graphic_elements"
    elif extract_charts:
        yield "enrich_chart"
    if embed_text:
        yield "embed_text"
