# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict

from retriever.chart.processor import extract_chart_data_from_primitives_df
from retriever.infographic.processor import extract_infographic_data_from_primitives_df
from retriever.table.processor import extract_table_data_from_primitives_df
from retriever.text_embed.processor import embed_text_from_primitives_df

StageHandler = Callable[..., Any]

# Registry can be extended by future package integrations without changing
# pipeline orchestration internals.
STAGE_REGISTRY: Dict[str, StageHandler] = {
    "enrich_infographic": extract_infographic_data_from_primitives_df,
    "enrich_table": extract_table_data_from_primitives_df,
    "enrich_chart": extract_chart_data_from_primitives_df,
    "embed_text": embed_text_from_primitives_df,
}
