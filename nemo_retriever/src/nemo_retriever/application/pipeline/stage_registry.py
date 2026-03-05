# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd

from nemo_retriever.chart.processor import extract_chart_data_from_primitives_df
from nemo_retriever.infographic.processor import extract_infographic_data_from_primitives_df
from nemo_retriever.table.processor import extract_table_data_from_primitives_df
from nemo_retriever.table.table_detection import table_structure_ocr_page_elements
from nemo_retriever.text_embed.processor import embed_text_from_primitives_df

StageHandler = Callable[..., Any]


def _enrich_table_structure(df: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Thin wrapper so ``table_structure_ocr_page_elements`` matches the ``(df, info)`` contract."""
    out_df = table_structure_ocr_page_elements(df, **kwargs)
    return out_df, {}


# Registry can be extended by future package integrations without changing
# pipeline orchestration internals.
STAGE_REGISTRY: Dict[str, StageHandler] = {
    "enrich_infographic": extract_infographic_data_from_primitives_df,
    "enrich_table": extract_table_data_from_primitives_df,
    "enrich_table_structure": _enrich_table_structure,
    "enrich_chart": extract_chart_data_from_primitives_df,
    "embed_text": embed_text_from_primitives_df,
}
