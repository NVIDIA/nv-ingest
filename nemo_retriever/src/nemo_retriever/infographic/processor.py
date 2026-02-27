# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema

from nemo_retriever.io.dataframe import validate_primitives_dataframe

logger = logging.getLogger(__name__)


@traceable_func(trace_name="retriever::infographic_extraction")
def extract_infographic_data_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    extractor_config: InfographicExtractorSchema,
    task_config: Optional[Dict[str, Any]] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich infographic primitives in-place by running OCR and writing table_content."""
    _ = trace_info
    validate_primitives_dataframe(df_primitives)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = extract_infographic_data_from_image_internal(
            df_extraction_ledger=df_primitives,
            task_config=task_config,
            extraction_config=extractor_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Infographic extraction failed")
        raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info
