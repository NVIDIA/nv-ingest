from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


@traceable_func(trace_name="retriever::chart_extraction")
def extract_chart_data_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    extractor_config: ChartExtractorSchema,
    task_config: Optional[Dict[str, Any]] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich chart primitives in-place by running YOLOX/OCR and writing chart content."""
    _validate_primitives_df(df_primitives)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = extract_chart_data_from_image_internal(
            df_extraction_ledger=df_primitives,
            task_config=task_config,
            extraction_config=extractor_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Chart extraction failed")
        raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info
