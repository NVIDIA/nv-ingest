from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

from retriever.metrics import Metrics, safe_metrics

logger = logging.getLogger(__name__)


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


def _is_infographic_candidate(meta: Any) -> bool:
    if not isinstance(meta, dict):
        return False
    content_md = meta.get("content_metadata", {}) or {}
    return (
        content_md.get("type") == "structured"
        and content_md.get("subtype") == "infographic"
        and meta.get("table_metadata") is not None
        and meta.get("content") not in (None, "")
    )


@traceable_func(trace_name="retriever::infographic_extraction")
def extract_infographic_data_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    extractor_config: InfographicExtractorSchema,
    task_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Metrics] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich infographic primitives in-place by running OCR and writing table_content."""
    metrics = safe_metrics(metrics)
    _validate_primitives_df(df_primitives)

    attrs = {"stage": "infographic_extractor"}
    metrics.inc("infographic.rows_in", int(len(df_primitives)), attrs=attrs)

    try:
        candidates = int(df_primitives["metadata"].apply(_is_infographic_candidate).sum())
    except Exception:
        candidates = 0
    metrics.inc("infographic.candidates", candidates, attrs=attrs)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        with metrics.timeit("infographic.extract.seconds", attrs=attrs):
            out_df, info = extract_infographic_data_from_image_internal(
                df_extraction_ledger=df_primitives,
                task_config=task_config,
                extraction_config=extractor_config,
                execution_trace_log=execution_trace_log,
            )
    except Exception:
        metrics.inc("infographic.failures", 1, attrs=attrs)
        logger.exception("Infographic extraction failed")
        raise

    metrics.inc("infographic.rows_out", int(len(out_df)), attrs=attrs)
    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info

