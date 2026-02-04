from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

from retriever.metrics import Metrics, safe_metrics

logger = logging.getLogger(__name__)


def make_pdf_task_config(
    *,
    method: str = "pdfium",
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    extract_page_as_image: bool = False,
    text_depth: str = "page",
    extract_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the `task_config` dict expected by `nv-ingest-api` PDF extraction internals."""
    params: Dict[str, Any] = {
        "extract_text": extract_text,
        "extract_images": extract_images,
        "extract_tables": extract_tables,
        "extract_charts": extract_charts,
        "extract_infographics": extract_infographics,
        "extract_page_as_image": extract_page_as_image,
        "text_depth": text_depth,
    }
    if extract_method is not None:
        # Some callsites use params["extract_method"] while others use task_config["method"].
        params["extract_method"] = extract_method

    return {"method": method, "params": params}


def _validate_pdf_ledger_df(df: pd.DataFrame) -> None:
    required = {"content", "source_id", "source_name", "document_type", "metadata"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"PDF ledger DataFrame is missing required columns: {missing}")


@traceable_func(trace_name="retriever::pdf_extraction")
def extract_pdf_primitives_from_ledger_df(
    df_ledger: pd.DataFrame,
    *,
    task_config: Dict[str, Any],
    extractor_config: PDFExtractorSchema,
    metrics: Optional[Metrics] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pure-Python PDF extraction (no Ray required).

    Returns:
      - extracted_df: columns ["document_type", "metadata", "uuid"]
      - info: includes `execution_trace_log`
    """
    metrics = safe_metrics(metrics)
    _validate_pdf_ledger_df(df_ledger)

    attrs = {"stage": "pdf_extraction", "method": str(task_config.get("method", ""))}
    metrics.inc("pdf.ledger.rows_in", int(len(df_ledger)), attrs=attrs)

    # `nv-ingest-api` engines treat this as a string-keyed mapping of trace timestamps.
    # The Ray pipeline stage uses `{}` as well; using a list can trigger
    # "list indices must be integers or slices, not str".
    execution_trace_log: Dict[str, Any] = {}
    try:
        with metrics.timeit("pdf.extract.seconds", attrs=attrs):
            extracted_df, info = extract_primitives_from_pdf_internal(
                df_extraction_ledger=df_ledger,
                task_config=task_config,
                extractor_config=extractor_config,
                execution_trace_log=execution_trace_log,
            )
    except Exception:
        metrics.inc("pdf.extract.failures", 1, attrs=attrs)
        logger.exception("PDF extraction failed")
        raise

    metrics.inc("pdf.primitives.rows_out", int(len(extracted_df)), attrs=attrs)
    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return extracted_df, info

