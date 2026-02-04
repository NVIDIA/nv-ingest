from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema

from .stage import extract_pdf_primitives_from_ledger_df

logger = logging.getLogger(__name__)


def extract_pdf_primitives_ray_data(
    ds: "ray.data.Dataset",  # quoted to avoid import at module load
    *,
    task_config: Dict[str, Any],
    extractor_config: PDFExtractorSchema,
    batch_size: int = 16,
) -> "ray.data.Dataset":
    """Ray Data adapter around the shared `nv-ingest-api` PDF extraction logic.

    The input dataset must provide the same ledger columns as the pandas path:
    ["content", "source_id", "source_name", "document_type", "metadata"].

    The output dataset rows correspond to extracted primitives with columns:
    ["document_type", "metadata", "uuid"].
    """
    # Import lazily so the pure-python path doesn't require Ray to be installed/initialized.
    import ray.data  # type: ignore

    def _map_batch(batch: pd.DataFrame) -> pd.DataFrame:
        extracted_df, _info = extract_pdf_primitives_from_ledger_df(
            batch,
            task_config=task_config,
            extractor_config=extractor_config,
        )
        return extracted_df

    logger.info(
        "Running Ray Data PDF extraction: batch_size=%s, method=%s",
        batch_size,
        task_config.get("method"),
    )

    return ds.map_batches(
        _map_batch,
        batch_format="pandas",
        batch_size=batch_size,
    )
