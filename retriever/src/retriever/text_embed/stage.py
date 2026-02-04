from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

from retriever.metrics import Metrics, safe_metrics
from retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_to_lancedb

logger = logging.getLogger(__name__)


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


def _count_non_null_embeddings(df: pd.DataFrame) -> int:
    def _has_embedding(meta: Any) -> bool:
        return isinstance(meta, dict) and meta.get("embedding") is not None

    try:
        return int(df["metadata"].apply(_has_embedding).sum())
    except Exception:
        return 0


@traceable_func(trace_name="retriever::text_embedding")
def embed_text_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    transform_config: TextEmbeddingSchema,
    task_config: Optional[Dict[str, Any]] = None,
    lancedb: Optional[LanceDBConfig] = None,
    metrics: Optional[Metrics] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate embeddings for supported content types and write to metadata."""
    metrics = safe_metrics(metrics)
    _validate_primitives_df(df_primitives)

    attrs = {"stage": "text_embedder"}
    metrics.inc("embed.rows_in", int(len(df_primitives)), attrs=attrs)
    before = _count_non_null_embeddings(df_primitives)
    metrics.inc("embed.rows_with_embedding_before", before, attrs=attrs)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        with metrics.timeit("embed.seconds", attrs=attrs):
            out_df, info = transform_create_text_embeddings_internal(
                df_primitives,
                task_config=task_config,
                transform_config=transform_config,
                execution_trace_log=execution_trace_log,
            )
    except Exception:
        metrics.inc("embed.failures", 1, attrs=attrs)
        logger.exception("Text embedding failed")
        raise

    after = _count_non_null_embeddings(out_df)
    metrics.inc("embed.rows_with_embedding_after", after, attrs=attrs)
    metrics.inc("embed.rows_out", int(len(out_df)), attrs=attrs)

    if lancedb is not None:
        try:
            with metrics.timeit("embed.lancedb_write.seconds", attrs=attrs):
                written = write_embeddings_to_lancedb(out_df, cfg=lancedb)
            metrics.inc("embed.lancedb_rows_written", int(written), attrs=attrs)
        except Exception:
            metrics.inc("embed.lancedb_write_failures", 1, attrs=attrs)
            logger.exception("Failed writing embeddings to LanceDB")
            raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info

