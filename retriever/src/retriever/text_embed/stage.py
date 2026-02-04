from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

from retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_to_lancedb

logger = logging.getLogger(__name__)


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


@traceable_func(trace_name="retriever::text_embedding")
def embed_text_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    transform_config: TextEmbeddingSchema,
    task_config: Optional[Dict[str, Any]] = None,
    lancedb: Optional[LanceDBConfig] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate embeddings for supported content types and write to metadata."""
    _validate_primitives_df(df_primitives)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = transform_create_text_embeddings_internal(
            df_primitives,
            task_config=task_config,
            transform_config=transform_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Text embedding failed")
        raise

    if lancedb is not None:
        try:
            write_embeddings_to_lancedb(out_df, cfg=lancedb)
        except Exception:
            logger.exception("Failed writing embeddings to LanceDB")
            raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info
