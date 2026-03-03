# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

from nemo_retriever.io.dataframe import validate_primitives_dataframe
from nemo_retriever.vector_store.lancedb_store import LanceDBConfig, write_embeddings_to_lancedb

logger = logging.getLogger(__name__)


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
    _ = trace_info
    validate_primitives_dataframe(df_primitives)

    if task_config is None:
        task_config = {}
    else:
        # Avoid mutating the caller's dict (we may inject a local embedder callable).
        task_config = dict(task_config)

    # Auto-fallback: if no embedding endpoint is configured, inject a local HF embedder callable.
    maybe_inject_local_hf_embedder(task_config, transform_config)

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


def maybe_inject_local_hf_embedder(task_config: Dict[str, Any], transform_config: TextEmbeddingSchema) -> None:
    """
    If no remote embedding endpoint is configured, inject a local HF embedder into task_config.

    This keeps the DataFrame embedding logic centralized in `nv_ingest_api.internal.transform.embed_text`
    while allowing retriever-local runs to operate without an embedding microservice.
    """
    # Respect explicit caller-provided embedder.
    if callable(task_config.get("embedder")):
        return

    # Resolve endpoint_url with explicit None override support.
    if "endpoint_url" in task_config:
        endpoint_url = task_config.get("endpoint_url")
    else:
        endpoint_url = getattr(transform_config, "embedding_nim_endpoint", None)

    endpoint_url = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
    has_endpoint = bool(endpoint_url)

    use_local = bool(task_config.get("use_local_hf_if_no_endpoint", True))
    if has_endpoint or not use_local:
        return

    # Lazy import: only load torch/HF when we truly need local embeddings.
    from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

    local_device = task_config.get("local_hf_device")
    local_cache_dir = task_config.get("local_hf_cache_dir")
    local_batch_size = int(task_config.get("local_hf_batch_size") or 64)

    embedder = LlamaNemotronEmbed1BV2Embedder(device=local_device, hf_cache_dir=local_cache_dir, normalize=True)

    def _embed(texts):
        prefix = f"{transform_config.input_type}: " if getattr(transform_config, "input_type", None) else ""
        vecs = embedder.embed([prefix + t for t in texts], batch_size=local_batch_size)
        return vecs.tolist()

    # Force the API transform to use the callable path by explicitly overriding endpoint_url to None.
    task_config["endpoint_url"] = None
    task_config["embedder"] = _embed
    task_config["local_batch_size"] = local_batch_size
