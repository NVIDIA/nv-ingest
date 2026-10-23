# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding runtime helpers shared by graph-based example pipelines."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import pandas as pd

from nemo_retriever.common.params.models import IMAGE_MODALITIES
from nemo_retriever.common.schemas.embedding import (
    embedding_runtime_modality,
    format_embedding_input,
    requires_text_admission,
)
from nemo_retriever.models import _DEFAULT_EMBED_MODEL, resolve_embed_model
from nemo_retriever.models.inference.embedding_input import (
    EmbeddingInputPolicy,
    prepare_embedding_inputs,
)
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    create_text_embeddings_for_df,
)
from nemo_retriever.models.nim.error_reporter import report_error

logger = logging.getLogger(__name__)


def _embed_group(
    group_df: pd.DataFrame,
    *,
    group_modality: str,
    model: Any,
    endpoint: Optional[str],
    api_key: Optional[str],
    text_column: str,
    inference_batch_size: int,
    remote_image_batch_size: int = 4,
    output_column: str,
    resolved_model_name: str,
    embed_model_provider_prefix: Optional[str] = None,
    nim_http_max_concurrent: int = 32,
    input_type: str = "passage",
    request_timeout_s: float = 600.0,
    truncate: str = "END",
) -> pd.DataFrame:
    """Embed a single modality group via ``create_text_embeddings_for_df``."""
    embedder = None
    multimodal_embedder = None

    if endpoint is None and model is not None:
        if group_modality in IMAGE_MODALITIES:
            multimodal_embedder = model
        else:
            skip_prefix = hasattr(model, "embed_queries")

            def embedder(texts: Sequence[str]) -> Sequence[Sequence[float]]:  # noqa
                if str(input_type).strip().lower() == "query" and skip_prefix:
                    vectors = model.embed_queries(texts, batch_size=int(inference_batch_size))
                else:
                    batch = texts if skip_prefix else [format_embedding_input(text, "passage: ") for text in texts]
                    vectors = model.embed(batch, batch_size=int(inference_batch_size))
                tolist = getattr(vectors, "tolist", None)
                if callable(tolist):
                    return tolist()
                return vectors  # type: ignore[return-value]

    default_local_image_batch_size = 8
    effective_batch_size = inference_batch_size
    if group_modality in IMAGE_MODALITIES:
        if endpoint is not None:
            effective_batch_size = min(inference_batch_size, remote_image_batch_size)
        else:
            effective_batch_size = min(inference_batch_size, default_local_image_batch_size)

    cfg = TextEmbeddingConfig(
        text_column=str(text_column),
        output_payload_column=str(output_column) if output_column else None,
        write_embedding_to_metadata=True,
        metadata_column="metadata",
        batch_size=int(effective_batch_size),
        encoding_format="float",
        input_type=str(input_type),
        truncate=str(truncate),
        dimensions=None,
        embedding_nim_endpoint=endpoint or "http://localhost:8012/v1",
        embedding_model=resolved_model_name or _DEFAULT_EMBED_MODEL,
        embedding_model_provider_prefix=embed_model_provider_prefix,
        embed_modality=group_modality,
        nim_http_max_concurrent=max(1, int(nim_http_max_concurrent)),
    )

    out_df, _ = create_text_embeddings_for_df(
        group_df,
        task_config={
            "api_key": api_key,
            "embedder": embedder,
            "multimodal_embedder": multimodal_embedder,
            "endpoint_url": endpoint,
            "embed_model_provider_prefix": embed_model_provider_prefix,
            "local_batch_size": int(effective_batch_size),
            "nim_http_max_concurrent": max(1, int(nim_http_max_concurrent)),
            "request_timeout_s": float(request_timeout_s),
        },
        transform_config=cfg,
    )
    from nemo_retriever.models.inference.input_audit import audit_handoff

    audit_handoff(out_df, output_column=output_column, text_column=text_column)
    return out_df


def embed_text_main_text_embed(
    batch_df: Any,
    *,
    model: Any = None,
    model_name: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    embed_invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    text_column: str = "text",
    inference_batch_size: int = 16,
    embed_inference_batch_size: int = 4,
    output_column: str = "text_embeddings_1b_v2",
    embedding_dim_column: str = "text_embeddings_1b_v2_dim",
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding",
    embed_modality: str = "text",
    embed_model_provider_prefix: Optional[str] = None,
    nim_http_max_concurrent: int = 32,
    input_type: str = "passage",
    request_timeout_s: float | None = None,
    embedding_input_policy: EmbeddingInputPolicy | None = None,
    **_extras: Any,
) -> Any:
    """Embed graph batches while preserving the legacy output columns."""
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_main_text_embed currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")
    if embed_inference_batch_size <= 0:
        raise ValueError("embed_inference_batch_size must be > 0")

    if request_timeout_s is None:
        request_timeout_s = float(_extras.get("request_timeout_s", 600.0))

    endpoint = (embedding_endpoint or embed_invoke_url or "").strip() or None
    if endpoint is None and model is None:
        raise ValueError("Either a local model or an embedding_endpoint must be provided.")

    resolved_model_name = resolve_embed_model(model_name)

    multimodal_planner = None
    if endpoint is None and callable(getattr(type(model), "plan_text_image", None)):
        from functools import partial

        multimodal_planner = partial(model.plan_text_image, query=str(input_type).strip().lower() == "query")
    if embedding_input_policy is not None or multimodal_planner is not None:
        preparation = prepare_embedding_inputs(
            batch_df,
            policy=embedding_input_policy,
            text_column=text_column,
            default_modality=embed_modality,
            multimodal_planner=multimodal_planner,
        )
        prepared_df = preparation.frame
        current_split_child_positions = preparation.split_child_positions
        current_split_parent_positions = preparation.split_parent_positions
    else:
        prepared_df = batch_df
        current_split_child_positions = frozenset()
        current_split_parent_positions = frozenset()
    input_row_count = len(batch_df.index)

    remote_text_admission = endpoint is not None and embedding_input_policy is not None
    original_index = prepared_df.index
    if remote_text_admission:
        # Partition and reassemble remote requests by row position, not caller
        # index labels (which may be unordered or repeated).
        prepared_df = prepared_df.reset_index(drop=True)

    request_routes = prepared_df.apply(
        lambda row: (
            embedding_runtime_modality(row, default_modality=embed_modality),
            (
                "NONE"
                if remote_text_admission and requires_text_admission(row, default_modality=embed_modality)
                else "END"
            ),
        ),
        axis=1,
    )
    routes = request_routes.unique().tolist() or [
        (embedding_runtime_modality({}, default_modality=embed_modality), "END")
    ]

    try:
        parts: List[pd.DataFrame] = []
        for modality, truncate in routes:
            group_df = prepared_df if len(routes) == 1 else prepared_df.loc[request_routes == (modality, truncate)]
            part = _embed_group(
                group_df,
                group_modality=modality,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                text_column=text_column,
                inference_batch_size=inference_batch_size,
                remote_image_batch_size=embed_inference_batch_size,
                output_column=output_column,
                resolved_model_name=resolved_model_name,
                embed_model_provider_prefix=embed_model_provider_prefix,
                nim_http_max_concurrent=nim_http_max_concurrent,
                input_type=input_type,
                request_timeout_s=float(request_timeout_s),
                truncate=truncate,
            )
            parts.append(part)
        out_df = parts[0] if len(routes) == 1 else pd.concat(parts).sort_index()
    except Exception as exc:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as _cache_exc:  # noqa: BLE001
            logger.debug("torch.cuda.empty_cache() failed during error cleanup: %s", _cache_exc)
        logger.error("Embedding failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        report_error("embed", exc)
        out_df = prepared_df.copy()
        public_error = f"{type(exc).__name__}: embedding batch failed; inspect embed-stage logs for the cause"
        out_df[output_column] = [{"embedding": [], "error": public_error} for _ in out_df.index]
        out_df[embedding_dim_column] = 0
        out_df[has_embedding_column] = False
        if "_embed_modality" in out_df.columns:
            out_df = out_df.drop(columns=["_embed_modality"])

    if remote_text_admission:
        out_df.index = original_index

    if embedding_dim_column:

        def dim(row: pd.Series) -> int:
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                embedding = metadata.get("embedding")
                if isinstance(embedding, list):
                    return int(len(embedding))
            payload = row.get(output_column) if output_column else None
            if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
                return int(len(payload.get("embedding") or []))
            return 0

        out_df[embedding_dim_column] = out_df.apply(dim, axis=1)
    else:
        out_df[embedding_dim_column] = [0 for _ in range(len(out_df.index))]

    out_df[has_embedding_column] = [bool(int(d) > 0) for d in out_df[embedding_dim_column].tolist()]

    embedded_flags = out_df[has_embedding_column].tolist()
    out_df["embedding_v1_num_detections"] = [int(f) for f in embedded_flags]
    counts_by_label: list[dict[str, int]] = []
    for position, (_, row) in enumerate(out_df.iterrows()):
        embedded = bool(row.get(has_embedding_column))
        counts = {"embedded": 1} if embedded else {"unembedded": 1}
        payload = row.get(output_column)
        if isinstance(payload, dict) and payload.get("error") is not None:
            counts["failed"] = 1
        if position in current_split_child_positions:
            counts["split_child"] = 1
            if position in current_split_parent_positions:
                counts["overlength"] = 1
                counts["split"] = 1
        counts_by_label.append(counts)
    out_df["embedding_v1_counts_by_label"] = counts_by_label

    totals = {
        label: sum(int(counts.get(label, 0)) for counts in counts_by_label)
        for label in (
            "overlength",
            "split",
            "split_child",
            "failed",
            "embedded",
            "unembedded",
        )
    }
    summary = (
        f"Embedding summary: input_rows={input_row_count} output_rows={len(out_df.index)} "
        f"overlength={totals['overlength']} split={totals['split']} truncated=0 "
        f"failed={totals['failed']} embedded={totals['embedded']} unembedded={totals['unembedded']} "
        f"split_children={totals['split_child']}"
    )
    if totals["failed"] or totals["unembedded"] or totals["overlength"]:
        logger.warning(summary)

    if "_embed_modality" in out_df.columns:
        # Internal embedding router column; StoreOperator consumes _image_b64.
        out_df = out_df.drop(columns=["_embed_modality"])

    return out_df
