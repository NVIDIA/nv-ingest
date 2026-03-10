# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_VL_EMBED_MODEL_IDS = frozenset(
    {
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "llama-nemotron-embed-vl-1b-v2",
    }
)

# Short name → full HF repo ID.
_EMBED_MODEL_ALIASES: dict[str, str] = {
    "nemo_retriever_v1": "nvidia/llama-nemotron-embed-1b-v2",
    "llama-nemotron-embed-vl-1b-v2": "nvidia/llama-nemotron-embed-vl-1b-v2",
}

_DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

# Reranker model aliases and default.
_RERANK_MODEL_ALIASES: dict[str, str] = {
    "llama-nemotron-rerank-1b-v2": "nvidia/llama-nemotron-rerank-1b-v2",
}

_DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"


def resolve_embed_model(model_name: str | None) -> str:
    """Resolve a model name/alias to a full HF repo ID.

    Returns ``_DEFAULT_EMBED_MODEL`` when *model_name* is ``None`` or empty.
    """
    if not model_name:
        return _DEFAULT_EMBED_MODEL
    return _EMBED_MODEL_ALIASES.get(model_name, model_name)


def is_vl_embed_model(model_name: str | None) -> bool:
    """Return True if *model_name* refers to the VL embedding model."""
    return resolve_embed_model(model_name) in _VL_EMBED_MODEL_IDS


def create_local_embedder(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    normalize: bool = True,
    max_length: int = 8192,
):
    """Create the appropriate local embedding model (VL or non-VL).

    Centralises the resolve -> branch -> construct pattern that was previously
    duplicated across batch, inprocess, fused, gpu_pool, recall, retriever,
    and text_embed code paths.
    """
    model_id = resolve_embed_model(model_name)

    if is_vl_embed_model(model_name):
        from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
            LlamaNemotronEmbedVL1BV2Embedder,
        )

        return LlamaNemotronEmbedVL1BV2Embedder(
            device=device,
            hf_cache_dir=hf_cache_dir,
            model_id=model_id,
        )

    from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import (
        LlamaNemotronEmbed1BV2Embedder,
    )

    return LlamaNemotronEmbed1BV2Embedder(
        device=device,
        hf_cache_dir=hf_cache_dir,
        normalize=normalize,
        max_length=max_length,
        model_id=model_id,
    )


def resolve_rerank_model(model_name: str | None) -> str:
    """Resolve a reranker model name/alias to a full HF repo ID.

    Returns ``_DEFAULT_RERANK_MODEL`` when *model_name* is ``None`` or empty.
    """
    if not model_name:
        return _DEFAULT_RERANK_MODEL
    return _RERANK_MODEL_ALIASES.get(model_name, model_name)


def create_local_reranker(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    max_length: int = 8192,
):
    """Create a local reranker model instance.

    Centralises the resolve -> construct pattern so callers don't need to
    know about the underlying class.
    """
    model_id = resolve_rerank_model(model_name)

    from nemo_retriever.model.local.llama_nemotron_reranker_1b_v2 import (
        LlamaNemotronReranker1BV2,
    )

    return LlamaNemotronReranker1BV2(
        device=device,
        hf_cache_dir=hf_cache_dir,
        max_length=max_length,
        model_id=model_id,
    )
