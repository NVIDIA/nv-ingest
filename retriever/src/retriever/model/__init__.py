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
    "nemo_retriever_v1": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "llama-nemotron-embed-vl-1b-v2": "nvidia/llama-nemotron-embed-vl-1b-v2",
}

_DEFAULT_EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"


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
