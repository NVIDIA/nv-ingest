# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of pinned HuggingFace model revisions.

Every ``from_pretrained`` call in the codebase should pass
``revision=get_hf_revision(model_id)`` so that we always download an
exact, immutable snapshot rather than tracking the mutable ``main``
branch.

To bump a model version, update the corresponding SHA in
``HF_MODEL_REVISIONS`` and re-test.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

HF_MODEL_REVISIONS: dict[str, str] = {
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "cefc2394cc541737b7867df197984cf23f05367f",
    "nvidia/llama-nemotron-embed-1b-v2": "cefc2394cc541737b7867df197984cf23f05367f",
    "nvidia/parakeet-ctc-1.1b": "a707e818195cb97c8f7da2fc36b221a29f69a5db",
    "nvidia/NVIDIA-Nemotron-Parse-v1.2": "f42c8040b12ee64370922d108778ab655b722c5d",
    "nvidia/llama-nemotron-embed-vl-1b-v2": "859e1f2dac29c56c37a5279cf55f53f3e74efc6b",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "intfloat/e5-large-unsupervised": "15af9288f69a6291f37bfb89b47e71abc747b206",
    "nvidia/llama-nemotron-rerank-1b-v2": "aee9a1be0bbd89489f8bd0ec5763614c8bb85878",
}


def get_hf_revision(model_id: str, *, strict: bool = True) -> str:
    """Return the pinned commit SHA for *model_id*.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier (e.g. ``"nvidia/parakeet-ctc-1.1b"``).
    strict:
        When ``True`` (the default), raise ``ValueError`` if *model_id* has
        no pinned revision.  When ``False``, log a warning and return
        ``None`` so that ``from_pretrained`` falls back to the ``main``
        branch.
    """
    revision = HF_MODEL_REVISIONS.get(model_id)
    if revision is not None:
        return revision

    msg = (
        f"No pinned HuggingFace revision for model '{model_id}'. "
        "Add an entry to HF_MODEL_REVISIONS in hf_model_registry.py to pin it."
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg + " Falling back to the default (main) branch.")
    return None  # type: ignore[return-value]
