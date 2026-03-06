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

from typing import Optional

HF_MODEL_REVISIONS: dict[str, str] = {
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "cefc2394cc541737b7867df197984cf23f05367f",
    "nvidia/parakeet-ctc-1.1b": "a707e818195cb97c8f7da2fc36b221a29f69a5db",
    "nvidia/NVIDIA-Nemotron-Parse-v1.2": "f42c8040b12ee64370922d108778ab655b722c5d",
    "nvidia/llama-nemotron-embed-vl-1b-v2": "859e1f2dac29c56c37a5279cf55f53f3e74efc6b",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "intfloat/e5-large-unsupervised": "15af9288f69a6291f37bfb89b47e71abc747b206",
}


def get_hf_revision(model_id: str) -> Optional[str]:
    """Return the pinned commit SHA for *model_id*, or ``None`` if not registered."""
    return HF_MODEL_REVISIONS.get(model_id)
