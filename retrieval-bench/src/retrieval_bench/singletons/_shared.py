# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities used across singleton retrievers."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def hash_corpus_ids10(corpus_ids: Sequence[str]) -> str:
    """Return a 10-char hex hash of corpus IDs for cache keying."""
    h = hashlib.sha256()
    for cid in corpus_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:10]


def slugify(value: str) -> str:
    """Make a filesystem-friendly string from a model/dataset name."""
    v = (value or "").strip().replace("/", "__")
    return v or "unnamed"


def try_preload_corpus_to_gpu(corpus_embeddings_cpu: "torch.Tensor", device: str) -> "Optional[torch.Tensor]":
    """
    Attempt to move corpus embeddings to GPU; return None on OOM.

    Handles different OOM exception types across PyTorch versions.
    """
    try:
        return corpus_embeddings_cpu.to(device, non_blocking=True)
    except Exception as e:
        oom_types = tuple(
            t
            for t in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )

        is_oom = False
        if oom_types and isinstance(e, oom_types):
            is_oom = True
        elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            is_oom = True

        if not is_oom:
            raise

        logger.debug("OOM preloading corpus to GPU; falling back to CPU scoring: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
