"""Compatibility package exposing retriever modules under nemo_retriever."""

from __future__ import annotations

import retriever as _retriever
from retriever import *  # noqa: F401,F403

__all__ = getattr(_retriever, "__all__", [])

# Reuse the original package path so `nemo_retriever.<submodule>` resolves
# without duplicating the full package tree.
__path__ = _retriever.__path__
