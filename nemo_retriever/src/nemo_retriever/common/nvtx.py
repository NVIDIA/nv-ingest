# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight NVTX helpers for GPU inference and batch profiling.

Usage::

    from nemo_retriever.common.nvtx import gpu_inference_range

    with gpu_inference_range("NemotronOCRv1", batch_size=8):
        result = self._model(input_data)

Use :func:`batch_phase` to add a stable, low-cardinality range around a
batch-level function. Batch ranges use the ``nrl.batch::`` prefix.

When ``nsys`` is launched with ``--capture-range=nvtx --nvtx-capture=gpu_inference``,
only the code inside these blocks is captured.  When no profiler is attached the
overhead is near-zero (a pair of C-level push/pop calls).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import cache, wraps
from types import ModuleType
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


@cache
def _get_nvtx() -> ModuleType | None:
    """Resolve the optional backend only when an instrumented function runs."""
    try:
        import torch.cuda.nvtx as nvtx
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "torch.cuda", "torch.cuda.nvtx"}:
            raise
        return None
    return nvtx


def batch_phase(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a function with a stable NVTX range for a batch phase.

    The range is a no-op when PyTorch or its NVTX support is unavailable. A
    CPU-only PyTorch NVTX stub is also treated as unavailable. Unexpected NVTX
    errors propagate to the caller; exceptions raised by the decorated function
    propagate after the range is closed.

    Args:
        label: Low-cardinality phase name appended to the ``nrl.batch::`` prefix.

    Returns:
        A decorator that wraps a function with the named NVTX range.

    Raises:
        RuntimeError: If the NVTX backend raises an unexpected runtime error.
    """

    range_name = f"nrl.batch::{label}"

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nvtx = _get_nvtx()
            if nvtx is None:
                return function(*args, **kwargs)
            try:
                nvtx.range_push(range_name)
            except RuntimeError as exc:
                if "NVTX functions not installed" not in str(exc):
                    raise
                return function(*args, **kwargs)
            try:
                return function(*args, **kwargs)
            finally:
                nvtx.range_pop()

        return wrapped

    return decorate


@contextmanager
def gpu_inference_range(model_name: str, batch_size: int = -1, **extra):
    """Push a nested NVTX range pair around GPU inference.

    The outer range is always named ``gpu_inference`` so that
    ``nsys --nvtx-capture=gpu_inference`` can trigger on it.
    The inner range carries the human-readable label visible in
    the Nsight Systems timeline (e.g. ``NemotronOCRv1 | bs=8``).
    """
    nvtx = _get_nvtx()
    if nvtx is None:
        yield
        return
    parts = [model_name]
    if batch_size >= 0:
        parts.append(f"bs={batch_size}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    label = " | ".join(parts)
    nvtx.range_push("gpu_inference")
    nvtx.range_push(label)
    try:
        yield
    finally:
        nvtx.range_pop()
        nvtx.range_pop()
