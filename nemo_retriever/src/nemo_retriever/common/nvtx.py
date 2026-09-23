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
from functools import wraps
from typing import ParamSpec, TypeVar

import torch.cuda.nvtx as _nvtx


_P = ParamSpec("_P")
_R = TypeVar("_R")


def batch_phase(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Mark a semantic batch boundary without synchronizing CUDA.

    The range is a no-op when PyTorch was built without NVTX support.
    """

    range_name = f"nrl.batch::{label}"

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                _nvtx.range_push(range_name)
            except RuntimeError as exc:
                if "NVTX functions not installed" not in str(exc):
                    raise
                return function(*args, **kwargs)
            try:
                return function(*args, **kwargs)
            finally:
                _nvtx.range_pop()

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
    parts = [model_name]
    if batch_size >= 0:
        parts.append(f"bs={batch_size}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    label = " | ".join(parts)
    _nvtx.range_push("gpu_inference")
    _nvtx.range_push(label)
    try:
        yield
    finally:
        _nvtx.range_pop()
        _nvtx.range_pop()
