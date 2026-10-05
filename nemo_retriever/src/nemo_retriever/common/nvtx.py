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

import nvtx as _nvtx

_P = ParamSpec("_P")
_R = TypeVar("_R")


def batch_phase(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a function with a stable NVTX range for a batch phase.

    Uses the standalone NVTX package without importing PyTorch or synchronizing
    CUDA. Exceptions raised by the decorated function propagate after the range
    is closed.

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
            with _nvtx.annotate(range_name, color="blue"):
                return function(*args, **kwargs)

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
    with _nvtx.annotate("gpu_inference", color="blue"), _nvtx.annotate(label, color="blue"):
        yield
