# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated NVTX import retained for compatibility with existing callers.

Library inference code uses ``nvtx.annotate`` directly. This module preserves
``gpu_inference_range`` during its deprecation period.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import warnings

import nvtx


@contextmanager
def gpu_inference_range(model_name: str, batch_size: int = -1, **extra: object) -> Iterator[None]:
    """Annotate inference with the legacy capture marker and dynamic label.

    Deprecated: use nested ``nvtx.annotate`` contexts directly. The outer
    label is ``gpu_inference``; the inner label contains ``model_name``, an
    optional nonnegative batch size, and any extra key/value pairs in order.
    Exceptions from the enclosed workload propagate unchanged.
    """
    warnings.warn(
        "nemo_retriever.common.nvtx.gpu_inference_range is deprecated; use nvtx.annotate directly.",
        DeprecationWarning,
        stacklevel=3,
    )
    parts = [model_name]
    if batch_size >= 0:
        parts.append(f"bs={batch_size}")
    parts.extend(f"{key}={value}" for key, value in extra.items())
    with nvtx.annotate("gpu_inference", color="blue"), nvtx.annotate(" | ".join(parts), color="blue"):
        yield
