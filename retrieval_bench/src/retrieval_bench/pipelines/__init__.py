# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Built-in retrieval pipelines for vidore-benchmark evaluation.
"""

from retrieval_bench.pipelines.backends import VALID_BACKENDS, init_backend
from retrieval_bench.pipelines.dense import DenseRetrievalPipeline
from retrieval_bench.pipelines.agentic import AgenticRetrievalPipeline

__all__ = [
    "DenseRetrievalPipeline",
    "AgenticRetrievalPipeline",
    "VALID_BACKENDS",
    "init_backend",
]
