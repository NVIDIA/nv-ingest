# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reranking stage using nvidia/llama-nemotron-rerank-1b-v2.

Exports
-------
NemotronRerankActor
    Ray Data-compatible stateful actor that initialises the cross-encoder once
    per worker and scores (query, document) pairs in batch DataFrames.
rerank_hits
    Convenience function to rerank a list of LanceDB hit dicts for a single
    query string, using either a local ``NemotronRerankV2`` model or a remote
    vLLM / NIM ``/rerank`` endpoint.
"""

from .rerank import NemotronRerankActor, rerank_hits

__all__ = [
    "NemotronRerankActor",
    "rerank_hits",
]
