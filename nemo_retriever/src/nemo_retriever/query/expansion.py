# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regroup losslessly expanded query rows without changing unsplit rankings."""

from __future__ import annotations

import json
from numbers import Integral
from typing import Any

QUERY_INDEX_COLUMN = "_nrl_query_index"


def _hit_key(hit: dict[str, Any]) -> str:
    if hit.get("chunk_id"):
        identity = {key: hit.get(key) for key in ("source_id", "document_id", "document_version", "chunk_id")}
    else:
        identity = {key: value for key, value in hit.items() if key not in {"_distance", "_score", "_relevance_score"}}
    return json.dumps(identity, sort_keys=True, default=str)


def regroup_query_hits(
    hits: list[list[dict[str, Any]]], query_indices: list[int], *, query_count: int, top_k: int
) -> list[list[dict[str, Any]]]:
    """Fuse chunk ranks with standard RRF (k=60), retaining one group per input."""
    if len(hits) != len(query_indices):
        raise ValueError("Query retrieval returned a different number of result groups than embedded rows")
    groups: list[list[list[dict[str, Any]]]] = [[] for _ in range(query_count)]
    for index, chunk_hits in zip(query_indices, hits):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < query_count:
            raise ValueError("Expanded query row has an invalid original query index")
        groups[index].append(chunk_hits)
    if any(not group for group in groups):
        raise ValueError("Query embedding output is missing an original query")

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0][:top_k])
            continue
        scores: dict[str, float] = {}
        first_hits: dict[str, dict[str, Any]] = {}
        for chunk_hits in group:
            seen = set()
            for rank, hit in enumerate(chunk_hits, start=1):
                key = _hit_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                first_hits.setdefault(key, hit)
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
        # Python's stable sort keeps first encounter order for score ties.
        ordered = sorted(first_hits, key=lambda key: -scores[key])
        result.append([{**first_hits[key], "_relevance_score": scores[key]} for key in ordered[:top_k]])
    return result
