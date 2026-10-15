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
    """Regroup chunk hits by original input, fusing split rankings with RRF.

    Parameters
    ----------
    hits : list[list[dict[str, Any]]]
        One ranked hit list per embedded query row, in the same order as
        ``query_indices``. Use an empty list for a row with no matches.
    query_indices : list[int]
        Original input position for each embedded row, as carried by
        ``_nrl_query_index``. Each entry must be a non-boolean integer in
        ``range(query_count)``. Split rows share their original input's index;
        identical query strings at different input positions have distinct
        indices. Rows can arrive in any order, but every original input must
        have at least one row.
    query_count : int
        Number of original input queries before embedding expansion.
    top_k : int
        Maximum number of hits to retain for each original input.

    Returns
    -------
    list[list[dict[str, Any]]]
        One hit list per original input, in input order, including empty lists
        for inputs with no matches. A single row retains its ranking and scores,
        truncated to ``top_k``. Multiple rows use equal-weight reciprocal rank
        fusion (RRF) with ``k=60`` and one-based ranks. A hit contributes at most
        once per row. Fused hit copies receive their summed RRF score in
        ``_relevance_score``; score ties retain first encounter order. Hits with
        a nonempty ``chunk_id`` are identified by source, document, version,
        and chunk IDs; otherwise, all fields except retrieval scores identify
        a hit. Input hit dictionaries are not modified.

    Raises
    ------
    ValueError
        If the result group count differs from the embedded row count, an
        original query index is invalid, or an original input has no row.
        Align ``hits`` with ``query_indices``, preserve each row's original
        integer index, and retain at least one row per original input. Set
        ``query_count`` to the number of original inputs, not expanded rows.
    """
    if len(hits) != len(query_indices):
        raise ValueError(
            "Query retrieval returned a different number of result groups than embedded rows; "
            "provide one hits entry per embedded query row in the same order as query_indices, "
            "using an empty list for a row with no matches"
        )
    groups: list[list[list[dict[str, Any]]]] = [[] for _ in range(query_count)]
    for index, chunk_hits in zip(query_indices, hits):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < query_count:
            raise ValueError(
                "Expanded query row has an invalid original query index; "
                "set each query_indices entry to a non-boolean integer in range(query_count) "
                "using its original input position, and set query_count to the original input count"
            )
        groups[index].append(chunk_hits)
    if any(not group for group in groups):
        raise ValueError(
            "Query embedding output is missing an original query; preserve at least one embedded row "
            "for every original input and set query_count to the original input count, not the expanded row count"
        )

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
