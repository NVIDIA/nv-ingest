# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for :class:`~nemo_retriever.retriever.Retriever` graph execution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.query.expansion import QUERY_INDEX_COLUMN

_RESERVED_RETRIEVE_KWARGS = frozenset({"query_texts", "refine_factor"})


def filter_retrieval_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keys reserved for graph coordination (not forwarded to ``VDB.retrieval``)."""
    return {k: v for k, v in kwargs.items() if k not in _RESERVED_RETRIEVE_KWARGS}


def hits_lists_to_rerank_dataframe(
    query_texts: list[str],
    hits_per_query: list[list[dict[str, Any]]],
) -> pd.DataFrame:
    """One row per (query, hit) with payload to rebuild hits after reranking.

    Returns a DataFrame with the columns ``query``, ``text``, ``_hit`` even when
    there are no hits — ``pd.DataFrame([])`` yields a column-less DataFrame,
    which crashes the downstream rerank actor with ``KeyError: 'query'`` on a
    legitimate empty-results path (empty/unmatched corpus, freshly-ingested
    table, etc.).
    """
    rows: list[dict[str, Any]] = []
    for index, (q, hits) in enumerate(zip(query_texts, hits_per_query)):
        for h in hits:
            rows.append({"query": q, "text": str(h.get("text", "")), "_hit": dict(h), QUERY_INDEX_COLUMN: index})
    if not rows:
        return pd.DataFrame(columns=["query", "text", "_hit", QUERY_INDEX_COLUMN])
    return pd.DataFrame(rows)


def rerank_long_dataframe_to_hits(
    df: pd.DataFrame,
    *,
    query_texts: list[str],
    top_k: int,
    score_column: str = "rerank_score",
) -> list[list[dict[str, Any]]]:
    """Group long rerank output by query (preserving *query_texts* order), take top_k per query."""
    if df.empty:
        return [[] for _ in query_texts]
    if score_column not in df.columns:
        raise ValueError(f"Rerank output missing score column {score_column!r}; columns={list(df.columns)}")

    work = df.copy()
    # Per-query score ordering (global sort in NemotronRerank is disabled for multi-query batches).
    has_query_indices = QUERY_INDEX_COLUMN in work.columns
    work["_q_order"] = (
        work[QUERY_INDEX_COLUMN]
        if has_query_indices
        else work["query"].map({q: i for i, q in enumerate(query_texts)}).fillna(len(query_texts))
    )
    work = work.sort_values(["_q_order", score_column], ascending=[True, False]).drop(columns=["_q_order"])

    out: list[list[dict[str, Any]]] = []
    for index, q in enumerate(query_texts):
        sub = work[work[QUERY_INDEX_COLUMN] == index] if has_query_indices else work[work["query"] == q]
        picked: list[dict[str, Any]] = []
        for _, row in sub.head(int(top_k)).iterrows():
            hit = dict(row["_hit"])
            score = row[score_column]
            if isinstance(score, (int, float)):
                hit["_rerank_score"] = float(score)
            picked.append(hit)
        out.append(picked)
    return out
