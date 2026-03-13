# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reranking stage using nvidia/llama-nemotron-rerank-1b-v2.

Provides:
  - ``rerank_hits``         – rerank a list of LanceDB hits for a single query
  - ``NemotronRerankActor`` – Ray Data-compatible stateful actor for batch DataFrames

Remote endpoint
---------------
When ``invoke_url`` is set the actor/function calls a vLLM (>=0.14) or NIM
server that exposes the OpenAI-compatible ``/rerank`` REST API::

    POST /rerank
    {
      "model": "nvidia/llama-nemotron-rerank-1b-v2",
      "query": "...",
      "documents": ["...", "..."],
      "top_n": N
    }

Local model
-----------
When no endpoint is configured the model is loaded directly from HuggingFace
(or ``hf_cache_dir``) using ``NemotronRerankV2``.

Ray Data actor usage::

    import ray
    ds = ds.map_batches(
        NemotronRerankActor,
        batch_size=64,
        batch_format="pandas",
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=4),
        fn_constructor_kwargs={
            "model_name": "nvidia/llama-nemotron-rerank-1b-v2",
            "query_column": "query",
            "text_column": "text",
            "score_column": "rerank_score",
            "max_length": 512,
            "batch_size": 32,
        },
    )
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import pandas as pd


_DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
_DEFAULT_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 32
_SCORE_COLUMN = "rerank_score"


# ---------------------------------------------------------------------------
# Remote endpoint helper
# ---------------------------------------------------------------------------


def _rerank_via_endpoint(
    query: str,
    documents: List[str],
    *,
    endpoint: str,
    model_name: str = _DEFAULT_MODEL,
    api_key: str = "",
) -> List[float]:
    """
    Call a vLLM / NIM ``/rerank`` REST endpoint and return per-document scores.

    The server must expose the OpenAI-compatible rerank API introduced in
    vLLM >= 0.14.0::

        POST {endpoint}/rerank
        {"model": ..., "query": ..., "documents": [...], "top_n": N}

    Parameters
    ----------
    query:
        The search query string.
    documents:
        List of document strings to score against the query.
    endpoint:
        Base URL of the reranking endpoint (e.g. ``http://localhost:8015
        ``).  The function will append ``/v1/ranking`` if the URL does not
        already end with ``/reranking``.
    model_name:
        Model identifier sent to the remote endpoint (default
        ``"nvidia/llama-nemotron-rerank-1b-v2"``).
    api_key:
        Bearer token for the remote endpoint (if required).

    Returns
    -------
    List[float]
        Scores aligned with *documents* (higher = more relevant).
        Documents not returned by ``top_n`` truncation receive ``-inf``.
    """
    import requests

    cleaned_endpoint = endpoint.rstrip("/")
    if not cleaned_endpoint.endswith("/reranking"):
        cleaned_endpoint = endpoint.rstrip("/") + "/v1/ranking"
    url = cleaned_endpoint
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    texts = [{"text": d} for d in documents]
    payload = {
        "model": model_name,
        "query": {"text": query},
        "passages": texts,
        "truncate": "END",
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    # Build score list aligned with input document order.
    scores = [float("-inf")] * len(documents)
    for item in data.get("rankings", []):
        idx = item.get("index")
        score = item.get("logit")
        if idx is not None and score is not None:
            scores[idx] = float(score)
    return scores


# ---------------------------------------------------------------------------
# Public helper: rerank LanceDB hits for a single query
# ---------------------------------------------------------------------------


def rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    model: Optional[Any] = None,
    invoke_url: Optional[str] = None,
    model_name: str = _DEFAULT_MODEL,
    api_key: str = "",
    max_length: int = _DEFAULT_MAX_LENGTH,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    top_n: Optional[int] = None,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """
    Rerank *hits* (list of LanceDB result dicts) by relevance to *query*.

    Each hit that has a ``text_key`` field is scored; hits without text are
    placed at the end.  The returned list is sorted highest-score first and
    each dict gains a ``"_rerank_score"`` field.

    Parameters
    ----------
    query:
        The search query.
    hits:
        LanceDB result dicts (as returned by ``Retriever.queries()``).
    model:
        A ``NemotronRerankV2`` instance (local GPU inference).  Ignored when
        *invoke_url* is set.
    invoke_url:
        Base URL of a vLLM / NIM ``/rerank`` endpoint.  Takes priority over
        *model*.
    model_name:
        Model identifier sent to the remote endpoint (default
        ``"nvidia/llama-nemotron-rerank-1b-v2"``).
    api_key:
        Bearer token for the remote endpoint.
    max_length:
        Tokenizer truncation length for local inference (max 8 192).
    batch_size:
        GPU forward-pass batch size for local inference.
    top_n:
        If set, only the top-N results (after reranking) are returned.
    text_key:
        Dict key used to extract document text from each hit (default
        ``"text"``).

    Returns
    -------
    List[dict]
        Hits sorted by ``"_rerank_score"`` descending.  Each dict has a new
        ``"_rerank_score"`` key with the raw logit (local) or relevance score
        (remote).
    """
    if not hits:
        return hits

    documents = [str(h.get(text_key) or "") for h in hits]
    if invoke_url:
        scores = _rerank_via_endpoint(
            query,
            documents,
            endpoint=invoke_url,
            model_name=model_name,
            api_key=api_key,
        )
    elif model is not None:
        scores = model.score(query, documents, max_length=max_length, batch_size=batch_size)
    else:
        raise ValueError("Either 'model' (NemotronRerankV2 instance) or 'invoke_url' must be provided.")

    ranked = sorted(
        [{"_rerank_score": s, **h} for s, h in zip(scores, hits)],
        key=lambda x: x["_rerank_score"],
        reverse=True,
    )

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


# ---------------------------------------------------------------------------
# Error payload helper (mirrors other actors in this project)
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "status": "error",
        "stage": stage,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


# ---------------------------------------------------------------------------
# Ray Data actor
# ---------------------------------------------------------------------------


class NemotronRerankActor:
    """
    Ray Data-compatible stateful actor for cross-encoder reranking.

    Initialises ``nvidia/llama-nemotron-rerank-1b-v2`` **once** per actor
    instance and reuses it across batches, avoiding repeated model loads.

    Each row in the input DataFrame is expected to have a *query* column and a
    *text* (document) column.  The actor appends a ``rerank_score`` column
    (name configurable) with the raw logit score.

    Usage with Ray Data::

        import ray
        ds = ds.map_batches(
            NemotronRerankActor,
            batch_size=64,
            batch_format="pandas",
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=4),
            fn_constructor_kwargs={
                "model_name": "nvidia/llama-nemotron-rerank-1b-v2",
                "query_column": "query",
                "text_column": "text",
                "score_column": "rerank_score",
                "max_length": 512,
                "batch_size": 32,
            },
        )

    Parameters
    ----------
    model_name:
        HuggingFace model ID (default ``"nvidia/llama-nemotron-rerank-1b-v2"``).
    invoke_url:
        Base URL of a vLLM / NIM ``/rerank`` endpoint.  When set the actor
        skips local model creation and delegates all scoring to the endpoint.
        Also accepted as ``rerank_invoke_url``.
    api_key:
        Bearer token for the remote endpoint.
    device:
        Torch device string (default: ``"cuda"`` if available, else ``"cpu"``).
    hf_cache_dir:
        Directory for HuggingFace model cache.
    query_column:
        DataFrame column containing query strings (default ``"query"``).
    text_column:
        DataFrame column containing document/passage text (default ``"text"``).
    score_column:
        Output column name for rerank scores (default ``"rerank_score"``).
    max_length:
        Tokenizer truncation length (default 512).
    batch_size:
        GPU forward-pass micro-batch size (default 32).
    sort_results:
        If ``True`` (default) rows in each batch are sorted by score descending.
    """

    __slots__ = ("_kwargs", "_model")

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)

        invoke_url = str(self._kwargs.get("rerank_invoke_url") or self._kwargs.get("invoke_url") or "").strip()
        if invoke_url and "invoke_url" not in self._kwargs:
            self._kwargs["invoke_url"] = invoke_url

        if invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronRerankV2

            self._model = NemotronRerankV2(
                model_name=str(self._kwargs.get("model_name", _DEFAULT_MODEL)),
                device=self._kwargs.get("device") or None,
                hf_cache_dir=str(self._kwargs["hf_cache_dir"]) if self._kwargs.get("hf_cache_dir") else None,
            )

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return _rerank_batch(batch_df, model=self._model, **self._kwargs, **override_kwargs)
        except BaseException as exc:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=exc)
                score_col = str(self._kwargs.get("score_column", _SCORE_COLUMN))
                out[score_col] = [payload for _ in range(len(out.index))]
                return out
            return [{"rerank_score": _error_payload(stage="actor_call", exc=exc)}]


# ---------------------------------------------------------------------------
# Batch processing function (called by actor and usable standalone)
# ---------------------------------------------------------------------------


def _rerank_batch(
    batch_df: pd.DataFrame,
    *,
    model: Optional[Any] = None,
    invoke_url: Optional[str] = None,
    model_name: str = _DEFAULT_MODEL,
    api_key: str = "",
    query_column: str = "query",
    text_column: str = "text",
    score_column: str = _SCORE_COLUMN,
    max_length: int = _DEFAULT_MAX_LENGTH,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    sort_results: bool = True,
    **_ignored: Any,
) -> pd.DataFrame:
    """
    Score each (query, document) row in *batch_df* and append *score_column*.

    When *sort_results* is ``True`` the returned DataFrame is sorted by score
    descending within the batch.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(batch_df)}")

    queries = batch_df[query_column].tolist()
    texts = batch_df[text_column].tolist()
    pairs = list(zip(queries, texts))

    if invoke_url:
        # Remote endpoint: score pair-by-pair (each row may have a different query).
        scores: List[float] = []
        for q, d in pairs:
            row_scores = _rerank_via_endpoint(
                q,
                [d],
                endpoint=invoke_url,
                model_name=model_name,
                api_key=api_key,
            )
            scores.append(row_scores[0])
    elif model is not None:
        scores = model.score_pairs(pairs, max_length=max_length, batch_size=batch_size)
    else:
        raise ValueError("Either 'model' or 'invoke_url' must be provided to NemotronRerankActor.")

    out = batch_df.copy()
    out[score_column] = scores

    if sort_results:
        out = out.sort_values(score_column, ascending=False).reset_index(drop=True)

    return out
