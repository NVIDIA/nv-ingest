# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone text embedding helper for retriever-local pandas DataFrames.

Goal:
- Mirror (as closely as practical) the batching/runner logic from
  `nv_ingest_api.internal.transform.embed_text.transform_create_text_embeddings_internal`,
  but adapt it to the **retriever-local** DataFrame structure used by
  `retriever.text_embed.text_embed.embed_text_1b_v2`.

Key differences vs the API transform:
- This module operates on a simple pandas.DataFrame that typically contains:
  - `text`: the text to embed (or other common text columns)
  - `metadata`: optional dict; if present, embeddings are written to `metadata["embedding"]`
- No imports from other files in this repo. Only stdlib + external deps (pandas/httpx).

Usage:

```python
import pandas as pd
from retriever.text_embed.main_text_embed import create_text_embeddings_for_df

# df must have a `text` column (recommended) and may have `metadata` dicts.
df = pd.DataFrame([{"text": "hello", "metadata": {"source_path": "/tmp/a.pdf"}}])

# Option A: local callable (recommended for retriever inprocess)
def local_embedder(texts):
    # return list[list[float]] matching len(texts)
    return [[0.0, 1.0] for _ in texts]

out_df, _info = create_text_embeddings_for_df(
    df,
    task_config={"embedder": local_embedder, "endpoint_url": None, "local_batch_size": 64},
)

# Embedding is written to out_df.loc[i, "metadata"]["embedding"] and _contains_embeddings is set.
```
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse  # noqa: F401

import pandas as pd

logger = logging.getLogger(__name__)

# Keep HTTP client logging quiet by default (parity with API transform).
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

EmbeddingCallable = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(slots=True)
class TextEmbeddingConfig:
    """
    Minimal config surface mirroring the API's TextEmbeddingSchema fields used by the transform.
    """

    # Remote / NIM-like settings
    api_key: Optional[str] = None
    embedding_nim_endpoint: Optional[str] = None  # e.g. "http://host:8000/v1"
    embedding_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    encoding_format: str = "float"  # OpenAI-compatible embeddings often accept "float"
    input_type: str = "passage"
    truncate: str = "END"
    batch_size: int = 128  # remote batch size
    dimensions: Optional[int] = None

    # Retriever-local dataframe settings
    text_column: str = "text"
    write_embedding_to_metadata: bool = True
    metadata_column: str = "metadata"
    # Optional extra output column containing a payload dict (similar to embed_text_1b_v2)
    output_payload_column: Optional[str] = None


# ------------------------------------------------------------------------------
# Batch Processing Utilities (copied from API transform with minimal edits)
# ------------------------------------------------------------------------------


def _batch_generator(iterable: Iterable[Any], batch_size: int = 10) -> Iterable[List[Any]]:
    """
    Yield list batches from any iterable.

    The API transform assumes sized/sliceable inputs; for robustness we also accept
    generators/iterators by materializing them once.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # If we can't take len() / slices, materialize.
    if not hasattr(iterable, "__len__") or not hasattr(iterable, "__getitem__"):
        iterable = list(iterable)

    seq = iterable  # now sized + sliceable
    iter_len = len(seq)  # type: ignore[arg-type]
    for idx in range(0, iter_len, batch_size):
        yield list(seq[idx : min(idx + batch_size, iter_len)])  # type: ignore[index]


def _generate_batches(prompts: Iterable[str], batch_size: int = 100) -> List[List[str]]:
    """
    Split prompts into concrete list batches.
    """
    return [batch for batch in _batch_generator(prompts, batch_size)]


# ------------------------------------------------------------------------------
# Content extraction for retriever-local DataFrames
# ------------------------------------------------------------------------------


def _text_from_row(row: pd.Series, *, text_column: str) -> Optional[str]:
    """
    Extract text from a row with small fallbacks (mirrors `retriever.text_embed.text_embed`).
    """
    v = row.get(text_column)
    if isinstance(v, str) and v.strip():
        return v

    for k in ("text", "content", "chunk", "page_text"):
        v2 = row.get(k)
        if isinstance(v2, str) and v2.strip():
            return v2

    return None


def _ensure_metadata_dict(row: pd.Series, *, metadata_column: str = "metadata") -> Dict[str, Any]:
    md = row.get(metadata_column)
    if isinstance(md, dict):
        return md
    return {}


# ------------------------------------------------------------------------------
# Remote embeddings (OpenAI-compatible HTTP) + Local callable runner
# ------------------------------------------------------------------------------


def _normalize_embeddings_endpoint(endpoint_url: str) -> str:
    """
    Normalize endpoint to a concrete embeddings URL.

    Accepts:
    - "http://host:8000/v1"
    - "http://host:8000/v1/embeddings"
    - "http://host:8000/embeddings"
    """
    s = (endpoint_url or "").strip().rstrip("/")
    if not s:
        raise ValueError("endpoint_url is empty")
    if s.endswith("/embeddings"):
        return s
    return f"{s}/embeddings"


def _http_embed_openai_compat(
    prompts: List[str],
    *,
    api_key: Optional[str],
    endpoint_url: str,
    model_name: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    dimensions: Optional[int] = None,
    timeout_s: float = 120.0,
) -> List[Optional[List[float]]]:
    """
    Best-effort HTTP embeddings call using an OpenAI-compatible schema.

    Expected response:
      {"data": [{"index": 0, "embedding": [...]}, ...]}
    """
    try:
        import httpx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Remote embedding requested but `httpx` is not installed.") from e

    url = _normalize_embeddings_endpoint(endpoint_url)
    headers: Dict[str, str] = {"accept": "application/json", "content-type": "application/json"}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Mimic the OpenAI Python client's `extra_body={...}` behavior by including
    # vendor-specific fields at the top-level JSON body.
    payload: Dict[str, Any] = {
        "model": model_name,
        "input": prompts,
        "encoding_format": encoding_format,
        "input_type": str(input_type),
        "truncate": str(truncate),
    }
    if dimensions is not None:
        payload["dimensions"] = int(dimensions)

    with httpx.Client(timeout=float(timeout_s)) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Parse embeddings.
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Unexpected embeddings response (missing 'data' list).")

    by_index: Dict[int, Optional[List[float]]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = it.get("index")
        emb = it.get("embedding")
        if isinstance(idx, int) and isinstance(emb, list):
            by_index[int(idx)] = emb

    # Preserve input order; unknown entries become None.
    return [by_index.get(i) for i in range(len(prompts))]


def _make_async_request(
    prompts: List[str],
    api_key: Optional[str],
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[str]] = None,
    dimensions: Optional[int] = None,
) -> dict:
    """
    Mirrors the API transform's request wrapper, but uses HTTP OpenAI-compatible embeddings.

    Notes:
    - `input_type` and `truncate` are sent as top-level JSON fields, matching the effective
      request body produced by the OpenAI Python client when using `extra_body={...}`.
    """
    _ = (filter_errors, modalities)  # reserved for parity/future support

    response: Dict[str, Any] = {}
    try:
        vecs = _http_embed_openai_compat(
            prompts,
            api_key=api_key,
            endpoint_url=str(embedding_nim_endpoint),
            model_name=str(embedding_model),
            encoding_format=str(encoding_format),
            input_type=str(input_type),
            truncate=str(truncate),
            dimensions=dimensions,
        )
        response["embedding"] = vecs
        response["info_msg"] = None
    except Exception as err:
        err_str = str(err)
        if len(err_str) > 500:
            err_str = err_str[:200] + "... [truncated] ..." + err_str[-100:]
        raise RuntimeError(f"Embedding error occurred: {err_str}") from err

    return response


def _async_request_handler(
    prompts: List[List[str]],
    api_key: Optional[str],
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[List[str]]] = None,
    dimensions: Optional[int] = None,
) -> List[dict]:
    if modalities is None:
        modalities = [None] * len(prompts)  # type: ignore[assignment]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _make_async_request,
                prompts=prompt_batch,
                api_key=api_key or None,
                embedding_nim_endpoint=str(embedding_nim_endpoint),
                embedding_model=str(embedding_model),
                encoding_format=str(encoding_format),
                input_type=str(input_type),
                truncate=str(truncate),
                filter_errors=bool(filter_errors),
                modalities=modality_batch,  # type: ignore[arg-type]
                dimensions=dimensions,
            )
            for prompt_batch, modality_batch in zip(prompts, modalities)
        ]
        results = [future.result() for future in futures]

    return results


def _async_runner(
    prompts: List[List[str]],
    api_key: Optional[str],
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[List[str]]] = None,
    dimensions: Optional[int] = None,
) -> dict:
    results = _async_request_handler(
        prompts,
        api_key,
        embedding_nim_endpoint,
        embedding_model,
        encoding_format,
        input_type,
        truncate,
        filter_errors,
        modalities=modalities,
        dimensions=dimensions,
    )

    flat_results = {"embeddings": [], "info_msgs": []}
    for batch_dict in results:
        info_msg = batch_dict.get("info_msg")
        for embedding in batch_dict.get("embedding") or []:
            flat_results["embeddings"].append(embedding)
            flat_results["info_msgs"].append(info_msg)

    return flat_results


def _callable_runner(
    prompts: List[List[str]],
    *,
    embedder: EmbeddingCallable,
    batch_size: int,
) -> dict:
    flat_embeddings: List[Optional[Sequence[float]]] = []
    flat_info_msgs: List[Optional[dict]] = []

    for prompt_batch in prompts:
        if not prompt_batch:
            continue
        for i in range(0, len(prompt_batch), max(1, int(batch_size))):
            chunk = prompt_batch[i : i + max(1, int(batch_size))]
            vecs = embedder(chunk)
            vecs_list = list(vecs)
            if len(vecs_list) != len(chunk):
                raise ValueError(
                    "Local embedder returned a mismatched number of embeddings "
                    f"(got={len(vecs_list)} expected={len(chunk)})"
                )
            flat_embeddings.extend(vecs_list)
            flat_info_msgs.extend([None] * len(vecs_list))

    return {"embeddings": flat_embeddings, "info_msgs": flat_info_msgs}


# ------------------------------------------------------------------------------
# Row update helpers (adapted for retriever-local DataFrames)
# ------------------------------------------------------------------------------


def _add_embeddings_retriever_df(
    row: pd.Series,
    embeddings: Dict[Any, Any],
    info_msgs: Dict[Any, Any],
    *,
    metadata_column: str,
    write_embedding_to_metadata: bool,
    output_payload_column: Optional[str],
) -> pd.Series:
    embedding = embeddings.get(row.name, None)
    info_msg = info_msgs.get(row.name, None)

    if write_embedding_to_metadata:
        md = _ensure_metadata_dict(row, metadata_column=metadata_column)
        md["embedding"] = embedding
        if info_msg:
            md["info_message_metadata"] = info_msg
        row[metadata_column] = md

    if output_payload_column:
        row[output_payload_column] = {"embedding": embedding, "info_msg": info_msg}

    row["_contains_embeddings"] = embedding is not None
    return row


# ------------------------------------------------------------------------------
# Public API (mirrors the API transform's surface, but for retriever-local df schema)
# ------------------------------------------------------------------------------


def create_text_embeddings_for_df(
    df_transform_ledger: pd.DataFrame,
    *,
    task_config: Dict[str, Any],
    transform_config: Optional[TextEmbeddingConfig] = None,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create embeddings for a retriever-local DataFrame and write them into row metadata.

    Parameters
    ----------
    df_transform_ledger:
        Input pandas.DataFrame. Recommended columns:
        - `text` (or provide `transform_config.text_column`)
        - `metadata` (optional dict; created if missing when writing embeddings)
    task_config:
        Controls runtime behavior. Keys (compatible with the API transform):
        - **api_key**: optional str
        - **endpoint_url**: optional str; if set, remote HTTP embeddings are used
        - **model_name**: optional str
        - **dimensions**: optional int
        - **embedder**: optional callable(texts)->vectors; used when endpoint_url is empty/None
        - **local_batch_size**: int; used to sub-batch for the callable embedder path
    transform_config:
        Optional TextEmbeddingConfig; if omitted, defaults are used.
    execution_trace_log:
        Optional dict to populate with trace info.

    Returns
    -------
    (out_df, info_dict)
    """
    if transform_config is None:
        transform_config = TextEmbeddingConfig()

    # Allow task_config to explicitly override values with None by checking key presence (API parity).
    api_key = task_config["api_key"] if "api_key" in task_config else transform_config.api_key
    endpoint_url = (
        task_config["endpoint_url"] if "endpoint_url" in task_config else transform_config.embedding_nim_endpoint
    )
    model_name = task_config["model_name"] if "model_name" in task_config else transform_config.embedding_model
    dimensions = task_config["dimensions"] if "dimensions" in task_config else transform_config.dimensions

    endpoint_url = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
    if isinstance(endpoint_url, str) and not endpoint_url:
        endpoint_url = None

    embedder: Optional[EmbeddingCallable] = task_config.get("embedder")
    local_batch_size = int(task_config.get("local_batch_size") or 4)

    if execution_trace_log is None:
        execution_trace_log = {}

    if df_transform_ledger.empty:
        return df_transform_ledger, {"trace_info": execution_trace_log}

    # Extract content and normalize empty or non-str to None (adapted for retriever-local schema).
    extracted_content = df_transform_ledger.apply(
        lambda r: _text_from_row(r, text_column=str(transform_config.text_column)), axis=1
    ).apply(lambda x: x.strip() if isinstance(x, str) and x.strip() else None)

    df_content = df_transform_ledger.copy()
    df_content["_content"] = extracted_content

    valid_content_mask = df_content["_content"].notna()
    if valid_content_mask.any():
        filtered_content_list = df_content.loc[valid_content_mask, "_content"].tolist()
        filtered_content_batches = _generate_batches(filtered_content_list, batch_size=int(transform_config.batch_size))

        if endpoint_url:
            # NOTE: this implementation uses HTTP embeddings with an OpenAI-compatible schema.
            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                str(endpoint_url),
                str(model_name),
                str(transform_config.encoding_format),
                str(transform_config.input_type),
                str(transform_config.truncate),
                False,
                modalities=None,
                dimensions=dimensions,
            )
        elif callable(embedder):
            content_embeddings = _callable_runner(
                filtered_content_batches,
                embedder=embedder,
                batch_size=local_batch_size,
            )
        else:
            raise ValueError(
                "No embedding endpoint configured (endpoint_url/embedding_nim_endpoint are empty) "
                "and no local embedder was provided in task_config['embedder']."
            )

        # Build a simple row index -> embedding map (API parity).
        embeddings_dict = dict(zip(df_content.loc[valid_content_mask].index, content_embeddings.get("embeddings", [])))
        info_msgs_dict = dict(zip(df_content.loc[valid_content_mask].index, content_embeddings.get("info_msgs", [])))
    else:
        embeddings_dict = {}
        info_msgs_dict = {}

    df_content = df_content.apply(
        _add_embeddings_retriever_df,
        embeddings=embeddings_dict,
        info_msgs=info_msgs_dict,
        metadata_column=str(transform_config.metadata_column),
        write_embedding_to_metadata=bool(transform_config.write_embedding_to_metadata),
        output_payload_column=transform_config.output_payload_column,
        axis=1,
    )

    # Drop helper column to keep the output clean.
    if "_content" in df_content.columns:
        df_content = df_content.drop(columns=["_content"])

    return df_content, {"trace_info": execution_trace_log}
