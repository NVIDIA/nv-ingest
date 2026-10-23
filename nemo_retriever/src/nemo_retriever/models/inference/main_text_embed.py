# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Text embedding helper for NeMo Retriever pandas DataFrames.

This module owns the batching and runner logic used by the graph pipeline. It
operates on a pandas.DataFrame that typically contains:
  - `text`: the text to embed (or other common text columns)
  - `metadata`: optional dict; embedding output is written to the `embedding` column
It uses the shared HTTP embedding URL normalization helpers with pandas/httpx.

Usage:

```python
import pandas as pd
from nemo_retriever.models.inference.main_text_embed import create_text_embeddings_for_df

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

# Embeddings are written to out_df["embedding"].
```
"""

from __future__ import annotations

import atexit
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from nemo_retriever.common.api.util.string_processing import (
    ensure_openai_embeddings_http_url,
    prepend_model_provider_prefix,
)
from nemo_retriever.common.params.models import IMAGE_MODALITIES
from nemo_retriever.common.schemas.embedding import embedding_text_input, select_embedding_text
from nemo_retriever.models import _DEFAULT_EMBED_MODEL

logger = logging.getLogger(__name__)

# Keep HTTP client logging quiet by default.
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

EmbeddingCallable = Callable[[Sequence[str]], Sequence[Sequence[float]]]

_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_HTTP_MAX_RETRIES = 5
_DEFAULT_HTTP_MAX_429_RETRIES = 3
_MAX_HTTP_RETRY_DELAY_S = 30.0
_MAX_HTTP_ERROR_BODY_CHARS = 2_000

EMBEDDING_DTYPE = pd.ArrowDtype(pa.list_(pa.float32()))
EMBEDDING_ERROR_DTYPE = pd.ArrowDtype(pa.string())


def embedding_error_message(stage: str, exc: BaseException) -> str:
    """Format an exception as a one-line message for the embedding error column.

    Args:
        stage: Name of the stage where the exception occurred.
        exc: Exception to describe.

    Returns:
        A single-line ``stage: ExceptionType: message`` string, omitting the message when it is empty.
    """
    message = " ".join(str(exc).split())
    return f"{stage}: {type(exc).__name__}: {message}" if message else f"{stage}: {type(exc).__name__}"


def assign_embedding_columns(
    df: pd.DataFrame,
    vectors: Sequence[Any],
    errors: Sequence[Optional[str]],
    *,
    output_column: str,
) -> pd.DataFrame:
    """Return a copy of ``df`` with typed embedding and error columns.

    Args:
        df: Input dataframe to copy.
        vectors: Per-row float sequences, or ``None`` or empty sequences for rows that were not embedded.
        errors: Per-row error messages or ``None``; values are preserved.
        output_column: Name of the vector column; its error column is named ``f"{output_column}_error"``.

    Returns:
        A new dataframe with ``output_column`` as an Arrow ``list<float32>`` column and
        ``f"{output_column}_error"`` as an Arrow string column. Missing and empty vectors are null;
        ``df`` is unchanged.

    Raises:
        ValueError: If ``vectors``, ``errors``, and ``df`` have different lengths.
    """
    if len(vectors) != len(errors) or len(vectors) != len(df):
        raise ValueError("vectors, errors, and dataframe must have the same length")

    mask = [vector is None or len(vector) == 0 for vector in vectors]
    present = [vector for vector, is_missing in zip(vectors, mask) if not is_missing]
    lengths = np.asarray(
        [len(vector) if not is_missing else 0 for vector, is_missing in zip(vectors, mask)], dtype=np.int32
    )
    offsets = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
    values = (
        np.concatenate([np.asarray(vector, dtype=np.float32).reshape(-1) for vector in present])
        if present
        else np.empty(0, dtype=np.float32)
    )

    array = pa.ListArray.from_arrays(
        pa.array(offsets, pa.int32()),
        pa.array(values, pa.float32()),
        mask=pa.array(mask, pa.bool_()),
    )
    vector_series = pd.Series(pd.arrays.ArrowExtensionArray(array), index=df.index)
    error_series = pd.Series(list(errors), index=df.index, dtype=EMBEDDING_ERROR_DTYPE)
    error_column = f"{output_column}_error"
    return df.assign(**{output_column: vector_series, error_column: error_series})


@dataclass(slots=True)
class TextEmbeddingConfig:
    """
    Configuration for DataFrame text embedding.
    """

    # Remote / NIM-like settings
    api_key: Optional[str] = None
    embedding_nim_endpoint: Optional[str] = None  # e.g. "http://host:8000/v1"
    embedding_model: str = _DEFAULT_EMBED_MODEL
    embedding_model_provider_prefix: Optional[str] = None
    encoding_format: str = "float"  # OpenAI-compatible embeddings often accept "float"
    input_type: str = "passage"
    truncate: str = "END"
    batch_size: int = 128  # remote batch size
    dimensions: Optional[int] = None

    # Retriever-local dataframe settings
    text_column: str = "text"
    output_column: str = "embedding"
    # Modality: "text" (default), "image", or "text_image"
    embed_modality: str = "text"
    # Parallel OpenAI-compatible embedding HTTP calls per Ray batch (NIM / vLLM).
    nim_http_max_concurrent: int = 32


# ------------------------------------------------------------------------------
# Batch processing utilities
# ------------------------------------------------------------------------------


def _batch_generator(iterable: Iterable[Any], batch_size: int = 10) -> Iterable[List[Any]]:
    """
    Yield list batches from any iterable.

    Accept generators and iterators by materializing them once.
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
    """Return the canonical selected source text without ordinary-row normalization."""
    selected = select_embedding_text(row, text_column=text_column)
    return selected.content if selected is not None else None


def _text_input_for_embedding(
    row: pd.Series,
    *,
    text_column: str,
    default_modality: str = "text",
) -> Optional[str]:
    """Preserve lossless split children while retaining legacy normalization for ordinary rows."""
    return embedding_text_input(row, text_column=text_column, default_modality=default_modality)


def _image_from_row(row: pd.Series) -> Optional[str]:
    """Extract ``_image_b64`` column value from a row."""
    v = row.get("_image_b64")
    if isinstance(v, str) and v.strip():
        return v
    return None


def _format_image_input_string(image_b64: str, mime: str = "image/png") -> str:
    """Format a base64 image as a data URL string for remote NIM embedding."""
    return f"data:{mime};base64,{image_b64}"


def _format_text_image_pair_input_string(text: str, image_b64: str, mime: str = "image/png") -> str:
    """Combine text and a data URL image for remote NIM text_image embedding."""
    data_url = f"data:{mime};base64,{image_b64}"
    return f"{text}\n{data_url}"


def _multimodal_callable_runner(
    df_slice: pd.DataFrame,
    *,
    embedder: Any,
    batch_size: int,
    embed_modality: str,
    text_column: str = "text",
) -> dict:
    """Run multimodal embedding (image-only or text+image) using a local VL embedder.

    Processes the DataFrame slice in batches, calling
    ``embedder.embed_images()`` or ``embedder.embed_text_image()``
    depending on *embed_modality*.

    For ``text_image`` mode, rows that have text but no image are
    embedded with the text-only ``embedder.embed()`` method as a
    graceful fallback (e.g. pdfium-extracted text without a rendered
    page image).  For ``image`` mode, rows without images get ``None``.

    Returns ``{"embeddings": [...]}``.
    """
    flat_embeddings: List[Optional[Sequence[float]]] = []

    n = len(df_slice)
    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        chunk = df_slice.iloc[start : start + bs]
        size = len(chunk)
        images_b64 = [_image_from_row(chunk.iloc[i]) or "" for i in range(size)]
        texts = [_text_from_row(chunk.iloc[i], text_column=text_column) or "" for i in range(size)]
        text_inputs = [
            _text_input_for_embedding(
                chunk.iloc[i],
                text_column=text_column,
                default_modality=embed_modality,
            )
            or ""
            for i in range(size)
        ]

        if embed_modality == "image":
            vecs = embedder.embed_images(images_b64, batch_size=bs)
            tolist = getattr(vecs, "tolist", None)
            vecs_list = tolist() if callable(tolist) else list(vecs)

            if len(vecs_list) != size:
                raise RuntimeError(f"Image embedder returned {len(vecs_list)} embeddings for {size} inputs")
            flat_embeddings.extend(vecs_list)

        else:  # text_image
            # Split rows into those with images (multimodal) and those
            # without (text-only fallback).
            has_image = [bool(b) for b in images_b64]

            # multimodal subset
            mm_texts = [t for t, h in zip(texts, has_image) if h]
            mm_images = [b for b, h in zip(images_b64, has_image) if h]
            mm_vecs_list: List[Optional[Sequence[float]]] = []
            if mm_images:
                vecs = embedder.embed_text_image(mm_texts, mm_images, batch_size=bs)
                tolist = getattr(vecs, "tolist", None)
                mm_vecs_list = tolist() if callable(tolist) else list(vecs)
                if len(mm_vecs_list) != len(mm_images):
                    raise RuntimeError(
                        f"Multimodal embedder returned {len(mm_vecs_list)} embeddings for {len(mm_images)} inputs"
                    )

            # text-only fallback subset
            fb_texts = [t for t, h in zip(text_inputs, has_image) if not h and t]
            fb_vecs_list: List[Optional[Sequence[float]]] = []
            if fb_texts:
                vecs = embedder.embed(fb_texts, batch_size=bs)
                tolist = getattr(vecs, "tolist", None)
                fb_vecs_list = tolist() if callable(tolist) else list(vecs)
                if len(fb_vecs_list) != len(fb_texts):
                    raise RuntimeError(
                        f"Text embedder returned {len(fb_vecs_list)} embeddings for {len(fb_texts)} inputs"
                    )

            # reassemble in original order
            mm_iter = iter(mm_vecs_list)
            fb_iter = iter(fb_vecs_list)
            for h, t in zip(has_image, text_inputs):
                if h:
                    flat_embeddings.append(next(mm_iter, None))
                elif t:
                    flat_embeddings.append(next(fb_iter, None))
                else:
                    flat_embeddings.append(None)

    return {"embeddings": flat_embeddings}


# ------------------------------------------------------------------------------
# Remote embeddings (OpenAI-compatible HTTP) + Local callable runner
# ------------------------------------------------------------------------------

_embed_rr_idx = 0
_embed_http_client: Optional[Any] = None
_embed_http_executor: Optional[ThreadPoolExecutor] = None
_embed_http_executor_workers: Optional[int] = None
_embed_http_resources_lock = threading.Lock()


def _get_embed_http_client() -> Any:
    """Return the process-local HTTP client reused by a Ray embed actor."""
    global _embed_http_client
    if _embed_http_client is None:
        with _embed_http_resources_lock:
            if _embed_http_client is None:
                try:
                    import httpx  # type: ignore
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError("Remote embedding requested but `httpx` is not installed.") from e
                _embed_http_client = httpx.Client()
    return _embed_http_client


def _get_embed_http_executor(max_workers: Optional[int]) -> ThreadPoolExecutor:
    """Return a process-local executor, replacing it only when concurrency changes."""
    global _embed_http_executor, _embed_http_executor_workers
    if _embed_http_executor is None or _embed_http_executor_workers != max_workers:
        with _embed_http_resources_lock:
            if _embed_http_executor is None or _embed_http_executor_workers != max_workers:
                previous = _embed_http_executor
                _embed_http_executor = ThreadPoolExecutor(max_workers=max_workers)
                _embed_http_executor_workers = max_workers
                if previous is not None:
                    previous.shutdown(wait=True)
    return _embed_http_executor


def _close_embed_http_resources() -> None:
    """Close process-local HTTP resources when the Ray actor exits."""
    global _embed_http_client, _embed_http_executor, _embed_http_executor_workers
    with _embed_http_resources_lock:
        if _embed_http_executor is not None:
            _embed_http_executor.shutdown(wait=True)
            _embed_http_executor = None
            _embed_http_executor_workers = None
        if _embed_http_client is not None:
            _embed_http_client.close()
            _embed_http_client = None


atexit.register(_close_embed_http_resources)


def _pick_embed_endpoint(csv_url: str) -> str:
    """Round-robin select one URL from a comma-separated NIM endpoint string.

    The service config allows comma-separated endpoints for load-balancing
    (e.g. ``http://host1:8012/v1/embeddings,http://host2:8112/v1/embeddings``).
    Each call rotates to the next endpoint.
    """
    global _embed_rr_idx
    urls = [u.strip() for u in csv_url.split(",") if u.strip()]
    if not urls:
        return csv_url
    if len(urls) == 1:
        return urls[0]
    chosen = urls[_embed_rr_idx % len(urls)]
    _embed_rr_idx += 1
    return chosen


def _normalize_embeddings_endpoint(endpoint_url: str) -> str:
    """
    Normalize endpoint to a concrete embeddings URL (delegates to shared `nemo_retriever.api` helper).
    """
    return ensure_openai_embeddings_http_url(endpoint_url)


def _http_embed_openai_compat(
    prompts: List[str],
    *,
    api_key: Optional[str],
    endpoint_url: str,
    model_name: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    modalities: Optional[List[str]] = None,
    model_provider_prefix: Optional[str] = None,
    dimensions: Optional[int] = None,
    timeout_s: float = 600.0,
    max_retries: int = _DEFAULT_HTTP_MAX_RETRIES,
    max_429_retries: int = _DEFAULT_HTTP_MAX_429_RETRIES,
    client: Optional[Any] = None,
) -> List[Optional[List[float]]]:
    """
    Best-effort HTTP embeddings call using an OpenAI-compatible schema.

    Expected response:
      {"data": [{"index": 0, "embedding": [...]}, ...]}
    """
    try:
        import httpx  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Remote embedding requested but `httpx` is not installed.") from e

    url = _normalize_embeddings_endpoint(_pick_embed_endpoint(endpoint_url))
    model_name = prepend_model_provider_prefix(model_name, model_provider_prefix) or model_name
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
    if modalities:
        if len(modalities) != len(prompts):
            raise ValueError("modalities must contain one value per embedding input")
        normalized_modalities = [str(modality) for modality in modalities]
        payload["modality"] = (
            normalized_modalities[0] if len(set(normalized_modalities)) == 1 else normalized_modalities
        )

    owns_client = client is None
    if client is None:
        client = httpx.Client()
    try:
        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                resp = client.post(url, headers=headers, json=payload, timeout=float(timeout_s))
            except httpx.TransportError:
                if attempt >= max(0, int(max_retries)):
                    raise
                delay_s = _http_retry_delay_s(attempt)
                logger.warning(
                    "Embedding request transport failure; retrying in %.1fs (attempt %d/%d)",
                    delay_s,
                    attempt + 1,
                    max(0, int(max_retries)),
                )
                time.sleep(delay_s)
                continue

            status_code = int(resp.status_code)
            retry_limit = max(0, int(max_429_retries if status_code == 429 else max_retries))
            if status_code in _RETRYABLE_HTTP_STATUS_CODES and attempt < retry_limit:
                delay_s = _http_retry_delay_s(attempt, retry_after=resp.headers.get("Retry-After"))
                logger.warning(
                    "Embedding endpoint returned HTTP %d; retrying in %.1fs (attempt %d/%d)",
                    status_code,
                    delay_s,
                    attempt + 1,
                    retry_limit,
                )
                time.sleep(delay_s)
                continue

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = resp.text.strip()
                if len(body) > _MAX_HTTP_ERROR_BODY_CHARS:
                    body = body[:_MAX_HTTP_ERROR_BODY_CHARS] + "... [truncated]"
                detail = f"; response body: {body}" if body else ""
                raise RuntimeError(f"Embedding endpoint returned HTTP {status_code}{detail}") from exc
            data = resp.json()
            break
    finally:
        if owns_client:
            client.close()

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


def _http_retry_delay_s(attempt: int, *, retry_after: Optional[str] = None) -> float:
    """Return a bounded retry delay, honoring numeric or HTTP-date Retry-After values."""
    if retry_after:
        try:
            return min(_MAX_HTTP_RETRY_DELAY_S, max(0.0, float(retry_after)))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                return min(_MAX_HTTP_RETRY_DELAY_S, max(0.0, retry_at.timestamp() - time.time()))
            except (TypeError, ValueError, OverflowError):
                pass
    ceiling_s = min(_MAX_HTTP_RETRY_DELAY_S, float(2 ** max(0, int(attempt))))
    return random.uniform(0.0, ceiling_s)


def _make_async_request(
    prompts: List[str],
    api_key: Optional[str],
    embedding_nim_endpoint: str,
    embedding_model: str,
    embedding_model_provider_prefix: Optional[str],
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[str]] = None,
    dimensions: Optional[int] = None,
    timeout_s: float = 600.0,
    client: Optional[Any] = None,
) -> dict:
    """
    Send an HTTP OpenAI-compatible embedding request.

    Notes:
    - `input_type` and `truncate` are sent as top-level JSON fields, matching the effective
      request body produced by the OpenAI Python client when using `extra_body={...}`.
    """
    _ = filter_errors  # reserved for parity/future support

    response: Dict[str, Any] = {}
    try:
        vecs = _http_embed_openai_compat(
            prompts,
            api_key=api_key,
            endpoint_url=str(embedding_nim_endpoint),
            model_name=str(embedding_model),
            model_provider_prefix=embedding_model_provider_prefix,
            encoding_format=str(encoding_format),
            input_type=str(input_type),
            truncate=str(truncate),
            modalities=modalities,
            dimensions=dimensions,
            timeout_s=timeout_s,
            client=client,
        )
        response["embedding"] = vecs
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
    embedding_model_provider_prefix: Optional[str],
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[List[str]]] = None,
    dimensions: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    timeout_s: float = 600.0,
) -> List[dict]:
    if modalities is None:
        modalities = [None] * len(prompts)  # type: ignore[assignment]

    pool_size = max_concurrent if max_concurrent and max_concurrent > 0 else None
    executor = _get_embed_http_executor(pool_size)
    client = _get_embed_http_client()
    futures = [
        executor.submit(
            _make_async_request,
            prompts=prompt_batch,
            api_key=api_key or None,
            embedding_nim_endpoint=str(embedding_nim_endpoint),
            embedding_model=str(embedding_model),
            embedding_model_provider_prefix=embedding_model_provider_prefix,
            encoding_format=str(encoding_format),
            input_type=str(input_type),
            truncate=str(truncate),
            filter_errors=bool(filter_errors),
            modalities=modality_batch,  # type: ignore[arg-type]
            dimensions=dimensions,
            timeout_s=timeout_s,
            client=client,
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
    embedding_model_provider_prefix: Optional[str],
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
    modalities: Optional[List[List[str]]] = None,
    dimensions: Optional[int] = None,
    max_concurrent: Optional[int] = None,
    timeout_s: float = 600.0,
) -> dict:
    results = _async_request_handler(
        prompts,
        api_key,
        embedding_nim_endpoint,
        embedding_model,
        embedding_model_provider_prefix,
        encoding_format,
        input_type,
        truncate,
        filter_errors,
        modalities=modalities,
        dimensions=dimensions,
        max_concurrent=max_concurrent,
        timeout_s=timeout_s,
    )

    flat_results = {"embeddings": []}
    for batch_dict in results:
        flat_results["embeddings"].extend(batch_dict.get("embedding") or [])

    return flat_results


def _callable_runner(
    prompts: List[List[str]],
    *,
    embedder: EmbeddingCallable,
    batch_size: int,
) -> dict:
    flat_embeddings: List[Optional[Sequence[float]]] = []

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

    return {"embeddings": flat_embeddings}


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def create_text_embeddings_for_df(
    df_transform_ledger: pd.DataFrame,
    *,
    task_config: Dict[str, Any],
    transform_config: Optional[TextEmbeddingConfig] = None,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create embeddings for a retriever-local DataFrame and write them to a typed vector column.

    Parameters
    ----------
    df_transform_ledger:
        Input pandas.DataFrame. Recommended columns:
        - `text` (or provide `transform_config.text_column`)
        - `metadata` (optional dict; created if missing when writing embeddings)
    task_config:
        Controls runtime behavior. Supported keys:
        - **api_key**: optional str
        - **endpoint_url**: optional str; if set, remote HTTP embeddings are used
        - **model_name**: optional str
        - **dimensions**: optional int
        - **embedder**: optional callable(texts)->vectors; used when endpoint_url is empty/None (injected by pipeline
                        or processor, e.g. from LlamaNemotronEmbed1BV2Embedder via create_local_embedder)
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

    # Allow task_config to explicitly override values with None by checking key presence.
    api_key = task_config["api_key"] if "api_key" in task_config else transform_config.api_key
    endpoint_url = (
        task_config["endpoint_url"] if "endpoint_url" in task_config else transform_config.embedding_nim_endpoint
    )
    model_name = task_config["model_name"] if "model_name" in task_config else transform_config.embedding_model
    model_provider_prefix = (
        task_config["model_provider_prefix"]
        if "model_provider_prefix" in task_config
        else task_config.get("embed_model_provider_prefix", transform_config.embedding_model_provider_prefix)
    )
    dimensions = task_config["dimensions"] if "dimensions" in task_config else transform_config.dimensions

    endpoint_url = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
    if isinstance(endpoint_url, str) and not endpoint_url:
        endpoint_url = None

    embedder: Optional[EmbeddingCallable] = task_config.get("embedder")
    local_batch_size = int(task_config.get("local_batch_size") or 4)

    if execution_trace_log is None:
        execution_trace_log = {}

    nim_http_raw = task_config.get("nim_http_max_concurrent")
    if nim_http_raw is None:
        nim_http_raw = getattr(transform_config, "nim_http_max_concurrent", 32)
    nim_http_max_concurrent = max(1, int(nim_http_raw))

    timeout_raw = task_config.get("request_timeout_s")
    if timeout_raw is None:
        timeout_raw = getattr(transform_config, "request_timeout_s", 600.0)
    request_timeout_s = float(timeout_raw)

    if df_transform_ledger.empty:
        return (
            assign_embedding_columns(
                df_transform_ledger,
                [],
                [],
                output_column=transform_config.output_column,
            ),
            {"trace_info": execution_trace_log},
        )

    embed_modality = transform_config.embed_modality
    multimodal_embedder = task_config.get("multimodal_embedder")  # local VL model for image/text_image

    # Extract content and normalize empty or non-str to None (adapted for retriever-local schema).
    if embed_modality == "image":
        # For image-only, valid rows are those with a non-empty _image_b64.
        extracted_content = df_transform_ledger.apply(lambda r: _image_from_row(r), axis=1).apply(
            lambda x: x if isinstance(x, str) and x.strip() else None
        )
    elif embed_modality == "text_image":
        # For text_image, a row is valid if it has either text or image (prefer both).
        def _text_image_content(r: pd.Series) -> Optional[str]:
            text = _text_from_row(r, text_column=str(transform_config.text_column))
            image = _image_from_row(r)
            if text or image:
                return text or "__image_only__"
            return None

        extracted_content = df_transform_ledger.apply(_text_image_content, axis=1)
    else:
        extracted_content = df_transform_ledger.apply(
            lambda r: _text_input_for_embedding(
                r,
                text_column=str(transform_config.text_column),
                default_modality=embed_modality,
            ),
            axis=1,
        )

    df_content = df_transform_ledger.copy()
    df_content["_content"] = extracted_content

    valid_content_mask = df_content["_content"].notna()
    vectors: List[Any] = [None] * len(df_content)
    if valid_content_mask.any():
        if embed_modality in IMAGE_MODALITIES and multimodal_embedder is not None:
            # Local multimodal path: use _multimodal_callable_runner
            content_embeddings = _multimodal_callable_runner(
                df_content.loc[valid_content_mask],
                embedder=multimodal_embedder,
                batch_size=local_batch_size,
                embed_modality=embed_modality,
                text_column=str(transform_config.text_column),
            )
        elif embed_modality in IMAGE_MODALITIES and endpoint_url:
            # Remote NIM path: format content as data URLs
            if embed_modality == "image":
                filtered_content_list = [
                    _format_image_input_string(img_b64)
                    for img_b64 in df_content.loc[valid_content_mask, "_content"].tolist()
                ]
                filtered_modalities = ["image"] * len(filtered_content_list)
            else:  # text_image
                filtered_content_list = []
                filtered_modalities = []
                for _, r in df_content.loc[valid_content_mask].iterrows():
                    text = (
                        _text_input_for_embedding(
                            r,
                            text_column=str(transform_config.text_column),
                            default_modality=embed_modality,
                        )
                        or ""
                    )
                    image = _image_from_row(r) or ""
                    if image and text.strip():
                        filtered_content_list.append(_format_text_image_pair_input_string(text, image))
                        filtered_modalities.append("text_image")
                    elif image:
                        # Image without text — send as image-only to avoid
                        # "Text part must be non-empty for text_image modality" errors.
                        filtered_content_list.append(_format_image_input_string(image))
                        filtered_modalities.append("image")
                    else:
                        filtered_content_list.append(text)
                        filtered_modalities.append("text")
            filtered_content_batches = _generate_batches(
                filtered_content_list, batch_size=int(transform_config.batch_size)
            )
            modality_batches = _generate_batches(filtered_modalities, batch_size=int(transform_config.batch_size))
            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                str(endpoint_url),
                str(model_name),
                model_provider_prefix,
                str(transform_config.encoding_format),
                str(transform_config.input_type),
                str(transform_config.truncate),
                False,
                modalities=modality_batches,
                dimensions=dimensions,
                max_concurrent=nim_http_max_concurrent,
                timeout_s=request_timeout_s,
            )
        else:
            # Text-only path (default)
            filtered_content_list = df_content.loc[valid_content_mask, "_content"].tolist()
            filtered_content_batches = _generate_batches(
                filtered_content_list, batch_size=int(transform_config.batch_size)
            )

            if endpoint_url:
                content_embeddings = _async_runner(
                    filtered_content_batches,
                    api_key,
                    str(endpoint_url),
                    str(model_name),
                    model_provider_prefix,
                    str(transform_config.encoding_format),
                    str(transform_config.input_type),
                    str(transform_config.truncate),
                    False,
                    modalities=[["text"] * len(batch) for batch in filtered_content_batches],
                    dimensions=dimensions,
                    max_concurrent=nim_http_max_concurrent,
                    timeout_s=request_timeout_s,
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

        for position, vector in zip(
            np.flatnonzero(valid_content_mask.to_numpy()),
            content_embeddings.get("embeddings", []),
        ):
            vectors[position] = vector

    # Drop helper column to keep the output clean.
    if "_content" in df_content.columns:
        df_content = df_content.drop(columns=["_content"])

    return (
        assign_embedding_columns(
            df_content,
            vectors,
            [None] * len(df_content),
            output_column=transform_config.output_column,
        ),
        {"trace_info": execution_trace_log},
    )
