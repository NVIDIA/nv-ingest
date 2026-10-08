# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nvtx
import requests

logger = logging.getLogger(__name__)


def _service_tracing() -> Any | None:
    try:
        from nemo_retriever.service import tracing
    except Exception:
        logger.debug("Service tracing helper unavailable for NIM request", exc_info=True)
        return None
    return tracing


def _capture_trace_context() -> dict[str, str] | None:
    service_tracing = _service_tracing()
    if service_tracing is None:
        return None
    try:
        return dict(service_tracing.inject_trace_context())
    except Exception as exc:
        logger.warning("OpenTelemetry trace context capture failed for NIM request: %s", exc)
        return None


def _extract_trace_context(service_tracing: Any, trace_context: Dict[str, str] | None) -> Any | None:
    if not trace_context:
        return None
    try:
        return service_tracing.extract_trace_context(trace_context)
    except Exception as exc:
        logger.warning("OpenTelemetry trace context extraction failed for NIM request: %s", exc)
        return None


def _set_span_attribute(span: Any, key: str, value: Any) -> None:
    setter = getattr(span, "set_attribute", None)
    if setter is None:
        return
    try:
        setter(key, value)
    except Exception:
        logger.debug("Ignoring NIM tracing span attribute failure", exc_info=True)


def _set_span_error(span: Any, exc_or_status: Any) -> None:
    if isinstance(exc_or_status, int):
        error_type = f"HTTP {exc_or_status}"
    else:
        error_type = type(exc_or_status).__name__
    _set_span_attribute(span, "error.type", error_type)

    setter = getattr(span, "set_status", None)
    if setter is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        setter(Status(StatusCode.ERROR))
    except Exception:
        logger.debug("Ignoring NIM tracing span status failure", exc_info=True)


def _safe_endpoint_attribute(invoke_url: str) -> str:
    parts = urlsplit(str(invoke_url))
    if not parts.scheme or not parts.netloc:
        return str(invoke_url).split("?", 1)[0].split("#", 1)[0]
    netloc = parts.hostname or ""
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _chunk_ranges(total: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]


def _parse_invoke_urls(invoke_url: str) -> List[str]:
    parts = [p.strip() for p in str(invoke_url or "").split(",")]
    urls = [p for p in parts if p]
    if not urls:
        raise ValueError("invoke_url is required")
    return urls


def _normalize_batch_response(response_json: Any, expected_count: int) -> List[Any]:
    """
    Normalize common NIM HTTP response shapes to one item per input image.
    """
    if isinstance(response_json, list):
        items = response_json
    elif isinstance(response_json, dict):
        data = response_json.get("data")
        output = response_json.get("output")
        predictions = response_json.get("predictions")
        if isinstance(data, list):
            items = data
        elif isinstance(output, list):
            items = output
        elif isinstance(predictions, list):
            items = predictions
        else:
            items = [response_json]
    else:
        raise RuntimeError(f"Unexpected response type: {type(response_json)!r}")

    out: List[Any] = []
    for item in items:
        out.append(item)

    if len(out) != int(expected_count):
        raise RuntimeError(f"Remote response count mismatch: expected {expected_count}, got {len(out)}")
    return out


@nvtx.annotate("nrl.remote::nim.request", color="blue")
def _post_with_retries(
    *,
    invoke_url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
    max_retries: int,
    max_429_retries: int,
    trace_context: Dict[str, str] | None = None,
) -> Any:
    base_delay = 2.0
    attempt = 0
    retries_429 = 0

    service_tracing = _service_tracing()
    span_parent_context = (
        _extract_trace_context(service_tracing, trace_context) if service_tracing is not None else None
    )

    while attempt < int(max_retries):
        request_headers = dict(headers)
        try:
            span_context = (
                service_tracing.start_span(
                    "nim.http.post",
                    context=span_parent_context,
                    attributes={
                        "http.method": "POST",
                        "nim.endpoint": _safe_endpoint_attribute(invoke_url),
                        "retry.attempt": attempt,
                    },
                )
                if service_tracing is not None
                else nullcontext()
            )
            with span_context as span:
                if service_tracing is not None:
                    try:
                        service_tracing.inject_trace_context(request_headers)
                    except Exception as exc:
                        logger.warning("OpenTelemetry trace propagation failed for NIM request: %s", exc)
                try:
                    response = requests.post(
                        invoke_url, headers=request_headers, json=payload, timeout=float(timeout_s)
                    )
                except (requests.Timeout, requests.RequestException) as exc:
                    _set_span_error(span, exc)
                    raise
                status_code = response.status_code
                _set_span_attribute(span, "http.status_code", status_code)
                if status_code >= 400:
                    _set_span_error(span, status_code)

            if status_code == 429:
                retries_429 += 1
                if retries_429 >= int(max_429_retries):
                    response.raise_for_status()
                backoff_time = base_delay * (2**retries_429)
                logger.warning(
                    "NIM endpoint %s returned 429 (rate limited). Retry %d/%d after %.1fs backoff.",
                    invoke_url,
                    retries_429,
                    max_429_retries,
                    backoff_time,
                )
                time.sleep(backoff_time)
                continue

            if status_code == 503 or (500 <= status_code < 600):
                if attempt == int(max_retries) - 1:
                    response.raise_for_status()
                backoff_time = base_delay * (2**attempt)
                logger.warning(
                    "NIM endpoint %s returned %d. Retry %d/%d after %.1fs backoff.",
                    invoke_url,
                    status_code,
                    attempt + 1,
                    max_retries,
                    backoff_time,
                )
                time.sleep(backoff_time)
                attempt += 1
                continue

            if 400 <= status_code < 500:
                raise requests.HTTPError(
                    f"HTTP {status_code} from {_safe_endpoint_attribute(invoke_url)}: {response.text}",
                    response=response,
                )
            response.raise_for_status()
            return response.json()

        except requests.Timeout as exc:
            if attempt == int(max_retries) - 1:
                raise TimeoutError(f"Request timed out after {attempt + 1} attempts.") from exc
            backoff_time = base_delay * (2**attempt)
            logger.warning(
                "NIM endpoint %s timed out (%.1fs). Retry %d/%d after %.1fs backoff.",
                invoke_url,
                timeout_s,
                attempt + 1,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)
            attempt += 1
        except requests.RequestException as exc:
            resp = getattr(exc, "response", None)
            if resp is not None and 400 <= resp.status_code < 500:
                raise  # client errors are not retryable
            if attempt == int(max_retries) - 1:
                raise
            backoff_time = base_delay * (2**attempt)
            logger.warning(
                "NIM endpoint %s request failed: %s. Retry %d/%d after %.1fs backoff.",
                invoke_url,
                exc,
                attempt + 1,
                max_retries,
                backoff_time,
            )
            time.sleep(backoff_time)
            attempt += 1

    raise RuntimeError(f"Failed to get a successful response after {max_retries} retries.")


def _mime_from_b64(b64: str) -> str:
    """Detect MIME type from the first bytes of a base64-encoded image."""
    if b64.startswith("/9j"):  # JPEG magic bytes FF D8 FF
        return "image/jpeg"
    return "image/png"


# ---------------------------------------------------------------------------
# Stateful NIM HTTP client
# ---------------------------------------------------------------------------


class NIMClient:
    """Persistent NIM HTTP client that reuses a single :class:`ThreadPoolExecutor`.

    Create one instance per Ray actor so the thread pool lives for the
    actor's entire lifetime instead of being torn down after every batch.
    """

    def __init__(self, *, max_pool_workers: int = 24) -> None:
        self._max_pool_workers = max(1, int(max_pool_workers))
        self._executor = ThreadPoolExecutor(max_workers=self._max_pool_workers)

    # -- image inference --------------------------------------------------

    def invoke_image_inference_batches(
        self,
        *,
        invoke_url: str,
        image_b64_list: Sequence[str],
        merge_levels: Optional[Sequence[str]] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        max_batch_size: int = 8,
        max_retries: int = 5,
        max_429_retries: int = 3,
    ) -> List[Any]:
        """Batched concurrent image inference using the persistent thread pool."""
        invoke_urls = _parse_invoke_urls(invoke_url)

        token = (api_key or "").strip()
        headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        n = len(image_b64_list)
        if n == 0:
            return []

        if merge_levels is not None and len(merge_levels) != n:
            raise ValueError(f"merge_levels length ({len(merge_levels)}) must match image_b64_list length ({n})")

        ranges = _chunk_ranges(n, int(max_batch_size))
        flattened: List[Optional[Any]] = [None] * n
        trace_context = _capture_trace_context()

        def _invoke_one_batch(start: int, end: int, endpoint_url: str) -> Tuple[int, int, List[Any]]:
            inputs = [
                {
                    "type": "image_url",
                    "url": f"data:{_mime_from_b64(b64)};base64,{b64}",
                }
                for b64 in image_b64_list[start:end]
            ]
            payload: Dict[str, Any] = {"input": inputs}
            if merge_levels is not None:
                payload["merge_levels"] = list(merge_levels[start:end])
            response_json = _post_with_retries(
                invoke_url=endpoint_url,
                payload=payload,
                headers=headers,
                timeout_s=float(timeout_s),
                max_retries=int(max_retries),
                max_429_retries=int(max_429_retries),
                trace_context=trace_context,
            )
            per_image = _normalize_batch_response(response_json, end - start)
            return start, end, per_image

        futures = {
            self._executor.submit(
                _invoke_one_batch,
                start,
                end,
                invoke_urls[idx % len(invoke_urls)],
            ): (start, end)
            for idx, (start, end) in enumerate(ranges)
        }
        try:
            for future in as_completed(futures):
                start, end = futures[future]
                _s, _e, per_image = future.result()
                if _s != start or _e != end:
                    raise RuntimeError("Internal batch ordering mismatch.")
                for i, item in enumerate(per_image):
                    flattened[start + i] = item
        except Exception as exc:
            # Do not leave a failed call running while its caller retries or
            # releases the input batch. Running requests retain their timeouts.
            for future in futures:
                future.cancel()
            wait(futures)
            errors = [exc]
            for future in futures:
                if future.cancelled():
                    continue
                error = future.exception()
                if error is not None and error is not exc:
                    if not isinstance(error, Exception):
                        raise error
                    errors.append(error)
            if len(errors) > 1:
                raise ExceptionGroup("Concurrent NIM image batches failed", errors) from None
            raise

        out: List[Any] = []
        for idx, item in enumerate(flattened):
            if item is None:
                raise RuntimeError(f"Missing response for item index {idx}")
            out.append(item)
        return out

    def invoke_page_elements_batches(
        self,
        *,
        invoke_url: str,
        image_b64_list: Sequence[str],
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        max_batch_size: int = 8,
        max_retries: int = 5,
        max_429_retries: int = 3,
    ) -> List[Any]:
        """Backward-compatible page-elements wrapper."""
        return self.invoke_image_inference_batches(
            invoke_url=invoke_url,
            image_b64_list=image_b64_list,
            api_key=api_key,
            timeout_s=timeout_s,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
            max_429_retries=max_429_retries,
        )

    # -- chat completions -------------------------------------------------

    def invoke_chat_completions(
        self,
        *,
        invoke_url: str,
        messages_list: Sequence[List[Dict[str, Any]]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        max_429_retries: int = 5,
    ) -> List[str]:
        """Concurrent chat completions using the persistent thread pool."""
        from nemo_retriever.models.nim.chat_completions import extract_chat_completion_text

        if not messages_list:
            return []

        token = (api_key or "").strip()
        headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        invoke_urls = _parse_invoke_urls(invoke_url)
        results: List[Optional[str]] = [None] * len(messages_list)
        trace_context = _capture_trace_context()

        def _invoke_one(idx: int, messages: List[Dict[str, Any]], endpoint_url: str) -> Tuple[int, str]:
            payload: Dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
            }
            if model:
                payload["model"] = model
            if extra_body:
                payload.update(extra_body)
            response_json = _post_with_retries(
                invoke_url=endpoint_url,
                payload=payload,
                headers=headers,
                timeout_s=float(timeout_s),
                max_retries=int(max_retries),
                max_429_retries=int(max_429_retries),
                trace_context=trace_context,
            )
            return idx, extract_chat_completion_text(response_json)

        futures = {
            self._executor.submit(_invoke_one, i, msgs, invoke_urls[i % len(invoke_urls)]): i
            for i, msgs in enumerate(messages_list)
        }
        for future in as_completed(futures):
            i, text = future.result()
            results[i] = text

        return [r if r is not None else "" for r in results]

    def invoke_chat_completions_images(
        self,
        *,
        invoke_url: str,
        image_b64_list: Sequence[str],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        task_prompt: Optional[str] = None,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = 1.1,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        max_429_retries: int = 5,
    ) -> List[str]:
        """Chat completions wrapper: one request per base64 image."""
        if not image_b64_list:
            return []

        messages_list: List[List[Dict[str, Any]]] = []
        for b64 in image_b64_list:
            mime = _mime_from_b64(b64)
            content: List[Dict[str, Any]] = []
            if task_prompt:
                content.append({"type": "text", "text": task_prompt})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            messages_list.append([{"role": "user", "content": content}])

        merged_extra: Dict[str, Any] = {}
        if repetition_penalty is not None:
            merged_extra["repetition_penalty"] = repetition_penalty
        if extra_body:
            merged_extra.update(extra_body)

        return self.invoke_chat_completions(
            invoke_url=invoke_url,
            messages_list=messages_list,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            extra_body=merged_extra,
            max_retries=max_retries,
            max_429_retries=max_429_retries,
        )

    # -- lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-compatible free functions (create a temporary pool per call)
# ---------------------------------------------------------------------------


def invoke_image_inference_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    merge_levels: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 24,
    max_retries: int = 5,
    max_429_retries: int = 3,
) -> List[Any]:
    """
    Invoke one or more image NIM HTTP endpoints with batched concurrent requests.

    `invoke_url` may be a single URL or a comma-separated URL list.
    When multiple URLs are provided, batches are distributed round-robin.

    Parameters
    ----------
    merge_levels
        Optional per-image merge level (``"word"``, ``"sentence"``, or
        ``"paragraph"``).  When provided, must have the same length as
        *image_b64_list*.  Passed as ``merge_levels`` in the JSON payload
        so the NIM can apply per-crop merging behaviour.

    Returns one response item per input image, in the same order.

    .. note::

       Prefer :class:`NIMClient` in long-lived actors to avoid creating a
       fresh ``ThreadPoolExecutor`` on every call.
    """
    client = NIMClient(max_pool_workers=max_pool_workers)
    try:
        return client.invoke_image_inference_batches(
            invoke_url=invoke_url,
            image_b64_list=image_b64_list,
            merge_levels=merge_levels,
            api_key=api_key,
            timeout_s=timeout_s,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
            max_429_retries=max_429_retries,
        )
    finally:
        client.shutdown()


def invoke_page_elements_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 24,
    max_retries: int = 5,
    max_429_retries: int = 3,
) -> List[Any]:
    """Backward-compatible alias for page-elements callers."""
    return invoke_image_inference_batches(
        invoke_url=invoke_url,
        image_b64_list=image_b64_list,
        api_key=api_key,
        timeout_s=timeout_s,
        max_batch_size=max_batch_size,
        max_pool_workers=max_pool_workers,
        max_retries=max_retries,
        max_429_retries=max_429_retries,
    )
