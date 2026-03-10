# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


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


def _post_with_retries(
    *,
    invoke_url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
    max_retries: int,
    max_429_retries: int,
) -> Any:
    base_delay = 2.0
    attempt = 0
    retries_429 = 0

    while attempt < int(max_retries):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=float(timeout_s))
            status_code = response.status_code

            if status_code == 429:
                retries_429 += 1
                if retries_429 >= int(max_429_retries):
                    response.raise_for_status()
                backoff_time = base_delay * (2**retries_429)
                time.sleep(backoff_time)
                continue

            if status_code == 503 or (500 <= status_code < 600):
                if attempt == int(max_retries) - 1:
                    response.raise_for_status()
                backoff_time = base_delay * (2**attempt)
                time.sleep(backoff_time)
                attempt += 1
                continue

            response.raise_for_status()
            return response.json()

        except requests.Timeout as exc:
            if attempt == int(max_retries) - 1:
                raise TimeoutError(f"Request timed out after {attempt + 1} attempts.") from exc
            backoff_time = base_delay * (2**attempt)
            time.sleep(backoff_time)
            attempt += 1
        except requests.RequestException:
            if attempt == int(max_retries) - 1:
                raise
            backoff_time = base_delay * (2**attempt)
            time.sleep(backoff_time)
            attempt += 1

    raise RuntimeError(f"Failed to get a successful response after {max_retries} retries.")


def invoke_image_inference_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
) -> List[Any]:
    """
    Invoke one or more image NIM HTTP endpoints with batched concurrent requests.

    `invoke_url` may be a single URL or a comma-separated URL list.
    When multiple URLs are provided, batches are distributed round-robin.

    Returns one response item per input image, in the same order.
    """
    invoke_urls = _parse_invoke_urls(invoke_url)

    token = (api_key or "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    n = len(image_b64_list)
    if n == 0:
        return []

    ranges = _chunk_ranges(n, int(max_batch_size))
    flattened: List[Optional[Any]] = [None] * n

    def _invoke_one_batch(start: int, end: int, endpoint_url: str) -> Tuple[int, int, List[Any]]:
        inputs = [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{b64}",
            }
            for b64 in image_b64_list[start:end]
        ]
        payload = {"input": inputs}
        response_json = _post_with_retries(
            invoke_url=endpoint_url,
            payload=payload,
            headers=headers,
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
            max_429_retries=int(max_429_retries),
        )
        per_image = _normalize_batch_response(response_json, end - start)
        return start, end, per_image

    with ThreadPoolExecutor(max_workers=max(1, int(max_pool_workers))) as executor:
        futures = {
            executor.submit(
                _invoke_one_batch,
                start,
                end,
                invoke_urls[idx % len(invoke_urls)],
            ): (start, end)
            for idx, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            start, end = futures[future]
            _s, _e, per_image = future.result()
            if _s != start or _e != end:
                raise RuntimeError("Internal batch ordering mismatch.")
            for i, item in enumerate(per_image):
                flattened[start + i] = item

    out: List[Any] = []
    for idx, item in enumerate(flattened):
        if item is None:
            raise RuntimeError(f"Missing response for item index {idx}")
        out.append(item)
    return out


def invoke_page_elements_batches(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    max_batch_size: int = 8,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
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
