# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-compatible HTTP embedding client.

vLLM's OpenAI-compatible embeddings API expects a minimal payload (model, input,
encoding_format). It does not use NIM-specific fields like input_type or truncate.
Use this module when pointing the retriever at a vLLM embedding server (e.g. for
llama-nemotron-embed-1b-v2) to avoid request schema mismatches.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Reduce HTTP client logging verbosity
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def _normalize_embeddings_endpoint(endpoint_url: str) -> str:
    """Normalize endpoint to a concrete /embeddings URL."""
    s = (endpoint_url or "").strip().rstrip("/")
    if not s:
        raise ValueError("endpoint_url is empty")
    if s.endswith("/embeddings"):
        return s
    return f"{s}/embeddings"


def get_model_id_from_server(endpoint_url: str, timeout_s: float = 10.0) -> str | None:
    """
    GET {endpoint}/models (or {endpoint}/v1/models) and return the first model id.
    Returns None if the request fails or the response has no models.
    """
    try:
        import httpx  # type: ignore
    except ImportError:
        return None
    base = (endpoint_url or "").strip().rstrip("/")
    if not base:
        return None
    # Try base/models (e.g. .../v1/models) then root .../models
    base_no_v1 = base.removesuffix("/v1").rstrip("/") if base.endswith("/v1") else base
    candidates = [f"{base}/models", f"{base_no_v1}/models"]
    for url in candidates:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if not isinstance(data, dict):
                    continue
                models = data.get("data")
                if isinstance(models, list) and len(models) > 0 and isinstance(models[0], dict):
                    return models[0].get("id")
        except Exception:
            continue
    return None


def embed_via_vllm_http(
    prompts: List[str],
    *,
    endpoint_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
    encoding_format: str = "float",
    batch_size: int = 256,
    timeout_s: float = 120.0,
    prefix: Optional[str] = None,
) -> List[List[float]]:
    """
    Request embeddings from a vLLM (or other OpenAI-compatible) server using a
    minimal payload: model, input, encoding_format. No input_type or truncate.

    Parameters
    ----------
    prompts : list of str
        Texts to embed. If prefix is set, each prompt is prefixed (e.g. "query: " or "passage: ").
    endpoint_url : str
        Base URL of the server (e.g. http://localhost:8000/v1).
    model_name : str
        Model name as returned by the server (e.g. from models.list()).
    api_key : str, optional
        Optional Bearer token.
    dimensions : int, optional
        Optional embedding dimension (if supported by server).
    encoding_format : str
        "float" or "base64".
    batch_size : int
        Max prompts per request.
    timeout_s : float
        Request timeout in seconds.

    Returns
    -------
    list of list of float
        One embedding vector per prompt, in order.
    """
    try:
        import httpx  # type: ignore
    except ImportError as e:
        raise RuntimeError("vLLM HTTP embedding requires httpx.") from e

    # Resolve model name from server if not provided
    resolved_model = (model_name or "").strip()
    if not resolved_model:
        resolved_model = get_model_id_from_server(endpoint_url, timeout_s=min(10.0, timeout_s)) or ""
        if not resolved_model:
            raise ValueError(
                "model_name is empty and could not discover model id from server; "
                "pass --embed-model-name or ensure server exposes /v1/models or /models"
            )

    headers: Dict[str, str] = {"accept": "application/json", "content-type": "application/json"}
    if (api_key or "").strip():
        headers["Authorization"] = f"Bearer {(api_key or '').strip()}"

    # Minimal OpenAI-compatible payload; no input_type or truncate.
    def make_payload(batch: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "input": batch,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        return payload

    if prefix:
        prompts = [str(prefix) + p for p in prompts]

    # Build candidate URLs: primary (e.g. .../v1/embeddings) and fallback (.../embeddings)
    base = (endpoint_url or "").strip().rstrip("/")
    base_no_v1 = base.removesuffix("/v1").rstrip("/") if base.endswith("/v1") else base
    candidate_urls = [
        _normalize_embeddings_endpoint(endpoint_url),
        f"{base_no_v1}/embeddings" if base_no_v1 != base else None,
    ]
    candidate_urls = [u for u in candidate_urls if u]

    last_error: Exception | None = None
    with httpx.Client(timeout=float(timeout_s)) as client:
        for url in candidate_urls:
            all_embeddings = []
            try:
                for i in range(0, len(prompts), batch_size):
                    batch = prompts[i : i + batch_size]
                    resp = client.post(url, headers=headers, json=make_payload(batch))
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("data") if isinstance(data, dict) else None
                    if not isinstance(items, list):
                        raise RuntimeError("Unexpected embeddings response (missing 'data' list).")
                    for j, it in enumerate(items):
                        if not isinstance(it, dict):
                            all_embeddings.append([])
                            continue
                        emb = it.get("embedding")
                        if isinstance(emb, list):
                            all_embeddings.append([float(x) for x in emb])
                        else:
                            all_embeddings.append([])
                return all_embeddings
            except Exception as e:
                last_error = e
                if hasattr(e, "response") and getattr(e.response, "status_code", None) == 404:
                    all_embeddings = []
                    continue
                raise
    if last_error is not None:
        tried = ", ".join(candidate_urls)
        raise RuntimeError(
            f"Embeddings request failed (tried: {tried}). "
            "Ensure the vLLM embedding server is running (e.g. `vllm serve <model_path> --runner pooling ...`) "
            "and that the base URL is correct (e.g. http://localhost:8000/v1)."
        ) from last_error
    return all_embeddings


__all__ = ["embed_via_vllm_http", "_normalize_embeddings_endpoint", "get_model_id_from_server"]
