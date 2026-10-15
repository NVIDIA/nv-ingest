# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-purpose OpenAI-compatible chat completions client.

This module is model-agnostic and can be used with any endpoint that
implements the ``/v1/chat/completions`` contract (build.nvidia.com,
self-hosted NIMs, OpenAI, vLLM, etc.).

Prefer :class:`~nemo_retriever.nim.nim.NIMClient` for long-lived actors;
the free functions below create a temporary ``ThreadPoolExecutor`` on
every call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from nemo_retriever.models.nim.nim import _parse_invoke_urls, _post_with_retries


def extract_chat_completion_text(response_json: Any, *, require_response: bool = False) -> str:
    """Extract generated text from an OpenAI-compatible chat completions response."""
    try:
        choice = response_json["choices"][0]["message"]
        # Some models return output via tool_calls
        tool_calls = choice.get("tool_calls")
        if tool_calls:
            arguments = tool_calls[0]["function"]["arguments"]
            if isinstance(arguments, str) or not require_response:
                return str(arguments).strip()
        content = choice.get("content")
        if isinstance(content, str):
            return content.strip()
        if content and not require_response:
            return str(content).strip()
    except (KeyError, IndexError, TypeError):
        pass
    if require_response:
        raise ValueError("Inference response is missing a completed chat message with text or tool output")
    return ""


def invoke_chat_completions(
    *,
    invoke_url: str,
    messages_list: Sequence[List[Dict[str, Any]]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
    temperature: float = 0.0,
    extra_body: Optional[Dict[str, Any]] = None,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
) -> List[str]:
    """Invoke an OpenAI-compatible chat completions endpoint concurrently.

    .. note::

       Prefer :meth:`NIMClient.invoke_chat_completions` in long-lived
       actors to avoid creating a fresh ``ThreadPoolExecutor`` per call.

    Parameters
    ----------
    messages_list
        A sequence of OpenAI-format message lists, one per request.
        Each entry is passed as the ``"messages"`` field in the payload.
    model
        Optional model identifier included in each request payload.
    extra_body
        Additional top-level keys merged into every request payload
        (e.g. ``{"repetition_penalty": 1.1, "max_tokens": 9000}``).

    Returns one extracted text string per entry in *messages_list*, in order.
    """
    from nemo_retriever.models.nim.nim import NIMClient

    client = NIMClient(max_pool_workers=max_pool_workers)
    try:
        return client.invoke_chat_completions(
            invoke_url=invoke_url,
            messages_list=messages_list,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            extra_body=extra_body,
            max_retries=max_retries,
            max_429_retries=max_429_retries,
        )
    finally:
        client.shutdown()


def invoke_chat_completion_step(
    *,
    invoke_url: str,
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    timeout_s: float = 120.0,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    max_retries: int = 10,
    max_429_retries: int = 5,
) -> Dict[str, Any]:
    """Single synchronous tool-aware chat completion call.

    Parameters
    ----------
    invoke_url
        Full ``/v1/chat/completions`` endpoint URL.
    messages
        OpenAI-format message list for this single request.
    tools
        List of OpenAI tool-spec dicts.  When provided, ``tool_choice`` is also
        forwarded so the model knows which tools it may call.
    tool_choice
        ``"auto"`` (default) lets the model decide; ``"none"`` suppresses tool
        use; or a specific tool name dict.
    temperature
        Sampling temperature.  ``None`` omits the field from the payload entirely
        so the endpoint/model default applies.
    """
    token = (api_key or "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    invoke_urls = _parse_invoke_urls(invoke_url)
    endpoint_url = invoke_urls[0]

    payload: Dict[str, Any] = {"messages": messages}
    # Unset (None) => omit, so the endpoint/model default applies.
    if temperature is not None:
        payload["temperature"] = temperature
    if model:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if extra_body:
        payload.update(extra_body)

    return _post_with_retries(
        invoke_url=endpoint_url,
        payload=payload,
        headers=headers,
        timeout_s=float(timeout_s),
        max_retries=int(max_retries),
        max_429_retries=int(max_429_retries),
    )


def invoke_chat_completions_images(
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
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
    require_response: bool = False,
) -> List[str]:
    """Convenience wrapper: one chat completion request per base64 image.

    Builds an OpenAI-format ``image_url`` message for each image and
    delegates to :func:`invoke_chat_completions`.
    """
    from nemo_retriever.models.nim.nim import NIMClient

    client = NIMClient(max_pool_workers=max_pool_workers)
    try:
        return client.invoke_chat_completions_images(
            invoke_url=invoke_url,
            image_b64_list=image_b64_list,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            task_prompt=task_prompt,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            extra_body=extra_body,
            max_retries=max_retries,
            max_429_retries=max_429_retries,
            require_response=require_response,
        )
    finally:
        client.shutdown()
