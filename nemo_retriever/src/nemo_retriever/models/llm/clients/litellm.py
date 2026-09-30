# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified LLM answer generation client.

LiteLLMClient wraps the litellm library which provides a single interface
for routing to NVIDIA NIM, OpenAI, HuggingFace Inference Endpoints, and
local vLLM / Ollama servers via a model name prefix convention.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nemo_retriever.common.params.models import (
    LLMInferenceParams,
    LLMRemoteClientParams,
    resolve_environment_reference,
    validate_llm_extra_params,
)
from nemo_retriever.models.llm.tasks.rag_answer import (
    RagAnswerTask,
    _build_rag_prompt as _task_build_rag_prompt,
    _deep_merge_dicts,
)
from nemo_retriever.models.llm.types import (
    GenerationResult,
    UnsupportedTextResponseError,
)

logger = logging.getLogger(__name__)
# Backwards-compatible helper export retained for existing callers.
_build_rag_prompt = _task_build_rag_prompt
_NO_AUTH_API_KEY = "no-api-key"


def _field(value: object, name: str, default: object = None) -> object:
    """Read a provider response field from either a mapping or an object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class LiteLLMClient:
    """Unified LLM client backed by litellm.

    A single model string change routes to any supported provider:
    - NVIDIA NIM:  nvidia_nim/<org>/<model>
    - OpenAI:      openai/<model>
    - Any OpenAI-compatible server (vLLM, Ollama): openai/<model> + api_base
    - HuggingFace: huggingface/<org>/<model>

    When ``api_key`` is omitted, LiteLLM performs provider-native environment
    lookup. Variable names are provider-specific. The default ``nvidia_nim``
    provider reads ``NVIDIA_NIM_API_KEY``, not ``NVIDIA_API_KEY``. Pass
    ``api_key="os.environ/NVIDIA_API_KEY"`` to forward the hosted-inference key.

    Configuration is split into two orthogonal Pydantic objects:

    * ``transport``: :class:`~nemo_retriever.common.params.LLMRemoteClientParams`
      owns provider endpoint, authentication, retry, and timeout.
    * ``sampling``: :class:`~nemo_retriever.common.params.LLMInferenceParams`
      owns ``temperature``, ``top_p``, and ``max_tokens``.

    Use :meth:`from_kwargs` for a flat, backwards-compatible constructor.
    """

    _DEFAULT_MODEL: str = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    supports_concurrent_calls: bool = True

    def __init__(
        self,
        transport: LLMRemoteClientParams,
        sampling: Optional[LLMInferenceParams] = None,
    ):
        self.transport = transport
        # Default to ``temperature=0.0, max_tokens=4096`` so the structured
        # constructor matches ``from_kwargs`` and keeps RAG-eval runs
        # deterministic.  ``LLMInferenceParams`` itself defaults to
        # ``max_tokens=1024`` for captioning/summarization workloads; RAG
        # answers routinely exceed that, so the client overrides it.
        self.sampling = sampling if sampling is not None else LLMInferenceParams(temperature=0.0, max_tokens=4096)

    @property
    def model(self) -> str:
        """Return the model identifier from the transport params."""
        return self.transport.model

    @classmethod
    def from_kwargs(
        cls,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        rag_system_prompt: Optional[str] = None,
        rag_system_prompt_prefix: Optional[str] = None,
        reasoning_enabled: bool = True,
    ) -> "LiteLLMClient":
        """Flat-kwarg constructor for zero-churn migration from the old signature.

        Splits the flat kwargs into the two structured params objects. All
        validation (temperature range, ``num_retries >= 0``, ``timeout > 0``)
        is delegated to the Pydantic models.
        """
        transport = LLMRemoteClientParams(
            model=model,
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
            timeout=timeout,
            extra_params=extra_params or {},
            rag_system_prompt=rag_system_prompt,
            rag_system_prompt_prefix=rag_system_prompt_prefix,
            reasoning_enabled=reasoning_enabled,
        )
        sampling = LLMInferenceParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return cls(transport=transport, sampling=sampling)

    def complete(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> tuple[str, float]:
        """Raw litellm completion call. Returns (content_text, latency_s)."""
        validate_llm_extra_params(self.transport.extra_params, source="LLMRemoteClientParams.extra_params")
        validate_llm_extra_params(extra_params, source="GenerationRequest.extra_params")
        import litellm

        sampling_kwargs = self.sampling.to_sampling_kwargs()
        if max_tokens is not None:
            sampling_kwargs["max_tokens"] = max_tokens

        call_kwargs: dict[str, Any] = {
            "model": self.transport.model,
            "messages": messages,
            "num_retries": self.transport.num_retries,
            "timeout": self.transport.timeout,
            **sampling_kwargs,
        }
        if self.transport.api_base:
            call_kwargs["api_base"] = self.transport.api_base
        if self.transport._uses_no_api_key("api_key"):
            call_kwargs["api_key"] = _NO_AUTH_API_KEY
        elif self.transport.api_key is not None:
            resolved_api_key = resolve_environment_reference(self.transport.api_key)
            if resolved_api_key:
                call_kwargs["api_key"] = resolved_api_key
        call_kwargs.update(_deep_merge_dicts(self.transport.extra_params, extra_params or {}))

        t0 = time.monotonic()
        try:
            response = litellm.completion(**call_kwargs)
        except Exception as exc:
            err = str(exc)
            if "temperature" in err and "top_p" in err:
                logger.error(
                    "Model %s rejected the request because both `temperature` "
                    "and `top_p` were specified. Some providers (e.g. Bedrock) "
                    "only accept one. Either remove `top_p` from the model "
                    "config or set `temperature` to null. Sent: "
                    "temperature=%s, top_p=%s",
                    self.transport.model,
                    call_kwargs.get("temperature"),
                    call_kwargs.get("top_p"),
                )
            raise
        latency = time.monotonic() - t0

        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise UnsupportedTextResponseError("provider response must contain exactly one choice")
        choice = choices[0]
        message = _field(choice, "message")
        if message is None:
            raise UnsupportedTextResponseError("provider response choice has no message")
        if _field(choice, "finish_reason") in {"tool_calls", "function_call"}:
            raise UnsupportedTextResponseError("tool-call responses are unsupported by the text completion contract")
        if _field(message, "tool_calls") or _field(message, "function_call"):
            raise UnsupportedTextResponseError("tool-call responses are unsupported by the text completion contract")
        if _field(message, "refusal"):
            raise UnsupportedTextResponseError("refusal responses are unsupported by the text completion contract")
        if any(_field(message, field_name) is not None for field_name in ("audio", "images", "videos")):
            raise UnsupportedTextResponseError("non-text response modalities are unsupported")
        content = _field(message, "content")
        if not isinstance(content, str):
            raise UnsupportedTextResponseError("provider response content must be plain text")
        content = content.strip()
        return content, latency

    def generate(
        self,
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: Optional[bool] = None,
    ) -> GenerationResult:
        """Generate an answer for the given query using retrieved chunks as context."""
        effective_reasoning_enabled = (
            self.transport.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        task = RagAnswerTask(
            system_prompt=self.transport.rag_system_prompt,
            system_prompt_prefix=self.transport.rag_system_prompt_prefix,
            reasoning_enabled=effective_reasoning_enabled,
        )
        result = task.execute(self, query=query, chunks=chunks)
        return GenerationResult(
            answer=result.text,
            latency_s=result.latency_s,
            model=result.model,
            error=result.error,
        )
