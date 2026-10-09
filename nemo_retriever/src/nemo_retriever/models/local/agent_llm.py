# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.model import BaseModel, ModelRunMode

logger = logging.getLogger(__name__)

# Conservative cap for short tool-call JSON responses. Larger evals should
# tune this with completion-token, truncation, and tool-parse telemetry.
_DEFAULT_MAX_TOKENS = 512
# Bound EngineCore join so a wedged child cannot hang process exit forever.
_VLLM_ENGINE_SHUTDOWN_TIMEOUT_S = 30.0


def _normalize_name(name: str) -> str:
    return name.strip().casefold()


_NANO_8B_MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
_SUPER_49B_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
_LEGACY_SUPER_49B_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
_NANO_8B_ALIASES = (
    "llama-3.1-nemotron-nano-8b-v1",
    "nemotron-nano-8b",
    "nemotron-8b",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
)
_SUPER_49B_ALIASES = (
    "llama-3.3-nemotron-super-49b-v1.5",
    "nemotron-super-49b",
    "super-49b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
)
_LEGACY_SUPER_49B_ALIASES = (
    "llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3_3-nemotron-super-49b-v1",
)

# Supported local agent LLM names. Keep this intentionally small: V1 only
# supports internal/NVIDIA models whose local tool-call behavior we have tested.
_AGENT_LLM_MODEL_ALIASES: dict[str, str] = {
    _normalize_name(name): _NANO_8B_MODEL_ID for name in (_NANO_8B_MODEL_ID, *_NANO_8B_ALIASES)
}
_AGENT_LLM_MODEL_ALIASES.update(
    {_normalize_name(name): _SUPER_49B_MODEL_ID for name in (_SUPER_49B_MODEL_ID, *_SUPER_49B_ALIASES)}
)
_AGENT_LLM_MODEL_ALIASES.update(
    {
        _normalize_name(name): _LEGACY_SUPER_49B_MODEL_ID
        for name in (_LEGACY_SUPER_49B_MODEL_ID, *_LEGACY_SUPER_49B_ALIASES)
    }
)
_SUPPORTED_AGENT_LLM_NAMES = tuple(
    sorted(
        (
            _NANO_8B_MODEL_ID,
            *_NANO_8B_ALIASES,
            _SUPER_49B_MODEL_ID,
            *_SUPER_49B_ALIASES,
            _LEGACY_SUPER_49B_MODEL_ID,
            *_LEGACY_SUPER_49B_ALIASES,
        )
    )
)
_NEMOTRON_JSON_TOOL_PROMPT_EXTRAS = {"chat_template_kwargs": {"tools_in_user_message": True}}


def supported_agent_llm_names() -> tuple[str, ...]:
    """Return supported local agent LLM model IDs and aliases."""

    return _SUPPORTED_AGENT_LLM_NAMES


def is_supported_agent_llm_model(name: str) -> bool:
    """Return whether *name* is a supported in-process local agent LLM."""

    return _normalize_name(name) in _AGENT_LLM_MODEL_ALIASES


def resolve_agent_llm_model_name(name: str) -> str:
    """Resolve a local agent LLM alias to its Hugging Face model ID."""

    return _AGENT_LLM_MODEL_ALIASES.get(_normalize_name(name), name)


@dataclass(frozen=True)
class LocalAgentLLMConfig:
    """Local vLLM configuration for the agent chat LLM."""

    model_path: str
    hf_cache_dir: Optional[str] = None
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_tokens: int = _DEFAULT_MAX_TOKENS


class VLLMAgentChatLLM(BaseModel):
    """In-process vLLM chat-completions adapter for agentic retrieval.

    The public call signature mirrors ``invoke_chat_completion_step`` and returns
    an OpenAI-compatible response dict. This lets the existing ReAct and
    selection agents consume local GPU inference without changing their loop
    semantics.
    """

    def __init__(self, config: LocalAgentLLMConfig) -> None:
        super().__init__()

        requested_model_path = config.model_path
        model_path = resolve_agent_llm_model_name(requested_model_path)
        if not is_supported_agent_llm_model(model_path):
            raise ValueError(
                f"Unsupported local agent LLM model {requested_model_path!r}. "
                "Custom in-process agent LLMs are not supported yet. "
                "Provide invoke_url for a custom/self-hosted OpenAI-compatible endpoint, "
                f"or choose one of: {', '.join(supported_agent_llm_names())}."
            )

        try:
            from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                'Local agentic LLM inference requires vLLM. Install with: pip install "nemo-retriever[local]"'
            ) from e

        apply_vllm_startup_defaults(tensor_parallel_size=int(config.tensor_parallel_size))
        _raise_if_cuda_unavailable()
        self._model_path = model_path
        self._max_tokens = int(config.max_tokens)
        self._request_extras = deepcopy(_NEMOTRON_JSON_TOOL_PROMPT_EXTRAS)
        self._lock = threading.Lock()

        configure_global_hf_cache_base(config.hf_cache_dir)
        revision = get_hf_revision(model_path)

        engine_kwargs: dict[str, Any] = {}
        if config.max_model_len is not None:
            engine_kwargs["max_model_len"] = int(config.max_model_len)
        if config.max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = int(config.max_num_seqs)
        if model_path == _SUPER_49B_MODEL_ID and int(config.tensor_parallel_size) > 1:
            # Super-49B v1.5 fails during CUDA graph warm-up with tensor
            # parallelism on supported H100 configurations. These are the
            # equivalent safe native-vLLM startup options.
            engine_kwargs["enforce_eager"] = True
            engine_kwargs["disable_custom_all_reduce"] = True

        self._sampling_params_cls = SamplingParams
        self._llm: Any | None = LLM(
            model=model_path,
            revision=revision,
            # Safe for this local path: supported model names are hard-allowlisted
            # above and revisions are pinned in hf_model_registry.py.
            trust_remote_code=True,
            tensor_parallel_size=int(config.tensor_parallel_size),
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            **engine_kwargs,
        )

    def __call__(
        self,
        *,
        invoke_url: Optional[str] = None,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str | dict[str, Any] = "auto",
        timeout_s: float = 120.0,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
        max_retries: int = 10,
        max_429_retries: int = 5,
    ) -> dict[str, Any]:
        """Run one chat completion on the in-process engine.

        ``temperature=None`` means *unset* and maps to ``0.0`` (greedy).
        """
        _ = (invoke_url, api_key, timeout_s, max_retries, max_429_retries)
        if model and model != self._model_path:
            logger.debug("Ignoring per-call model=%r for local agent LLM already loaded as %r", model, self._model_path)

        sampling_params = self._sampling_params_cls(
            temperature=0.0 if temperature is None else float(temperature),
            max_tokens=int(max_tokens) if max_tokens is not None else self._max_tokens,
        )
        chat_kwargs = self._build_chat_kwargs(extra_body)
        active_tools = tools if tools and tool_choice != "none" else None
        if active_tools:
            chat_kwargs["tools"] = active_tools

        local_messages = self._normalize_messages(messages, tools=active_tools)
        with self._lock:
            self._require_loaded()
            outputs = self._llm.chat(local_messages, sampling_params=sampling_params, **chat_kwargs)

        request_output = outputs[0]
        completion = request_output.outputs[0]
        text = str(getattr(completion, "text", "") or "").strip()
        tool_calls = _tool_calls_from_completion(completion) or parse_tool_calls_from_text(text)
        finish_reason = "tool_calls" if tool_calls else str(getattr(completion, "finish_reason", None) or "stop")

        message: dict[str, Any] = {"role": "assistant"}
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = tool_calls
        else:
            message["content"] = text

        return {
            "id": _new_response_id(),
            "object": "chat.completion",
            "model": self._model_path,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage_from_outputs(request_output, completion),
        }

    def _build_chat_kwargs(self, extra_body: Optional[dict[str, Any]]) -> dict[str, Any]:
        chat_kwargs = deepcopy(self._request_extras)
        for key, value in (extra_body or {}).items():
            if key in {"parallel_tool_calls", "reasoning_effort"}:
                continue
            chat_kwargs[key] = value
        chat_kwargs.setdefault("use_tqdm", False)
        return chat_kwargs

    def _normalize_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        tool_prompt = (
            "You have access to the following tools. Your entire response must be a JSON array of one or more "
            "tool calls. Each item must be shaped as "
            '{"name": "tool_name", "arguments": {"arg_name": "arg_value"}}. '
            "Use only the listed tool names. Do not include prose, markdown, or text outside the JSON array.\n"
            f"<AVAILABLE_TOOLS>{json.dumps(list(tools), ensure_ascii=False)}</AVAILABLE_TOOLS>"
            if tools
            else None
        )
        for idx, message in enumerate(messages):
            msg = dict(message)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = _flatten_text_content(content)
            if msg.get("role") == "assistant" and msg.get("tool_calls") and not msg.get("content"):
                msg["content"] = _format_assistant_tool_calls_content(msg.get("tool_calls") or [])
            if tool_prompt and idx == 0 and msg.get("role") == "system":
                msg["content"] = f"{tool_prompt}\n\n{msg.get('content') or ''}"
                tool_prompt = None
            normalized.append(msg)
        if tool_prompt:
            normalized.insert(0, {"role": "system", "content": tool_prompt})
        return _collapse_consecutive_tool_messages(normalized)

    def unload(self) -> None:
        """Shut down the EngineCore child and release GPU memory.

        Unlike embed/rerank holders that only ``del`` the ``LLM`` and rely on
        process exit, the agent chat path must call ``engine_core.shutdown()``:
        vLLM V1 generation spawns a multiprocess ``VLLM::EngineCore`` that can
        keep the parent process alive after results are returned.
        """

        with self._lock:
            llm = self._llm
            self._llm = None

        if llm is None:
            return

        _shutdown_vllm_engine(llm)
        try:
            del llm
        except Exception:
            logger.debug("Ignoring error while deleting local agent vLLM LLM", exc_info=True)

        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self._llm is None:
            raise RuntimeError(
                "VLLMAgentChatLLM has been unloaded; create a new local agent LLM before generating responses."
            )

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def model_type(self) -> str:
        return "agent-chat-llm"

    @property
    def model_runmode(self) -> ModelRunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {"type": "chat_messages", "format": "openai_chat_completions"}

    @property
    def output(self) -> Any:
        return {"type": "chat_completion", "format": "openai_compatible"}

    @property
    def input_batch_size(self) -> int:
        return 1


def _raise_if_cuda_unavailable() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The local agent LLM vLLM backend requires CUDA, but torch reports no available CUDA device. "
            "Run on a GPU host or provide invoke_url for a remote OpenAI-compatible endpoint."
        )


def _call_shutdown(shutdown: Any, *, timeout_s: float) -> None:
    try:
        shutdown(timeout=timeout_s)
    except TypeError:
        shutdown()


def _shutdown_vllm_engine(llm: Any) -> None:
    """Best-effort vLLM V1 EngineCore / executor shutdown before dropping the LLM."""

    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None) if engine_core is not None else None
        if callable(shutdown):
            _call_shutdown(shutdown, timeout_s=_VLLM_ENGINE_SHUTDOWN_TIMEOUT_S)
            return
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            _call_shutdown(shutdown, timeout_s=_VLLM_ENGINE_SHUTDOWN_TIMEOUT_S)
    except Exception:
        logger.warning(
            "Local agent vLLM engine shutdown failed; dropping the LLM reference anyway",
            exc_info=True,
        )


def create_vllm_agent_chat_llm(config: LocalAgentLLMConfig) -> VLLMAgentChatLLM:
    """Create a local agent LLM owned by the caller (no process-global cache).

    Mirrors embed/rerank: the CLI job holds one instance for the whole
    run and calls ``unload()`` when the job finishes.
    """

    return VLLMAgentChatLLM(config)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Parse common offline tool-call JSON into OpenAI ``tool_calls`` shape."""

    payload = _load_json_payload(text)
    if payload is None:
        return []
    return _coerce_tool_calls(payload)


def _load_json_payload(text: str) -> Any | None:
    cleaned = _strip_code_fence(str(text or "").strip())
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    starts = [idx for idx, ch in enumerate(cleaned) if ch in "[{"]
    for start in starts:
        try:
            value, _end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _coerce_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and "tool_calls" in payload:
        return _coerce_tool_calls(payload.get("tool_calls"))
    if isinstance(payload, Mapping):
        call = _coerce_single_tool_call(payload)
        return [call] if call is not None else []
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            call = _coerce_single_tool_call(item)
            if call is not None:
                calls.append(call)
        return calls
    return []


def _coerce_single_tool_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None

    if isinstance(item.get("function"), Mapping):
        function = dict(item["function"])
        if "arguments" not in function:
            return None
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = item.get("name") or item.get("tool_name")
        if "arguments" in item:
            arguments = item.get("arguments")
        elif "parameters" in item:
            arguments = item.get("parameters")
        else:
            return None

    if not name:
        return None

    return {
        "id": str(item.get("id") or _new_tool_call_id()),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _arguments_to_json_string(arguments),
        },
    }


def _arguments_to_json_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {})


def _tool_calls_from_completion(completion: Any) -> list[dict[str, Any]]:
    for attr in ("tool_calls", "tool_call"):
        value = getattr(completion, attr, None)
        if value:
            return _coerce_tool_calls(value)
    return []


def _usage_from_outputs(request_output: Any, completion: Any) -> dict[str, int]:
    prompt_tokens = len(getattr(request_output, "prompt_token_ids", None) or [])
    completion_tokens = len(getattr(completion, "token_ids", None) or [])
    usage: dict[str, int] = {}
    if prompt_tokens:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens:
        usage["completion_tokens"] = completion_tokens
    if usage:
        usage["total_tokens"] = prompt_tokens + completion_tokens
    return usage


def _flatten_text_content(content: Sequence[Any]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, Mapping):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item))
        else:
            parts.append(str(item))
    return "\n\n".join(part for part in parts if part)


def _collapse_consecutive_tool_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge parallel OpenAI tool results into one turn for local chat templates.

    Llama-family chat templates used by Nemotron require user/tool and
    assistant roles to alternate. The agent may execute multiple OpenAI tool
    calls from one assistant response, which creates consecutive ``tool`` turns.
    Collapsing them keeps the transcript semantically equivalent for local
    generation without changing the operator contract.
    """

    collapsed: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}

    for message in messages:
        msg = dict(message)
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_id = str(tool_call.get("id") or "")
                function = tool_call.get("function") or {}
                if tool_call_id and isinstance(function, Mapping):
                    tool_names_by_id[tool_call_id] = str(function.get("name") or "tool")

        if msg.get("role") != "tool":
            collapsed.append(msg)
            continue

        msg["content"] = _format_tool_message_content(msg, tool_names_by_id)
        if collapsed and collapsed[-1].get("role") == "tool":
            previous = str(collapsed[-1].get("content") or "")
            current = str(msg.get("content") or "")
            collapsed[-1]["content"] = "\n\n".join(part for part in (previous, current) if part)
            continue
        collapsed.append(msg)

    return collapsed


def _format_tool_message_content(message: Mapping[str, Any], tool_names_by_id: Mapping[str, str]) -> str:
    tool_call_id = str(message.get("tool_call_id") or "")
    tool_name = tool_names_by_id.get(tool_call_id, "tool")
    content = str(message.get("content") or "")
    if tool_call_id:
        return f"Tool result for {tool_name} ({tool_call_id}):\n{content}"
    return f"Tool result for {tool_name}:\n{content}"


def _format_assistant_tool_calls_content(tool_calls: Sequence[Any]) -> str:
    """Render OpenAI assistant tool calls for local templates that read only content."""

    serializable_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, Mapping):
            continue
        serializable_calls.append(
            {
                "id": tool_call.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return "Assistant tool calls:\n" + json.dumps(serializable_calls, ensure_ascii=False)


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _new_response_id() -> str:
    return f"chatcmpl_{uuid.uuid4().hex[:12]}"
