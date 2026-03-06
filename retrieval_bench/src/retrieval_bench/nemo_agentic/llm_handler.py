# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implements a class and related utilities to make LLM API calls."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import aiofiles
import litellm
from litellm.exceptions import BadRequestError, ContentPolicyViolationError, ContextWindowExceededError
from litellm.types.utils import ModelResponse

from .configs import LLMConfig
from .logging_utils import get_logger_with_config

logger, _ = get_logger_with_config()

LLM_ERROR_PREFIX = "LLMError:"

# Reduce noisy provider/help banners on handled API errors.
if hasattr(litellm, "suppress_debug_info"):
    litellm.suppress_debug_info = True


def is_error(response: Any) -> bool:
    """Return True if the response from the LLM handler is an error string."""
    return isinstance(response, str) and response.startswith(LLM_ERROR_PREFIX)


def normalize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize message content from list-of-content-blocks to plain strings.

    Some OpenAI-compatible endpoints only accept string content for certain roles.
    This converts text-only `content` lists (e.g., [{"type":"text","text":"..."}])
    into a plain string. Messages with non-text blocks (e.g., image_url) are left as-is.
    """

    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            all_text = True
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    all_text = False
                    break
            if all_text:
                if len(text_parts) == 0:
                    msg["content"] = None
                elif len(text_parts) == 1:
                    msg["content"] = text_parts[0]
                else:
                    msg["content"] = "\n".join(text_parts)
        normalized.append(msg)
    return normalized


def write_json(obj: Any, log_dir: Union[str, Path], filename: Union[str, Path]):
    """Write an object to a json file."""
    if log_dir is None:
        return
    path = Path(log_dir, filename)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


async def awrite_json(obj: Any, log_dir: Union[str, Path], filename: Union[str, Path]):
    """Write an object to a json file async."""
    if log_dir is None:
        return
    path = Path(log_dir, filename)
    path.parent.mkdir(exist_ok=True, parents=True)
    async with aiofiles.open(path.as_posix(), "w") as f:
        await f.write(json.dumps(obj, indent=2))


async def make_acall_with_ratelimit_pause(*args: Any, **kwargs: Any):
    """Make an LLM call and pause on ratelimits (fixed 60s, max 3 attempts)."""
    i = 0
    while True:
        try:
            return await litellm.acompletion(*args, **kwargs)
        except litellm.RateLimitError:
            if i > 2:
                raise
            logger.info(f"Rate Limit. Sleep for a min (i={i}).")
            await asyncio.sleep(60)
            i += 1


class LLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        """LLM client to make calls to chat completion endpoint via LiteLLM."""
        self.config = llm_config

        self.completion_kwargs = dict(
            model=self.config.model,
            tool_choice=self.config.tool_choice,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            num_retries=self.config.num_retries,
            max_completion_tokens=self.config.max_completion_tokens,
        )
        if self.config.drop_params:
            self.completion_kwargs["drop_params"] = self.config.drop_params
        if self.config.allowed_openai_params is not None and len(self.config.allowed_openai_params) != 0:
            self.completion_kwargs["allowed_openai_params"] = self.config.allowed_openai_params
        if self.config.reasoning_effort is not None:
            self.completion_kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.api_key is not None:
            self.completion_kwargs["api_key"] = self.config.api_key

        self._api_key = None
        if self.config.api_key is not None and self.config.api_key.strip().startswith("os.environ/"):
            self._api_key = os.environ[self.config.api_key.strip().removeprefix("os.environ/")]

    async def log_extra_data_log_dir(
        self,
        subdir: Optional[str] = None,
        info: Optional[Any] = None,
        filename: str = "extra_info.json",
    ) -> None:
        if info is None or self.config.raw_log_pardir is None or subdir is None:
            return
        if not filename.endswith(".json"):
            raise ValueError(f"filename must end with '.json', got {filename!r}")
        json_log_dir = Path(self.config.raw_log_pardir, subdir)
        await awrite_json(obj=info, log_dir=json_log_dir, filename=filename)

    def pre_completion(self, messages: list[dict], tools: Optional[list[dict]] = None, **kwargs: Any) -> tuple:
        """Prepare request kwargs and logging context before calling LiteLLM."""
        logging_kwargs = kwargs.pop("logging_kwargs", None)
        if logging_kwargs is None or "step" not in logging_kwargs:
            curr_step = uuid4().hex
        else:
            curr_step = logging_kwargs["step"]

        json_log_dir = None
        if self.config.raw_log_pardir is not None and logging_kwargs is not None and "subdir" in logging_kwargs:
            json_log_dir = Path(self.config.raw_log_pardir, logging_kwargs["subdir"])

        # Merge in steps so per-call kwargs can intentionally override defaults
        # without raising `TypeError` on duplicate keys.
        request_kwargs = {"messages": messages}
        request_kwargs.update(self.completion_kwargs)
        request_kwargs.update(kwargs)
        if tools is not None:
            request_kwargs["tools"] = tools
        else:
            request_kwargs.pop("tool_choice", None)

        io_log_kwargs = {
            "input_json": dict(obj=request_kwargs, log_dir=json_log_dir, filename=f"{curr_step}_prompt.json")
        }
        logging_ctx = {
            "logging_kwargs": logging_kwargs,
            "json_log_dir": json_log_dir,
            "curr_step": curr_step,
        }
        return request_kwargs, logging_ctx, io_log_kwargs

    def post_completion(self, request_kwargs: dict, response: ModelResponse, logging_ctx: dict):
        """Finish logging after the completion method is finished."""
        metadata_kv = {}
        additional_headers = getattr(response, "_hidden_params", {}).get("additional_headers", {}) or {}
        remaining_tpm = additional_headers.get("llm_provider-x-ratelimit-remaining-tokens", "-1")
        remaining_rq = additional_headers.get("llm_provider-x-ratelimit-remaining-requests", "-1")
        metadata_kv["TPM"] = f"{int(remaining_tpm):,}"
        metadata_kv["RQ"] = f"{int(remaining_rq):,}"

        json_log_dir = logging_ctx["json_log_dir"]
        curr_step = logging_ctx["curr_step"]

        io_log_kwargs = {}
        io_log_kwargs["output_json"] = dict(
            obj=response.model_dump(), log_dir=json_log_dir, filename=f"{curr_step}_response.json"
        )
        return response, metadata_kv, io_log_kwargs

    async def acompletion(self, messages: list[dict], tools: Optional[list[dict]] = None, **kwargs: Any):
        """Call the chat completion endpoint and return results."""
        io_log_kwargs: dict[str, Any] = {}
        return_metadata = kwargs.pop("return_metadata", False)

        request_kwargs, logging_ctx, pre_io = self.pre_completion(messages=messages, tools=tools, **kwargs)
        io_log_kwargs.update(pre_io)
        if self.config.instant_log:
            await awrite_json(**io_log_kwargs["input_json"])

        try:
            if self._api_key is not None:
                request_kwargs["api_key"] = self._api_key
            response = await make_acall_with_ratelimit_pause(**request_kwargs)
        except (ContentPolicyViolationError, ContextWindowExceededError, BadRequestError) as e:
            if isinstance(e, BadRequestError):
                if not (
                    "ContentPolicyViolationError".lower() in str(e).lower()
                    or "ContextWindowExceededError".lower() in str(e).lower()
                    or ("context" in str(e).lower() and "window" in str(e).lower())
                ):
                    raise
            if self.config.strict_error_handling:
                raise
            print(LLM_ERROR_PREFIX + " " + str(e))
            return LLM_ERROR_PREFIX + " " + str(e)

        _, metadata_kv, post_io = self.post_completion(
            request_kwargs=request_kwargs, response=response, logging_ctx=logging_ctx
        )
        io_log_kwargs.update(post_io)

        if self.config.instant_log:
            await awrite_json(**io_log_kwargs["output_json"])

        if return_metadata:
            output = dict(response=response, metadata_kv=metadata_kv, io_log_kwargs=io_log_kwargs)
            try:
                r = response.model_dump()
                r.pop("choices", None)
                output["api_response_extras"] = r
            except Exception:
                pass
            return output
        return response
