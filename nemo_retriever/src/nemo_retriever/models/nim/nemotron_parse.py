# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-team document parsing over an OpenAI-compatible served endpoint."""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Sequence

from nemo_retriever.models.nim.nim import NIMClient, _parse_invoke_urls


def invoke_document_parse_images(
    *,
    invoke_url: str,
    image_b64_list: Sequence[str],
    model: str,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    max_pool_workers: int = 16,
    max_retries: int = 10,
    max_429_retries: int = 5,
    nim_client: NIMClient | None = None,
) -> list[dict[str, Any]]:
    """Parse pages concurrently with the model package's prompts and sampling.

    Results retain input order. Each failed page has raw output, completion
    metadata, and an error; a page failure does not discard successful siblings.
    The optional package is imported only for this served model contract.
    """
    try:
        from nemotron_parse.inference import DocumentParser
    except ImportError as exc:
        raise ImportError(
            "Served Nemotron Parse 3.5 requires the optional nemotron-parse package. "
            "Install the model team's pinned package with --no-deps: "
            "pip install --no-deps 'git+https://huggingface.co/nvidia/Nemotron-3.5-Parse"
            "@e7ea79242a87ea0ad3919b672bfedb429a683206'. "
            "Refer to the Parse documentation for vLLM plugin isolation."
        ) from exc
    from PIL import Image

    endpoints = _parse_invoke_urls(invoke_url)
    client = nim_client or NIMClient(max_pool_workers=max_pool_workers)

    def parse_one(item: tuple[int, str]) -> dict[str, Any]:
        index, encoded = item
        endpoint = endpoints[index % len(endpoints)]

        def backend(image: Any, *, prompt: str, stage: str, settings: Any) -> dict[str, Any]:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            sampling = asdict(settings)
            response = client.invoke_chat_completions_images(
                invoke_url=endpoint,
                image_b64_list=[base64.b64encode(buffer.getvalue()).decode("ascii")],
                model=model,
                api_key=api_key,
                timeout_s=timeout_s,
                task_prompt=prompt,
                temperature=sampling.pop("temperature"),
                repetition_penalty=sampling.pop("repetition_penalty"),
                extra_body={**sampling, "skip_special_tokens": False},
                max_retries=max_retries,
                max_429_retries=max_429_retries,
                return_metadata=True,
            )[0]
            return {
                "raw_out": response["text"],
                **{k: response[k] for k in ("finish_reason", "usage")},
            }

        try:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as original:
                image = original.convert("RGB")
            image.info.clear()
            with DocumentParser(model=model, backend=backend) as parser:
                document = parser.parse(image, mode="single")
            first = document["stages"]["fullpage"]
            return {
                "raw_output": first["raw_out"],
                "routing_output": document["raw_out"],
                "finish_reason": first.get("finish_reason"),
                "usage": first.get("usage"),
                "inference": document["inference"],
                "stages": document["stages"],
                "error": None,
            }
        except Exception as exc:
            response = getattr(exc, "response", {})
            if not isinstance(response, dict):
                response = {}
            return {
                "raw_output": response.get("raw_out"),
                "routing_output": "",
                "finish_reason": response.get("finish_reason"),
                "usage": response.get("usage"),
                "error": {
                    "stage": "nemotron_parse_pages_" + str(getattr(exc, "stage", "request")),
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_pool_workers))) as pool:
            return list(pool.map(parse_one, enumerate(image_b64_list)))
    finally:
        if nim_client is None:
            client.shutdown()
