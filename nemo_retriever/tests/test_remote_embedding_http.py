# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for remote OpenAI-compatible embedding requests."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    _http_embed_openai_compat,
    _http_retry_delay_s,
    create_text_embeddings_for_df,
)


def _request(**overrides):
    kwargs = {
        "prompts": ['web text <img src="data:image/png;base64,not-an-image">'],
        "api_key": None,
        "endpoint_url": "http://embedding.test/v1",
        "model_name": "test/model",
        "encoding_format": "float",
        "input_type": "passage",
        "truncate": "END",
        "modalities": ["text"],
        "max_retries": 2,
        "max_429_retries": 1,
    }
    kwargs.update(overrides)
    return _http_embed_openai_compat(**kwargs)


def test_remote_embedding_sends_explicit_text_modality():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("httpx.Client", return_value=client):
        assert _request() == [[0.1, 0.2]]

    assert payloads[0]["modality"] == "text"


def test_remote_embedding_retries_transient_status():
    statuses = iter([503, 200])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.3]}]},
            )
        return httpx.Response(status, text="temporarily unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with (
        patch("httpx.Client", return_value=client),
        patch("nemo_retriever.models.inference.main_text_embed.time.sleep") as sleep,
    ):
        assert _request() == [[0.3]]

    assert calls == 2
    sleep.assert_called_once()


def test_remote_embedding_does_not_retry_bad_request_and_preserves_detail():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="set modality='text'")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("httpx.Client", return_value=client):
        with pytest.raises(RuntimeError, match="HTTP 400.*set modality='text'"):
            _request()

    assert calls == 1


def test_retry_delay_honors_numeric_retry_after():
    assert _http_retry_delay_s(0, retry_after="7") == 7.0


def test_text_dataframe_routes_explicit_text_modalities():
    captured = {}

    def fake_runner(*args, **kwargs):
        captured["modalities"] = kwargs["modalities"]
        return {"embeddings": [[0.1], [0.2]]}

    config = TextEmbeddingConfig(
        embedding_nim_endpoint="http://embedding.test/v1",
        batch_size=2,
        embed_modality="text",
    )
    frame = pd.DataFrame({"text": ["ordinary text", "data:image/png;base64,not-an-image"]})

    with patch(
        "nemo_retriever.models.inference.main_text_embed._async_runner",
        side_effect=fake_runner,
    ):
        create_text_embeddings_for_df(
            frame,
            task_config={"endpoint_url": "http://embedding.test/v1"},
            transform_config=config,
        )

    assert captured["modalities"] == [["text", "text"]]
