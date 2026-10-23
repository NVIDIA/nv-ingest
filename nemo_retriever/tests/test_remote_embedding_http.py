# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for remote OpenAI-compatible embedding requests."""

from __future__ import annotations

import builtins
import json
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
from nemo_retriever.models.inference.main_text_embed import (
    TextEmbeddingConfig,
    _async_request_handler,
    _close_embed_http_resources,
    _get_embed_http_client,
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


def test_async_request_handler_reuses_http_client_and_executor():
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    request_args = (
        [["text"]],
        None,
        "http://embedding.test/v1",
        "test/model",
        None,
        "float",
        "passage",
        "END",
        False,
    )
    _close_embed_http_resources()
    try:
        with patch("httpx.Client", return_value=client) as client_factory:
            assert _async_request_handler(*request_args, max_concurrent=2)[0]["embedding"] == [[0.5]]
            assert _async_request_handler(*request_args, max_concurrent=2)[0]["embedding"] == [[0.5]]
        assert client_factory.call_count == 1
        assert requests == 2
    finally:
        _close_embed_http_resources()


def test_retry_delay_honors_numeric_retry_after():
    assert _http_retry_delay_s(0, retry_after="7") == 7.0


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_http_client_import_preserves_unexpected_failures(monkeypatch, error_type):
    original_import = builtins.__import__
    failure = error_type("httpx import failed")

    def import_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise failure
        return original_import(name, *args, **kwargs)

    _close_embed_http_resources()
    monkeypatch.setattr(builtins, "__import__", import_httpx)
    with pytest.raises(RuntimeError) as caught:
        _get_embed_http_client()

    if error_type is ImportError:
        assert caught.value.__cause__ is failure
        assert "not installed" in str(caught.value)
    else:
        assert caught.value is failure


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


@pytest.mark.parametrize("input_type", ["passage", "query"])
def test_compose_alias_resolves_pinned_input_policy_and_stays_in_http_requests(monkeypatch, input_type):
    from nemo_retriever.models import EmbedModelSpec, NEMOTRON_35_EMBED_BF16_MODEL, resolve_embed_model
    from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS
    from nemo_retriever.models.inference import embedding_input

    alias = "nvidia/nemotron-3.5-embed-1b"
    checkpoint = NEMOTRON_35_EMBED_BF16_MODEL
    revision = HF_MODEL_REVISIONS[checkpoint]
    spec = EmbedModelSpec(
        model_id=checkpoint,
        revision=revision,
        family="vl",
        output_dimension=2048,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_tokens=32768,
        query_prefix_declared=True,
        document_prefix_declared=True,
        model_type="mistral3",
    )
    resolve_spec = MagicMock(return_value=spec)
    tokenizer = MagicMock()
    tokenizer.encode_batch.side_effect = lambda texts, **kwargs: [list(text.encode()) for text in texts]
    load_tokenizer = MagicMock(return_value=tokenizer)
    load_template = MagicMock(return_value="{{ messages[0].content ~ messages[1].content }}")
    monkeypatch.setattr(embedding_input, "resolve_embed_model_spec", resolve_spec)
    monkeypatch.setattr(embedding_input, "load_chunk_tokenizer", load_tokenizer)
    monkeypatch.setattr(embedding_input, "load_retrieval_template", load_template)

    # Exercise admission with the same logical model and no explicit checkpoint/revision as Compose.
    kwargs = {"embed_model_name": alias, "embedding_endpoint": "http://embedding.test/v1", "input_type": input_type}
    policy = embedding_input.configure_embedding_input_policy(kwargs)
    resolve_spec.assert_called_once_with(checkpoint, revision=None, hf_cache_dir=None)
    load_tokenizer.assert_called_once_with(checkpoint, cache_dir=None, revision=revision)
    load_template.assert_called_once_with(checkpoint, revision, None)
    assert policy.plan(["short"])[0].formatted_tokens == len(f"{input_type}: short")
    assert kwargs["embed_model_name"] == resolve_embed_model(alias) == alias

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == alias
        assert payload["input_type"] == input_type
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with patch("httpx.Client", return_value=client):
        assert _request(model_name=kwargs["embed_model_name"], input_type=input_type) == [[0.1, 0.2]]
