# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Served Parse uses the model package and keeps per-page completion evidence."""

import base64
import io

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.models.nim import nim
from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

RAW = "<x_0.1><y_0.1>Hello<x_0.9><y_0.2><class_Text>"


def page(color):
    stream = io.BytesIO()
    Image.new("RGB", (24, 32), color).save(stream, format="PNG")
    return {"page_image": {"image_b64": base64.b64encode(stream.getvalue()).decode()}}


def completion(text=RAW, reason="stop"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": reason}],
        "usage": {"completion_tokens": 9},
    }


def test_chat_metadata_is_opt_in(monkeypatch):
    monkeypatch.setattr(nim, "_post_with_retries", lambda **kwargs: completion(reason="length"))
    client = nim.NIMClient()
    try:
        args = dict(
            invoke_url="http://example/v1/chat/completions",
            messages_list=[[{"role": "user", "content": "x"}]],
        )
        assert client.invoke_chat_completions(**args) == [RAW]
        assert client.invoke_chat_completions(**args, return_metadata=True) == [
            {"text": RAW, "finish_reason": "length", "usage": {"completion_tokens": 9}}
        ]
    finally:
        client.shutdown()


def test_served_model_uses_package_settings_and_preserves_successful_siblings(
    monkeypatch,
):
    pytest.importorskip("nemotron_parse.inference")
    requests = []

    def post(**kwargs):
        payload = kwargs["payload"]
        requests.append(payload)
        encoded = payload["messages"][0]["content"][-1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            reason = "length" if image.getpixel((0, 0))[0] == 255 else "stop"
        return completion(reason=reason)

    monkeypatch.setattr(nim, "_post_with_retries", post)
    result = nemotron_parse_pages(
        pd.DataFrame([page("white"), page("black")]),
        invoke_url="http://example/v1/chat/completions",
        nemotron_parse_model="nvidia/Nemotron-3.5-Parse",
        extract_text=True,
    )
    failed, good = result["nemotron_parse_v1_2"].tolist()
    assert failed["raw_output"] == RAW
    assert failed["finish_reason"] == "length"
    assert failed["error"]["stage"] == "nemotron_parse_pages_fullpage"
    assert good["bbox_coordinate_space"] == "normalized_page_content"
    assert failed["bbox_coordinate_space"] == "normalized_page_content"
    assert good["error"] is None
    assert good["finish_reason"] == "stop"
    assert good["usage"] == {"completion_tokens": 9}
    assert result["text"].tolist() == ["", "Hello"]
    assert len(requests) == 2
    for request in requests:
        assert request["temperature"] == 0.2
        assert request["repetition_penalty"] == 1.05
        assert request["top_k"] == 50
        assert request["skip_special_tokens"] is False
        assert "<predict_text_in_pic>" in request["messages"][0]["content"][0]["text"]


def test_chat_length_completion_is_flagged_with_raw_evidence(monkeypatch):
    monkeypatch.setattr(nim, "_post_with_retries", lambda **kwargs: completion(reason="length"))
    result = nemotron_parse_pages(
        pd.DataFrame([page("black")]),
        invoke_url="http://example/v1/chat/completions",
        extract_text=True,
    )
    meta = result.iloc[0]["nemotron_parse_v1_2"]
    assert meta["raw_output"] == RAW
    assert meta["finish_reason"] == "length"
    assert meta["error"]["type"] == "IncompleteModelOutputError"


def test_native_batch_failure_attaches_raw_only_to_its_page(monkeypatch):
    monkeypatch.setattr(
        nim,
        "_post_with_retries",
        lambda **kwargs: {"content": "partial", "finish_reason": "length"},
    )
    result = nemotron_parse_pages(
        pd.DataFrame([page("black")]),
        invoke_url="http://example/v1/parse",
        extract_text=True,
    )
    meta = result.iloc[0]["nemotron_parse_v1_2"]
    assert meta["raw_output"] == "partial"
    assert meta["finish_reason"] == "length"
    assert meta["error"] is not None
