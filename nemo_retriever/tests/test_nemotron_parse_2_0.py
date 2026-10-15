# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse 2.0 request contracts, local loading, and output compatibility."""

from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.operators.extract.parse.nemotron_parse import (
    NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
    NemotronParseGPUActor,
    _route_parsed_elements,
    nemotron_parse_pages,
)

TAGGED_OUTPUT = (
    "<x_0><y_0>Body text<x_1><y_1><class_Text>"
    "<x_0><y_0>A | B<x_1><y_1><class_Table>"
    "<x_0><y_0>Revenue: 42<x_1><y_1><class_Chart>"
    "<x_0><y_0>Picture text<x_1><y_1><class_Picture>"
)


@pytest.mark.parametrize(
    ("endpoint", "model", "decoding"),
    [
        (
            "http://parse:8000/v1/chat/completions",
            "nvidia/nemotron-parse-v2.0",
            {"max_tokens": 8192, "top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": False},
        ),
        (
            "https://integrate.api.nvidia.com/v1/chat/completions",
            "nvidia/nemotron-parse-2.0",
            {"max_tokens": 8192},
        ),
    ],
)
def test_v2_wire_payload_and_tagged_routing(monkeypatch, endpoint, model, decoding):
    sent = []

    def fake_post(**kwargs):
        sent.append(kwargs)
        return {"choices": [{"message": {"content": TAGGED_OUTPUT}}]}

    monkeypatch.setattr("nemo_retriever.models.nim.nim._post_with_retries", fake_post)
    batch = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}, {}], "text": ["Native", "Untouched"]})
    result = nemotron_parse_pages(
        batch,
        invoke_url=endpoint,
        nemotron_parse_model=model,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )

    assert len(sent) == 1
    assert sent[0]["invoke_url"] == endpoint
    assert sent[0]["payload"] == {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": NEMOTRON_PARSE_DEFAULT_TASK_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            }
        ],
        **decoding,
    }
    assert result["text"].tolist() == ["Body text", "Untouched"]
    for channel in ("table", "chart", "infographic"):
        assert len(result.at[0, channel]) == 1
        assert result.at[1, channel] == []
        assert result[channel].tolist() == result[f"{channel}_parse"].tolist()
    assert result.at[0, "chart"][0]["text"] == "Revenue: 42"
    assert result.at[0, "nemotron_parse_v1_2"]["error"] is None


def test_v2_disabled_structured_channels_preserve_text():
    tables, charts, pictures, text = _route_parsed_elements(
        TAGGED_OUTPUT, extract_tables=False, extract_charts=False, extract_infographics=False
    )
    assert (tables, charts, pictures) == ([], [], [])
    assert all(value in text for value in ("Body text", "Revenue: 42", "Picture text"))


@pytest.mark.parametrize("model", ["nvidia/nemotron-parse-v2.0", "nvidia/nemotron-parse-2.0"])
def test_v2_errors_keep_existing_metadata(monkeypatch, model):
    def fail(**kwargs):
        raise RuntimeError("Parse unavailable")

    monkeypatch.setattr("nemo_retriever.operators.extract.parse.nemotron_parse.invoke_chat_completions_images", fail)
    batch = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}], "text": ["Native"]})
    result = nemotron_parse_pages(
        batch,
        invoke_url="http://parse:8000/v1/chat/completions",
        nemotron_parse_model=model,
        extract_text=True,
    )
    assert result.at[0, "text"] == "Native"
    error = result.at[0, "nemotron_parse_v1_2"]["error"]
    assert (error["stage"], error["type"], error["message"]) == (
        "nemotron_parse_pages",
        "RuntimeError",
        "Parse unavailable",
    )


@pytest.mark.parametrize("model", ["custom/parse", "nvidia/nemotron-parse-v2.0", "nvidia/nemotron-parse-2.0"])
def test_direct_local_actor_rejects_unsupported_model_before_loading(monkeypatch, model):
    import nemo_retriever.models.local as local_models

    loader = MagicMock()
    monkeypatch.setitem(local_models.__dict__, "NemotronParse20", loader)
    with pytest.raises(ValueError, match="Local Nemotron Parse supports only"):
        NemotronParseGPUActor(nemotron_parse_model=model)
    loader.assert_not_called()


def test_local_v2_uses_pinned_checkpoint_and_preserves_tags(monkeypatch):
    from nemo_retriever.models.local import NemotronParse20
    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    monkeypatch.setenv("TQDM_DISABLE", "1")
    output = SimpleNamespace(outputs=[SimpleNamespace(text=TAGGED_OUTPUT)])
    with (
        patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
        patch.object(mod, "configure_global_hf_cache_base"),
        patch.object(NemotronParse20, "_vllm_runtime", return_value=nullcontext()),
        patch("vllm.LLM") as llm,
        patch("vllm.SamplingParams") as sampling,
    ):
        llm.return_value.generate.return_value = [output]
        model = NemotronParse20()
        assert model.invoke_batch([Image.new("RGB", (2, 2))]) == [TAGGED_OUTPUT]

    assert llm.call_args.kwargs["model"] == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert llm.call_args.kwargs["revision"] == "b6742064f4a8cf22a10383ece5e7fbead355ac04"
    assert sampling.call_args.kwargs == {
        "temperature": 0,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "max_tokens": 9000,
        "skip_special_tokens": False,
    }
    assert llm.return_value.generate.call_args.kwargs["use_tqdm"] is False
    assert model.model_name == "NVIDIA-Nemotron-Parse-2.0"
    assert model.model_name in model.output["description"]


@pytest.mark.parametrize(
    ("model", "decoding"),
    [
        ("nvidia/nemotron-parse-v2.0", {"max_tokens": 8192, "top_k": 1, "skip_special_tokens": False}),
        ("nvidia/nemotron-parse-2.0", {"max_tokens": 8192}),
    ],
)
def test_lower_level_v2_payload_matches_endpoint_decoding(model, decoding):
    from nemo_retriever.models.nim.primitives.model_interface.nemotron_parse import NemotronParseModelInterface

    interface = NemotronParseModelInterface(model_name=model)
    payload = interface._prepare_nemotron_parse_payload(["aW1hZ2U="])
    assert payload["messages"][0]["content"][0] == {"type": "text", "text": NEMOTRON_PARSE_DEFAULT_TASK_PROMPT}
    for field, value in decoding.items():
        assert payload[field] == value
    if model == "nvidia/nemotron-parse-2.0":
        assert not {"repetition_penalty", "top_k", "skip_special_tokens"} & payload.keys()
    else:
        assert payload["repetition_penalty"] == 1.1


@pytest.mark.parametrize("picture_text", [None, False, True])
def test_self_hosted_v2_reserves_context_for_task_prompt(monkeypatch, picture_text):
    """Reproduce the 2.0.8-variant NIM's 9,000-token total-context admission."""

    def context_limited_post(**kwargs):
        payload = kwargs["payload"]
        prompt = payload["messages"][0]["content"][0]["text"]
        # Character count bounds these short tag-only task prompts conservatively.
        if payload["max_tokens"] + len(prompt) > 9000:
            raise ValueError("NIM context limit includes the task prompt")
        return {"choices": [{"message": {"content": TAGGED_OUTPUT}}]}

    monkeypatch.setattr("nemo_retriever.models.nim.nim._post_with_retries", context_limited_post)
    batch = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}], "text": ["Native"]})
    result = nemotron_parse_pages(
        batch,
        invoke_url="http://parse:8000/v1/chat/completions",
        nemotron_parse_model="nvidia/nemotron-parse-v2.0",
        nemotron_parse_predict_text_in_pic=picture_text,
        extract_text=True,
    )
    assert result.at[0, "text"].startswith("Body text")
    assert result.at[0, "nemotron_parse_v1_2"]["error"] is None
