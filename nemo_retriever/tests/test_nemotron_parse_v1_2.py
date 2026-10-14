# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local vLLM-backed Nemotron Parse model."""

import os
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_dependencies_support_transformers_tokenizers_floor():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    tokenizers = next(Requirement(dependency) for dependency in dependencies if dependency.startswith("tokenizers"))

    assert tokenizers.specifier.contains("0.23.1")
    assert not tokenizers.specifier.contains("0.23.0")


def test_applies_vllm_startup_defaults_before_constructing_llm(monkeypatch):
    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    def assert_startup_defaults(**_kwargs):
        assert os.environ["VLLM_DEEP_GEMM_WARMUP"] == "skip"
        return MagicMock()

    with (
        patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
        patch.object(mod, "configure_global_hf_cache_base"),
        patch.object(mod, "get_hf_revision", return_value="test-revision"),
        patch("vllm.LLM", side_effect=assert_startup_defaults),
        patch("vllm.SamplingParams"),
    ):
        mod.NemotronParseV12()


def test_v2_local_wrapper_defaults_to_approved_model() -> None:
    from nemo_retriever.models.local import nemotron_parse_2_0 as mod
    from nemo_retriever.operators.extract.parse.nemotron_parse import NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL

    assert NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL == "nvidia/NVIDIA-Nemotron-Parse-v1.2"

    with patch.object(mod.NemotronParseV12, "__init__", return_value=None) as init:
        mod.NemotronParse20()

    assert init.call_args.kwargs["model_path"] == "nvidia/NVIDIA-Nemotron-Parse-2.0"


def test_local_actor_keeps_v1_2_default(monkeypatch) -> None:
    import nemo_retriever.models.local as local_models
    from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseGPUActor

    loader = MagicMock()
    monkeypatch.setitem(local_models.__dict__, "NemotronParseV12", loader)

    actor = NemotronParseGPUActor()
    actor._ensure_model()

    loader.assert_called_once()


def test_v2_local_actor_forwards_configured_model(monkeypatch) -> None:
    import nemo_retriever.models.local as local_models
    from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseGPUActor

    loader = MagicMock()
    monkeypatch.setitem(local_models.__dict__, "NemotronParse20", loader)

    actor = NemotronParseGPUActor(nemotron_parse_model="nvidia/NVIDIA-Nemotron-Parse-2.0")
    actor._ensure_model()

    assert loader.call_args.kwargs["model_path"] == "nvidia/NVIDIA-Nemotron-Parse-2.0"


def test_v2_chart_tag_routes_to_chart_channel() -> None:
    from nemo_retriever.operators.extract.parse.nemotron_parse import _route_parsed_elements

    tables, charts, infographics, text = _route_parsed_elements(
        "<x_0.1><y_0.2>Revenue: 42<x_0.8><y_0.9><class_Chart>",
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )

    assert tables == []
    assert charts == [{"bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9], "text": "Revenue: 42"}]
    assert infographics == []
    assert text is None


def test_nim_payload_uses_tagged_prompt_for_v2() -> None:
    from nemo_retriever.models.nim.primitives.model_interface.nemotron_parse import NemotronParseModelInterface

    interface = NemotronParseModelInterface(model_name="nvidia/nemotron-parse-v2.0")
    payload = interface._prepare_nemotron_parse_payload(["aW1hZ2U="])

    assert [item["type"] for item in payload["messages"][0]["content"]] == ["text", "image_url"]


def test_nim_payload_keeps_hosted_build_contract_image_only() -> None:
    from nemo_retriever.models.nim.primitives.model_interface.nemotron_parse import NemotronParseModelInterface

    interface = NemotronParseModelInterface(model_name="nvidia/nemotron-parse")
    payload = interface._prepare_nemotron_parse_payload(["aW1hZ2U="])

    assert [item["type"] for item in payload["messages"][0]["content"]] == ["image_url"]
