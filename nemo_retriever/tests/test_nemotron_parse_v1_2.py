# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local vLLM-backed Nemotron Parse model."""

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
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


def _local_wrapper(outputs):
    from nemo_retriever.models.local.nemotron_parse_v1_2 import NemotronParseV12

    with patch.object(NemotronParseV12, "__init__", return_value=None):
        model = NemotronParseV12()

    model._task_prompt = "prompt"
    model._sampling_params = object()
    model.preprocess = MagicMock(side_effect=lambda image: image)
    model._llm = MagicMock()
    model._llm.generate.return_value = outputs
    return model


def test_local_wrapper_retains_finish_reason_without_changing_legacy_text_contract() -> None:
    raw_output = "\n<x_0><y_0>Hello<x_1><y_1><class_Text>\t"
    model = _local_wrapper([SimpleNamespace(outputs=[SimpleNamespace(text=raw_output, finish_reason="length")])])

    assert model.invoke_batch_with_finish_reasons(["image"], task_prompt="prompt") == [(raw_output, "length")]
    assert model.invoke_batch(["image"], task_prompt="prompt") == [raw_output.strip()]


def test_local_wrapper_never_shifts_results_when_a_completion_is_missing() -> None:
    model = _local_wrapper(
        [SimpleNamespace(outputs=[]), SimpleNamespace(outputs=[SimpleNamespace(text="page", finish_reason="stop")])]
    )

    with pytest.raises(IndexError):
        model.invoke_batch(["first", "second"])

