# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from types import ModuleType

import pytest


NANO_BF16 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
NANO_FP8 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
NANO_NVFP4 = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"
NANO_REMOTE = "nvidia/nemotron-nano-12b-v2-vl"
OMNI_BF16 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
OMNI_FP8 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
OMNI_NVFP4 = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
OMNI_REMOTE = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def test_canonical_caption_defaults_are_omni():
    from nemo_retriever.common.modality.caption.model_profiles import (
        DEFAULT_LOCAL_CAPTION_MODEL_ID,
        DEFAULT_REMOTE_CAPTION_MODEL_ID,
    )
    from nemo_retriever.common.params import CaptionParams

    assert DEFAULT_LOCAL_CAPTION_MODEL_ID == OMNI_BF16
    assert DEFAULT_REMOTE_CAPTION_MODEL_ID == OMNI_REMOTE
    assert CaptionParams().model_name == OMNI_BF16
    assert "approximately 62 GiB" in CaptionParams.model_json_schema()["properties"]["model_name"]["description"]


_LOCAL_CAPTIONER_NEMO_IMPORT_MODULES = (
    "nemo_retriever.models.local.nemotron_vlm_captioner",
    "nemo_retriever.models.model",
)
_LOCAL_CAPTIONER_DEPENDENCY_MODULES = (
    "torch",
    "torch.cuda",
    "torch.cuda.nvtx",
    "torch.nn",
    "vllm",
)
_LOCAL_CAPTIONER_IMPORT_MODULES = _LOCAL_CAPTIONER_NEMO_IMPORT_MODULES + _LOCAL_CAPTIONER_DEPENDENCY_MODULES
_LOCAL_CAPTIONER_PARENT_ATTRS = (
    ("nemo_retriever.models.local", "nemotron_vlm_captioner"),
    ("nemo_retriever.utils", "nvtx"),
    ("nemo_retriever.models", "model"),
    ("torch", "cuda"),
    ("torch", "nn"),
    ("torch.cuda", "nvtx"),
)
_MISSING = object()


def _reload_model_profiles_with_fake_revision_lookup(monkeypatch, fake_get_hf_revision):
    import nemo_retriever.common.modality.caption as caption_pkg
    import nemo_retriever.models.hf_model_registry as hf_model_registry

    module_name = "nemo_retriever.common.modality.caption.model_profiles"
    original_module = sys.modules.get(module_name, _MISSING)
    original_parent_attr = caption_pkg.__dict__.get("model_profiles", _MISSING)

    monkeypatch.setattr(hf_model_registry, "get_hf_revision", fake_get_hf_revision)
    sys.modules.pop(module_name, None)
    caption_pkg.__dict__.pop("model_profiles", None)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        caption_pkg.__dict__.pop("model_profiles", None)
        if original_module is not _MISSING:
            sys.modules[module_name] = original_module
        if original_parent_attr is not _MISSING:
            caption_pkg.model_profiles = original_parent_attr


class _VllmImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise AssertionError("vLLM must not be imported before unsupported local model validation")
        return None


def _clear_local_captioner_import_state():
    for module_name in _LOCAL_CAPTIONER_IMPORT_MODULES:
        sys.modules.pop(module_name, None)
    for parent_name, attr_name in _LOCAL_CAPTIONER_PARENT_ATTRS:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            parent.__dict__.pop(attr_name, None)


@pytest.fixture
def isolated_local_captioner_imports():
    module_snapshot = {
        module_name: sys.modules.get(module_name, _MISSING) for module_name in _LOCAL_CAPTIONER_IMPORT_MODULES
    }
    attr_snapshot = {}
    for parent_name, attr_name in _LOCAL_CAPTIONER_PARENT_ATTRS:
        parent = sys.modules.get(parent_name)
        attr_snapshot[(parent_name, attr_name)] = (
            _MISSING if parent is None else parent.__dict__.get(attr_name, _MISSING)
        )

    _clear_local_captioner_import_state()
    try:
        yield
    finally:
        _clear_local_captioner_import_state()
        for module_name, module in module_snapshot.items():
            if module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module
        for (parent_name, attr_name), attr_value in attr_snapshot.items():
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if attr_value is _MISSING:
                parent.__dict__.pop(attr_name, None)
            else:
                setattr(parent, attr_name, attr_value)


def test_model_profiles_import_uses_non_strict_revision_lookup(monkeypatch):
    calls = []

    def fake_get_hf_revision(model_id, *, strict=True):
        calls.append((model_id, strict))
        if strict:
            raise ValueError(f"strict lookup used for {model_id}")
        return None

    module = _reload_model_profiles_with_fake_revision_lookup(monkeypatch, fake_get_hf_revision)

    assert module.caption_model_revisions() == {
        NANO_BF16: None,
        NANO_FP8: None,
        NANO_NVFP4: None,
        OMNI_BF16: None,
        OMNI_FP8: None,
        OMNI_NVFP4: None,
    }
    assert calls
    assert {strict for _model_id, strict in calls} == {False}


def test_nano_resolution_remains_unchanged():
    from nemo_retriever.common.modality.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl", target="local") == NANO_BF16
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-bf16", target="local") == NANO_BF16
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-fp8", target="local") == NANO_FP8
    assert resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad", target="local") == NANO_NVFP4
    assert resolve_caption_model_name(NANO_BF16, target="remote") == NANO_REMOTE
    assert resolve_caption_model_name(NANO_FP8, target="remote") == "nvidia/nemotron-nano-12b-v2-vl-fp8"
    assert resolve_caption_model_name(NANO_NVFP4, target="remote") == "nvidia/nemotron-nano-12b-v2-vl-nvfp4-qad"


def test_legacy_local_captioner_resolve_caption_model_name_shim_warns():
    from nemo_retriever.models.local.nemotron_vlm_captioner import resolve_caption_model_name

    with pytest.warns(DeprecationWarning, match="nemo_retriever.caption.model_profiles"):
        resolved = resolve_caption_model_name("nvidia/nemotron-nano-12b-v2-vl", target="local")

    assert resolved == NANO_BF16


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (OMNI_BF16, OMNI_BF16),
        (OMNI_FP8, OMNI_FP8),
        (OMNI_NVFP4, OMNI_NVFP4),
        (OMNI_REMOTE, OMNI_BF16),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16", OMNI_BF16),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8", OMNI_FP8),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4", OMNI_NVFP4),
    ],
)
def test_omni_local_names_resolve_to_hf_ids(name, expected):
    from nemo_retriever.common.modality.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name(name, target="local") == expected


@pytest.mark.parametrize(
    "name",
    [
        OMNI_BF16,
        OMNI_FP8,
        OMNI_NVFP4,
        OMNI_REMOTE,
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4",
    ],
)
def test_omni_remote_names_resolve_to_hosted_model(name):
    from nemo_retriever.common.modality.caption.model_profiles import resolve_caption_model_name

    assert resolve_caption_model_name(name, target="remote") == OMNI_REMOTE


def test_unknown_remote_model_name_passes_through():
    from nemo_retriever.common.modality.caption.model_profiles import (
        get_caption_model_profile,
        resolve_caption_model_name,
    )

    assert get_caption_model_profile("acme/custom-vlm", target="remote", strict=False) is None
    assert resolve_caption_model_name("acme/custom-vlm", target="remote") == "acme/custom-vlm"


def test_unknown_local_profile_raises_clear_error():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    with pytest.raises(ValueError) as exc_info:
        get_caption_model_profile("acme/custom-vlm", target="local")

    message = str(exc_info.value)
    assert "Unsupported caption model" in message
    assert "target='local'" in message
    assert NANO_BF16 in message
    assert OMNI_BF16 in message


def test_omni_profile_has_request_defaults_and_current_capabilities():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_FP8, target="local")

    assert profile.family == "nemotron-3-nano-omni"
    assert profile.variant == "FP8"
    assert profile.local_model_id == OMNI_FP8
    assert profile.remote_model_id == OMNI_REMOTE
    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.request_extras_for("remote") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.capabilities.image_captioning is True
    assert profile.capabilities.audio_input is False
    assert profile.capabilities.video_input is False
    assert profile.capabilities.document_intelligence is False
    assert profile.capabilities.reasoning_control is True


def test_request_extras_are_defensive_copies():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_BF16, target="local")
    extras = profile.request_extras_for("local")
    extras["chat_template_kwargs"]["enable_thinking"] = True

    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_merge_request_extras_deep_merges_with_user_values_winning():
    from nemo_retriever.common.modality.caption.model_profiles import merge_request_extras

    defaults = {
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 0},
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
    }
    user = {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 1,
    }

    assert merge_request_extras(defaults, user) == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 0},
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
        "top_k": 1,
    }
    assert defaults["chat_template_kwargs"]["enable_thinking"] is False


def test_supported_local_names_include_ids_and_aliases():
    from nemo_retriever.common.modality.caption.model_profiles import supported_caption_model_names

    names = supported_caption_model_names(target="local")

    assert NANO_BF16 in names
    assert OMNI_BF16 in names
    assert "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8" in names


def test_public_request_extra_fields_are_immutable():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_BF16, target="local")

    with pytest.raises(TypeError):
        profile.local_request_extras["chat_template_kwargs"]["enable_thinking"] = True
    with pytest.raises(TypeError):
        profile.remote_request_extras["chat_template_kwargs"]["enable_thinking"] = True

    assert profile.request_extras_for("local") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert profile.request_extras_for("remote") == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize(
    ("model_name", "expected_engine"),
    [
        (NANO_FP8, {"dtype": "auto"}),
        (OMNI_FP8, {"dtype": "auto"}),
    ],
)
def test_modelopt_fp8_profiles_preserve_checkpoint_quantization(model_name, expected_engine):
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(model_name, target="local")

    engine_kwargs = profile.engine_kwargs_for_local()

    assert engine_kwargs == expected_engine
    assert "quantization" not in engine_kwargs
    assert "hf_overrides" not in engine_kwargs


def test_public_nano_fp8_engine_kwargs_are_immutable():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(NANO_FP8, target="local")

    with pytest.raises(TypeError):
        profile.local_engine_kwargs["dtype"] = "bfloat16"

    assert profile.engine_kwargs_for_local() == {"dtype": "auto"}


def test_public_omni_fp8_engine_kwargs_are_immutable():
    from nemo_retriever.common.modality.caption.model_profiles import get_caption_model_profile

    profile = get_caption_model_profile(OMNI_FP8, target="local")

    with pytest.raises(TypeError):
        profile.local_engine_kwargs["dtype"] = "bfloat16"

    assert profile.engine_kwargs_for_local() == {"dtype": "auto"}


def _install_fake_torch():
    fake_torch = ModuleType("torch")
    fake_cuda = ModuleType("torch.cuda")
    fake_nvtx = ModuleType("torch.cuda.nvtx")
    fake_nn = ModuleType("torch.nn")

    class FakeModule:
        pass

    for fake_module in (fake_torch, fake_cuda, fake_nvtx, fake_nn):
        fake_module._nemo_retriever_test_fake = True
    fake_nvtx.range_push = lambda *_args, **_kwargs: None
    fake_nvtx.range_pop = lambda: None
    fake_cuda.nvtx = fake_nvtx
    fake_nn.Module = FakeModule
    fake_torch.cuda = fake_cuda
    fake_torch.nn = fake_nn
    sys.modules["torch"] = fake_torch
    sys.modules["torch.cuda"] = fake_cuda
    sys.modules["torch.cuda.nvtx"] = fake_nvtx
    sys.modules["torch.nn"] = fake_nn


def _install_fake_vllm():
    _install_fake_torch()
    fake_vllm = ModuleType("vllm")
    fake_vllm._nemo_retriever_test_fake = True

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTextOutput:
        text = " generated caption "

    class FakeRequestOutput:
        outputs = [FakeTextOutput()]

    class FakeLLM:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.deep_gemm_warmup = os.environ.get("VLLM_DEEP_GEMM_WARMUP")
            self.chat_calls = []
            FakeLLM.instances.append(self)

        def chat(self, conversations, sampling_params, **kwargs):
            self.chat_calls.append(
                {
                    "conversations": conversations,
                    "sampling_params": sampling_params,
                    "kwargs": kwargs,
                }
            )
            return [FakeRequestOutput() for _ in conversations]

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    sys.modules["vllm"] = fake_vllm
    return FakeLLM, FakeSamplingParams


@pytest.mark.parametrize(
    ("model_name", "expected_revision", "expected_engine"),
    [
        (
            NANO_FP8,
            "7394488badb786e1decc0e00e308de1cab9560e6",
            {"dtype": "auto"},
        ),
        (
            OMNI_BF16,
            "24e67ea000b7c2837fc8f9488aa2008524fac8ba",
            {"dtype": "bfloat16"},
        ),
        (
            OMNI_FP8,
            "6647b845a4b786c6e2c7adb1b6a909e1aa71fac2",
            {"dtype": "auto"},
        ),
        (
            OMNI_NVFP4,
            "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe",
            {"dtype": "auto", "quantization": "modelopt"},
        ),
    ],
)
def test_local_captioner_uses_profile_metadata(
    isolated_local_captioner_imports, model_name, expected_revision, expected_engine
):
    FakeLLM, _FakeSamplingParams = _install_fake_vllm()

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    captioner = NemotronVLMCaptioner(
        model_path=model_name,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.25,
    )

    llm_kwargs = FakeLLM.instances[-1].kwargs
    assert captioner.model_name == model_name
    assert llm_kwargs["model"] == model_name
    assert llm_kwargs["revision"] == expected_revision
    assert llm_kwargs["trust_remote_code"] is True
    assert llm_kwargs["tensor_parallel_size"] == 2
    assert llm_kwargs["gpu_memory_utilization"] == 0.25
    assert llm_kwargs["max_num_seqs"] == 256
    for key, value in expected_engine.items():
        assert llm_kwargs[key] == value
    if model_name in {NANO_FP8, OMNI_FP8}:
        assert "quantization" not in llm_kwargs
        assert "hf_overrides" not in llm_kwargs


def test_local_captioner_applies_vllm_startup_defaults_before_constructing_llm(
    isolated_local_captioner_imports, monkeypatch
):
    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)
    FakeLLM, _FakeSamplingParams = _install_fake_vllm()

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    NemotronVLMCaptioner(model_path=OMNI_BF16)

    assert FakeLLM.instances[-1].deep_gemm_warmup == "skip"
    assert FakeLLM.instances[-1].kwargs["gpu_memory_utilization"] == 0.95


@pytest.mark.parametrize(
    ("model_name", "expected_utilization"),
    [
        (NANO_BF16, 0.5),
        (OMNI_BF16, 0.95),
    ],
)
def test_local_captioner_uses_profile_gpu_memory_default(
    isolated_local_captioner_imports, model_name, expected_utilization
):
    FakeLLM, _FakeSamplingParams = _install_fake_vllm()

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    NemotronVLMCaptioner(model_path=model_name)

    assert FakeLLM.instances[-1].kwargs["gpu_memory_utilization"] == expected_utilization


def test_caption_params_preserves_an_omitted_gpu_memory_utilization():
    from nemo_retriever.common.params import CaptionParams

    assert CaptionParams(model_name=OMNI_BF16).gpu_memory_utilization is None
    assert CaptionParams(model_name=OMNI_BF16, gpu_memory_utilization=0.8).gpu_memory_utilization == 0.8


def test_local_captioner_passes_omni_no_think_chat_kwargs(isolated_local_captioner_imports):
    FakeLLM, _FakeSamplingParams = _install_fake_vllm()

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    captioner = NemotronVLMCaptioner()

    assert captioner.model_name == OMNI_BF16

    assert captioner.caption_batch(["abc123"]) == ["generated caption"]
    chat_kwargs = FakeLLM.instances[-1].chat_calls[-1]["kwargs"]
    assert chat_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat_kwargs["use_tqdm"] is False


def test_local_captioner_user_extra_body_overrides_profile_extras(isolated_local_captioner_imports):
    FakeLLM, _FakeSamplingParams = _install_fake_vllm()

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    captioner = NemotronVLMCaptioner(model_path=OMNI_BF16)
    captioner.caption_batch(
        ["abc123"],
        extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 32}},
    )

    chat_kwargs = FakeLLM.instances[-1].chat_calls[-1]["kwargs"]
    assert chat_kwargs["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 32}
    assert chat_kwargs["use_tqdm"] is False


def test_local_captioner_rejects_unknown_model_before_vllm_import(isolated_local_captioner_imports, monkeypatch):
    _install_fake_torch()
    sys.modules.pop("vllm", None)
    monkeypatch.setattr(sys, "meta_path", [_VllmImportBlocker(), *sys.meta_path])

    from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

    with pytest.raises(ValueError, match="Unsupported caption model"):
        NemotronVLMCaptioner(model_path="acme/custom-vlm")


def test_local_captioner_fake_imports_do_not_leak():
    leaked_modules = sorted(
        module_name
        for module_name in _LOCAL_CAPTIONER_IMPORT_MODULES
        if getattr(sys.modules.get(module_name), "_nemo_retriever_test_fake", False)
    )

    assert leaked_modules == []
