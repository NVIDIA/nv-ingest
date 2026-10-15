# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_retriever.models import hf_model_registry as registry


def test_extraction_hf_repos_have_pinned_revisions():
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v1"] == "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-ocr-v2"] == "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca"
    assert registry.HF_MODEL_REVISIONS["nvidia/nemotron-page-elements-v3"] == "df62dbb631502575ac4d43b44d700b1674ab1d56"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Parse-v1.2"] == "2bd0189bffd6cdded6280d9f22a4077b25a504e3"
    )
    assert registry.get_hf_revision("nvidia/NVIDIA-Nemotron-Parse-2.0") == "b6742064f4a8cf22a10383ece5e7fbead355ac04"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/nemotron-table-structure-v1"] == "9350162faa1110320af62699105780b0c87b73ad"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"]
        == "5d250e2e111dc5e1434131bdf3d590c27a878ade"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"]
        == "7394488badb786e1decc0e00e308de1cab9560e6"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"]
        == "b8d3c170d9ee3a078917ef9bfd508eff988d6de7"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"]
        == "24e67ea000b7c2837fc8f9488aa2008524fac8ba"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"]
        == "6647b845a4b786c6e2c7adb1b6a909e1aa71fac2"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"]
        == "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
    )


def test_local_hf_nemotron_models_use_transformers_5_compatible_revisions() -> None:
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/llama-nemotron-embed-1b-v2"] == "113abe4acafa848e77ead9c0623205e511932348"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/llama-nemotron-embed-vl-1b-v2"]
        == "582e3bf72aee355e3c59ed89de53543c5b0657ee"
    )
    assert registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Embed-1B-BF16"] == "c0c9fea93ea424587517f2c59e20db9f1d6bf615"
    assert registry.HF_MODEL_REVISIONS["nvidia/Nemotron-3-Embed-1B-NVFP4"] == "f630128278eb245579d3ea022b4f659fbd614318"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/llama-nemotron-rerank-vl-1b-v2"]
        == "9c20c4aedf9ec87b6b7346c3bc4754ea030dab35"
    )
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"]
        == "420ba7d28211abf116b8b103ab700d92619daf98"
    )


def test_local_hf_asr_and_agent_models_use_approved_revisions() -> None:
    assert registry.HF_MODEL_REVISIONS["nvidia/parakeet-ctc-1.1b"] == "20e63a0fed6aedba145b74b826dbd41df0941730"
    assert (
        registry.HF_MODEL_REVISIONS["nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"]
        == "420ba7d28211abf116b8b103ab700d92619daf98"
    )


def test_hf_hub_download_with_pinned_revision_injects_known_revision(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/cache/model.bin"

    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    out = registry.hf_hub_download_with_pinned_revision(
        repo_id="nvidia/nemotron-ocr-v1",
        filename="checkpoints/det_model.pt",
    )

    assert out == "/cache/model.bin"
    assert calls == [
        (
            (),
            {
                "repo_id": "nvidia/nemotron-ocr-v1",
                "filename": "checkpoints/det_model.pt",
                "revision": "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb",
            },
        )
    ]


def test_hf_hub_download_with_pinned_revision_preserves_explicit_revision(monkeypatch):
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/cache/model.bin"

    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    registry.hf_hub_download_with_pinned_revision(
        "nvidia/nemotron-ocr-v1",
        filename="checkpoints/det_model.pt",
        revision="custom-sha",
    )

    assert calls[0][0] == ("nvidia/nemotron-ocr-v1",)
    assert calls[0][1]["revision"] == "custom-sha"


def test_hf_hub_download_with_pinned_revision_adds_startup_context(monkeypatch):
    class LocalEntryNotFoundError(Exception):
        pass

    LocalEntryNotFoundError.__module__ = "huggingface_hub.errors"

    def fake_download(*args, **kwargs):
        raise LocalEntryNotFoundError("cache miss")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(registry, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError) as exc_info:
        registry.hf_hub_download_with_pinned_revision(
            repo_id="nvidia/nemotron-ocr-v1",
            filename="checkpoints/det_model.pt",
        )

    message = str(exc_info.value)
    assert "nvidia/nemotron-ocr-v1" in message
    assert "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb" in message
    assert "checkpoints/det_model.pt" in message
    assert "HF_HUB_OFFLINE=0" in message
    assert "HF token is unset" in message


def test_install_pinned_hf_hub_download_warns_when_module_lacks_downloader(caplog):
    module = SimpleNamespace(__name__="upstream_without_downloader")

    registry.install_pinned_hf_hub_download(module)

    assert "revision pinning was NOT applied" in caplog.text
    assert "upstream_without_downloader" in caplog.text
