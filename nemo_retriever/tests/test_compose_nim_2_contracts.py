# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose contracts for the native extraction and embedding NIMs."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "dev/compose/service-mode.compose.yaml"
CORE_PRESET = ROOT / "dev/compose/presets/nims-core.env"
LOCAL_PRESET = ROOT / "dev/compose/presets/local-models.env"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_extraction_nims_use_distinct_native_models_and_supported_model_paths() -> None:
    services = _compose()["services"]
    expected = {
        "nim-page-elements": (
            "nvidia/nemotron-page-elements-v3",
            "page-elements",
            "NIM_PAGE_ELEMENTS_CACHE_PATH",
        ),
        "nim-table-structure": (
            "nvidia/nemotron-table-structure-v1",
            "table-structure",
            "NIM_TABLE_STRUCTURE_CACHE_PATH",
        ),
        "nim-ocr": ("nvidia/nemotron-ocr-v2", "ocr", "NIM_OCR_CACHE_PATH"),
    }

    for service_name, (model_name, model_dir, path_variable) in expected.items():
        service = services[service_name]
        env = service["environment"]
        assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
        assert env["NIM_ENGINE_MODEL_NAME"] == model_name
        assert env["NIM_ENGINE_MODEL_PATH"] == f"${{{path_variable}:-/model}}/{model_dir}"
        assert service["volumes"] == [
            f"nim_{service_name.removeprefix('nim-').replace('-', '_')}_cache:${{{path_variable}:-/model}}"
        ]
        assert service["healthcheck"]["test"] == [
            "CMD",
            "curl",
            "--fail",
            "--silent",
            "http://localhost:8000/v1/health/ready",
        ]
        assert not any(name.startswith("NIM_TRITON_") for name in env)

    assert services["nim-ocr"]["environment"]["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"
    assert "/model-store" not in COMPOSE.read_text(encoding="utf-8")


def test_embedding_nim_uses_native_runtime_and_matching_client_model() -> None:
    compose = _compose()
    service = compose["services"]["nim-embedding"]
    env = service["environment"]
    model = "nvidia/nemotron-3.5-embed-1b"
    assert service["image"] == ("${NIM_EMBED_IMAGE:-nvcr.io/nim/nvidia/nemotron-embed}:${NIM_EMBED_TAG:-latest}")
    # Pinning either setting would disable NIM's automatic GPU-specific selection.
    assert "NIM_ENGINE_MODEL_NAME" not in env
    assert "NIM_ENGINE_PRECISION" not in env
    assert env["NIM_SERVER_SERVED_MODEL_NAME"] == f"${{NIM_EMBED_MODEL:-{model}}}"
    assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
    assert env["NIM_SERVER_BIND_ADDR"] == "0.0.0.0:8000"
    assert "NIM_HTTP_API_PORT" not in env
    assert not any(name.startswith("NIM_TRITON_") for name in env)
    assert env["NIM_ENGINE_MODEL_PATH"] == "${NIM_EMBED_MODEL_PATH:-/model}/embed"
    assert service["volumes"] == [
        "nim_embedding_cache:/opt/cache",
        "nim_embedding_model:${NIM_EMBED_MODEL_PATH:-/model}",
    ]
    assert service["healthcheck"]["test"] == [
        "CMD",
        "curl",
        "--fail",
        "--silent",
        "http://localhost:8000/v1/health/ready",
    ]
    # A service-level healthcheck replaces the entire inherited mapping.
    inherited_healthcheck = compose["x-nim-service"]["healthcheck"]
    for setting in ("interval", "timeout", "retries", "start_period"):
        assert service["healthcheck"][setting] == inherited_healthcheck[setting]
    # The preset must select the served alias for both ingestion and queries.
    assert f"NIM_EMBED_MODEL={model}" in CORE_PRESET.read_text(encoding="utf-8").splitlines()
    config = compose["configs"]["retriever_service_config"]["content"]
    assert 'embed_model_name: "${NIM_EMBED_MODEL:-nvidia/nemotron-3-embed-1b}"' in config
    assert 'embed_model: "${NIM_EMBED_MODEL:-nvidia/nemotron-3-embed-1b}"' in config
    command = compose["services"]["vectordb"]["command"]
    assert command[command.index("--embed-model") + 1] == "${NIM_EMBED_MODEL:-nvidia/nemotron-3-embed-1b}"


def test_caption_nim_uses_the_helm_aligned_omni_image_tag() -> None:
    caption_image = _compose()["services"]["nim-caption"]["image"]
    assert caption_image == (
        "${NIM_CAPTION_IMAGE:-nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}:"
        "${NIM_CAPTION_TAG:-2.0.4-variant}"
    )


def test_zero_profile_defaults_to_hosted_endpoints() -> None:
    compose = _compose()
    config = compose["configs"]["retriever_service_config"]["content"]

    hosted_endpoints = (
        "${NIM_PAGE_ELEMENTS_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3}",
        "${NIM_TABLE_STRUCTURE_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1}",
        "${NIM_OCR_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2}",
    )
    for endpoint in hosted_endpoints:
        assert endpoint in config


def test_nims_core_preset_uses_internal_endpoints() -> None:
    core_preset = set(CORE_PRESET.read_text(encoding="utf-8").splitlines())
    internal_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=http://nim-page-elements:8000/v1/page-elements",
        "NIM_TABLE_STRUCTURE_URL=http://nim-table-structure:8000/v1/table-structure",
        "NIM_OCR_URL=http://nim-ocr:8000/v1/ocr",
        "NIM_EMBED_URL=http://nim-embedding:8000/v1/embeddings",
    }
    assert internal_endpoints <= core_preset
    assert "/v1/infer" not in CORE_PRESET.read_text(encoding="utf-8")


def test_local_models_preset_disables_remote_endpoints() -> None:
    local_preset = set(LOCAL_PRESET.read_text(encoding="utf-8").splitlines())
    disabled_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=",
        "NIM_TABLE_STRUCTURE_URL=",
        "NIM_OCR_URL=",
        "NIM_EMBED_URL=",
    }
    assert disabled_endpoints <= local_preset
