# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.models.create_local_embedder factory."""

import json
import sys
import warnings
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_retriever.models import (
    EmbedModelSpec,
    NEMOTRON_3_EMBED_BF16_MODEL,
    NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_3_EMBED_NVFP4_MODEL,
    create_local_embedder,
    create_local_query_embedder,
    is_vl_embed_model,
    resolve_embed_model,
)


@pytest.fixture(autouse=True)
def _patch_embedders(monkeypatch):
    """Prevent real model downloads by stubbing all four embedder classes.

    The ``nemo_retriever.models.local`` package uses a custom ``__getattr__``
    that only exposes specific class names — not submodule names.  Because
    ``monkeypatch.setattr`` resolves each path segment via ``getattr``, it
    cannot traverse to the submodule.  We work around this by injecting fake
    modules directly into ``sys.modules``, which Python checks first when
    handling ``from … import`` statements.
    """
    fake_text_vllm = MagicMock(name="LlamaNemotronEmbed1BV2Embedder")
    fake_text_hf = MagicMock(name="LlamaNemotronEmbed1BV2HFEmbedder")
    fake_vl_hf = MagicMock(name="LlamaNemotronEmbedVL1BV2Embedder")
    fake_vl_vllm = MagicMock(name="LlamaNemotronEmbedVL1BV2VLLMEmbedder")

    text_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder")
    text_mod.LlamaNemotronEmbed1BV2Embedder = fake_text_vllm

    text_hf_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder")
    text_hf_mod.LlamaNemotronEmbed1BV2HFEmbedder = fake_text_hf

    vl_mod = ModuleType("nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder")
    vl_mod.LlamaNemotronEmbedVL1BV2Embedder = fake_vl_hf
    vl_mod.LlamaNemotronEmbedVL1BV2VLLMEmbedder = fake_vl_vllm

    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder", text_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder", text_hf_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder", vl_mod)
    n35_mod = ModuleType("nemo_retriever.models.local.nemotron35_embedder")
    n35_mod.Nemotron35Embedder = lambda **kwargs: (fake_text_hf if kwargs["backend"] == "hf" else fake_text_vllm)(
        **kwargs
    )
    monkeypatch.setitem(sys.modules, "nemo_retriever.models.local.nemotron35_embedder", n35_mod)

    def resolve_spec(model_id, *, revision=None, hf_cache_dir=None):
        config_path = Path(model_id) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            family = "vl" if config["model_type"] == "llama_nemotron_vl" else "text"
            quantization = config.get("quantization_config") or {}
            requires_vllm = quantization.get("quant_method") == "modelopt"
            dimension_config = (config.get("llm_config") or {}) if family == "vl" else config
            output_dimension = dimension_config.get("hidden_size", 2048)
            resolved_revision = None
        else:
            family = "vl" if "embed-vl" in model_id or "vlm-embed" in model_id else "text"
            quantization = {}
            requires_vllm = False
            output_dimension = 4096 if "8b" in model_id.lower() else 2048
            resolved_revision = revision or "a" * 40
        instruction_model = model_id == "nvidia/llama-embed-nemotron-8b"
        return EmbedModelSpec(
            model_id=model_id,
            revision=resolved_revision,
            family=family,
            output_dimension=output_dimension,
            query_prefix=(
                "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
                if instruction_model
                else "query: "
            ),
            document_prefix="" if instruction_model else "passage: ",
            quantization=quantization.get("quant_algo"),
            requires_vllm=requires_vllm,
            model_type="mistral3" if "Nemotron-3.5-" in model_id else None,
        )

    monkeypatch.setattr("nemo_retriever.models.resolve_embed_model_spec", resolve_spec)

    yield fake_text_vllm, fake_text_hf, fake_vl_hf, fake_vl_vllm


@pytest.fixture
def local_checkpoint(tmp_path):
    """Create a local checkpoint whose architecture is declared by config.json."""

    def create(model_type, *, modelopt=False):
        config = {"model_type": model_type}
        if modelopt:
            config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "FP8"}
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return tmp_path

    return create


@pytest.mark.parametrize(
    "checkpoint_model",
    [NEMOTRON_3_EMBED_BF16_MODEL, NEMOTRON_3_EMBED_NVFP4_MODEL],
)
def test_precision_checkpoints_share_logical_model_identity(checkpoint_model: str) -> None:
    assert resolve_embed_model(checkpoint_model) == NEMOTRON_3_EMBED_MODEL


# ---------------------------------------------------------------------------
# create_local_embedder — Nemotron 3 default profile
# ---------------------------------------------------------------------------


def test_default_returns_bf16_text_vllm_embedder(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    result = create_local_embedder()
    fake_text_vllm.assert_called_once()
    assert fake_text_vllm.call_args.kwargs["model_id"] == "nvidia/Nemotron-3.5-Embed-1B-BF16"
    assert result is fake_text_vllm.return_value


def test_default_uses_nvfp4_on_native_hardware(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: True)
    result = create_local_embedder(None)
    fake_text_vllm.assert_called_once()
    assert fake_text_vllm.call_args.kwargs["model_id"] == "nvidia/Nemotron-3.5-Embed-1B-NVFP4"
    assert result is fake_text_vllm.return_value


def test_alias_resolved_to_text_embedder(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    result = create_local_embedder("nemo_retriever_v1")
    call_kwargs = fake_text_vllm.call_args
    assert call_kwargs.kwargs["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert result is fake_text_vllm.return_value


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        pytest.param(None, False, id="default"),
        pytest.param("llama-3.2-nemoretriever-1b-vlm-embed-v1", True, id="legacy-vl-alias"),
        pytest.param("nvidia/llama-nemotron-embed-1b-v2", False, id="text"),
    ],
)
def test_is_vl_embed_model_preserves_legacy_contract(model_name, expected):
    assert is_vl_embed_model(model_name) is expected


def test_default_model_explicit_vllm_backend(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    result = create_local_embedder(backend="vllm")
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_default_model_hf_backend_returns_bf16_text_embedder(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_embedder(backend="hf")
    fake_text_hf.assert_called_once()
    assert fake_text_hf.call_args.kwargs["model_id"] == "nvidia/Nemotron-3.5-Embed-1B-BF16"
    assert result is fake_text_hf.return_value


def test_kwargs_forwarded_to_default_vllm_embedder(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    create_local_embedder(
        device="cuda:1",
        hf_cache_dir="/tmp/cache",
        gpu_memory_utilization=0.6,
        normalize=False,
    )
    kw = fake_text_vllm.call_args.kwargs
    assert kw["device"] == "cuda:1"
    assert kw["hf_cache_dir"] == "/tmp/cache"
    assert kw["gpu_memory_utilization"] == 0.6
    assert kw["normalize"] is False


def test_kwargs_forwarded_to_default_hf_embedder(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    create_local_embedder(
        backend="hf",
        device="cuda:0",
        hf_cache_dir="/models",
    )
    kw = fake_text_hf.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["model_id"] == "nvidia/Nemotron-3.5-Embed-1B-BF16"


def test_unknown_model_passes_through(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    create_local_embedder("custom-org/my-embed-model")
    kw = fake_text_vllm.call_args.kwargs
    assert kw["model_id"] == "custom-org/my-embed-model"


def test_8b_prompt_metadata_is_forwarded_to_text_loader(_patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders

    create_local_embedder("nvidia/llama-embed-nemotron-8b")

    kw = fake_text_vllm.call_args.kwargs
    assert kw["query_prefix"].startswith("Instruct:")
    assert kw["document_prefix"] == ""


# ---------------------------------------------------------------------------
# create_local_embedder — VL model
# ---------------------------------------------------------------------------


def test_vl_model_default_returns_vllm_embedder(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


def test_vl_short_alias_returns_vllm_embedder(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_embedder("llama-nemotron-embed-vl-1b-v2")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


def test_vl_model_hf_backend_returns_hf_embedder(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="hf")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_kwargs_forwarded_to_vl_hf_embedder(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    create_local_embedder(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        backend="hf",
        device="cuda:0",
        hf_cache_dir="/models",
    )
    kw = fake_vl_hf.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["model_id"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


# ---------------------------------------------------------------------------
# Invalid backend
# ---------------------------------------------------------------------------


def test_invalid_backend_raises_value_error(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_embedder(backend="auto")


def test_invalid_backend_raises_for_vl(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="auto")


# ---------------------------------------------------------------------------
# create_local_query_embedder
# ---------------------------------------------------------------------------


def test_query_embedder_defaults_to_vllm(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    result = create_local_query_embedder()
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_query_embedder_explicit_hf(_patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    result = create_local_query_embedder(backend="hf")
    fake_text_hf.assert_called_once()
    assert result is fake_text_hf.return_value


def test_query_embedder_vllm_uses_vllm_embedder(monkeypatch, _patch_embedders):
    fake_text_vllm, _, _, _ = _patch_embedders
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    result = create_local_query_embedder(backend="vllm")
    fake_text_vllm.assert_called_once()
    assert result is fake_text_vllm.return_value


def test_query_embedder_invalid_backend_raises(_patch_embedders):
    with pytest.raises(ValueError, match="backend must be"):
        create_local_query_embedder(backend="auto")


def test_query_embedder_vl_hf_uses_hf_vl(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    result = create_local_query_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="hf")
    fake_vl_hf.assert_called_once()
    assert result is fake_vl_hf.return_value


def test_query_embedder_vl_vllm_uses_vllm_vl(_patch_embedders):
    _, _, _, fake_vl_vllm = _patch_embedders
    result = create_local_query_embedder("nvidia/llama-nemotron-embed-vl-1b-v2", backend="vllm")
    fake_vl_vllm.assert_called_once()
    assert result is fake_vl_vllm.return_value


# ---------------------------------------------------------------------------
# Local checkpoint directories — routed by config.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_type", "backend", "embedder_index"),
    [
        pytest.param("llama_nemotron_vl", "vllm", 3, id="vl-vllm"),
        pytest.param("llama_nemotron_vl", "hf", 2, id="vl-hf"),
        pytest.param("llama_bidirec", "vllm", 0, id="text-vllm"),
        pytest.param("llama_bidirec", "hf", 1, id="text-hf"),
        pytest.param("ministral3", "vllm", 0, id="ministral3-vllm"),
        pytest.param("ministral3", "hf", 1, id="ministral3-hf"),
    ],
)
def test_local_checkpoint_routes_from_config(
    local_checkpoint,
    _patch_embedders,
    model_type,
    backend,
    embedder_index,
):
    checkpoint = local_checkpoint(model_type)

    result = create_local_embedder(str(checkpoint), backend=backend)

    selected = _patch_embedders[embedder_index]
    selected.assert_called_once()
    assert selected.call_args.kwargs["model_id"] == str(checkpoint)
    assert selected.call_args.kwargs["revision"] is None
    assert result is selected.return_value


def test_modelopt_checkpoint_rejects_hf_before_loader(local_checkpoint, _patch_embedders):
    checkpoint = local_checkpoint("llama_nemotron_vl", modelopt=True)
    with pytest.raises(ValueError, match="requires backend='vllm'"):
        create_local_embedder(str(checkpoint), backend="hf")


def test_relative_local_checkpoint_routes_locally(tmp_path, monkeypatch, _patch_embedders):
    _, fake_text_hf, _, _ = _patch_embedders
    checkpoint = tmp_path / "my-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"llama_bidirec"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    create_local_embedder("./my-checkpoint", backend="hf")

    assert fake_text_hf.call_args.kwargs["model_id"] == "./my-checkpoint"


def test_query_embedder_forwards_recorded_revision(_patch_embedders):
    _, _, fake_vl_hf, _ = _patch_embedders
    revision = "b" * 40
    create_local_query_embedder(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        backend="hf",
        revision=revision,
    )
    assert fake_vl_hf.call_args.kwargs["revision"] == revision


# ---------------------------------------------------------------------------
# Real-class smoke test (requires torch; skipped if not installed)
# ---------------------------------------------------------------------------


def test_llama_nemotron_text_embedder_deprecates_device(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import importlib

    monkeypatch.delitem(
        sys.modules,
        "nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder",
        raising=False,
    )
    mod = importlib.import_module("nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder")
    monkeypatch.setattr(
        "nemo_retriever.models.inference.vllm.create_vllm_llm",
        MagicMock(return_value=MagicMock()),
    )
    Embed = mod.LlamaNemotronEmbed1BV2Embedder

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always", category=DeprecationWarning)
        Embed(device="cuda:0")
    assert any("no longer uses 'device'" in str(w.message) for w in wrec)

    with warnings.catch_warnings(record=True) as wrec2:
        warnings.simplefilter("always", category=DeprecationWarning)
        Embed(device=None)
    assert not any("no longer uses 'device'" in str(w.message) for w in wrec2)
