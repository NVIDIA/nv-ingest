# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.model.create_local_embedder factory."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from nemo_retriever.model import create_local_embedder


@pytest.fixture(autouse=True)
def _patch_embedders(monkeypatch):
    """Prevent real model downloads by stubbing both embedder classes.

    The ``nemo_retriever.model.local`` package uses a custom ``__getattr__``
    that only exposes specific class names — not submodule names.  Because
    ``monkeypatch.setattr`` resolves each path segment via ``getattr``, it
    cannot traverse to the submodule.  We work around this by injecting fake
    modules directly into ``sys.modules``, which Python checks first when
    handling ``from … import`` statements.
    """
    fake_text = MagicMock(name="LlamaNemotronEmbed1BV2Embedder")
    fake_vl = MagicMock(name="LlamaNemotronEmbedVL1BV2Embedder")

    text_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder")
    text_mod.LlamaNemotronEmbed1BV2Embedder = fake_text

    vl_mod = ModuleType("nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder")
    vl_mod.LlamaNemotronEmbedVL1BV2Embedder = fake_vl

    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder", text_mod)
    monkeypatch.setitem(sys.modules, "nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder", vl_mod)

    yield fake_text, fake_vl


def test_default_returns_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder()
    fake_text.assert_called_once()
    assert result is fake_text.return_value


def test_none_model_name_returns_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder(None)
    fake_text.assert_called_once()
    assert result is fake_text.return_value


def test_alias_resolved_to_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    result = create_local_embedder("nemo_retriever_v1")
    call_kwargs = fake_text.call_args
    assert call_kwargs.kwargs["model_id"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert result is fake_text.return_value


def test_vl_model_returns_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    result = create_local_embedder("nvidia/llama-nemotron-embed-vl-1b-v2")
    fake_vl.assert_called_once()
    assert result is fake_vl.return_value


def test_vl_short_alias_returns_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    result = create_local_embedder("llama-nemotron-embed-vl-1b-v2")
    fake_vl.assert_called_once()
    assert result is fake_vl.return_value


def test_kwargs_forwarded_to_text_embedder(_patch_embedders):
    fake_text, _ = _patch_embedders
    create_local_embedder(
        device="cuda:1",
        hf_cache_dir="/tmp/cache",
        normalize=False,
        max_length=4096,
    )
    kw = fake_text.call_args.kwargs
    assert kw["device"] == "cuda:1"
    assert kw["hf_cache_dir"] == "/tmp/cache"
    assert kw["normalize"] is False
    assert kw["max_length"] == 4096


def test_kwargs_forwarded_to_vl_embedder(_patch_embedders):
    _, fake_vl = _patch_embedders
    create_local_embedder(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        device="cuda:0",
        hf_cache_dir="/models",
    )
    kw = fake_vl.call_args.kwargs
    assert kw["device"] == "cuda:0"
    assert kw["hf_cache_dir"] == "/models"
    assert kw["model_id"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_unknown_model_passes_through(_patch_embedders):
    fake_text, _ = _patch_embedders
    create_local_embedder("custom-org/my-embed-model")
    kw = fake_text.call_args.kwargs
    assert kw["model_id"] == "custom-org/my-embed-model"
