# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.params.utils."""

import pytest

from nemo_retriever.params.models import EmbedParams
from nemo_retriever.params.utils import build_embed_kwargs, coerce_params


class TestCoerceParams:
    def test_none_params_constructs_from_kwargs(self):
        result = coerce_params(None, EmbedParams, {"embed_modality": "image"})
        assert isinstance(result, EmbedParams)
        assert result.embed_modality == "image"

    def test_params_without_kwargs_returned_unchanged(self):
        original = EmbedParams(embed_modality="text")
        result = coerce_params(original, EmbedParams, {})
        assert result is original

    def test_params_with_kwargs_applies_overrides(self):
        original = EmbedParams(embed_modality="text")
        result = coerce_params(original, EmbedParams, {"embed_modality": "image"})
        assert result.embed_modality == "image"
        assert result is not original


class TestBuildEmbedKwargs:
    def test_normalises_embed_invoke_url(self):
        params = EmbedParams(embed_invoke_url="http://nim:8000/v1")
        kwargs = build_embed_kwargs(params)
        assert kwargs["embedding_endpoint"] == "http://nim:8000/v1"

    def test_does_not_overwrite_existing_embedding_endpoint(self):
        params = EmbedParams(
            embed_invoke_url="http://old:8000/v1",
        )
        kwargs = build_embed_kwargs(params)
        assert "embedding_endpoint" in kwargs

    def test_includes_batch_tuning_when_requested(self):
        params = EmbedParams()
        with_bt = build_embed_kwargs(params, include_batch_tuning=True)
        without_bt = build_embed_kwargs(params, include_batch_tuning=False)
        # batch_tuning keys should be present when included
        assert isinstance(with_bt, dict)
        assert isinstance(without_bt, dict)

    def test_excludes_nested_sub_models(self):
        params = EmbedParams()
        kwargs = build_embed_kwargs(params)
        assert "runtime" not in kwargs
        assert "batch_tuning" not in kwargs
        assert "fused_tuning" not in kwargs
