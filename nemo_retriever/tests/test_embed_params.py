# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for EmbedParams._normalize_modality and IMAGE_MODALITIES constant.
"""

import warnings

import pytest

from nemo_retriever.model import _DEFAULT_EMBED_MODEL
from nemo_retriever.params.models import EmbedParams, IMAGE_MODALITIES


def test_normalize_modality_image_text_to_text_image():
    """'image_text' is normalized to 'text_image' on all three modality fields."""
    params = EmbedParams(
        embed_modality="image_text",
        text_elements_modality="image_text",
        structured_elements_modality="image_text",
    )
    assert params.embed_modality == "text_image"
    assert params.text_elements_modality == "text_image"
    assert params.structured_elements_modality == "text_image"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("text", "text"),
        ("image", "image"),
        ("text_image", "text_image"),
        (None, None),
    ],
)
def test_normalize_modality_passthrough(value, expected):
    """Values other than 'image_text' pass through unchanged."""
    kwargs = {}
    if value is not None:
        kwargs["embed_modality"] = value
    kwargs["text_elements_modality"] = value
    kwargs["structured_elements_modality"] = value

    params = EmbedParams(**kwargs)

    if value is not None:
        assert params.embed_modality == expected
    assert params.text_elements_modality == expected
    assert params.structured_elements_modality == expected


def test_image_modalities_constant():
    """IMAGE_MODALITIES is a frozenset containing the three image-related modalities."""
    assert IMAGE_MODALITIES == {"image", "text_image", "image_text"}
    assert isinstance(IMAGE_MODALITIES, frozenset)


# ===================================================================
# embed_granularity
# ===================================================================


class TestEmbedParamsGranularity:
    def test_default_is_element(self):
        params = EmbedParams()
        assert params.embed_granularity == "element"

    def test_page_accepted(self):
        params = EmbedParams(embed_granularity="page")
        assert params.embed_granularity == "page"

    def test_invalid_value_rejected(self):
        with pytest.raises(Exception):
            EmbedParams(embed_granularity="invalid")

    def test_warning_on_per_type_modality_with_page(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EmbedParams(
                embed_granularity="page",
                text_elements_modality="image",
            )
            assert len(w) == 1
            assert "ignored" in str(w[0].message).lower()

    def test_no_warning_on_element_granularity_with_overrides(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EmbedParams(
                embed_granularity="element",
                text_elements_modality="image",
                structured_elements_modality="text_image",
            )
            assert len(w) == 0


# ===================================================================
# vLLM embedding
# ===================================================================


def test_embed_vllm_params_defaults():
    """embed_use_vllm and embed_model_name default to False and None."""
    params = EmbedParams()
    assert params.embed_use_vllm is False
    assert params.embed_model_name is None


def test_embed_vllm_params_accepted():
    """embed_use_vllm and embed_model_name are accepted."""
    params = EmbedParams(
        embed_use_vllm=True,
        embed_model_name=_DEFAULT_EMBED_MODEL,
    )
    assert params.embed_use_vllm is True
    assert params.embed_model_name == _DEFAULT_EMBED_MODEL
