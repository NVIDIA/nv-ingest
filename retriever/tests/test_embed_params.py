# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for EmbedParams._normalize_modality and IMAGE_MODALITIES constant.
"""

import pytest

from retriever.params.models import EmbedParams, IMAGE_MODALITIES


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


@pytest.mark.parametrize("value,expected", [
    ("text", "text"),
    ("image", "image"),
    ("text_image", "text_image"),
    (None, None),
])
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
