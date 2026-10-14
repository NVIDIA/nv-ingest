# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared embedding image decoding contracts without model loading."""

import base64
import io

import pytest

from nemo_retriever.models.inference.image import decode_base64_image


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_decode_base64_image_returns_owned_rgb_image(mode):
    from PIL import Image

    buffer = io.BytesIO()
    with Image.new(mode, (3, 2)) as source:
        source.save(buffer, format="PNG")
    image = decode_base64_image(base64.b64encode(buffer.getvalue()).decode())
    try:
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        assert len(image.tobytes()) == 18
    finally:
        image.close()


@pytest.mark.parametrize("value", ["", "not base64!", "bm90IGFuIGltYWdl"])
def test_decode_base64_image_wraps_invalid_input(value):
    with pytest.raises(ValueError, match="Failed to decode base64 image:"):
        decode_base64_image(value)
