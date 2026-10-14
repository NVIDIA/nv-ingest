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


@pytest.mark.parametrize("value", ["", "not base64!", "bm90IGFuIGltYWdl", "non-ASCII: é"])
def test_decode_base64_image_wraps_invalid_input(value):
    with pytest.raises(ValueError, match="Failed to decode base64 image:"):
        decode_base64_image(value)


@pytest.mark.parametrize("error_type", ["truncated", "decompression_bomb"])
def test_decode_base64_image_wraps_expected_image_errors(monkeypatch, error_type):
    from PIL import Image

    error = OSError("truncated image") if error_type == "truncated" else Image.DecompressionBombError("image too large")

    def fail_open(*args, **kwargs):
        raise error

    monkeypatch.setattr(Image, "open", fail_open)
    with pytest.raises(ValueError, match="Failed to decode base64 image:") as caught:
        decode_base64_image("aW1hZ2U=")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("stage", ["base64", "image"])
def test_decode_base64_image_preserves_unexpected_errors(monkeypatch, stage):
    from PIL import Image

    error = RuntimeError("unexpected implementation failure")

    def fail(*args, **kwargs):
        raise error

    if stage == "base64":
        monkeypatch.setattr(base64, "b64decode", fail)
    else:
        monkeypatch.setattr(Image, "open", fail)
    with pytest.raises(RuntimeError) as caught:
        decode_base64_image("aW1hZ2U=")
    assert caught.value is error


def test_decode_base64_image_does_not_wrap_image_programming_value_error(monkeypatch):
    from PIL import Image

    error = ValueError("unexpected image implementation failure")

    def fail_open(*args, **kwargs):
        raise error

    monkeypatch.setattr(Image, "open", fail_open)
    with pytest.raises(ValueError) as caught:
        decode_base64_image("aW1hZ2U=")
    assert caught.value is error
