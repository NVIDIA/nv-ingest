# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral image decoding for local multimodal inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode a base64 image as an independently owned RGB image.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded image bytes, without a data-URL prefix.

    Returns
    -------
    PIL.Image.Image
        Decoded RGB image. The caller owns and must close the returned image.

    Raises
    ------
    ValueError
        If the input cannot be decoded or read as an image.
    ImportError
        If Pillow is not installed.
    """
    import base64
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(base64.b64decode(image_b64))) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc
