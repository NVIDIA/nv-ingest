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
        Unexpected implementation errors propagate without being relabeled.
    ImportError
        If Pillow is not installed.
    """
    import base64
    import binascii
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(image_b64)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            from nemo_retriever.models.inference.input_audit import current_rows

            audit = current_rows()
            if audit is not None:
                audit.image(image_bytes, rgb)
            return rgb
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc
