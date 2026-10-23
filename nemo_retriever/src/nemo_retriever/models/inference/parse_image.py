# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared image normalization for local Nemotron Parse adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

ImageInput = Image.Image | str | Path | np.ndarray | torch.Tensor


def normalize_parse_image(input_data: ImageInput, *, legacy_float_scaling: bool = False) -> Image.Image:
    """Return an independently owned RGB page without resizing or mutating input.

    Args:
        input_data: PIL image, image path (with user expansion), HWC/CHW array,
            or tensor. Arrays and tensors accept one, three, or four channels
            and an optional leading batch dimension of size one. HWC takes
            precedence when both first and last dimensions look like channels.
            BF16 tensors are converted to float32 before NumPy conversion.
        legacy_float_scaling: Preserve Parse v1.2's heuristic: multiply floating
            pixels by 255 when their maximum is at most 1.5, then clip. Otherwise
            scale only floating pixels entirely in [0, 1]. Integer pixels are
            never scaled. All pixels are clipped to [0, 255] and cast to uint8.

    Returns:
        RGB PIL image with alpha discarded and the original spatial dimensions.
        Image paths are closed before returning; the caller owns the result.

    Raises:
        TypeError: If the input type or pixel dtype is unsupported.
        ValueError: If the shape, channel count, or finite-pixel contract is invalid.
        OSError: If an image path cannot be opened or decoded.
    """
    if isinstance(input_data, Image.Image):
        return input_data.convert("RGB")
    if isinstance(input_data, (str, Path)):
        with Image.open(Path(input_data).expanduser()) as image:
            return image.convert("RGB")
    if isinstance(input_data, torch.Tensor):
        tensor = input_data.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        input_data = tensor.numpy()
    if not isinstance(input_data, np.ndarray):
        raise TypeError(f"Unsupported input type for Nemotron Parse: {type(input_data)!r}")
    pixels = input_data
    if pixels.ndim == 4 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 3 or not pixels.size:
        raise ValueError(f"Expected a nonempty HWC/CHW image, got {pixels.shape}")
    if pixels.shape[0] in (1, 3, 4) and pixels.shape[-1] not in (1, 3, 4):
        pixels = pixels.transpose(1, 2, 0)
    if pixels.shape[-1] not in (1, 3, 4) or not np.isfinite(pixels).all():
        raise ValueError("Expected finite pixels with 1, 3, or 4 channels.")
    if np.issubdtype(pixels.dtype, np.floating):
        scale_float = pixels.max() <= 1.5 if legacy_float_scaling else pixels.min() >= 0 and pixels.max() <= 1
        if scale_float:
            pixels = pixels * 255
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    return Image.fromarray(pixels).convert("RGB")
