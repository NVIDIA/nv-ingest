# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for shared Parse normalization and adapter delegation."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
import torch

from nemo_retriever.models.inference.parse_image import ImageInput, normalize_parse_image
from nemo_retriever.models.local import nemotron_parse35, nemotron_parse_v1_2


@pytest.fixture(params=["helper", "parse35", "parse12"])
def normalize(request: pytest.FixtureRequest) -> Callable[[ImageInput], Image.Image]:
    """Exercise both adapters without loading either checkpoint or vLLM."""
    if request.param == "parse35":
        return nemotron_parse35.NemotronParse35(backend="transformers", device="cpu").preprocess
    if request.param == "parse12":
        return nemotron_parse_v1_2.NemotronParseV12.__new__(nemotron_parse_v1_2.NemotronParseV12).preprocess
    return normalize_parse_image


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("layout", ["HWC", "CHW"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_channels_layouts_and_singleton_batches(
    normalize: Callable[[ImageInput], Image.Image], channels: int, layout: str, batched: bool, as_tensor: bool
) -> None:
    pixels = np.empty((20, 16, channels), dtype=np.uint8)
    pixels[:] = [11, 22, 33, 44][:channels]
    if layout == "CHW":
        pixels = pixels.transpose(2, 0, 1)
    if batched:
        pixels = pixels[None]
    original = pixels.copy()
    with normalize(torch.from_numpy(pixels) if as_tensor else pixels) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 20)
        assert image.getpixel((0, 0)) == ((11, 11, 11) if channels == 1 else (11, 22, 33))
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize("height", [1, 3, 4])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_hwc_wins_ambiguous_layouts(normalize: Callable[[ImageInput], Image.Image], height: int, channels: int) -> None:
    with normalize(np.ones((height, 20, channels), dtype=np.uint8)) as image:
        assert image.size == (20, height)
        assert image.getpixel((0, 0)) == (1, 1, 1)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_pil_input_is_independently_owned(normalize: Callable[[ImageInput], Image.Image], mode: str) -> None:
    original = Image.new(mode, (16, 20), 0)
    result = normalize(original)
    original.close()
    try:
        assert result.mode == "RGB" and result.size == (16, 20)
        assert result.getpixel((0, 0)) == (0, 0, 0)
    finally:
        result.close()


def test_bf16_rgba_tensor_is_supported(normalize: Callable[[ImageInput], Image.Image]) -> None:
    with normalize(torch.ones((1, 4, 20, 16), dtype=torch.bfloat16)) as image:
        assert image.size == (16, 20)
        assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("shape", [(16,), (20, 16), (2, 20, 16, 3), (0, 16, 3), (20, 16, 2)])
def test_invalid_shapes_fail(normalize: Callable[[ImageInput], Image.Image], shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        normalize(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_pixels_fail(normalize: Callable[[ImageInput], Image.Image], value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize(np.full((20, 16, 3), value))


def test_unsupported_input_fails(normalize: Callable[[ImageInput], Image.Image]) -> None:
    with pytest.raises(TypeError, match="Unsupported input type"):
        normalize(object())


def test_image_path_is_closed_and_owned(
    normalize: Callable[[ImageInput], Image.Image], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "page.png"
    with Image.new("RGBA", (16, 20), (11, 22, 33, 44)) as image:
        image.save(path)
    opened = Image.open(path)
    monkeypatch.setattr(Image, "open", lambda _: opened)
    with normalize(path) as result:
        assert opened.fp is None
        assert result.getpixel((0, 0)) == (11, 22, 33)


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize(
    "values,expected35,expected12",
    [
        ([0, 0.5, 1], (0, 127, 255), (0, 127, 255)),
        ([0, 1, 1.5], (0, 1, 1), (0, 255, 255)),
        ([-0.5, 0.5, 1], (0, 0, 1), (0, 127, 255)),
        ([0, 1, 1.6], (0, 1, 1), (0, 1, 1)),
        ([0, 128, 300], (0, 128, 255), (0, 128, 255)),
    ],
)
def test_adapters_preserve_their_float_scaling_policies(
    values: list[float], expected35: tuple[int, ...], expected12: tuple[int, ...], as_tensor: bool
) -> None:
    pixels = np.empty((20, 16, 3), dtype=np.float32)
    pixels[:] = values
    input_data = torch.from_numpy(pixels) if as_tensor else pixels
    parser35 = nemotron_parse35.NemotronParse35(backend="transformers", device="cpu")
    parser12 = nemotron_parse_v1_2.NemotronParseV12.__new__(nemotron_parse_v1_2.NemotronParseV12)
    with parser35.preprocess(input_data) as image:
        assert image.getpixel((0, 0)) == expected35
    with parser12.preprocess(input_data) as image:
        assert image.getpixel((0, 0)) == expected12


@pytest.mark.parametrize("module,legacy", [(nemotron_parse35, False), (nemotron_parse_v1_2, True)])
def test_adapters_delegate_to_shared_normalizer(monkeypatch: pytest.MonkeyPatch, module: object, legacy: bool) -> None:
    helper = MagicMock()
    monkeypatch.setattr(module, "normalize_parse_image", helper)
    if legacy:
        parser = module.NemotronParseV12.__new__(module.NemotronParseV12)
    else:
        parser = module.NemotronParse35(backend="transformers", device="cpu")
    pixels = np.zeros((20, 16, 3))
    assert parser.preprocess(pixels) is helper.return_value
    if legacy:
        helper.assert_called_once_with(pixels, legacy_float_scaling=True)
    else:
        helper.assert_called_once_with(pixels)
