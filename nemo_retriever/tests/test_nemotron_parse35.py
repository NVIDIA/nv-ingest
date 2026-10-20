# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline adapter and pipeline regression tests for Transformers Parse 3.5."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL, NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.local.nemotron_parse35 import NemotronParse35
from nemo_retriever.operators.extract.parse.nemotron_parse import (
    NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
    NemotronParseGPUActor,
    nemotron_parse_pages,
)

TAGGED_OUTPUT = "<x_0.1><y_0.2>Body text<x_0.9><y_0.8><class_Text>"


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace downloads and model execution without importing the vLLM backend."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2]])
    processor = MagicMock()
    processor.return_value = BatchFeature({"input_ids": torch.tensor([[1]])})
    processor.batch_decode.return_value = [TAGGED_OUTPUT]
    snapshot_download = MagicMock(return_value="/snapshot")
    model_loader = MagicMock(return_value=model)
    processor_loader = MagicMock(return_value=processor)
    generation_config = SimpleNamespace(max_new_tokens=9000)
    generation_loader = MagicMock(return_value=generation_config)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", model_loader)
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", processor_loader)
    monkeypatch.setattr("transformers.GenerationConfig.from_pretrained", generation_loader)
    monkeypatch.setattr(
        "nemo_retriever.models.local.nemotron_parse35.configure_global_hf_cache_base", lambda _: "/cache"
    )
    return SimpleNamespace(
        model=model,
        processor=processor,
        snapshot_download=snapshot_download,
        model_loader=model_loader,
        processor_loader=processor_loader,
        generation_loader=generation_loader,
        generation_config=generation_config,
    )


def test_lazy_loading_pins_native_assets_and_preserves_tags(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(device="cpu", max_tokens=123)
    runtime.snapshot_download.assert_not_called()
    assert parser.invoke(Image.new("RGB", (16, 16))) == TAGGED_OUTPUT
    runtime.snapshot_download.assert_called_once()
    download_kwargs = runtime.snapshot_download.call_args.kwargs
    assert download_kwargs["revision"] == get_hf_revision(NEMOTRON_PARSE35_MODEL)
    assert download_kwargs["cache_dir"] == "/cache"
    runtime.model_loader.assert_called_once_with(
        "/snapshot", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    runtime.processor_loader.assert_called_once_with("/snapshot", trust_remote_code=True)
    runtime.generation_loader.assert_called_once_with("/snapshot")
    assert runtime.processor.call_args.kwargs["add_special_tokens"] is False
    assert runtime.processor.call_args.kwargs["text"] == NEMOTRON_PARSE_DEFAULT_TASK_PROMPT
    assert runtime.processor.batch_decode.call_args.kwargs["skip_special_tokens"] is False
    assert runtime.generation_config.max_new_tokens == 123
    assert runtime.model.generate.call_args.kwargs["generation_config"] is runtime.generation_config


def test_batch_order_prompt_override_and_empty_batch(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(device="cpu")
    assert parser.invoke_batch([]) == []
    runtime.snapshot_download.assert_not_called()
    runtime.processor.batch_decode.side_effect = [["first"], ["second"]]
    assert parser.invoke_batch([Image.new("RGB", (16, 16))] * 2, task_prompt="picture text") == ["first", "second"]
    assert runtime.model.generate.call_count == 2
    assert all(call.kwargs["text"] == "picture text" for call in runtime.processor.call_args_list)
    runtime.snapshot_download.assert_called_once()


def test_unload_then_reload(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(device="cpu")
    page = Image.new("RGB", (16, 16))
    parser(page)
    parser.unload()
    runtime.model.to.assert_called_with("cpu")
    assert parser._model is None
    assert parser._processor is None
    assert parser._generation_config is None
    parser.unload()
    assert parser(page) == TAGGED_OUTPUT
    assert runtime.model_loader.call_count == 2


def test_cuda_unload_explicitly_clears_cache(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35()
    parser._model = runtime.model
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache") as empty_cache:
        parser.unload()
    runtime.model.to.assert_called_once_with("cpu")
    empty_cache.assert_called_once()


def test_model_transfer_failure_releases_tensors(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(device="cpu")
    runtime.model.eval.return_value.to.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        parser(Image.new("RGB", (16, 16)))
    runtime.model.to.assert_called_with("cpu")
    assert parser._model is None


@pytest.mark.parametrize(
    "image",
    [
        Image.new("L", (16, 20)),
        np.zeros((20, 16, 3), dtype=np.uint8),
        np.zeros((3, 20, 16), dtype=np.float32),
        torch.ones((1, 3, 20, 16), dtype=torch.bfloat16),
        np.ones((20, 16, 1), dtype=np.float32),
    ],
)
def test_supported_images_keep_geometry(image: object) -> None:
    normalized = NemotronParse35(device="cpu").preprocess(image)
    assert normalized.mode == "RGB"
    assert normalized.size == (16, 20)


def test_image_path_is_closed_after_loading(tmp_path) -> None:
    path = tmp_path / "page.png"
    Image.new("RGB", (16, 20)).save(path)
    normalized = NemotronParse35(device="cpu").preprocess(path)
    assert normalized.size == (16, 20)
    assert normalized.getpixel((0, 0)) == (0, 0, 0)


def test_integer_tensor_pixels_are_not_scaled_as_floats() -> None:
    parser = NemotronParse35(device="cpu")
    assert parser.preprocess(torch.ones((3, 20, 16), dtype=torch.uint8)).getpixel((0, 0)) == (1, 1, 1)
    assert parser.preprocess(torch.ones((3, 20, 16))).getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("shape", [(2, 20, 16, 3), (16,), (0, 16, 3), (20, 16, 2)])
def test_invalid_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        NemotronParse35(device="cpu").preprocess(np.zeros(shape))


def test_invalid_inputs_do_not_download_weights(runtime: SimpleNamespace) -> None:
    parser = NemotronParse35(device="cpu")
    with pytest.raises(TypeError):
        parser.invoke(object())
    with pytest.raises(ValueError, match="finite"):
        parser.invoke(np.full((20, 16, 3), np.nan))
    runtime.snapshot_download.assert_not_called()
    with pytest.raises(ValueError, match="positive"):
        NemotronParse35(max_tokens=0)


@pytest.mark.parametrize("model_id", [None, NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL, NEMOTRON_PARSE35_MODEL])
def test_gpu_actor_dispatches_explicit_model_without_changing_default(model_id: str | None) -> None:
    with (
        patch("nemo_retriever.models.local.NemotronParseV12") as legacy_parser,
        patch("nemo_retriever.models.local.NemotronParse35") as new_parser,
    ):
        actor = NemotronParseGPUActor(nemotron_parse_model=model_id)
        actor._ensure_model()
        actor._ensure_model()
    expected_parser = new_parser if model_id == NEMOTRON_PARSE35_MODEL else legacy_parser
    other_parser = legacy_parser if model_id == NEMOTRON_PARSE35_MODEL else new_parser
    expected_parser.assert_called_once_with(task_prompt=NEMOTRON_PARSE_DEFAULT_TASK_PROMPT)
    other_parser.assert_not_called()
    assert actor.get_constructor_kwargs()["nemotron_parse_model"] == model_id


def test_remote_actor_does_not_load_local_model() -> None:
    with patch("nemo_retriever.models.local.NemotronParse35") as new_parser:
        actor = NemotronParseGPUActor(
            nemotron_parse_model=NEMOTRON_PARSE35_MODEL, invoke_url="http://parse:8000/v1/chat/completions"
        )
        actor._ensure_model()
    new_parser.assert_not_called()


def test_unknown_local_actor_model_is_not_silently_replaced() -> None:
    actor = NemotronParseGPUActor(nemotron_parse_model="unsupported")
    with pytest.raises(ValueError, match="Unsupported local"):
        actor._ensure_model()


def test_native_output_uses_existing_pipeline_routing(runtime: SimpleNamespace) -> None:
    import base64
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (16, 20)).save(buffer, format="PNG")
    frame = pd.DataFrame([{"page_image": {"image_b64": base64.b64encode(buffer.getvalue()).decode()}}])
    runtime.processor.batch_decode.return_value = [
        TAGGED_OUTPUT
        + "<x_0.1><y_0.3>Picture text<x_0.8><y_0.7><class_Picture>"
        + "<x_0.2><y_0.4>Chart text<x_0.7><y_0.6><class_Chart>"
        + "<x_0.2><y_0.4>Table text<x_0.7><y_0.6><class_Table>"
    ]
    result = nemotron_parse_pages(
        frame,
        model=NemotronParse35(device="cpu"),
        extract_text=True,
        extract_infographics=True,
        extract_charts=True,
        extract_tables=True,
    )
    assert result.iloc[0]["text"] == "Body text"
    assert result.iloc[0]["infographic"][0]["text"] == "Picture text"
    assert result.iloc[0]["chart"][0]["text"] == "Chart text"
    assert result.iloc[0]["table"][0]["text"] == "Table text"
    assert result.iloc[0]["infographic"][0]["bbox_xyxy_norm"] == [0.1, 0.3, 0.8, 0.7]
