# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image captioning pipeline stage."""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _make_test_png_b64(size: tuple[int, int] = (64, 64)) -> str:
    img = Image.new("RGB", size, color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_page_df(num_images=2, captioned=False):
    b64 = _make_test_png_b64()
    images = [
        {"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "done" if captioned else "", "image_b64": b64}
        for _ in range(num_images)
    ]
    return pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])


def test_caption_images_writes_back():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["cap1", "cap2"]
    result = caption_images(_make_page_df(), model=mock_model)
    assert result.iloc[0]["images"][0]["text"] == "cap1"
    assert result.iloc[0]["images"][1]["text"] == "cap2"


def test_caption_images_skips_already_captioned():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_model = MagicMock()
    result = caption_images(_make_page_df(captioned=True), model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == "done"


def test_caption_images_uses_page_image_for_direct_image_when_images_is_empty():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    b64 = _make_test_png_b64()
    df = pd.DataFrame([{"path": "/data/direct.png", "text": "", "page_image": {"image_b64": b64}, "images": []}])
    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["a red square"]

    result = caption_images(df, model=mock_model)

    mock_model.caption_batch.assert_called_once()
    assert result.iloc[0]["images"] == [
        {
            "image_b64": b64,
            "text": "a red square",
            "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
        }
    ]


def test_caption_images_uses_page_image_for_direct_image_with_ray_object_array():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    b64 = _make_test_png_b64()
    df = pd.DataFrame(
        [
            {
                "path": "/data/direct.png",
                "text": "",
                "page_image": {"image_b64": b64},
                "images": np.array([], dtype=object),
            }
        ]
    )
    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["a red square"]

    result = caption_images(df, model=mock_model)

    mock_model.caption_batch.assert_called_once()
    assert result.iloc[0]["images"][0]["text"] == "a red square"


def test_caption_images_persists_caption_in_arrow_backed_images_column():
    pa = pytest.importorskip("pyarrow")

    from nemo_retriever.graph.executor import arrow_table_to_pandas
    from nemo_retriever.operators.extract.caption.caption import caption_images

    b64 = _make_test_png_b64()
    df = arrow_table_to_pandas(
        pa.Table.from_pylist(
            [
                {
                    "path": "/data/direct.png",
                    "text": "",
                    "page_image": {"image_b64": b64},
                    "images": [{"image_b64": b64, "text": "", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}],
                }
            ]
        )
    )
    assert isinstance(df["images"].dtype, pd.ArrowDtype)
    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["a red square"]

    result = caption_images(df, model=mock_model)

    mock_model.caption_batch.assert_called_once()
    assert result.iloc[0]["images"][0]["text"] == "a red square"


def test_caption_images_captions_arrow_backed_infographic_column():
    pa = pytest.importorskip("pyarrow")

    from nemo_retriever.graph.executor import arrow_table_to_pandas
    from nemo_retriever.operators.extract.caption.caption import caption_images

    df = arrow_table_to_pandas(
        pa.Table.from_pylist(
            [
                {
                    "path": "/data/document.pdf",
                    "text": "page text",
                    "page_image": {"image_b64": _make_test_png_b64((256, 256))},
                    "images": [],
                    "infographic": [{"bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0], "text": "ocr text"}],
                }
            ]
        )
    )
    assert isinstance(df["infographic"].dtype, pd.ArrowDtype)
    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["an infographic"]

    result = caption_images(df, model=mock_model, caption_infographics=True)

    mock_model.caption_batch.assert_called_once()
    infographic = result.iloc[0]["infographic"][0]
    assert infographic["caption"] == "an infographic"
    assert infographic["text"] == "ocr text"


def test_caption_images_does_not_use_pdf_page_image_when_images_is_empty():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    df = pd.DataFrame(
        [
            {
                "path": "/data/document.pdf",
                "text": "page text",
                "page_image": {"image_b64": _make_test_png_b64()},
                "images": [],
            }
        ]
    )
    mock_model = MagicMock()

    result = caption_images(df, model=mock_model)

    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"] == []


@patch("nemo_retriever.operators.extract.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_populates_images(mock_extract):
    _ext = pytest.importorskip("nemo_retriever.operators.extract.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")

    mock_img = MagicMock(image=_make_test_png_b64(), bbox=(10, 20, 100, 200), max_width=612, max_height=792)
    mock_extract.return_value = [mock_img]

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]), extract_images=True
    )
    images = result.iloc[0]["images"]
    assert len(images) == 1
    assert images[0]["text"] == ""
    assert abs(images[0]["bbox_xyxy_norm"][0] - 10 / 612) < 1e-6


@patch("nemo_retriever.operators.extract.pdf.extract.extract_nested_simple_images_from_pdfium_page")
@patch("nemo_retriever.operators.extract.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_adds_nested_images_when_enabled(mock_extract, mock_extract_nested):
    _ext = pytest.importorskip("nemo_retriever.operators.extract.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")

    rendered = MagicMock(image=_make_test_png_b64(), bbox=(10, 20, 100, 200), max_width=612, max_height=792)
    nested = MagicMock(image=_make_test_png_b64(), bbox=(20, 30, 80, 90), max_width=612, max_height=792)
    mock_extract.return_value = [rendered]
    mock_extract_nested.return_value = [nested, nested]

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]),
        extract_images=True,
        extract_nested_images=True,
    )

    images = result.iloc[0]["images"]
    assert len(images) == 3
    assert images[0]["bbox_xyxy_norm"] == pytest.approx([10 / 612, 20 / 792, 100 / 612, 200 / 792])
    assert images[1]["bbox_xyxy_norm"] == pytest.approx([20 / 612, 30 / 792, 80 / 612, 90 / 792])
    assert images[1] == images[2]


@patch("nemo_retriever.operators.extract.pdf.extract.extract_nested_simple_images_from_pdfium_page")
@patch("nemo_retriever.operators.extract.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_does_not_extract_nested_images_by_default(mock_extract, mock_extract_nested):
    _ext = pytest.importorskip("nemo_retriever.operators.extract.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")
    mock_extract.return_value = []

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]), extract_images=True
    )

    mock_extract_nested.assert_not_called()


@patch("nemo_retriever.operators.extract.pdf.extract.extract_nested_simple_images_from_pdfium_page")
@patch("nemo_retriever.operators.extract.pdf.extract.extract_image_like_objects_from_pdfium_page")
def test_pdf_extraction_records_nested_image_failure(mock_extract, mock_extract_nested):
    _ext = pytest.importorskip("nemo_retriever.operators.extract.pdf.extract")
    pdfium = pytest.importorskip("pypdfium2")
    from nemo_retriever.models.nim.error_reporter import drain_errors

    base_image = MagicMock()
    base_image.image = "base-image"
    base_image.bbox = [10, 20, 100, 200]
    base_image.max_width = 612
    base_image.max_height = 792
    mock_extract.return_value = [base_image]
    mock_extract_nested.side_effect = RuntimeError("nested image budget exceeded")
    drain_errors()

    doc = pdfium.PdfDocument.new()
    doc.new_page(612, 792)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _ext.pdf_extraction(
        pd.DataFrame([{"bytes": buf.getvalue(), "path": "t.pdf", "page_number": 1}]),
        extract_images=True,
        extract_nested_images=True,
    )

    row = result.iloc[0]
    assert row["page_image"] is not None
    assert len(row["images"]) == 1
    assert row["images"][0]["image_b64"] == "base-image"
    assert row["metadata"]["error"]["stage"] == "nested_images"
    assert row["metadata"]["error"]["type"] == "RuntimeError"
    assert row["metadata"]["error"]["message"] == "nested image budget exceeded"
    errors = drain_errors()
    assert len(errors) == 1
    assert errors[0].stage == "pdf_extraction:nested_images"
    assert errors[0].message == "nested image budget exceeded"


def test_explode_includes_captioned_images():
    from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

    b64 = _make_test_png_b64()
    df = pd.DataFrame(
        [
            {
                "text": "page",
                "page_image": {"image_b64": b64},
                "images": [{"text": "a dog", "bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "image_b64": b64}],
                "tables": [],
                "charts": [],
                "infographics": [],
            }
        ]
    )
    result = explode_content_to_rows(df, content_columns=("table", "chart", "infographic", "images"))
    assert len(result) == 2  # page text + image caption

    # Default columns exclude images
    result2 = explode_content_to_rows(df)
    assert len(result2) == 1


def test_context_text_prepended_to_prompt():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["captioned with context"]

    df = _make_page_df(num_images=1)
    df.at[0, "text"] = "The quick brown fox jumps over the lazy dog."

    result = caption_images(df, model=mock_model, context_text_max_chars=100)

    assert result.iloc[0]["images"][0]["text"] == "captioned with context"
    # The prompt passed to caption_batch should contain the page text.
    call_kwargs = mock_model.caption_batch.call_args[1]
    assert "quick brown fox" in call_kwargs["prompt"]
    assert "Text near this image:" in call_kwargs["prompt"]


def test_caption_images_skips_small_images():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    tiny_b64 = _make_test_png_b64(size=(1, 1))
    images = [{"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.8], "text": "", "image_b64": tiny_b64}]
    df = pd.DataFrame([{"text": "page", "images": images, "tables": [], "charts": [], "infographics": []}])

    mock_model = MagicMock()
    result = caption_images(df, model=mock_model)
    mock_model.caption_batch.assert_not_called()
    assert result.iloc[0]["images"][0]["text"] == ""


# ---------------------------------------------------------------------------
# LLMInferenceParams & CaptionParams tests
# ---------------------------------------------------------------------------


class TestLLMInferenceParamsValidation:
    """Validate LLMInferenceParams range checks and defaults."""

    def test_defaults(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        p = LLMInferenceParams()
        assert p.temperature == 1.0
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_negative_temperature_rejected(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="temperature must be between"):
            LLMInferenceParams(temperature=-0.1)

    def test_top_p_out_of_range_rejected(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="top_p must be between"):
            LLMInferenceParams(top_p=1.5)
        with pytest.raises(ValueError, match="top_p must be between"):
            LLMInferenceParams(top_p=-0.1)

    def test_zero_max_tokens_rejected(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            LLMInferenceParams(max_tokens=0)

    def test_valid_top_p_accepted(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        p = LLMInferenceParams(top_p=0.95)
        assert p.top_p == 0.95

    def test_to_sampling_kwargs_includes_top_p_when_set(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        kw = LLMInferenceParams(temperature=0.3, top_p=0.9, max_tokens=512).to_sampling_kwargs()
        assert kw == {"temperature": 0.3, "top_p": 0.9, "max_tokens": 512}

    def test_to_sampling_kwargs_omits_top_p_when_none(self):
        from nemo_retriever.common.params.models import LLMInferenceParams

        kw = LLMInferenceParams(temperature=0.5).to_sampling_kwargs()
        assert kw == {"temperature": 0.5, "max_tokens": 1024}
        assert "top_p" not in kw


class TestCaptionParamsInheritance:
    """Verify CaptionParams inherits LLM fields and model_dump stays flat."""

    def test_inherits_llm_fields(self):
        from nemo_retriever.common.params import CaptionParams

        p = CaptionParams()
        assert p.temperature == 1.0
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_model_dump_is_flat(self):
        from nemo_retriever.common.params import CaptionParams

        d = CaptionParams(temperature=0.7, top_p=0.9, max_tokens=512).model_dump()
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.9
        assert d["max_tokens"] == 512
        assert "llm" not in d

    def test_backward_compat_temperature_only(self):
        from nemo_retriever.common.params import CaptionParams

        p = CaptionParams(temperature=0.5)
        assert p.temperature == 0.5
        assert p.top_p is None
        assert p.max_tokens == 1024

    def test_validation_inherited(self):
        from nemo_retriever.common.params import CaptionParams

        with pytest.raises(ValueError, match="temperature must be between"):
            CaptionParams(temperature=-1.0)

    def test_null_temperature_rejected(self):
        from nemo_retriever.common.params import CaptionParams

        with pytest.raises(ValueError, match="cannot be None for captioning"):
            CaptionParams(temperature=None)


class TestCaptionImageParamThreading:
    """Verify top_p and max_tokens flow through to the model / client."""

    def test_top_p_forwarded_to_local_model(self):
        from nemo_retriever.operators.extract.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model, top_p=0.9)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    def test_max_tokens_forwarded_to_local_model(self):
        from nemo_retriever.operators.extract.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model, max_tokens=512)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    def test_defaults_forwarded_when_omitted(self):
        from nemo_retriever.operators.extract.caption.caption import caption_images

        mock_model = MagicMock()
        mock_model.caption_batch.return_value = ["cap"]
        caption_images(_make_page_df(num_images=1), model=mock_model)

        call_kwargs = mock_model.caption_batch.call_args[1]
        assert call_kwargs["top_p"] is None
        assert call_kwargs["max_tokens"] == 1024

    @patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
    def test_top_p_and_max_tokens_forwarded_to_remote(self, mock_create_client):
        from nemo_retriever.operators.extract.caption.caption import caption_images

        mock_nim = MagicMock()
        mock_nim.infer.return_value = ["remote cap"]
        mock_create_client.return_value = mock_nim

        caption_images(
            _make_page_df(num_images=1),
            endpoint_url="http://fake-vlm/v1",
            top_p=0.85,
            max_tokens=256,
        )

        infer_kwargs = mock_nim.infer.call_args[1]
        assert infer_kwargs["top_p"] == 0.85
        assert infer_kwargs["max_tokens"] == 256
        assert infer_kwargs["temperature"] == 1.0

    @patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
    def test_remote_omits_top_p_when_none(self, mock_create_client):
        from nemo_retriever.operators.extract.caption.caption import caption_images

        mock_nim = MagicMock()
        mock_nim.infer.return_value = ["remote cap"]
        mock_create_client.return_value = mock_nim

        caption_images(
            _make_page_df(num_images=1),
            endpoint_url="http://fake-vlm/v1",
        )

        infer_kwargs = mock_nim.infer.call_args[1]
        assert "top_p" not in infer_kwargs
        assert infer_kwargs["max_tokens"] == 1024


def test_caption_params_accepts_extra_body():
    from nemo_retriever.common.params import CaptionParams

    params = CaptionParams(extra_body={"chat_template_kwargs": {"enable_thinking": True}})

    assert params.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert params.model_dump(mode="python")["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_vlm_model_interface_forwards_request_extras():
    from nemo_retriever.models.nim.primitives.model_interface.vlm import VLMModelInterface

    interface = VLMModelInterface()
    payloads, batch_data = interface.format_input(
        {"base64_images": ["abc123"], "prompt": "Caption this."},
        protocol="http",
        max_batch_size=8,
        model_name="nvidia/test-vlm",
        temperature=0.25,
        max_tokens=99,
        chat_template_kwargs={"enable_thinking": False},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        media_options={"image": {"detail": "high"}},
        extra_body={"custom_request_id": "caption-123"},
    )

    assert len(payloads) == 1
    assert batch_data == [{"base64_images": ["abc123"], "prompt": "Caption this."}]
    assert payloads[0]["model"] == "nvidia/test-vlm"
    assert payloads[0]["temperature"] == 0.25
    assert payloads[0]["max_tokens"] == 99
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payloads[0]["mm_processor_kwargs"] == {"max_dynamic_patch": 4}
    assert payloads[0]["media_options"] == {"image": {"detail": "high"}}
    assert payloads[0]["custom_request_id"] == "caption-123"


def test_vlm_model_interface_extra_body_overrides_payload_fields():
    from nemo_retriever.models.nim.primitives.model_interface.vlm import VLMModelInterface

    interface = VLMModelInterface()
    payloads, _batch_data = interface.format_input(
        {"base64_images": ["abc123"], "prompt": "Caption this."},
        protocol="http",
        max_batch_size=8,
        model_name="nvidia/test-vlm",
        temperature=0.25,
        chat_template_kwargs={"enable_thinking": False},
        extra_body={
            "temperature": 0.05,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert len(payloads) == 1
    assert payloads[0]["temperature"] == 0.05
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_remote_omni_uses_hosted_model_and_profile_extras(mock_create_client):
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    result = caption_images(
        _make_page_df(num_images=1),
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    )

    assert result.iloc[0]["images"][0]["text"] == "remote cap"
    infer_kwargs = mock_nim.infer.call_args[1]
    assert infer_kwargs["model_name"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    assert infer_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_caption_cpu_actor_default_extra_body_does_not_repack_profile_extras(mock_create_client):
    from nemo_retriever.operators.extract.caption.caption import CaptionCPUActor
    from nemo_retriever.common.params import CaptionParams

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    params = CaptionParams(
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )
    actor = CaptionCPUActor(params)

    result = actor.process(_make_page_df(num_images=1))

    assert result.iloc[0]["images"][0]["text"] == "remote cap"
    infer_kwargs = mock_nim.infer.call_args[1]
    assert infer_kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in infer_kwargs


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_caption_cpu_actor_defaults_to_hosted_endpoint_when_api_key_is_configured(mock_create_client):
    from nemo_retriever.operators.extract.caption.caption import CaptionCPUActor
    from nemo_retriever.common.params import CaptionParams

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    actor = CaptionCPUActor(CaptionParams(api_key="nvapi-test"))
    result = actor.process(_make_page_df(num_images=1))

    assert result.iloc[0]["images"][0]["text"] == "remote cap"
    mock_create_client.assert_called_once()
    assert mock_create_client.call_args.args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
    infer_kwargs = mock_nim.infer.call_args.kwargs
    assert infer_kwargs["model_name"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_caption_cpu_actor_default_endpoint_reads_api_key_from_runtime_env(mock_create_client, monkeypatch):
    from nemo_retriever.operators.extract.caption.caption import CaptionCPUActor
    from nemo_retriever.common.params import CaptionParams

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    params = CaptionParams()
    assert params.api_key is None

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-env")
    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    actor = CaptionCPUActor(params)
    actor.process(_make_page_df(num_images=1))

    assert actor._kwargs["api_key"] == "nvapi-env"
    mock_create_client.assert_called_once_with(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        "nvapi-env",
    )


def test_caption_cpu_actor_default_endpoint_requires_api_key(monkeypatch):
    from nemo_retriever.operators.extract.caption.caption import CaptionCPUActor
    from nemo_retriever.common.params import CaptionParams

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="no API key is configured"):
        CaptionCPUActor(CaptionParams())


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_remote_omni_user_extra_body_overrides_profile_defaults(mock_create_client):
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    caption_images(
        _make_page_df(num_images=1),
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 32}},
    )

    infer_kwargs = mock_nim.infer.call_args[1]
    assert infer_kwargs["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 32}


def test_caption_batch_remote_request_extras_override_sampling_defaults():
    from nemo_retriever.operators.extract.caption.caption import _caption_batch_remote

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]

    result = _caption_batch_remote(
        [_make_test_png_b64()],
        nim_client=mock_nim,
        model_name="nvidia/test-vlm",
        prompt="Caption this.",
        system_prompt=None,
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        request_extras={
            "temperature": 0.05,
            "top_p": 0.1,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert result == ["remote cap"]
    infer_kwargs = mock_nim.infer.call_args[1]
    assert infer_kwargs["model_name"] == "nvidia/test-vlm"
    assert infer_kwargs["temperature"] == 0.05
    assert infer_kwargs["top_p"] == 0.1
    assert infer_kwargs["max_tokens"] == 32
    assert infer_kwargs["chat_template_kwargs"] == {"enable_thinking": False}


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_remote_extra_body_arbitrary_keys_reach_formatted_payload(mock_create_client):
    from nemo_retriever.models.nim.primitives.model_interface.vlm import VLMModelInterface
    from nemo_retriever.operators.extract.caption.caption import caption_images

    formatted_payloads = []
    infer_kwargs = []

    class FormattingNimClient:
        def infer(self, data, **kwargs):
            infer_kwargs.append(kwargs)
            payloads, _batch_data = VLMModelInterface().format_input(
                data,
                protocol="http",
                max_batch_size=8,
                **kwargs,
            )
            formatted_payloads.extend(payloads)
            return ["remote cap"]

    mock_create_client.return_value = FormattingNimClient()
    user_extra_body = {
        "custom_request_id": "caption-123",
        "temperature": 0.05,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 32},
    }

    result = caption_images(
        _make_page_df(num_images=1),
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        temperature=0.7,
        extra_body=user_extra_body,
    )

    assert result.iloc[0]["images"][0]["text"] == "remote cap"
    assert infer_kwargs[0]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 32}
    assert formatted_payloads[0]["custom_request_id"] == "caption-123"
    assert formatted_payloads[0]["temperature"] == 0.05
    assert formatted_payloads[0]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_budget": 32}
    assert user_extra_body == {
        "custom_request_id": "caption-123",
        "temperature": 0.05,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 32},
    }


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_remote_omni_partial_extra_body_preserves_profile_defaults_in_formatted_payload(mock_create_client):
    from nemo_retriever.models.nim.primitives.model_interface.vlm import VLMModelInterface
    from nemo_retriever.operators.extract.caption.caption import caption_images

    formatted_payloads = []

    class FormattingNimClient:
        def infer(self, data, **kwargs):
            payloads, _batch_data = VLMModelInterface().format_input(
                data,
                protocol="http",
                max_batch_size=8,
                **kwargs,
            )
            formatted_payloads.extend(payloads)
            return ["remote cap"]

    mock_create_client.return_value = FormattingNimClient()
    user_extra_body = {"chat_template_kwargs": {"reasoning_budget": 32}}

    caption_images(
        _make_page_df(num_images=1),
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        extra_body=user_extra_body,
    )

    assert formatted_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_budget": 32}
    assert user_extra_body == {"chat_template_kwargs": {"reasoning_budget": 32}}


@patch("nemo_retriever.operators.extract.caption.caption._create_remote_client")
def test_unknown_remote_model_passes_through_without_profile_extras(mock_create_client):
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_nim = MagicMock()
    mock_nim.infer.return_value = ["remote cap"]
    mock_create_client.return_value = mock_nim

    caption_images(
        _make_page_df(num_images=1),
        endpoint_url="https://example.test/v1/chat/completions",
        model_name="acme/custom-vlm",
    )

    infer_kwargs = mock_nim.infer.call_args[1]
    assert infer_kwargs["model_name"] == "acme/custom-vlm"
    assert "chat_template_kwargs" not in infer_kwargs


def test_caption_images_threads_resolved_local_model_name_to_loader(monkeypatch):
    from nemo_retriever.operators.extract.caption import caption as caption_module

    created_kwargs = []

    class FakeLocalCaptioner:
        def caption_batch(self, base64_images, **kwargs):
            return ["local cap" for _ in base64_images]

    def fake_create_local_model(kwargs):
        created_kwargs.append(kwargs)
        return FakeLocalCaptioner()

    monkeypatch.setattr(caption_module, "_cached_local_model", None)
    monkeypatch.setattr(caption_module, "_create_local_model", fake_create_local_model)

    result = caption_module.caption_images(
        _make_page_df(num_images=1),
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8",
    )

    assert result.iloc[0]["images"][0]["text"] == "local cap"
    assert created_kwargs[0]["model_name"] == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"


def test_caption_images_local_cache_keys_by_resolved_loader_kwargs(monkeypatch):
    from nemo_retriever.operators.extract.caption import caption as caption_module

    created_kwargs = []

    class FakeLocalCaptioner:
        def __init__(self, label):
            self.label = label

        def caption_batch(self, base64_images, **kwargs):
            return [self.label for _ in base64_images]

    def fake_create_local_model(kwargs):
        created_kwargs.append(dict(kwargs))
        return FakeLocalCaptioner(f"local cap {len(created_kwargs)}")

    monkeypatch.setattr(caption_module, "_cached_local_model", None)
    monkeypatch.setattr(caption_module, "_create_local_model", fake_create_local_model)

    default_result = caption_module.caption_images(_make_page_df(num_images=1), max_tokens=512)
    default_again = caption_module.caption_images(_make_page_df(num_images=1), max_tokens=512)
    default_different_tokens = caption_module.caption_images(_make_page_df(num_images=1), max_tokens=256)
    omni_result = caption_module.caption_images(
        _make_page_df(num_images=1),
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8",
        max_tokens=512,
    )
    omni_again = caption_module.caption_images(
        _make_page_df(num_images=1),
        model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
        max_tokens=512,
    )

    assert default_result.iloc[0]["images"][0]["text"] == "local cap 1"
    assert default_again.iloc[0]["images"][0]["text"] == "local cap 1"
    assert default_different_tokens.iloc[0]["images"][0]["text"] == "local cap 1"
    assert omni_result.iloc[0]["images"][0]["text"] == "local cap 2"
    assert omni_again.iloc[0]["images"][0]["text"] == "local cap 2"
    assert [kwargs["model_name"] for kwargs in created_kwargs] == [
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    ]


def test_caption_images_forwards_local_user_extra_body_only_when_supplied():
    from nemo_retriever.operators.extract.caption.caption import caption_images

    mock_model = MagicMock()
    mock_model.caption_batch.return_value = ["cap"]

    caption_images(
        _make_page_df(num_images=1),
        model=mock_model,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    call_kwargs = mock_model.caption_batch.call_args[1]
    assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
