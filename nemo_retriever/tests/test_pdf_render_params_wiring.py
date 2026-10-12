# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for forwarding public PDF rendering parameters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import EmbedParams, ExtractParams
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.graph_ops import multi_type_extract_operator as multi_type_module
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor


def _render_params() -> ExtractParams:
    return ExtractParams(
        extract_tables=False,
        extract_charts=False,
        render_mode="full_dpi",
        image_format="png",
        jpeg_quality=73,
        extract_nested_images=True,
    )


def test_pdf_graph_forwards_rendering_params() -> None:
    graph = build_graph(extraction_mode="pdf", extract_params=_render_params())

    extract_kwargs = get_node_kwargs(graph, "PDFExtractionActor")

    assert extract_kwargs["render_mode"] == "full_dpi"
    assert extract_kwargs["image_format"] == "png"
    assert extract_kwargs["jpeg_quality"] == 73
    assert extract_kwargs["extract_nested_images"] is True


class _IdentityStage:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, data):
        return data


def test_multitype_pdf_graph_forwards_rendering_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _CapturingPDFExtractionActor(_IdentityStage):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(multi_type_module, "DocToPdfConversionActor", _IdentityStage)
    monkeypatch.setattr(multi_type_module, "PDFSplitActor", _IdentityStage)
    monkeypatch.setattr(multi_type_module, "PDFExtractionActor", _CapturingPDFExtractionActor)

    operator = MultiTypeExtractCPUActor(extraction_mode="pdf", extract_params=_render_params())
    monkeypatch.setattr(operator, "_run_detection_pipeline", lambda data: data)
    monkeypatch.setattr(operator, "_maybe_chunk", lambda data, _mode: data)

    operator._run_pdf_pipeline(pd.DataFrame())

    assert captured["render_mode"] == "full_dpi"
    assert captured["image_format"] == "png"
    assert captured["jpeg_quality"] == 73
    assert captured["extract_nested_images"] is True


@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_sdk_auto_page_image_embedding_skips_unused_page_elements(monkeypatch, modality: str) -> None:
    remote_calls: list[dict[str, object]] = []

    def _fake_async_runner(prompt_batches, _api_key, endpoint_url, model_name, *_args, **kwargs):
        prompt_count = sum(len(batch) for batch in prompt_batches)
        remote_calls.append(
            {
                "prompt_batches": prompt_batches,
                "endpoint_url": endpoint_url,
                "model_name": model_name,
                "modalities": kwargs["modalities"],
            }
        )
        return {
            "embeddings": [[1.0] * 2048 for _ in range(prompt_count)],
        }

    class _UnexpectedPageElementsActor:
        def __init__(self, **_kwargs) -> None:
            pytest.fail("Page Elements must not be initialized when it is disabled")

    monkeypatch.setattr(
        "nemo_retriever.models.inference.main_text_embed._async_runner",
        _fake_async_runner,
    )
    monkeypatch.setattr(multi_type_module, "PageElementDetectionActor", _UnexpectedPageElementsActor)
    document = Path(__file__).resolve().parents[2] / "data" / "multimodal_test.pdf"
    model = "nvidia/llama-nemotron-embed-vl-1b-v2"
    endpoint = "http://vl-embed.example/v1/embeddings"

    result = (
        create_ingestor(run_mode="inprocess", allow_no_gpu=True)
        .files([str(document)])
        .extract(
            params=ExtractParams(
                extract_text=True,
                extract_images=False,
                extract_tables=False,
                extract_charts=False,
                extract_infographics=False,
                extract_page_as_image=False,
                use_page_elements=False,
            ),
            extraction_mode="auto",
        )
        .embed(
            params=EmbedParams(
                model_name=model,
                embed_model_name=model,
                embed_invoke_url=endpoint,
                embed_modality=modality,
                embed_granularity="page",
            )
        )
        .ingest()
    )

    assert len(result) == 3
    assert result["page_number"].tolist() == [1, 2, 3]
    assert (
        result["page_image"]
        .map(lambda page_image: isinstance(page_image, dict) and bool(page_image.get("image_b64")))
        .all()
    )
    assert result["_image_b64"].map(lambda image_b64: isinstance(image_b64, str) and bool(image_b64)).all()
    assert result["embedding"].notna().all()

    assert len(remote_calls) == 1
    remote_call = remote_calls[0]
    assert remote_call["endpoint_url"] == endpoint
    assert remote_call["model_name"] == model
    prompt_batches = remote_call["prompt_batches"]
    assert isinstance(prompt_batches, list)
    assert sum(len(batch) for batch in prompt_batches) == 3
    assert all(isinstance(prompt, str) and prompt for prompt_batch in prompt_batches for prompt in prompt_batch)
    modalities = remote_call["modalities"]
    assert isinstance(modalities, list)
    assert [item for batch in modalities for item in batch] == [modality] * 3
