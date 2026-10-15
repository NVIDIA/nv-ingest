# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Parse options reach both graph paths and their inference requests."""

import base64
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.ray_resource_hueristics import Resources
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.ingest.plan import (
    IngestExtractOptions,
    IngestPlanRequest,
    IngestSourceOptions,
    resolve_ingest_plan,
)
from nemo_retriever.operators.extract.parse import nemotron_parse as parse
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor


PROMPT_WITH_PICTURE_TEXT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
PROMPT_WITHOUT_PICTURE_TEXT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
TAGGED_OUTPUT = (
    "<x_0><y_0>Body text<x_1><y_1><class_Text>"
    "<x_0><y_0>A | B<x_1><y_1><class_Table>"
    "<x_0><y_0>Revenue: 42<x_1><y_1><class_Chart>"
    "<x_0><y_0>Picture text<x_1><y_1><class_Picture>"
)


def _nodes(graph):
    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)

    for root in graph.roots:
        yield from walk(root)


@pytest.mark.parametrize("graph_path", ["batch", "inprocess"])
@pytest.mark.parametrize("predict_text_in_pic", [None, True, False])
@pytest.mark.parametrize(
    ("endpoint", "model"),
    [
        (None, "nvidia/NVIDIA-Nemotron-Parse-2.0"),
        ("http://parse:8000/v1/chat/completions", "nvidia/nemotron-parse-v2.0"),
        ("https://integrate.api.nvidia.com/v1/chat/completions", "nvidia/nemotron-parse-2.0"),
    ],
)
def test_typed_parse_options_reach_inference(tmp_path, monkeypatch, graph_path, predict_text_in_pic, endpoint, model):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    plan = resolve_ingest_plan(
        IngestPlanRequest(
            source=IngestSourceOptions(documents=[str(pdf)]),
            extract=IngestExtractOptions(
                method="nemotron_parse",
                nemotron_parse_model=model,
                nemotron_parse_invoke_url=endpoint,
                nemotron_parse_predict_text_in_pic=predict_text_in_pic,
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=True,
            ),
        )
    )
    assert plan.extract_params.nemotron_parse_model == model
    assert plan.extract_params.nemotron_parse_invoke_url == endpoint
    assert plan.extract_params.nemotron_parse_predict_text_in_pic is predict_text_in_pic

    image_buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(image_buffer, format="PNG")
    image_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    batch = pd.DataFrame({"path": [str(pdf)], "page_image": [{"image_b64": image_b64}], "text": ["Native"]})
    expected_prompt = PROMPT_WITH_PICTURE_TEXT if predict_text_in_pic else PROMPT_WITHOUT_PICTURE_TEXT
    local_loader = MagicMock()
    local_loader.return_value.invoke_batch.return_value = [TAGGED_OUTPUT]
    monkeypatch.setattr("nemo_retriever.models.local.NemotronParse20", local_loader)
    http_calls = []

    def fake_post(**kwargs):
        http_calls.append(kwargs)
        return {"choices": [{"message": {"content": TAGGED_OUTPUT}}]}

    monkeypatch.setattr("nemo_retriever.models.nim.nim._post_with_retries", fake_post)
    resources = Resources(cpu_count=8, gpu_count=1)
    if graph_path == "batch":
        graph = build_graph(extraction_mode="pdf", extract_params=plan.extract_params).resolve(resources)
        node = next(
            node
            for node in _nodes(graph)
            if node.operator_class in (parse.NemotronParseGPUActor, parse.NemotronParseCPUActor)
        )
        result = node.operator_class(**node.operator_kwargs).run(batch)
    else:
        monkeypatch.setattr("nemo_retriever.common.ray_resource_hueristics.gather_local_resources", lambda: resources)
        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.DocToPdfConversionActor.run",
            lambda self, data: data,
        )
        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.PDFSplitActor.run",
            lambda self, data: data,
        )
        result = MultiTypeExtractCPUActor(extraction_mode="pdf", extract_params=plan.extract_params)._run_pdf_pipeline(
            batch
        )

    if endpoint is None:
        local_loader.assert_called_once_with(model_path=model, task_prompt=expected_prompt)
        args, kwargs = local_loader.return_value.invoke_batch.call_args
        assert len(args[0]) == 1 and args[0][0].shape == (2, 2, 3)
        assert kwargs == {"task_prompt": expected_prompt}
        assert not http_calls
    else:
        local_loader.assert_not_called()
        assert len(http_calls) == 1
        assert http_calls[0]["invoke_url"] == endpoint
        payload = http_calls[0]["payload"]
        assert payload["model"] == model
        assert payload["messages"][0]["content"] == [
            {"type": "text", "text": expected_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        if model == "nvidia/nemotron-parse-v2.0":
            assert {
                key: payload[key] for key in ("max_tokens", "top_k", "skip_special_tokens", "repetition_penalty")
            } == {
                "max_tokens": 8192,
                "top_k": 1,
                "skip_special_tokens": False,
                "repetition_penalty": 1.1,
            }
        else:
            assert payload["max_tokens"] == 8192
            assert not {"top_k", "skip_special_tokens", "repetition_penalty"} & payload.keys()
    assert result.at[0, "text"] == "Body text"
    assert result.at[0, "chart"][0]["text"] == "Revenue: 42"
    assert result.at[0, "infographic"][0]["text"] == "Picture text"
    assert result.at[0, "nemotron_parse_v1_2"]["error"] is None


@pytest.mark.parametrize("entrypoint", ["typed", "params", "cpu", "gpu", "pages"])
@pytest.mark.parametrize("predict_text_in_pic", [True, False])
@pytest.mark.parametrize(
    ("endpoint", "model", "error"),
    [
        ("https://integrate.api.nvidia.com/v1/chat/completions", None, "image-only hosted-default"),
        ("http://parse:8000/v1/chat/completions", "nvidia/nemotron-parse", "image-only hosted-default"),
        ("http://parse:8000/v1/infer", "nvidia/nemotron-parse-v2.0", "cannot carry a task prompt"),
        (
            "http://parse:8000/v1/chat/completions,http://parse2:8000/v1/infer",
            "nvidia/nemotron-parse-v2.0",
            "cannot carry a task prompt",
        ),
    ],
)
def test_unsupported_picture_text_controls_fail_before_dispatch(
    tmp_path, monkeypatch, entrypoint, predict_text_in_pic, endpoint, model, error
):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    client = MagicMock()
    dispatch = MagicMock()
    monkeypatch.setattr(parse, "NIMClient", client)
    monkeypatch.setattr(parse, "invoke_chat_completions_images", dispatch)
    monkeypatch.setattr(parse, "invoke_image_inference_batches", dispatch)
    kwargs = {
        "nemotron_parse_invoke_url": endpoint,
        "nemotron_parse_model": model,
        "nemotron_parse_predict_text_in_pic": predict_text_in_pic,
    }
    with pytest.raises(ValueError, match=error):
        if entrypoint == "typed":
            resolve_ingest_plan(
                IngestPlanRequest(
                    source=IngestSourceOptions(documents=[str(pdf)]),
                    extract=IngestExtractOptions(method="nemotron_parse", **kwargs),
                )
            )
        elif entrypoint == "params":
            ExtractParams(method="nemotron_parse", **kwargs)
        elif entrypoint == "pages":
            parse.nemotron_parse_pages(pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]}), **kwargs)
        else:
            actor_class = parse.NemotronParseCPUActor if entrypoint == "cpu" else parse.NemotronParseGPUActor
            actor_class(**kwargs)
    client.assert_not_called()
    dispatch.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nemotron_parse_model", "nvidia/NVIDIA-Nemotron-Parse-2.0"),
        ("nemotron_parse_invoke_url", "http://parse:8000/v1/chat/completions"),
        ("nemotron_parse_predict_text_in_pic", True),
        ("nemotron_parse_predict_text_in_pic", False),
    ],
)
def test_typed_parse_controls_require_parse_method(tmp_path, field, value):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    with pytest.raises(ValueError, match="method='nemotron_parse'"):
        resolve_ingest_plan(
            IngestPlanRequest(
                source=IngestSourceOptions(documents=[str(pdf)]),
                extract=IngestExtractOptions(**{field: value}),
            )
        )


@pytest.mark.parametrize(
    ("task_prompt", "control", "expected_prompt"),
    [
        (
            "<predict_bbox><predict_no_text_in_pic><output_markdown>",
            True,
            "<predict_bbox><predict_text_in_pic><output_markdown>",
        ),
        (
            "<predict_classes><predict_text_in_pic><output_markdown>",
            False,
            "<predict_classes><predict_no_text_in_pic><output_markdown>",
        ),
        ("<predict_classes><output_markdown>", True, "<predict_classes><output_markdown><predict_text_in_pic>"),
        (
            "<predict_text_in_pic><predict_no_text_in_pic><output_markdown>",
            False,
            "<predict_no_text_in_pic><output_markdown>",
        ),
        (
            "<predict_classes><predict_text_in_pic><output_markdown>",
            None,
            "<predict_classes><predict_text_in_pic><output_markdown>",
        ),
    ],
)
def test_direct_actor_control_overrides_only_picture_text_token(monkeypatch, task_prompt, control, expected_prompt):
    client = MagicMock()
    client.invoke_chat_completions_images.return_value = [TAGGED_OUTPUT]
    monkeypatch.setattr(parse, "NIMClient", lambda **kwargs: client)
    actor = parse.NemotronParseCPUActor(
        nemotron_parse_invoke_url="http://parse:8000/v1/chat/completions",
        nemotron_parse_model="nvidia/nemotron-parse-v2.0",
        task_prompt=task_prompt,
        nemotron_parse_predict_text_in_pic=control,
    )
    actor.run(pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]}))
    assert client.invoke_chat_completions_images.call_args.kwargs["task_prompt"] == expected_prompt
