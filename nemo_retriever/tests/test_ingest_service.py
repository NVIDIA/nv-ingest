# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_retriever.common.params import DedupParams, EmbedParams, ExtractParams, TextChunkParams
from nemo_retriever.common.policy import validate_pipeline_spec
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.ingest.service import (
    ServiceIngestCaptionOptions,
    ServiceIngestDedupOptions,
    ServiceIngestPlanRequest,
    ServiceIngestRequest,
    ServiceIngestSourceOptions,
    build_service_ingestor,
    execute_service_ingest_request,
    resolve_service_ingest_request,
)
from nemo_retriever.service.config import PipelineOverridesConfig
from nemo_retriever.service.service_ingestor import ServiceIngestor, ServiceIngestResult


def test_build_service_ingestor_wires_extract_embed_and_chunking(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    ingestor = build_service_ingestor(
        ServiceIngestRequest(
            documents=[str(pdf)],
            input_type="pdf",
            extract_params=ExtractParams(method="ocr", extract_text=False, dpi=300),
            embed_params=EmbedParams(embed_granularity="page"),
            text_chunk_params=TextChunkParams(max_tokens=64, overlap_tokens=8),
            enable_text_chunk=True,
        )
    )

    assert isinstance(ingestor, ServiceIngestor)
    payload = ingestor._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "pdf"
    assert payload["extract_params"]["method"] == "ocr"
    assert payload["extract_params"]["extract_text"] is False
    assert payload["extract_params"]["dpi"] == 300
    assert "batch_tuning" not in payload["extract_params"]
    assert payload["split_config"]["pdf"]["max_tokens"] == 64
    assert payload["split_config"]["pdf"]["overlap_tokens"] == 8
    assert payload["embed_params"]["embed_granularity"] == "page"
    assert "model_name" not in payload["embed_params"]

    validate_pipeline_spec(
        PipelineSpec.model_validate(ingestor._pipeline_spec),
        PipelineOverridesConfig().to_policy(),
    )


def test_resolve_service_caption_leaves_automatic_dedup_to_worker(tmp_path: Path) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4")
    resolved = resolve_service_ingest_request(
        ServiceIngestPlanRequest(
            source=ServiceIngestSourceOptions(documents=[str(document)]),
            caption=ServiceIngestCaptionOptions(enabled=True),
        )
    )

    assert resolved.dedup_params is None


def test_build_service_ingestor_wires_explicit_no_dedup_opt_out(tmp_path: Path) -> None:
    pdf = tmp_path / "captioned.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    request = resolve_service_ingest_request(
        ServiceIngestPlanRequest(
            source=ServiceIngestSourceOptions(documents=[str(pdf)]),
            caption=ServiceIngestCaptionOptions(enabled=True),
            dedup=ServiceIngestDedupOptions(enabled=False),
        )
    )

    ingestor = build_service_ingestor(request)
    payload = ingestor._pipeline_payload()

    assert request.dedup_params == DedupParams(content_hash=False, bbox_iou=False)
    assert payload is not None
    assert payload["dedup_params"] == {"content_hash": False, "bbox_iou": False}
    assert payload["stage_order"].index("dedup") < payload["stage_order"].index("caption")
    validate_pipeline_spec(
        PipelineSpec.model_validate(ingestor._pipeline_spec),
        PipelineOverridesConfig().to_policy(caption_enabled=True),
    )


def test_build_service_ingestor_does_not_forward_environment_api_key(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("NGC_API_KEY", "environment-secret")
    stage_params: dict[str, object] = {}
    original_extract = ServiceIngestor.extract
    original_embed = ServiceIngestor.embed

    def capture_extract(self, params=None, **kwargs):
        stage_params["extract"] = params
        return original_extract(self, params, **kwargs)

    def capture_embed(self, params=None, **kwargs):
        stage_params["embed"] = params
        return original_embed(self, params, **kwargs)

    monkeypatch.setattr(ServiceIngestor, "extract", capture_extract)
    monkeypatch.setattr(ServiceIngestor, "embed", capture_embed)

    ingestor = build_service_ingestor(
        ServiceIngestRequest(
            documents=[str(pdf)],
            input_type="pdf",
            extract_params=ExtractParams(method="pdfium", use_table_structure=True),
            embed_params=EmbedParams(embed_granularity="page"),
        )
    )

    payload = ingestor._pipeline_payload()
    assert payload is not None
    assert isinstance(stage_params["extract"], ExtractParams)
    assert isinstance(stage_params["embed"], EmbedParams)
    assert payload["extract_params"]["method"] == "pdfium"
    assert payload["extract_params"]["use_table_structure"] is True
    assert payload["embed_params"]["embed_granularity"] == "page"
    assert not ({"api_key", "page_elements_api_key", "ocr_api_key"} & payload["extract_params"].keys())
    assert "api_key" not in payload["embed_params"]


def test_execute_service_ingest_request_raises_for_document_failures(monkeypatch, tmp_path: Path) -> None:
    request = ServiceIngestRequest(documents=[str(tmp_path / "doc.pdf")], input_type="pdf")
    failed_result = SimpleNamespace(failures=[("doc.pdf", "HTTP 400: invalid request")])
    monkeypatch.setattr(
        "nemo_retriever.ingest.service.build_service_ingestor",
        lambda _request: SimpleNamespace(ingest=lambda **_kwargs: failed_result),
    )

    with pytest.raises(RuntimeError, match=r"failed for 1 document\(s\).+doc.pdf.+HTTP 400"):
        execute_service_ingest_request(request)


def test_execute_service_ingest_request_can_materialize_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request = ServiceIngestRequest(documents=[str(tmp_path / "doc.pdf")], input_type="pdf")
    result = SimpleNamespace(dataframe=[{"row": 1}, {"row": 2}], failures=[])
    captured: dict[str, object] = {}

    def ingest(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        "nemo_retriever.ingest.service.build_service_ingestor",
        lambda _request: SimpleNamespace(ingest=ingest),
    )

    execution = execute_service_ingest_request(request, return_results=True)

    assert captured == {"return_results": True}
    assert execution.n_rows == 2
    assert execution.result_n_rows == 2


def test_execute_service_ingest_request_defaults_to_event_row_counts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request = ServiceIngestRequest(documents=[str(tmp_path / "doc.pdf")], input_type="pdf")
    result = ServiceIngestResult(
        [
            {"event": "document_complete", "document_id": "a", "status": "completed", "result_rows": 2},
            {"event": "document_complete", "document_id": "b", "status": "failed", "result_rows": 99},
            {"event": "document_complete", "document_id": "c", "status": "completed", "result_rows": 3},
        ]
    )
    captured: dict[str, object] = {}

    def ingest(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        "nemo_retriever.ingest.service.build_service_ingestor",
        lambda _request: SimpleNamespace(ingest=ingest),
    )

    execution = execute_service_ingest_request(request)

    assert captured == {"return_results": False}
    assert execution.n_rows == 5
    assert execution.result_n_rows == 5
    assert execution.to_summary_dict()["result_n_rows"] == 5
