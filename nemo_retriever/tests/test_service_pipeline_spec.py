# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 unit tests for the per-request PipelineSpec wire format.

Covers three layers:

* the client-side ``ServiceIngestor`` fluent builders translate to the
  right ``_pipeline_spec`` shape;
* the server-side ``validate_pipeline_spec`` policy accepts well-formed
  specs and rejects trust-sensitive overrides; and
* the worker-side merge preserves server-owned keys regardless of the
  client spec.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nemo_retriever.common.params import DedupParams, EmbedParams, ExtractParams
from nemo_retriever.common.vdb import lancedb_capabilities
from nemo_retriever.service.config import PipelineOverridesConfig
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.service.services.pipeline_executor import (
    _build_graph_ingestor_from_spec,
    _DEFAULT_VECTORDB_WRITE_TIMEOUT_S,
    _merge_server_owned,
    _post_records_to_vectordb,
    _request_needs_asr_params,
    _resolve_extract_params,
    _resolve_service_extraction_mode,
    _run_pipeline_in_process,
    _TRUST_OWNED_EMBED_KEYS,
    _TRUST_OWNED_EXTRACT_KEYS,
)
from nemo_retriever.common.schemas.collections import IngestOperation
from nemo_retriever.service.services.pipeline_pool import DocumentWriteContext
from nemo_retriever.service.utils.file_type import infer_extraction_mode_from_filename
from nemo_retriever.service.service_ingestor import ServiceIngestor
from nemo_retriever.service.client import InMemoryUpload


class _TinyTokenizer:
    def __init__(self) -> None:
        self._text = ""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self._text = text
        return list(range(len(text)))

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(self._text[i] for i in ids)


@pytest.fixture(autouse=True)
def _no_remote_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # _ParamsModel auto-resolves unset *api_key fields from these env vars,
    # which would then trip ServiceIngestor's server-owned-key guard.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)


# ----------------------------------------------------------------------
# Client side: fluent → spec dict
# ----------------------------------------------------------------------


def test_serviceingestor_empty_spec_is_none() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    assert ing._pipeline_payload() is None


def test_compact_result_schema_populates_pipeline_payload() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    payload = ing._pipeline_payload(result_schema="compact")
    assert payload is not None
    assert payload["result_schema"] == "compact"
    assert PipelineSpec.model_validate(payload).result_schema == "compact"


def test_service_inline_text_builds_in_memory_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = ServiceIngestor(base_url="http://retriever.example")
    monkeypatch.setattr("tempfile.mkdtemp", lambda *args, **kwargs: pytest.fail("inline text must remain in memory"))

    ingestor.texts(["first", "first"]).extract(split_config={"text": {"max_tokens": 12}})

    assert ingestor._collect_inputs() == [
        InMemoryUpload(
            filename="inline://00000000",
            content=b"first",
            content_type="text/plain; charset=utf-8",
            classification_filename="inline-00000000.txt",
        ),
        InMemoryUpload(
            filename="inline://00000001",
            content=b"first",
            content_type="text/plain; charset=utf-8",
            classification_filename="inline-00000001.txt",
        ),
    ]
    assert ingestor._pipeline_payload()["extraction_mode"] == "auto"
    assert ingestor._pipeline_payload()["split_config"] == {"text": {"max_tokens": 12}}


def test_service_inline_text_replaces_and_validates_inputs() -> None:
    ingestor = ServiceIngestor(base_url="http://retriever.example").texts("first").texts(["second"])

    assert [item.filename for item in ingestor._collect_inputs()] == ["inline://00000000"]
    assert [item.content for item in ingestor._collect_inputs()] == [b"second"]

    with pytest.raises(TypeError, match=r"texts\[1\] must be a string"):
        ServiceIngestor(base_url="http://retriever.example").texts(["valid", None])


@pytest.mark.parametrize("files_first", [True, False])
def test_service_inline_text_composes_with_files_and_uses_auto_routing(tmp_path, files_first: bool) -> None:
    document = tmp_path / "document.txt"
    document.write_text("document", encoding="utf-8")

    ingestor = ServiceIngestor(base_url="http://retriever.example")
    if files_first:
        ingestor.files(str(document)).texts(["inline"])
    else:
        ingestor.texts(["inline"]).files(str(document))
    ingestor.extract(split_config={"text": {"max_tokens": 12}})

    inputs = ingestor._collect_inputs()
    assert inputs[0] == document
    assert inputs[1] == InMemoryUpload(
        filename="inline://00000000",
        content=b"inline",
        content_type="text/plain; charset=utf-8",
        classification_filename="inline-00000000.txt",
    )
    assert ingestor._pipeline_payload()["extraction_mode"] == "auto"
    assert ingestor._pipeline_payload()["split_config"] == {"text": {"max_tokens": 12}}


@pytest.mark.parametrize("inline_texts", [[], ["", "  \n"]])
def test_service_empty_inline_text_does_not_hide_files(tmp_path, inline_texts: list[str]) -> None:
    document = tmp_path / "document.txt"
    document.write_text("document", encoding="utf-8")

    ingestor = ServiceIngestor(base_url="http://retriever.example").files(str(document)).texts(inline_texts)

    assert ingestor._collect_inputs()[0] == document
    assert ingestor._pipeline_payload() is None


@pytest.mark.parametrize(("inline_texts", "expected_mode"), [([], "pdf"), ([""], "auto")])
def test_service_empty_inline_list_preserves_explicit_extraction_mode(
    tmp_path, inline_texts: list[str], expected_mode: str
) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 stub")

    ingestor = (
        ServiceIngestor(base_url="http://retriever.example")
        .files(str(document))
        .texts(inline_texts)
        .extract(extraction_mode="pdf")
    )

    assert ingestor._pipeline_payload()["extraction_mode"] == expected_mode


@pytest.mark.parametrize(
    ("result_schema", "expected_columns"),
    [
        ("legacy", ["text", "content", "path", "page_number", "metadata"]),
        ("compact", ["text", "source_id", "element_type", "page_number"]),
    ],
)
def test_service_blank_inline_corpus_short_circuits_with_schema(
    result_schema: str,
    expected_columns: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestor = ServiceIngestor(base_url="http://retriever.example").texts(["", "  \n"]).embed()
    monkeypatch.setattr(
        ingestor,
        "ingest_stream",
        lambda **kwargs: pytest.fail("empty inline corpus must not contact the service"),
    )

    result = ingestor.ingest(result_schema=result_schema)

    assert result.job_id is None
    assert result.failures == []
    assert result.dataframe.empty
    assert result.dataframe.columns.tolist() == expected_columns
    assert ingestor._collect_inputs() == []


@pytest.mark.parametrize("input_method", [None, "files", "texts", "buffers"])
def test_service_streaming_ingest_requires_input_sources(input_method: str | None) -> None:
    ingestor = ServiceIngestor(base_url="http://retriever.example")
    if input_method is not None:
        getattr(ingestor, input_method)([])

    with pytest.raises(ValueError, match="No input sources configured"):
        ingestor.ingest_stream()

    async def consume_async_stream() -> list[dict]:
        return [event async for event in ingestor.aingest_stream()]

    with pytest.raises(ValueError, match="No input sources configured"):
        asyncio.run(consume_async_stream())


def test_legacy_pipeline_payload_disables_bulk_result_payloads() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.all_tasks()
    payload = ing._pipeline_payload(result_schema="legacy")
    assert payload is not None
    assert payload["result_schema"] == "legacy"
    assert payload["return_embeddings"] is False
    assert payload["return_images"] is False

    spec = PipelineSpec.model_validate(payload)
    assert spec.return_embeddings is False
    assert spec.return_images is False


def test_legacy_pipeline_payload_accepts_bulk_result_flags() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")

    payload = ing._pipeline_payload(result_schema="legacy", return_embeddings=True, return_images=True)

    assert payload is not None
    assert payload["return_embeddings"] is True
    assert payload["return_images"] is True


def test_execute_time_result_schema_overrides_stored_spec_value() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing._pipeline_spec["result_schema"] = "compact"
    assert ing._pipeline_payload(result_schema="legacy") is None


def test_extract_mode_only_omits_extract_params() -> None:
    """``.extract(extraction_mode='pdf')`` must not send client model defaults."""
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(extraction_mode="pdf").all_tasks()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "pdf"
    assert payload["stage_order"] == ["extract", "dedup", "embed"]
    assert "extract_params" not in payload


def test_extract_records_stage_and_params() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(ExtractParams(extract_text=False, dpi=300))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "auto"
    assert payload["stage_order"] == ["extract"]
    assert payload["extract_params"]["extract_text"] is False
    assert payload["extract_params"]["dpi"] == 300
    assert "page_elements_invoke_url" not in payload["extract_params"]
    assert "api_key" not in payload["extract_params"]
    assert "use_page_elements" not in payload["extract_params"]
    assert "batch_tuning" not in payload["extract_params"]


def test_extract_params_passes_default_policy_allowlist() -> None:
    """Regression: public ExtractParams must not send model defaults to nrl-service."""
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(
        params=ExtractParams(
            extract_text=True,
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        )
    )
    spec = PipelineSpec.model_validate(ing._pipeline_spec)
    validate_pipeline_spec(spec, PipelineOverridesConfig().to_policy())
    assert set(spec.extract_params) <= {
        "extract_text",
        "extract_images",
        "extract_tables",
        "extract_charts",
        "extract_infographics",
        "table_output_format",
    }


def test_extract_image_files_sets_image_mode() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract_image_files()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "image"


def test_dedup_and_embed_add_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract().dedup(DedupParams(iou_threshold=0.7)).embed(EmbedParams(inference_batch_size=64))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]
    assert payload["dedup_params"]["iou_threshold"] == 0.7
    assert payload["embed_params"]["inference_batch_size"] == 64


def test_pdf_split_config_round_trips_via_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.pdf_split_config(pages_per_chunk=16)
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["pdf_split"]["pages_per_chunk"] == 16


def test_split_method_records_split_config() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.split({"pdf": {"max_tokens": 512, "overlap_tokens": 32}})
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["split_config"] == {"pdf": {"max_tokens": 512, "overlap_tokens": 32}}


def test_all_tasks_seeds_canonical_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.all_tasks()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]


def test_client_rejects_server_owned_keys() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.extract(ExtractParams(page_elements_invoke_url="http://attacker/"))


def test_policy_rejects_client_nemotron_parse_model_override() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"method": "nemotron_parse", "nemotron_parse_model": "attacker/model"})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec, policy)


def test_future_phase_methods_raise_informative_error() -> None:
    """Methods deferred to follow-up phases still produce a clear error.

    ``store`` / ``webhook`` / ``vdb_upload`` (sinks) moved out in Phase 2,
    ``save_to_disk`` in Phase 3, ``caption`` in Phase 4, ``vdb_upload``
    sidecar metadata in Phase 6. ``udf`` is the only remaining stub.
    """
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(NotImplementedError, match="Phase 5"):
        ing.udf("noop")


# ----------------------------------------------------------------------
# Policy: accept / reject
# ----------------------------------------------------------------------


def test_validate_returns_none_for_empty_spec() -> None:
    policy = PipelineOverridesConfig().to_policy()
    assert validate_pipeline_spec(None, policy) is None
    assert validate_pipeline_spec(PipelineSpec(), policy) is None


def test_validate_accepts_default_allowlist() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(
        extract_params={"extract_text": False, "dpi": 300},
        embed_params={"inference_batch_size": 64},
        dedup_params={"iou_threshold": 0.5},
        stage_order=["extract", "dedup", "embed"],
    )
    out = validate_pipeline_spec(spec, policy)
    assert out is spec  # returned by reference when unchanged


def test_validate_rejects_endpoint_url() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"page_elements_invoke_url": "http://attacker/"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403
    assert "trust-sensitive" in exc.value.detail


def test_validate_rejects_api_key() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(embed_params={"api_key": "leaked-token"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


def test_validate_rejects_unallowed_key_in_allow_list_mode() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"not_a_real_field": True})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec, policy)


def test_validate_allows_extra_key_when_operator_widens() -> None:
    cfg = PipelineOverridesConfig(extra_extract_keys=["weird_dev_flag"])
    spec = PipelineSpec(extract_params={"weird_dev_flag": True, "dpi": 300})
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_reject_mode_blocks_any_override() -> None:
    cfg = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(extract_params={"dpi": 300})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 403


def test_validate_reject_mode_allows_compact_result_schema_only() -> None:
    cfg = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(result_schema="compact")
    assert validate_pipeline_spec(spec, cfg.to_policy()) is spec


def test_validate_reject_mode_blocks_extraction_mode_piggyback_on_compact_schema() -> None:
    cfg = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(result_schema="compact", extraction_mode="audio")
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 403


def test_validate_allow_all_mode_still_blocks_endpoints() -> None:
    cfg = PipelineOverridesConfig(mode="allow_all")
    policy = cfg.to_policy()
    # "shape" keys pass freely:
    spec = PipelineSpec(extract_params={"any_dev_only_flag": True})
    assert validate_pipeline_spec(spec, policy) is spec
    # but the denylist still bites:
    spec2 = PipelineSpec(extract_params={"ocr_invoke_url": "http://x/"})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec2, policy)


def test_validate_rejects_caption_without_endpoint() -> None:
    """Without an operator-configured caption endpoint, the stage is forbidden."""
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=False)
    spec = PipelineSpec(caption_params={"prompt": "Describe"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


# ----------------------------------------------------------------------
# Worker merge: server-owned keys always win
# ----------------------------------------------------------------------


def test_merge_preserves_server_extract_endpoints() -> None:
    base = {
        "page_elements_invoke_url": "http://server/page_elements",
        "ocr_invoke_url": "http://server/ocr",
        "api_key": "server-token",
        "nemotron_parse_invoke_url": "http://server/parse",
        "nemotron_parse_model": "nvidia/nemotron-parse-v1.2",
        "dpi": 150,
    }
    override = {
        "dpi": 600,
        "page_elements_invoke_url": "http://attacker/",
        "nemotron_parse_model": "attacker/model",
    }
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EXTRACT_KEYS)
    assert merged["dpi"] == 600
    assert merged["page_elements_invoke_url"] == "http://server/page_elements"
    assert merged["ocr_invoke_url"] == "http://server/ocr"
    assert merged["api_key"] == "server-token"
    assert merged["nemotron_parse_model"] == "nvidia/nemotron-parse-v1.2"


@pytest.mark.parametrize("method", ["pdfium", "pdfium_hybrid", "ocr"])
def test_resolve_extract_params_drops_parse_fields_for_other_methods(method: str) -> None:
    base = {
        "method": "nemotron_parse",
        "nemotron_parse_invoke_url": "http://server/parse",
        "nemotron_parse_model": "nvidia/nemotron-parse-v1.2",
        "api_key": "server-token",
    }
    resolved = _resolve_extract_params(base, {"method": method})
    assert resolved.method == method
    assert resolved.nemotron_parse_invoke_url is None
    assert resolved.nemotron_parse_model is None
    assert resolved.api_key == "server-token"


def test_resolve_extract_params_preserves_parse_fields_for_parse_method() -> None:
    base = {
        "method": "nemotron_parse",
        "nemotron_parse_invoke_url": "http://server/parse",
        "nemotron_parse_model": "nvidia/nemotron-parse-v1.2",
    }
    resolved = _resolve_extract_params(base, {"method": "nemotron_parse"})
    assert resolved.method == "nemotron_parse"
    assert resolved.nemotron_parse_invoke_url == "http://server/parse"
    assert resolved.nemotron_parse_model == "nvidia/nemotron-parse-v1.2"


def test_merge_preserves_server_embed_endpoints() -> None:
    base = {"embed_invoke_url": "http://server/embed", "api_key": "k"}
    override = {"embed_invoke_url": "http://attacker/", "inference_batch_size": 8}
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EMBED_KEYS)
    assert merged["embed_invoke_url"] == "http://server/embed"
    assert merged["api_key"] == "k"
    assert merged["inference_batch_size"] == 8


def test_build_graph_ingestor_applies_spec_extraction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``extraction_mode='image'`` calls extract_image_files on the GraphIngestor."""
    base_extract = {"page_elements_invoke_url": "http://server/page_elements"}
    spec = {"extraction_mode": "image", "extract_params": {"dpi": 300}, "stage_order": ["extract"]}

    ingestor, mode, has_vdb = _build_graph_ingestor_from_spec(
        "stub.png",
        b"\x89PNG\r\n",
        base_extract,
        None,
        spec,
    )
    assert mode == "image"
    assert has_vdb is False
    assert ingestor._extraction_mode == "image"
    assert ingestor._extract_params is not None
    assert ingestor._extract_params.dpi == 300
    assert ingestor._extract_params.page_elements_invoke_url == "http://server/page_elements"


# ----------------------------------------------------------------------
# ASR-params gating
# ----------------------------------------------------------------------
#
# Regression coverage for the bug where the worker's ``ASRParams`` (built
# from ``serviceConfig.nimEndpoints.audioGrpcEndpoint``) leaked into every
# per-request ingestor and forced PDF uploads through the audio-only
# graph, crashing inside ``MediaChunkActor`` with
# ``RuntimeError: MediaChunkActor requires media dependencies; missing:
# ffmpeg, ffprobe``.


@pytest.mark.parametrize(
    ("extraction_mode", "filename", "expected"),
    [
        # Explicit audio/video intent: always attach.
        ("audio", "lecture.mp3", True),
        ("audio", "recording.wav", True),
        ("video", "talk.mp4", True),
        ("AUDIO", "recording.WAV", True),
        # auto + media extension: attach so MultiTypeExtractOperator can
        # dispatch the audio rows.
        ("auto", "lecture.mp3", True),
        ("auto", "talk.mp4", True),
        ("auto", "podcast.m4a", True),
        ("auto", "clip.mov", True),
        # auto + non-media extension: DO NOT attach. This is the PDF bug.
        ("auto", "report.pdf", False),
        ("auto", "scan.docx", False),
        ("auto", "spec.pptx", False),
        ("auto", "diagram.png", False),
        ("auto", "page.html", False),
        ("auto", "notes.txt", False),
        # Explicit non-media modes: never attach regardless of filename.
        ("pdf", "report.pdf", False),
        ("pdf", "weird.mp3", False),
        ("image", "diagram.png", False),
        ("text", "notes.txt", False),
        ("html", "page.html", False),
        # Unknown extension under auto: be conservative, don't attach.
        ("auto", "unknown.xyz", False),
        ("auto", "no_extension", False),
        # Missing/empty mode: same as unknown — don't attach.
        ("", "report.pdf", False),
        (None, "report.pdf", False),
    ],
)
def test_request_needs_asr_params(extraction_mode: str | None, filename: str, expected: bool) -> None:
    assert _request_needs_asr_params(extraction_mode, filename) is expected


def test_build_graph_ingestor_does_not_attach_asr_params_for_pdf_upload() -> None:
    """Regression: a worker with ``base_asr`` configured must not pin the
    cluster-wide ASR params onto PDF ingest requests.

    Before the fix the worker unconditionally executed
    ``ingestor._asr_params = asr_params`` whenever ``base_asr`` was
    truthy, which forced :func:`build_graph` into the audio-only branch
    and crashed inside :class:`MediaChunkActor` when ffmpeg was absent.
    """
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "report.pdf",
        b"%PDF-1.4 stub",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert mode == "pdf"
    assert (
        ingestor._asr_params is None
    ), f"PDF ingestion must not carry worker-wide ASR params. Got: {ingestor._asr_params!r}"


def test_build_graph_ingestor_attaches_asr_params_for_audio_upload() -> None:
    """A genuine audio upload under ``extraction_mode='auto'`` must still
    carry the ASR params so MultiTypeExtractOperator can dispatch ASR.
    """
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "lecture.mp3",
        b"ID3\x03",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert ingestor._asr_params is not None
    assert tuple(ingestor._asr_params.audio_endpoints) == ("audio:50051", None)


def test_build_graph_ingestor_preserves_canonical_video_defaults() -> None:
    """Auto-routed MP4 uploads must build the full video extraction branch."""
    base_extract = {"ocr_invoke_url": "https://server.example/v1/ocr"}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "talk.mp4",
        b"video bytes",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert mode == "video"
    assert ingestor._extraction_mode == "video"
    assert ingestor._extract_params.ocr_invoke_url == "https://server.example/v1/ocr"
    assert ingestor._audio_chunk_params.enabled is True
    assert ingestor._audio_chunk_params.split_type == "size"
    assert ingestor._audio_chunk_params.split_interval == 500000
    assert tuple(ingestor._asr_params.audio_endpoints) == ("audio:50051", None)
    assert ingestor._video_frame_params.enabled is True
    assert ingestor._video_frame_params.fps == 0.5
    assert ingestor._video_frame_params.dedup is True
    assert ingestor._video_text_dedup_params.enabled is True
    assert ingestor._video_text_dedup_params.max_dropped_frames == 2
    assert ingestor._av_fuse_params.enabled is True


def test_build_graph_ingestor_keeps_video_frames_when_asr_is_unconfigured() -> None:
    """An unconfigured ASR endpoint disables audio, not frame OCR."""
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "silent.mp4",
        b"video bytes",
        {"ocr_invoke_url": "https://server.example/v1/ocr"},
        None,
        spec,
        base_asr=None,
    )

    assert mode == "video"
    assert ingestor._audio_chunk_params.enabled is False
    assert ingestor._asr_params is None
    assert ingestor._video_frame_params.enabled is True
    assert ingestor._video_text_dedup_params.enabled is True
    assert ingestor._av_fuse_params.enabled is True


def test_build_graph_ingestor_attaches_asr_params_for_explicit_audio_mode() -> None:
    """``extraction_mode='audio'`` must always attach the worker ASR params."""
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "audio", "stage_order": ["extract"]}

    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        # Filename without a media extension — explicit mode wins.
        "stream.bin",
        b"binary",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert mode == "audio"
    assert ingestor._asr_params is not None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.txt", "text"),
        ("README.md", "text"),
        ("payload.json", "text"),
        ("setup.sh", "text"),
        ("inline://00000000", "text"),
        ("page.html", "html"),
        ("report.pdf", "pdf"),
        ("diagram.png", "image"),
        ("clip.mp4", "video"),
        ("unknown.xyz", None),
    ],
)
def test_infer_extraction_mode_from_filename(filename: str, expected: str | None) -> None:
    assert infer_extraction_mode_from_filename(filename) == expected


@pytest.mark.parametrize(
    ("extraction_mode", "filename", "resolved"),
    [
        ("auto", "notes.txt", "text"),
        ("auto", "inline://00000000", "text"),
        ("auto", "page.html", "html"),
        ("auto", "report.pdf", "pdf"),
        ("pdf", "notes.txt", "pdf"),
        ("text", "page.html", "text"),
    ],
)
def test_resolve_service_extraction_mode(extraction_mode: str, filename: str, resolved: str) -> None:
    assert _resolve_service_extraction_mode(extraction_mode, filename) == resolved


def test_build_graph_ingestor_routes_txt_and_html_inputs() -> None:
    base_extract: dict[str, object] = {}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    txt_ingestor, txt_mode, _ = _build_graph_ingestor_from_spec(
        "README.md",
        b"# The quick brown fox",
        base_extract,
        None,
        spec,
    )
    assert txt_mode == "text"
    assert txt_ingestor._extraction_mode == "text"

    inline_ingestor, inline_mode, _ = _build_graph_ingestor_from_spec(
        "inline://00000000",
        b"The quick brown fox",
        base_extract,
        None,
        None,
    )
    assert inline_mode == "text"
    assert inline_ingestor._extraction_mode == "text"

    html_ingestor, html_mode, _ = _build_graph_ingestor_from_spec(
        "page.html",
        b"<html><body><h1>Hi</h1></body></html>",
        base_extract,
        None,
        spec,
    )
    assert html_mode == "html"
    assert html_ingestor._extraction_mode == "html"
    assert html_ingestor._html_params is not None


def test_run_pipeline_in_process_rejects_empty_text_like_output() -> None:
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    with pytest.raises(ValueError, match="Extraction produced no rows"):
        _run_pipeline_in_process("empty.txt", b"", {}, None, None, spec)


def test_run_pipeline_in_process_html_txt_produce_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    monkeypatch.setattr(
        "nemo_retriever.common.modality.html.convert._get_txt_tokenizer", lambda *_, **__: _TinyTokenizer()
    )
    monkeypatch.setattr("nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *_, **__: _TinyTokenizer())
    html_rows, _, _ = _run_pipeline_in_process(
        "page.html",
        b"<html><body><h1>Title</h1><p>body</p></body></html>",
        {},
        None,
        None,
        spec,
    )
    txt_rows, _, _ = _run_pipeline_in_process(
        "notes.txt",
        b"Line one\nLine two\n",
        {},
        None,
        None,
        spec,
    )
    assert html_rows >= 1
    assert txt_rows >= 1


def test_run_pipeline_in_process_preserves_service_inline_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *_, **__: _TinyTokenizer())

    row_count, rows, _ = _run_pipeline_in_process(
        "inline://00000003",
        "café service".encode("utf-8"),
        {},
        None,
        None,
        None,
    )

    assert row_count == 1
    assert rows[0]["text"] == "café service"
    assert rows[0]["path"] == "inline://00000003"
    assert rows[0]["metadata"]["source_path"] == "inline://00000003"


def test_run_pipeline_in_process_uses_effective_embedding_output_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    class _Ingestor:
        _embed_params = EmbedParams(model_name="embed-model", output_column="custom_embeddings")

        def ingest(self):
            return pd.DataFrame(
                {
                    "text": ["hello"],
                    "earlier_payload": [{"embedding": [9.0]}],
                    "custom_embeddings": [{"embedding": [0.1]}],
                }
            )

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor._build_graph_ingestor_from_spec",
        lambda *_args, **_kwargs: (_Ingestor(), "pdf", False),
    )

    row_count, rows, _ = _run_pipeline_in_process(
        "document.pdf",
        b"%PDF-1.4 stub",
        {},
        None,
        pipeline_spec={"result_schema": "compact", "return_embeddings": True},
    )

    assert row_count == 1
    assert rows[0]["embedding"] == [0.1]


def test_build_graph_ingestor_omits_asr_params_when_worker_unconfigured() -> None:
    """When the worker has no ASR endpoint, nothing should be attached
    regardless of filename or extraction mode.
    """
    base_extract: dict[str, object] = {}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "lecture.mp3",
        b"ID3\x03",
        base_extract,
        None,
        spec,
        base_asr=None,
    )

    assert ingestor._asr_params is None


def test_run_pipeline_posts_canonical_pdf_table_image_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: dict[str, object] = {}
    graph_rows = [
        {
            "text": "table content",
            "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
            "path": "/documents/report.pdf",
            "page_number": 1,
            "_page_number": 7,
            "_content_type": "table_caption",
            "_stored_image_uri": "s3://artifacts/table.png",
            "_bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            "metadata": {"content_metadata": {"page_number": 1}},
        }
    ]

    class _Ingestor:
        def ingest(self):
            return graph_rows

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        posted["url"] = request.full_url
        posted["timeout"] = timeout
        posted["json"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor._build_graph_ingestor_from_spec",
        lambda *_args, **_kwargs: (_Ingestor(), "pdf", False),
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor._sanitize_result_data",
        lambda _result, **_kwargs: [],
    )
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    row_count, _, _ = _run_pipeline_in_process(
        "report.pdf",
        b"%PDF-1.4 stub",
        {},
        None,
        vectordb_url="http://vectordb:7671",
        write_context=DocumentWriteContext(
            scope="tenant-a",
            collection_name="papers",
            storage_document_id="document-1",
            content_sha256="a" * 64,
            document_version="version-2",
            document_metadata={
                "category": "Finance_Investment",
                "source_path": "Finance_Investment/report.pdf",
                "source_filename": "report.pdf",
                "page_number": 999,
            },
        ),
        job_id="job-1",
    )

    assert row_count == 1
    assert posted["url"] == "http://vectordb:7671/internal/vectordb/write"
    payload = posted["json"]
    assert isinstance(payload, dict)
    record = payload["records"][0][0]
    assert record["document_type"] == "text"
    metadata = record["metadata"]
    assert metadata["embedding"] == [0.1, 0.2]
    assert metadata["content"] == "table content"
    assert metadata["source_metadata"] == {
        "source_id": "/documents/report.pdf",
        "source_name": "report.pdf",
    }
    assert metadata["content_metadata"] == {
        "page_number": 7,
        "type": "table",
        "fidelity": "ocr",
        "stored_image_uri": "s3://artifacts/table.png",
        "uploaded_image_uri": "s3://artifacts/table.png",
        "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
        "category": "Finance_Investment",
        "source_path": "Finance_Investment/report.pdf",
        "source_filename": "report.pdf",
    }
    assert payload["scope"] == "tenant-a"
    assert payload["collection_name"] == "papers"
    assert payload["document_id"] == "document-1"
    assert "rows" not in payload


def test_post_records_to_vectordb_uses_canonical_internal_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        posted["url"] = request.full_url
        posted["headers"] = dict(request.headers)
        posted["timeout"] = timeout
        posted["json"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "canonical chunk",
                    "content_metadata": {"page_number": 7},
                },
            }
        ]
    ]

    _post_records_to_vectordb(
        records,
        "http://vectordb:7671/",
        "document.pdf",
        context=DocumentWriteContext(
            scope="tenant-a",
            collection_name="papers",
            storage_document_id="document-1",
            content_sha256="a" * 64,
            document_version="version-2",
            operation=IngestOperation.REPLACE,
        ),
        job_id="job-1",
        internal_api_token="internal-token",
    )

    assert posted["url"] == "http://vectordb:7671/internal/vectordb/write"
    assert posted["timeout"] == _DEFAULT_VECTORDB_WRITE_TIMEOUT_S
    assert posted["headers"]["X-nrl-internal-token"] == "internal-token"
    assert posted["json"] == {
        "records": records,
        "scope": "tenant-a",
        "collection_name": "papers",
        "document_id": "document-1",
        "job_id": "job-1",
        "filename": "document.pdf",
        "content_sha256": "a" * 64,
        "document_version": "version-2",
        "operation": "replace",
    }
    assert "rows" not in posted["json"]
    assert "artifact_prefix" not in posted["json"]


def test_post_records_to_vectordb_fails_the_document_when_the_write_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write the storage tier never acknowledged must not be reported as ingested."""

    def _urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    records = [[{"document_type": "text", "metadata": {"embedding": [0.1], "content": "chunk"}}]]

    # A legacy fixed-table write, i.e. no collection context.
    with pytest.raises(RuntimeError, match="VectorDB write failed for gamma.txt"):
        _post_records_to_vectordb(
            records,
            "http://vectordb:7671",
            "gamma.txt",
            context=DocumentWriteContext(),
        )


def test_post_records_to_vectordb_honors_the_configured_write_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(request, timeout):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    _post_records_to_vectordb(
        [[{"document_type": "text", "metadata": {"embedding": [0.1], "content": "chunk"}}]],
        "http://vectordb:7671",
        "gamma.txt",
        context=DocumentWriteContext(),
        timeout_s=45.0,
    )

    assert seen["timeout"] == 45.0


def test_default_write_timeout_outlasts_the_index_readiness_waits() -> None:
    """A durable write must not fail because the VectorDB waited on its own indexes.

    The VectorDB acknowledges a write only after index maintenance, and a hybrid
    table waits once per index. If those advisory waits can outlast the worker's
    timeout, a document whose rows are already committed fails on the wait alone.
    """
    worst_case_index_wait_s = (
        lancedb_capabilities.MAX_INDEX_READY_WAITS_PER_WRITE * lancedb_capabilities.INDEX_READY_TIMEOUT.total_seconds()
    )

    assert worst_case_index_wait_s < _DEFAULT_VECTORDB_WRITE_TIMEOUT_S


def test_service_worker_installs_url_source_map_before_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    transport = "url-source://00000000/url-00000000.pdf"
    source_url = "https://example.test/report.pdf"
    observed = {}

    class _Ingestor:
        _embed_params = None

        def _set_source_map(self, source_map):
            observed["source_map"] = source_map

        def ingest(self):
            assert observed["source_map"] == {transport: source_url}
            return pd.DataFrame([{"path": f"{transport}_1", "metadata": {"source_path": f"{transport}_1"}}])

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor._build_graph_ingestor_from_spec",
        lambda *_args, **_kwargs: (_Ingestor(), "pdf", False),
    )

    row_count, rows, _ = _run_pipeline_in_process(
        transport,
        b"%PDF-1.4 stub",
        {},
        None,
        write_context=DocumentWriteContext(document_metadata={"_nrl_source_url": source_url}),
    )

    assert row_count == 1
    assert rows[0]["path"] == f"{source_url}_1"
    assert rows[0]["metadata"]["source_path"] == f"{source_url}_1"
