from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image
from ray.data.block import BlockAccessor
from ray.data.extensions import ArrowPythonObjectArray

import nemo_retriever
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.ingestor import IngestorCreateParams, _merge_params, create_ingestor
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    IngestExecuteParams,
    NO_API_KEY,
    RemoteRetryParams,
    TextChunkParams,
    VdbUploadParams,
)
from nemo_retriever.operators.vdb import IngestVdbOperator


class _InlineTextTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


def _graph_node_names(graph) -> list[str]:
    names: list[str] = []

    def visit(node) -> None:
        names.append(node.name)
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return names


def _effective_graph_node_names(ingestor: GraphIngestor) -> list[str]:
    effective = ingestor._resolve_effective_extraction_inputs()
    graph = build_graph(
        extraction_mode=effective.extraction_mode,
        extract_params=effective.extract_params,
        text_params=effective.text_params,
        html_params=effective.html_params,
        audio_chunk_params=effective.audio_chunk_params,
        asr_params=effective.asr_params,
        video_frame_params=effective.video_frame_params,
        video_text_dedup_params=effective.video_text_dedup_params,
        av_fuse_params=effective.av_fuse_params,
        split_config=ingestor._split_config,
    )
    return _graph_node_names(graph)


def _run_graph_ingest_with_result(ingestor: GraphIngestor, result, monkeypatch, **ingest_kwargs):
    if not ingestor._documents and not ingestor._buffers and not ingestor._inline_texts:
        ingestor.files(["document.pdf"])
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="pdf"),
    )

    def _execute_single_graph(effective_extraction, *, dedup_params, post_extract_order):
        assert effective_extraction.extraction_mode == "pdf"
        assert dedup_params is ingestor._dedup_params
        assert isinstance(post_extract_order, tuple)
        return result

    monkeypatch.setattr(ingestor, "_execute_single_graph", _execute_single_graph)
    return ingestor.ingest(**ingest_kwargs)


def test_merge_params_none_returns_kwargs() -> None:
    merged = _merge_params(None, {"documents": ["a.pdf"]})
    assert merged == {"documents": ["a.pdf"]}


def test_merge_params_with_model_copy_updates_values() -> None:
    params = IngestorCreateParams(documents=["before.pdf"], ray_log_to_driver=True)
    merged = _merge_params(params, {"documents": ["after.pdf"], "ray_log_to_driver": False})
    assert isinstance(merged, IngestorCreateParams)
    assert merged.documents == ["after.pdf"]
    assert merged.ray_log_to_driver is False


def test_create_ingestor_parses_kwargs_and_returns_graph_ingestor() -> None:
    ingestor = create_ingestor(run_mode="inprocess", documents=["doc.pdf"], base_url="http://example:7670")
    assert isinstance(ingestor, GraphIngestor)
    assert ingestor._run_mode == "inprocess"
    assert ingestor._documents == ["doc.pdf"]


def test_create_ingestor_passes_error_policy_to_graph_ingestor() -> None:
    ingestor = create_ingestor(run_mode="inprocess", error_policy="collect")
    assert isinstance(ingestor, GraphIngestor)
    assert ingestor._error_policy == "collect"


def test_graph_ingestion_error_is_exported_from_top_level_package() -> None:
    assert "GraphIngestionError" in nemo_retriever.__all__
    assert nemo_retriever.GraphIngestionError is GraphIngestionError


def test_create_ingestor_rejects_unknown_kwargs() -> None:
    with pytest.raises(Exception):
        create_ingestor(run_mode="inprocess", unknown_field=True)


def test_create_ingestor_rejects_unknown_run_modes() -> None:
    with pytest.raises(ValueError, match="supports run modes"):
        create_ingestor(run_mode="parallel")  # type: ignore[arg-type]


@pytest.mark.parametrize("run_mode", ["inprocess", "batch", "service"])
@pytest.mark.parametrize("input_method", [None, "files", "texts", "buffers"])
def test_ingest_requires_input_sources(
    run_mode: str,
    input_method: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_kwargs = {"run_mode": run_mode}
    if run_mode == "service":
        create_kwargs["base_url"] = "http://retriever.example"
    ingestor = create_ingestor(**create_kwargs)
    if input_method is not None:
        getattr(ingestor, input_method)([])

    if run_mode == "batch":
        monkeypatch.setattr(
            ingestor,
            "_ensure_batch_runtime",
            lambda: pytest.fail("input validation must run before starting Ray"),
        )
    elif run_mode == "service":
        monkeypatch.setattr(
            ingestor,
            "ingest_stream",
            lambda **kwargs: pytest.fail("input validation must run before contacting the service"),
        )
    else:
        monkeypatch.setattr(
            ingestor,
            "_execute_single_graph",
            lambda *args, **kwargs: pytest.fail("input validation must run before executing the graph"),
        )

    with pytest.raises(
        ValueError,
        match=r"No input sources configured\. Call files\(\), texts\(\), or buffers\(\) with at least one source",
    ):
        ingestor.extract(params=ExtractParams(extract_text=True)).ingest()


def test_texts_accepts_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )

    result = create_ingestor(run_mode="inprocess").texts("first").ingest()

    assert result["text"].tolist() == ["first"]
    assert result["path"].tolist() == ["inline://00000000"]


def test_texts_replaces_prior_inline_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )

    result = create_ingestor(run_mode="inprocess").texts("first").texts(["second", "third"]).ingest()

    assert result["text"].tolist() == ["second", "third"]
    assert result["path"].tolist() == ["inline://00000000", "inline://00000001"]


@pytest.mark.parametrize("values", [["valid", None], ["valid", 3], [object()]])
def test_texts_rejects_non_string_values_with_index(values) -> None:
    bad_index = next(index for index, value in enumerate(values) if not isinstance(value, str))

    with pytest.raises(TypeError, match=rf"texts\[{bad_index}\] must be a string"):
        create_ingestor(run_mode="inprocess").texts(values)


def test_texts_rejects_non_sequence_input() -> None:
    with pytest.raises(TypeError, match="string or sequence of strings"):
        create_ingestor(run_mode="inprocess").texts(iter(["one", "two"]))


@pytest.mark.parametrize("files_first", [True, False])
def test_texts_can_mix_with_text_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files_first: bool,
) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )
    document = tmp_path / "document.txt"
    document.write_text("document", encoding="utf-8")

    ingestor = create_ingestor(run_mode="inprocess")
    if files_first:
        ingestor.files([str(document)]).texts(["inline"])
    else:
        ingestor.texts(["inline"]).files([str(document)])

    result = ingestor.extract(split_config={"text": {"max_tokens": 10}}).ingest()

    assert result["text"].tolist() == ["document", "inline"]
    assert result["path"].tolist() == [str(document.resolve()), "inline://00000000"]


def test_texts_can_mix_with_text_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )

    result = (
        create_ingestor(run_mode="inprocess")
        .texts(["inline"])
        .buffers(("document.txt", BytesIO(b"document")))
        .extract(split_config={"text": {"max_tokens": 10}})
        .ingest()
    )

    assert result["text"].tolist() == ["document", "inline"]
    assert result["path"].tolist() == [
        str(Path("document.txt").resolve()),
        "inline://00000000",
    ]


def test_texts_allow_explicit_text_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )

    result = (
        create_ingestor(run_mode="inprocess")
        .texts(["one two"])
        .extract(extraction_mode="text", text_params=TextChunkParams(max_tokens=1))
        .ingest()
    )

    assert result["text"].tolist() == ["one", "two"]


def test_texts_plan_alongside_other_modalities_regardless_of_call_order(tmp_path: Path) -> None:
    image = tmp_path / "scan.bmp"
    image.write_bytes(b"bmp")

    images_first = GraphIngestor(run_mode="inprocess").files([str(image)]).extract_image_files().texts(["inline"])
    texts_first = GraphIngestor(run_mode="inprocess").texts(["inline"]).files([str(image)]).extract_image_files()

    for ingestor in (images_first, texts_first):
        branches = ingestor._plan_default_extraction_branches()
        assert [(branch.family, branch.input_paths) for branch in branches] == [
            ("image", (str(image),)),
            ("txt", ("inline://00000000",)),
        ]

    empty_inline = GraphIngestor(run_mode="inprocess").files([str(image)]).texts([])
    assert [(branch.family, branch.input_paths) for branch in empty_inline._plan_default_extraction_branches()] == [
        ("image", (str(image),)),
    ]

    explicit_image = GraphIngestor(run_mode="inprocess").files([str(image)]).texts([]).extract_image_files()
    assert explicit_image._plan_default_extraction_branches() is None


def test_empty_and_blank_inline_corpus_short_circuits_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = create_ingestor(run_mode="inprocess").texts(["", "  \n"])
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.build_graph",
        lambda *args, **kwargs: pytest.fail("empty inline corpus should not execute the graph"),
    )

    result = ingestor.embed().vdb_upload().ingest()

    assert result.empty
    assert list(result.columns) == ["text", "content", "path", "page_number", "metadata"]


@pytest.mark.parametrize("inline_texts", [[], ["", "  \n"]])
def test_empty_inline_text_does_not_short_circuit_file_ingestion(
    inline_texts: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer",
        lambda *args, **kwargs: _InlineTextTokenizer(),
    )
    document = tmp_path / "document.txt"
    document.write_text("document", encoding="utf-8")

    result = create_ingestor(run_mode="inprocess").files([str(document)]).texts(inline_texts).ingest()

    assert result["text"].tolist() == ["document"]
    assert result["path"].tolist() == [str(document.resolve())]


def test_empty_batch_inline_corpus_returns_dataframe_without_starting_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = create_ingestor(run_mode="batch").texts(["", "  \n"])
    monkeypatch.setattr(
        ingestor,
        "_ensure_batch_runtime",
        lambda: pytest.fail("empty inline corpus should not start Ray"),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.build_graph",
        lambda *args, **kwargs: pytest.fail("empty inline corpus should not execute the graph"),
    )

    result = ingestor.ingest()

    assert isinstance(result, pd.DataFrame)
    assert ingestor.get_dataset() is result
    assert result.empty
    assert list(result.columns) == ["text", "content", "path", "page_number", "metadata"]


def test_graph_ingestor_action_methods_materialize_default_params() -> None:
    ingestor = GraphIngestor(run_mode="inprocess")

    ingestor.extract()
    assert isinstance(ingestor._extract_params, ExtractParams)

    ingestor.extract_image_files()
    assert isinstance(ingestor._extract_params, ExtractParams)

    ingestor.extract(split_config={"text": {"max_tokens": 512}})
    assert isinstance(ingestor._split_config["text"], TextChunkParams)

    ingestor.extract_html()
    assert isinstance(ingestor._html_params, HtmlChunkParams)

    ingestor.extract_audio()
    assert isinstance(ingestor._audio_chunk_params, AudioChunkParams)
    assert isinstance(ingestor._asr_params, ASRParams)

    ingestor.dedup()
    assert isinstance(ingestor._dedup_params, DedupParams)

    ingestor.caption()
    assert isinstance(ingestor._caption_params, CaptionParams)

    ingestor.embed()
    assert isinstance(ingestor._embed_params, EmbedParams)


def test_caption_auto_dedup_does_not_mutate_repeated_ingest_state(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess").files(["document.pdf"]).caption()
    configured_stage_order = list(ingestor._stage_order)
    calls: list[tuple[DedupParams | None, tuple[str, ...]]] = []
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="pdf"),
    )

    def _execute_single_graph(effective_extraction, *, dedup_params, post_extract_order):
        assert effective_extraction.extraction_mode == "pdf"
        calls.append((dedup_params, post_extract_order))
        return pd.DataFrame()

    monkeypatch.setattr(ingestor, "_execute_single_graph", _execute_single_graph)

    ingestor.ingest()
    ingestor.ingest()

    assert ingestor._dedup_params is None
    assert ingestor._stage_order == configured_stage_order == ["caption"]
    assert [stage_order for _, stage_order in calls] == [("dedup", "caption"), ("dedup", "caption")]
    for dedup_params, _ in calls:
        assert dedup_params is not None
        assert dedup_params.content_hash is True
        assert dedup_params.bbox_iou is True
        assert dedup_params.iou_threshold == 0.45


def test_image_only_caption_does_not_enable_or_persist_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess").files(["image.png"]).extract_image_files().caption()
    configured_stage_order = list(ingestor._stage_order)
    calls: list[tuple[DedupParams | None, tuple[str, ...]]] = []
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="image"),
    )

    def _execute_single_graph(effective_extraction, *, dedup_params, post_extract_order):
        assert effective_extraction.extraction_mode == "image"
        calls.append((dedup_params, post_extract_order))
        return pd.DataFrame()

    monkeypatch.setattr(ingestor, "_execute_single_graph", _execute_single_graph)

    ingestor.ingest()

    assert calls == [(None, ("caption",))]
    assert ingestor._dedup_params is None
    assert ingestor._stage_order == configured_stage_order == ["extract", "caption"]


def test_explicit_disabled_dedup_preserves_sdk_state(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_dedup = DedupParams(content_hash=False, bbox_iou=False)
    ingestor = GraphIngestor(run_mode="inprocess").files(["document.pdf"]).extract().dedup(configured_dedup).caption()
    configured_stage_order = list(ingestor._stage_order)
    calls: list[tuple[DedupParams | None, tuple[str, ...]]] = []
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="pdf"),
    )

    def _execute_single_graph(effective_extraction, *, dedup_params, post_extract_order):
        assert effective_extraction.extraction_mode == "pdf"
        calls.append((dedup_params, post_extract_order))
        return pd.DataFrame()

    monkeypatch.setattr(ingestor, "_execute_single_graph", _execute_single_graph)

    ingestor.ingest()

    assert calls == [(configured_dedup, ("dedup", "caption"))]
    assert ingestor._dedup_params is configured_dedup
    assert ingestor._stage_order == configured_stage_order == ["extract", "dedup", "caption"]


def test_extract_unified_defaults() -> None:
    """`.extract()` defaults: infer extraction_mode at graph-build time and no chunking unless opted in."""
    ingestor = GraphIngestor(run_mode="inprocess").extract()
    assert ingestor._extraction_mode is None
    assert all(ingestor._split_config[k] is None for k in ("text", "html", "pdf", "audio", "image", "video"))


def test_extract_rejects_unknown_kwargs() -> None:
    """`.extract()` must fail loudly on kwargs that are not fields of ExtractParams.

    The audio-video.md hosted-Parakeet snippet historically passed
    ``extract_method``/``extract_audio_params`` (and ``document_type``) into
    ``.extract()``; those silently flowed through ``_coerce()`` /
    ``params.model_copy(update=...)``, bypassing ``ExtractParams``'s
    ``extra="forbid"`` config. Audio credentials never reached the ASR actor,
    which then fell back to the local HF model with no signal to the user.
    Regression-guard the strict validation that fixes that silent drop.
    """
    ingestor = GraphIngestor(run_mode="inprocess")

    with pytest.raises(TypeError, match="garbage_kwarg"):
        ingestor.extract(garbage_kwarg="x")

    with pytest.raises(TypeError) as exc_info:
        ingestor.extract(
            document_type="mp3",
            extract_method="audio",
            extract_audio_params={
                "grpc_endpoint": "grpc.nvcf.nvidia.com:443",
                "auth_token": "fake-key",
                "function_id": "fake-function-id",
                "segment_audio": True,
            },
        )
    message = str(exc_info.value)
    # Pin the rejected-keys list as a single repr so this test fails loudly if
    # any of these keys ever become real ExtractParams fields.
    expected_rejected = repr(sorted(["document_type", "extract_method", "extract_audio_params"]))
    assert expected_rejected in message
    assert "asr_params" in message


@pytest.mark.parametrize("run_mode", ["inprocess", "batch"])
def test_extract_validates_overrides_to_existing_params(run_mode: str) -> None:
    params = ExtractParams()

    with pytest.raises(ValueError, match="pdfium_hybrid"):
        GraphIngestor(run_mode=run_mode).extract(params, method="unsupported")


@pytest.mark.parametrize("run_mode", ["inprocess", "batch"])
@pytest.mark.parametrize("overrides", [{}, {"dpi": 300}])
def test_extract_preserves_explicit_api_key_suppression(
    monkeypatch: pytest.MonkeyPatch, run_mode: str, overrides: dict[str, int]
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-review-secret")
    params = ExtractParams(api_key=NO_API_KEY)

    resolved = GraphIngestor(run_mode=run_mode).extract(params, **overrides)._extract_params

    assert resolved.api_key is None
    assert resolved._uses_no_api_key("api_key")


def test_extract_default_pdf_only_builds_dedicated_pdf_graph(tmp_path) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    ingestor = GraphIngestor(run_mode="inprocess").files([str(document)]).extract()

    node_names = _effective_graph_node_names(ingestor)
    assert "MultiTypeExtractOperator" not in node_names
    assert node_names[:4] == [
        "DocToPdfConversionActor",
        "PDFSplitActor",
        "PDFExtractionActor",
        "PageElementDetectionActor",
    ]


@pytest.mark.parametrize(("suffix", "image_format"), [(".bmp", "BMP"), (".tiff", "TIFF"), (".tif", "TIFF")])
def test_extract_default_direct_images_materialize_page_image(monkeypatch, tmp_path, suffix, image_format) -> None:
    image_path = tmp_path / f"page{suffix}"
    Image.new("RGB", (12, 8), color=(40, 90, 130)).save(image_path, format=image_format)

    def passthrough_detection(self, batch_df):
        return batch_df

    def fail_pdf_split(self, batch_df):
        raise AssertionError("direct image extraction routed through PDFSplitActor")

    monkeypatch.setattr(
        "nemo_retriever.operators.graph_ops.multi_type_extract_operator._MultiTypeExtractBase._run_detection_pipeline",
        passthrough_detection,
    )
    monkeypatch.setattr("nemo_retriever.operators.extract.pdf.split.PDFSplitActor.run", fail_pdf_split)

    result = (
        create_ingestor(run_mode="inprocess")
        .files([str(image_path)])
        .extract(
            params=ExtractParams(
                extract_text=True,
                extract_images=True,
                extract_tables=False,
                extract_charts=False,
                extract_infographics=False,
            )
        )
        .ingest()
    )

    assert len(result) == 1
    page_image = result.iloc[0]["page_image"]
    assert isinstance(page_image, dict)
    assert page_image["image_b64"]
    assert result.iloc[0]["metadata"]["source_path"] == str(image_path.resolve())


def test_extract_default_mixed_pdf_and_image_plans_ordered_branches(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.bmp"
    pdf.write_bytes(b"%PDF-1.4\n")
    image.write_bytes(b"bmp")

    ingestor = GraphIngestor(run_mode="inprocess").files([str(image), str(pdf)]).extract()

    branches = ingestor._plan_default_extraction_branches()
    assert [(branch.family, branch.extraction_mode, branch.input_paths) for branch in branches] == [
        ("pdf", "pdf", (str(pdf),)),
        ("image", "image", (str(image),)),
    ]


def test_extract_explicit_pdf_rejects_image_input(tmp_path) -> None:
    image = tmp_path / "scan.bmp"
    image.write_bytes(b"bmp")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(image)]).extract(extraction_mode="pdf")

    with pytest.raises(ValueError, match="extraction_mode='pdf'"):
        ingestor.ingest()


def test_extract_default_rejects_unknown_input_type(tmp_path) -> None:
    document = tmp_path / "payload.bin"
    document.write_bytes(b"unknown")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(document)]).extract()

    with pytest.raises(ValueError, match="Unsupported input file type"):
        ingestor.ingest()


def test_extract_audio_does_not_enable_post_extraction_chunking_by_default() -> None:
    audio_ingestor = GraphIngestor(run_mode="inprocess").extract_audio()
    assert audio_ingestor._split_config["audio"] is None


def test_extract_split_configures_automatically_routed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _InlineTextTokenizer()
    )

    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .texts(["one two three"])
        .extract(split_config={"text": {"max_tokens": 2}})
        .ingest()
    )

    assert result["text"].tolist() == ["one two", "three"]


def test_graph_ingestor_return_failures_returns_service_tuples_from_path(monkeypatch) -> None:
    result = pd.DataFrame(
        {
            "path": ["doc-a.pdf"],
            "page_elements_v3": [
                {
                    "timing": None,
                    "error": {
                        "stage": "remote_inference",
                        "type": "ConnectionError",
                        "message": "connection refused",
                    },
                }
            ],
            "metadata": [
                {
                    "error": {
                        "stage": "local_postprocess",
                        "message": "ignored because diagnostics narrow the scan",
                    }
                }
            ],
        }
    )
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    returned, failures = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        return_failures=True,
    )

    assert returned is result
    assert failures == [
        (
            "doc-a.pdf",
            "row 0, column page_elements_v3, path error: " "remote_inference: ConnectionError: connection refused",
        )
    ]


def test_graph_ingestor_return_failures_params_model_uses_metadata_source_path(monkeypatch) -> None:
    result = pd.DataFrame(
        {
            "metadata": [
                {
                    "source_path": "nested-doc.pdf",
                    "error": {
                        "stage": "local_postprocess",
                        "message": "boom",
                    },
                }
            ],
        }
    )
    ingestor = GraphIngestor(run_mode="inprocess")

    returned, failures = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        params=IngestExecuteParams(return_failures=True),
    )

    assert returned is result
    assert failures == [("nested-doc.pdf", "row 0, column metadata, path error: local_postprocess: boom")]


def test_graph_ingestor_return_failures_kwargs_override_params(monkeypatch) -> None:
    result = pd.DataFrame({"text": ["ok"]})
    ingestor = GraphIngestor(run_mode="inprocess")

    returned = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        params=IngestExecuteParams(return_failures=True),
        return_failures=False,
    )

    assert returned is result
    assert not isinstance(returned, tuple)


def test_graph_ingestor_return_failures_empty_when_no_row_errors(monkeypatch) -> None:
    result = pd.DataFrame({"page_elements_v3": [{"detections": [], "error": None}], "text": ["ok"]})
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    returned, failures = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        return_failures=True,
    )

    assert returned is result
    assert failures == []


def test_graph_ingestor_return_failures_batch_mode_uses_source_metadata(monkeypatch) -> None:
    result = pd.DataFrame(
        {
            "metadata": [
                {
                    "source_metadata": {"source_id": "batch-doc.pdf"},
                    "errors": [
                        {
                            "stage": "batch_stage",
                            "message": "row failed",
                        }
                    ],
                }
            ],
        }
    )
    ingestor = GraphIngestor(run_mode="batch")

    returned, failures = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        params={"return_failures": True},
    )

    assert returned is result
    assert len(failures) == 1
    assert failures[0][0] == "batch-doc.pdf"
    assert "row 0, column metadata, path errors" in failures[0][1]
    assert "row failed" in failures[0][1]


def test_graph_ingestor_return_failures_falls_back_to_row_index(monkeypatch) -> None:
    result = pd.DataFrame(
        {
            "metadata": [
                {
                    "errors": [
                        {
                            "stage": "batch_stage",
                            "message": "row failed",
                        }
                    ],
                }
            ],
        }
    )
    ingestor = GraphIngestor(run_mode="inprocess")

    returned, failures = _run_graph_ingest_with_result(
        ingestor,
        result,
        monkeypatch,
        return_failures=True,
    )

    assert returned is result
    assert len(failures) == 1
    assert failures[0][0] == "row 0"
    assert "row failed" in failures[0][1]


def test_graph_ingestor_row_value_logs_best_effort_lookup_failures(caplog) -> None:
    class BrokenRow:
        def get(self, key):
            raise RuntimeError(f"cannot read {key}")

    with caplog.at_level("DEBUG", logger="nemo_retriever.ingestor.graph_ingestor"):
        assert GraphIngestor._row_value(BrokenRow(), "path") is None

    assert "Failed to read source identifier field 'path'" in caplog.text


@pytest.mark.integration
def test_graph_ingestor_raises_for_explicit_remote_stage_errors() -> None:
    ingestor = GraphIngestor(
        run_mode="inprocess",
        documents=["data/test.pdf"],
        show_progress=False,
    ).extract(
        page_elements_invoke_url="http://127.0.0.1:1/v1/nonexistent",
        extract_text=False,
        extract_images=False,
        extract_tables=False,
        extract_charts=True,
        extract_infographics=False,
        inference_batch_size=1,
        remote_retry=RemoteRetryParams(
            remote_max_pool_workers=1,
            remote_max_retries=1,
            remote_max_429_retries=1,
        ),
    )

    with pytest.raises(RuntimeError, match="page_elements_v3"):
        ingestor.ingest()


@pytest.mark.integration
def test_graph_ingestor_collect_policy_returns_explicit_remote_stage_errors() -> None:
    result = (
        GraphIngestor(
            run_mode="inprocess",
            documents=["data/test.pdf"],
            show_progress=False,
            error_policy="collect",
        )
        .extract(
            page_elements_invoke_url="http://127.0.0.1:1/v1/nonexistent",
            extract_text=False,
            extract_images=False,
            extract_tables=False,
            extract_charts=True,
            extract_infographics=False,
            inference_batch_size=1,
            remote_retry=RemoteRetryParams(
                remote_max_pool_workers=1,
                remote_max_retries=1,
                remote_max_429_retries=1,
            ),
        )
        .ingest()
    )

    assert "page_elements_v3" in result.columns
    payload = result.iloc[0]["page_elements_v3"]
    assert payload["error"]["type"] == "ConnectionError"


def test_strict_remote_error_policy_ignores_unrelated_error_columns() -> None:
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )
    result = pd.DataFrame(
        {
            "page_elements_v3": [{"detections": [], "error": None}],
            "metadata": [
                {
                    "error": {
                        "stage": "local_postprocess",
                        "type": "ValueError",
                        "message": "local stage failed",
                    }
                }
            ],
        }
    )

    ingestor._raise_for_stage_errors(result)


def test_strict_remote_error_policy_accepts_batch_dataset_rows() -> None:
    class RayLikeDataset:
        columns = ["page_elements_v3", "metadata"]

        def iter_batches(self, *, batch_format: str):
            assert batch_format == "pyarrow"
            yield pa.Table.from_pylist(
                [
                    {
                        "page_elements_v3": {
                            "timing": None,
                            "error": {
                                "stage": "remote_inference",
                                "type": "ConnectionError",
                                "message": "connection refused",
                            },
                        },
                        "metadata": {"source": "test.pdf"},
                    }
                ]
            )

    ingestor = GraphIngestor(run_mode="batch").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    with pytest.raises(GraphIngestionError, match="page_elements_v3") as exc_info:
        ingestor._raise_for_stage_errors(RayLikeDataset())

    assert exc_info.value.records[0]["row_index"] == 0


def test_graph_ingestion_error_sanitizes_remote_message_fields() -> None:
    sensitive_tail = "TAIL_SHOULD_NOT_APPEAR"
    long_message = "π" + ("x" * 600) + sensitive_tail

    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "BadRequest",
                    "message": long_message,
                },
            }
        ]
    )

    rendered = str(err)
    assert "π" not in rendered
    assert sensitive_tail not in rendered

    raw_string_err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "text_embeddings_1b_v2",
                "path": "error",
                "error": long_message,
            }
        ]
    )
    raw_rendered = str(raw_string_err)
    assert "π" not in raw_rendered
    assert sensitive_tail not in raw_rendered


def test_graph_ingestion_error_accepts_plain_string_records() -> None:
    err = GraphIngestionError(["Embedding error occurred: timed out"])
    rendered = str(err)
    assert "timed out" in rendered
    assert "path error" in rendered


def test_graph_ingestion_error_preserves_readable_remote_message_text() -> None:
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "ConnectionError",
                    "message": "connection\nrefused",
                },
            }
        ]
    )

    rendered = str(err)
    assert "connection refused" in rendered
    assert "c o n n e c t i o n" not in rendered


def test_get_error_rows_accepts_inprocess_dataframe_stage_error_columns() -> None:
    ingestor = GraphIngestor(run_mode="inprocess")
    df = pd.DataFrame(
        {
            "table_structure_ocr_v1": [
                {
                    "timing": None,
                    "error": {
                        "stage": "remote_inference",
                        "type": "ConnectionError",
                        "message": "connection refused",
                    },
                },
                {"timing": None, "error": None},
            ],
            "text": ["first page", "second page"],
        }
    )

    errors = ingestor.get_error_rows(df)

    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "first page"


def test_get_error_rows_maps_batch_dataset_with_columns_property() -> None:
    class RayLikeDataset:
        columns = ["page_elements_v3", "text"]

        def __getitem__(self, key: str):
            raise AssertionError(f"expected map_batches path, got pandas access for {key}")

        def map_batches(self, fn, *, batch_format: str, fn_kwargs: dict[str, object]):
            assert batch_format == "pyarrow"
            batch = pa.Table.from_pylist(
                [
                    {
                        "page_elements_v3": {
                            "timing": None,
                            "error": {
                                "stage": "remote_inference",
                                "type": "ConnectionError",
                                "message": "connection refused",
                            },
                        },
                        "text": "first page",
                    },
                    {"page_elements_v3": {"timing": None, "error": None}, "text": "second page"},
                ]
            )
            return fn(batch, **fn_kwargs)

    errors = GraphIngestor(run_mode="batch").get_error_rows(RayLikeDataset())

    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "first page"


@pytest.mark.parametrize("error_row", [None, 1])
def test_batch_ingest_finalization_handles_ray_pickled_object_arrays(
    error_row: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_values = []
    for row_index in range(3):
        error = None
        if row_index == error_row:
            error = {
                "stage": "remote_inference",
                "type": "ConnectionError",
                "message": "connection refused",
            }
        stage_values.append({"timing": None, "error": error})

    table = pa.Table.from_arrays(
        [
            pa.array(["first.pdf", "second.pdf", "third.pdf"]),
            pa.array(stage_values),
            ArrowPythonObjectArray.from_objects([np.array([], dtype=object) for _ in range(3)]),
        ],
        names=["path", "page_elements_v3", "images"],
    )
    batch_df = BlockAccessor.for_block(table).to_pandas()
    assert str(batch_df["images"].dtype) == "python_object()"

    ingestor = GraphIngestor(run_mode="batch").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    if error_row is None:
        result = _run_graph_ingest_with_result(ingestor, batch_df, monkeypatch)

        assert result is batch_df
        assert len(result) == 3
        assert str(result["images"].dtype) == "python_object()"
    else:
        with pytest.raises(GraphIngestionError) as exc_info:
            _run_graph_ingest_with_result(ingestor, batch_df, monkeypatch)

        records = exc_info.value.records
        assert len(records) == 1
        assert records[0]["row_index"] == error_row
        assert records[0]["source_identifier"] == "second.pdf"
        assert records[0]["column"] == "page_elements_v3"
        assert records[0]["path"] == "error"


def test_batch_ingest_finalization_skips_unrelated_arrow_extension_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array(["first.pdf"]),
            pa.array([{"timing": None, "error": None}]),
            pa.array([[{"stored_image_uri": "file:///tmp/stored-images/first.png"}]]),
        ],
        names=["path", "page_elements_v3", "images"],
    )
    batch_df = table.to_pandas(types_mapper=pd.ArrowDtype)
    assert pa.types.is_list(batch_df["images"].dtype.pyarrow_dtype)

    original_iter = pd.arrays.ArrowExtensionArray.__iter__

    def fail_for_arrow_list_column(values):
        # Reproduce the RC3 failure without requiring the customer NIM output:
        # only whole-column iteration of the stored image payload is invalid.
        if pa.types.is_list(values.dtype.pyarrow_dtype):
            raise pa.ArrowIndexError("index with value of 1 is out-of-bounds for array of length 1")
        return original_iter(values)

    monkeypatch.setattr(pd.arrays.ArrowExtensionArray, "__iter__", fail_for_arrow_list_column)
    ingestor = GraphIngestor(run_mode="batch").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    result = _run_graph_ingest_with_result(ingestor, batch_df, monkeypatch)

    assert result is batch_df


def _vdb_sink(graph):
    node = graph.roots[0]
    while not isinstance(node.operator, IngestVdbOperator):
        node = node.children[0]
    return node.operator._vdb


def test_lancedb_sink_resolves_index_mode_each_time_the_graph_is_built(tmp_path: Path) -> None:
    import lancedb

    params = VdbUploadParams(vdb_kwargs={"uri": str(tmp_path), "table_name": "docs", "overwrite": False})

    def build():
        return build_graph(extraction_mode="pdf", extract_params=ExtractParams(), vdb_upload_params=params)

    assert _vdb_sink(build()).hybrid is True
    lancedb.connect(str(tmp_path)).create_table("docs", data=[{"vector": [0.1, 0.2], "text": "alpha"}])
    assert _vdb_sink(build()).hybrid is False


def test_lancedb_sink_records_the_embed_stage_model(tmp_path: Path) -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(),
        embed_params=EmbedParams(),
        vdb_upload_params=VdbUploadParams(vdb_kwargs={"uri": str(tmp_path)}),
        stage_order=("embed",),
    )

    assert _vdb_sink(graph).embedding_model_name == "nvidia/nemotron-3-embed-1b"


def test_directory_inputs_expand_to_supported_files(tmp_path: Path) -> None:
    (tmp_path / "nested.pdf").mkdir()
    (tmp_path / "empty").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "nested.pdf" / "b.txt").write_text("b")
    (tmp_path / "notes.xyz").write_text("unsupported")
    expected = [str((tmp_path / "a.pdf").resolve()), str((tmp_path / "nested.pdf" / "b.txt").resolve())]

    assert GraphIngestor().files(str(tmp_path))._documents == expected
    assert GraphIngestor().files(str(tmp_path / "nested.pdf"))._documents == expected[1:]
    assert create_ingestor(documents=[str(tmp_path)])._documents == expected
    assert GraphIngestor(documents=str(tmp_path / "a.pdf"))._documents == [str(tmp_path / "a.pdf")]
    with pytest.raises(FileNotFoundError, match="No supported ingest files found under directory"):
        GraphIngestor().files(str(tmp_path / "empty"))


def test_explicit_media_methods_use_the_shared_media_defaults() -> None:
    from nemo_retriever.ingestor.manifest import (
        DEFAULT_AUDIO_SPLIT_INTERVAL,
        DEFAULT_VIDEO_FRAME_FPS,
        default_asr_params,
    )

    audio = GraphIngestor().extract_audio()
    video = GraphIngestor().extract_video()

    assert audio._audio_chunk_params.split_interval == DEFAULT_AUDIO_SPLIT_INTERVAL
    assert audio._asr_params == video._asr_params == default_asr_params()
    assert video._video_frame_params.fps == DEFAULT_VIDEO_FRAME_FPS
    # Other split types keep the model's interval.
    assert GraphIngestor().extract_audio(split_type="time")._audio_chunk_params.split_interval == 450
