# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import importlib
import itertools
import json
import logging
import os
import re
import sys
from typing import Any
from unittest.mock import create_autospec

import pytest
from pydantic import ValidationError
import typer.rich_utils as typer_rich_utils
from typer.testing import CliRunner

import nemo_retriever.ingest.execution as ingest_execution
import nemo_retriever.ingest.plan as ingest_plan
import nemo_retriever.ingest.service as ingest_service
import nemo_retriever.cli.ingest_workflow as ingest_workflow
import nemo_retriever.cli.ingest.graph_commands as ingest_cli_graph
import nemo_retriever.cli.ingest.shared as ingest_cli_shared
import nemo_retriever.cli.shared as cli_shared
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    StoreParams,
    TextChunkParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")


@pytest.fixture(autouse=True)
def _successful_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most tests fake GraphIngestor; default row counts should look like a successful write.
    counts = itertools.count(1)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: next(counts))


def _make_fake_ingestor() -> Any:
    fake_ingestor = create_autospec(GraphIngestor, instance=True, spec_set=True)
    fake_ingestor.files.return_value = fake_ingestor
    fake_ingestor.extract.return_value = fake_ingestor
    fake_ingestor.dedup.return_value = fake_ingestor
    fake_ingestor.caption.return_value = fake_ingestor
    fake_ingestor.embed.return_value = fake_ingestor
    fake_ingestor.store.return_value = fake_ingestor
    fake_ingestor.vdb_upload.return_value = fake_ingestor
    fake_ingestor.ingest.return_value = [{"status": "ok"}]
    return fake_ingestor


def test_root_help_lists_only_product_workflows() -> None:
    result = RUNNER.invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "service" in result.output
    assert "ingest" in result.output
    assert "query" in result.output
    assert "harness" in result.output
    for developer_command in (
        "audio",
        "image",
        "pdf",
        "local",
        "chart",
        "compare",
        "eval",
        "benchmark",
        "recall",
        "skill-eval",
        "txt",
        "html",
        "pipeline",
    ):
        assert f"│ {developer_command} " not in result.output


@pytest.mark.parametrize(
    "removed_command",
    ("txt", "html", "local", "audio", "image", "pdf", "chart", "compare", "pipeline"),
)
def test_removed_root_commands_are_not_callable(removed_command: str) -> None:
    result = RUNNER.invoke(cli_main.app, [removed_command, "--help"])

    assert result.exit_code == 2
    assert f"No such command '{removed_command}'" in result.output


def test_root_ingest_help_explains_cpu_hosted_embedding_default() -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", "local", "--help"])

    assert result.exit_code == 0
    assert "CPU-only hosts" in result.output
    assert "NVIDIA_API_KEY" in result.output
    assert "another endpoint." in result.output


def test_service_root_is_operator_only() -> None:
    result = RUNNER.invoke(cli_main.app, ["service", "--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "mcp-stdio" in result.output
    assert "│ ingest " not in result.output
    assert "retriever ingest service" in result.output


def test_root_ingest_runs_default_execution_chain(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    document = tmp_path / "multimodal_test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(ingest_execution, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 7)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "inprocess"}]
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.dpi == 200
    assert extract_params.extract_infographics is False
    assert extract_params.extract_page_as_image is True
    assert fake_ingestor.extract.call_args.kwargs == {}
    assert fake_ingestor.embed.call_args.args == ()
    vdb_upload_params = fake_ingestor.vdb_upload.call_args.args[0]
    assert vdb_upload_params.vdb_op == "lancedb"
    assert vdb_upload_params.vdb_kwargs == {
        "uri": "lancedb",
        "table_name": "nemo-retriever",
        "overwrite": True,
        "hybrid": True,
        "embedding_model_name": "nvidia/nemotron-3-embed-1b",
    }
    assert "Ingested 1 file(s) → 7 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_without_mode_defaults_to_local(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    document = tmp_path / "default-local.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(ingest_execution, "create_ingestor", fake_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 0, result.output
    assert create_calls == [{"run_mode": "inprocess"}]
    assert fake_ingestor.files.call_args.args == ([str(document)],)


def test_root_ingest_without_mode_accepts_local_options_before_documents(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "default-local-options.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "--append", "--lancedb-uri", "/tmp/default-lancedb", str(document)],
    )

    assert result.exit_code == 0, result.output
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/default-lancedb",
        "table_name": "nemo-retriever",
        "overwrite": False,
        "hybrid": True,
        "embedding_model_name": "nvidia/nemotron-3-embed-1b",
    }


def test_root_ingest_service_mode_uses_service_ingest_core(tmp_path, monkeypatch) -> None:
    import nemo_retriever.service.service_ingestor as service_ingestor_module

    document = tmp_path / "service.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    captured: dict[str, Any] = {}

    class _FakeServiceIngestor(list):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            captured["init"] = kwargs
            self.dataframe = None

        def files(self, files: list[str]):
            captured["files"] = files
            return self

        def extract(self, params=None, *, split_config=None, extraction_mode="auto", **_kwargs):
            captured["extract_params"] = params
            captured["split_config"] = split_config
            captured["extraction_mode"] = extraction_mode
            return self

        def dedup(self, params=None, **_kwargs):
            captured["dedup_params"] = params
            return self

        def caption(self, params=None, **_kwargs):
            captured["caption_params"] = params
            return self

        def embed(self, params=None, **_kwargs):
            captured["embed_params"] = params
            return self

        def ingest(self, *args: Any, **kwargs: Any):
            captured["ingest_kwargs"] = kwargs
            return self

    monkeypatch.setattr(service_ingestor_module, "ServiceIngestor", _FakeServiceIngestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "service",
            str(document),
            "--service-url",
            "http://retriever-service:7670",
            "--service-concurrency",
            "3",
            "--service-api-token",
            "service-token",
            "--dpi",
            "300",
            "--extract-images",
            "--extract-nested-images",
            "--embed-granularity",
            "page",
            "--dedup",
            "--dedup-iou-threshold",
            "0.6",
            "--caption",
            "--caption-context-text-max-chars",
            "12",
            "--text-chunk",
            "--text-chunk-max-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["init"] == {
        "base_url": "http://retriever-service:7670",
        "max_concurrency": 3,
        "api_token": "service-token",
    }
    assert captured["files"] == [str(document)]
    assert captured["extraction_mode"] == "auto"
    assert captured["extract_params"].dpi == 300
    assert captured["extract_params"].extract_images is True
    assert captured["extract_params"].extract_nested_images is True
    assert captured["split_config"]["pdf"]["max_tokens"] == 64
    assert captured["dedup_params"].iou_threshold == 0.6
    assert captured["caption_params"].context_text_max_chars == 12
    assert captured["embed_params"].embed_granularity == "page"
    assert captured["ingest_kwargs"] == {"return_results": True}
    assert "through retriever service http://retriever-service:7670" in result.output


def test_service_family_resolution_expands_glob_patterns_for_auto_input(tmp_path) -> None:
    document = tmp_path / "chunked.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    image = tmp_path / "diagram.png"
    image.write_bytes(b"png")
    request = ingest_service.ServiceIngestRequest(
        documents=[str(tmp_path / "*")],
        input_type="auto",
        enable_text_chunk=True,
        text_chunk_params=TextChunkParams(max_tokens=64, overlap_tokens=8),
        caption_params=CaptionParams(),
    )

    split_config = ingest_service.service_split_config_for_request(request)
    dedup_params, dedup_scope = ingest_service.resolve_service_dedup_for_request(request)

    chunk_config = {"max_tokens": 64, "overlap_tokens": 8, "encoding": "utf-8"}
    assert split_config == {"pdf": chunk_config, "image": chunk_config}
    assert dedup_params is None
    assert dedup_scope == {
        "mode": "caption_default",
        "enabled_families": ["pdf"],
        "exempt_families": ["image"],
    }


def test_root_ingest_service_dry_run_redacts_token(tmp_path, monkeypatch) -> None:
    import nemo_retriever.service.service_ingestor as service_ingestor_module

    document = tmp_path / "service.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fail_service_ingestor(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ServiceIngestor should not be created for --dry-run")

    monkeypatch.setattr(service_ingestor_module, "ServiceIngestor", fail_service_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "service",
            str(document),
            "--service-api-token",
            "service-token",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_mode"] == "service"
    assert payload["documents"] == [str(document)]
    assert payload["service"]["service_api_token"] == "<redacted>"
    assert payload["service"]["service_url"] == "http://localhost:7670"


@pytest.mark.parametrize(
    ("filenames", "expected_dedup", "expected_scope"),
    [
        pytest.param(
            ["document.pdf"],
            {"content_hash": True, "bbox_iou": True, "iou_threshold": 0.45},
            None,
            id="pdf",
        ),
        pytest.param(["diagram.png"], None, None, id="image"),
        pytest.param(
            ["document.pdf", "diagram.png"],
            None,
            {
                "mode": "caption_default",
                "enabled_families": ["pdf"],
                "exempt_families": ["image"],
            },
            id="mixed-pdf-image",
        ),
    ],
)
def test_root_ingest_service_caption_dry_run_reports_automatic_dedup_scope(
    tmp_path,
    filenames: list[str],
    expected_dedup: dict[str, object] | None,
    expected_scope: dict[str, object] | None,
) -> None:
    documents = []
    for filename in filenames:
        document = tmp_path / filename
        document.write_bytes(b"input")
        documents.append(str(document))

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "service", *documents, "--caption", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dedup"] == expected_dedup
    assert payload.get("dedup_scope") == expected_scope


def test_root_ingest_service_no_dedup_dry_run_reports_explicit_opt_out(tmp_path) -> None:
    document = tmp_path / "captioned.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "service", str(document), "--caption", "--no-dedup", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dedup"] == {
        "content_hash": False,
        "bbox_iou": False,
        "iou_threshold": 0.45,
    }
    assert "dedup_scope" not in payload


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lancedb-uri", "custom-db"),
        ("--embed-invoke-url", "http://embed.example/v1"),
        ("--ray-address", "ray://localhost:10001"),
    ],
)
def test_root_ingest_service_mode_rejects_local_only_options(tmp_path, flag: str, value: str) -> None:
    document = tmp_path / "service.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", "service", str(document), flag, value])

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert flag in result.output


def test_root_ingest_passes_vdb_options_and_run_mode(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    first_document = tmp_path / "a.pdf"
    globbed_document = tmp_path / "b" / "c.pdf"
    first_document.write_bytes(b"%PDF-1.4\n")
    globbed_document.parent.mkdir()
    globbed_document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(ingest_execution, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 12)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "batch",
            str(first_document),
            str(globbed_document.parent),
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "batch"}]
    assert fake_ingestor.files.call_args.args == ([str(first_document), str(globbed_document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert fake_ingestor.extract.call_args.kwargs == {}
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/lancedb",
        "table_name": "docs",
        "overwrite": True,
        "hybrid": True,
        "embedding_model_name": "nvidia/nemotron-3-embed-1b",
    }
    assert "Ingested 2 file(s) → 12 row(s) in LanceDB /tmp/lancedb/docs." in result.output


def test_root_ingest_append_forwards_overwrite_false(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "multimodal_test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), "--append"])

    assert result.exit_code == 0
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "lancedb",
        "table_name": "nemo-retriever",
        "overwrite": False,
        "hybrid": True,
        "embedding_model_name": "nvidia/nemotron-3-embed-1b",
    }


def test_root_ingest_fails_when_no_rows_landed(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "silent-stage-failure.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 0)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 1
    assert "retriever ingest produced 0 rows" in result.output
    assert "NVIDIA_API_KEY/NGC_API_KEY" in result.output
    assert "Ingested 1 file(s)" not in result.output


def test_root_ingest_fails_when_current_run_is_empty_but_table_has_stale_rows(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    fake_ingestor.ingest.return_value = []
    document = tmp_path / "stale-table.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 3)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 1
    assert "retriever ingest produced 0 rows before LanceDB write" in result.output
    assert "may still contain rows from an earlier run" in result.output
    assert "Ingested 1 file(s)" not in result.output


def test_root_ingest_append_fails_when_row_count_does_not_increase(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "silent-append-failure.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    counts = iter([3, 3])

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: next(counts))

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), "--append"])

    assert result.exit_code == 1
    assert "did not add rows" in result.output
    assert "row count stayed at 3" in result.output


def test_root_ingest_passes_nim_url_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "nim-routed.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**_kwargs: Any) -> Any:
        return fake_ingestor

    monkeypatch.setattr(ingest_execution, "create_ingestor", fake_create_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--page-elements-invoke-url",
            "http://page-elements:8000/v1/infer",
            "--ocr-invoke-url",
            "http://ocr:8000/v1/infer",
            "--ocr-version",
            "v1",
            "--table-structure-invoke-url",
            "http://table-structure:8000/v1/infer",
            "--embed-invoke-url",
            "http://embed:8000/v1/embeddings",
            "--embed-model-name",
            "nvidia/llama-nemotron-embed-1b-v2",
            "--embed-model-provider-prefix",
            "nvidia",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.page_elements_invoke_url == "http://page-elements:8000/v1/infer"
    assert extract_params.ocr_invoke_url == "http://ocr:8000/v1/infer"
    assert extract_params.ocr_version == "v1"
    assert extract_params.table_structure_invoke_url == "http://table-structure:8000/v1/infer"

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.embed_invoke_url == "http://embed:8000/v1/embeddings"
    assert embed_params.embedding_endpoint == "http://embed:8000/v1/embeddings"
    assert embed_params.model_name == "nvidia/llama-nemotron-embed-1b-v2"
    assert embed_params.embed_model_name == "nvidia/llama-nemotron-embed-1b-v2"
    assert embed_params.embed_model_provider_prefix == "nvidia"
    vdb_kwargs = fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs
    assert vdb_kwargs["embedding_model_name"] == "nvidia/llama-nemotron-embed-1b-v2"
    assert vdb_kwargs["vector_dim"] is None
    assert "embedding_model_revision" not in vdb_kwargs


def test_root_ingest_passes_embedding_overrides_without_stage_flags(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "jp20-style.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--embed-modality",
            "text_image",
            "--embed-granularity",
            "element",
            "--text-elements-modality",
            "text",
            "--structured-elements-modality",
            "image",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.use_page_elements is True
    assert extract_params.use_table_structure is False
    assert extract_params.table_output_format == "pseudo_markdown"

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.embed_modality == "text_image"
    assert embed_params.embed_granularity == "element"
    assert embed_params.text_elements_modality == "text"
    assert embed_params.structured_elements_modality == "image"


def test_root_ingest_text_chunk_builds_split_config(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "chunked.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--text-chunk",
            "--text-chunk-max-tokens",
            "512",
            "--text-chunk-overlap-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert fake_ingestor.extract.call_args.kwargs["split_config"] == {
        "pdf": {
            "max_tokens": 512,
            "overlap_tokens": 64,
            "tokenizer_model_id": None,
            "encoding": "utf-8",
            "tokenizer_cache_dir": None,
        }
    }


@pytest.mark.parametrize(
    ("filename", "param_key", "param_type"),
    [
        ("notes.txt", "text_params", TextChunkParams),
        ("page.html", "html_params", HtmlChunkParams),
    ],
)
def test_root_ingest_text_chunk_uses_dedicated_text_params(
    monkeypatch,
    tmp_path,
    filename: str,
    param_key: str,
    param_type: type[TextChunkParams | HtmlChunkParams],
) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / filename
    document.write_text("chunk me", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--text-chunk",
            "--text-chunk-max-tokens",
            "512",
            "--text-chunk-overlap-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    extract_kwargs = fake_ingestor.extract.call_args.kwargs
    assert "split_config" not in extract_kwargs
    chunk_params = extract_kwargs[param_key]
    assert isinstance(chunk_params, param_type)
    assert chunk_params.max_tokens == 512
    assert chunk_params.overlap_tokens == 64


def test_root_ingest_passes_ocr_lang_option(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "english-ocr.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), "--ocr-lang", "english"])

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "pdfium_hybrid"
    assert extract_params.ocr_version == "v2"
    assert extract_params.ocr_lang == "english"


def test_root_ingest_rejects_ocr_lang_with_legacy_ocr_version(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "legacy-ocr.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--ocr-version",
            "v1",
            "--ocr-lang",
            "english",
        ],
    )

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "ocr_lang is only supported when ocr_version='v2'" in result.output
    assert "Traceback" not in result.output
    fake_ingestor.extract.assert_not_called()


def test_root_ingest_passes_batch_tuning_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    document = tmp_path / "batch-tuned.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(ingest_execution, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 42)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "batch",
            str(document),
            "--ray-address",
            "ray://cluster:10001",
            "--no-ray-log-to-driver",
            "--pdf-extract-workers",
            "4",
            "--pdf-extract-batch-size",
            "2",
            "--pdf-extract-cpus-per-task",
            "1.5",
            "--page-elements-workers",
            "3",
            "--page-elements-batch-size",
            "8",
            "--page-elements-cpus-per-actor",
            "0.5",
            "--ocr-workers",
            "5",
            "--ocr-batch-size",
            "6",
            "--ocr-cpus-per-actor",
            "0.75",
            "--embed-workers",
            "7",
            "--embed-batch-size",
            "16",
            "--embed-cpus-per-actor",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert create_calls == [
        {
            "run_mode": "batch",
            "ray_address": "ray://cluster:10001",
            "ray_log_to_driver": False,
        }
    ]

    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.batch_tuning.pdf_extract_workers == 4
    assert extract_params.batch_tuning.pdf_extract_batch_size == 2
    assert extract_params.batch_tuning.pdf_extract_num_cpus == 1.5
    assert extract_params.batch_tuning.page_elements_workers == 3
    assert extract_params.batch_tuning.page_elements_batch_size == 8
    assert extract_params.batch_tuning.page_elements_cpus_per_actor == 0.5
    assert extract_params.batch_tuning.ocr_workers == 5
    assert extract_params.batch_tuning.ocr_inference_batch_size == 6
    assert extract_params.batch_tuning.ocr_cpus_per_actor == 0.75

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.batch_tuning.embed_workers == 7
    assert embed_params.batch_tuning.embed_batch_size == 16
    assert embed_params.batch_tuning.embed_cpus_per_actor == 0.25
    assert "Ingested 1 file(s) → 42 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_passes_public_parity_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "parity.pdf"
    image_store = tmp_path / "images"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 14)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "batch",
            str(document),
            "--api-key",
            "nvapi-secret",
            "--dedup",
            "--dedup-iou-threshold",
            "0.6",
            "--caption",
            "--caption-invoke-url",
            "http://vlm:8000/v1/chat/completions",
            "--store-images-uri",
            str(image_store),
            "--method",
            "nemotron_parse",
            "--pdf-split-batch-size",
            "2",
            "--nemotron-parse-workers",
            "3",
            "--nemotron-parse-batch-size",
            "4",
            "--nemotron-parse-gpus-per-actor",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "dedup",
        "caption",
        "embed",
        "store",
        "vdb_upload",
        "ingest",
    ]

    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "nemotron_parse"
    assert extract_params.api_key == "nvapi-secret"
    assert extract_params.batch_tuning.pdf_split_batch_size == 2
    assert extract_params.batch_tuning.nemotron_parse_workers == 3
    assert extract_params.batch_tuning.nemotron_parse_batch_size == 4
    assert extract_params.batch_tuning.gpu_nemotron_parse == 0.5

    dedup_params = fake_ingestor.dedup.call_args.args[0]
    assert isinstance(dedup_params, DedupParams)
    assert dedup_params.iou_threshold == 0.6

    caption_params = fake_ingestor.caption.call_args.args[0]
    assert isinstance(caption_params, CaptionParams)
    assert caption_params.api_key == "nvapi-secret"

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.api_key == "nvapi-secret"

    store_params = fake_ingestor.store.call_args.args[0]
    assert isinstance(store_params, StoreParams)
    assert store_params.storage_uri == str(image_store.resolve())
    assert "Ingested 1 file(s) → 14 row(s) in LanceDB lancedb/nemo-retriever." in result.output


@pytest.mark.parametrize(
    "dedup_args",
    [
        pytest.param([], id="unspecified"),
        pytest.param(["--no-dedup"], id="explicitly-disabled"),
    ],
)
def test_root_ingest_rejects_dedup_threshold_without_enabling_dedup(
    monkeypatch,
    tmp_path,
    dedup_args: list[str],
) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "dedup-threshold.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "local", str(document), *dedup_args, "--dedup-iou-threshold", "0.6"],
    )

    assert result.exit_code == 1
    assert "Dedup options require --dedup" in result.output
    fake_ingestor.dedup.assert_not_called()
    fake_ingestor.embed.assert_not_called()


def test_resolved_ingest_plan_runs_through_workflow(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "programmatic-plan.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    plan = ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(document)], input_type="pdf"),
            extract=ingest_plan.IngestExtractOptions(
                table_output_format="markdown",
                batch=ingest_plan.IngestExtractBatchOptions(
                    page_elements_gpus_per_actor=0.2,
                    ocr_gpus_per_actor=0.3,
                    table_structure_workers=6,
                    table_structure_batch_size=12,
                    table_structure_cpus_per_actor=0.4,
                    table_structure_gpus_per_actor=0.25,
                    pdf_split_batch_size=9,
                    nemotron_parse_workers=10,
                    nemotron_parse_batch_size=11,
                    nemotron_parse_gpus_per_actor=0.6,
                ),
            ),
            embed=ingest_plan.IngestEmbedOptions(
                local_ingest_embed_backend="hf",
                batch=ingest_plan.IngestEmbedBatchOptions(
                    embed_workers_min=1,
                    embed_workers_initial=4,
                    embed_workers_max=8,
                    embed_gpus_per_actor=0.5,
                ),
            ),
        )
    )
    result = ingest_workflow.run_ingest_workflow(plan)

    assert result["n_documents"] == 1
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.use_table_structure is True
    assert extract_params.table_output_format == "markdown"
    assert extract_params.batch_tuning.gpu_page_elements == 0.2
    assert extract_params.batch_tuning.gpu_ocr == 0.3
    assert extract_params.batch_tuning.table_structure_workers == 6
    assert extract_params.batch_tuning.table_structure_batch_size == 12
    assert extract_params.batch_tuning.table_structure_cpus_per_actor == 0.4
    assert extract_params.batch_tuning.gpu_table_structure == 0.25
    assert extract_params.batch_tuning.pdf_split_batch_size == 9
    assert extract_params.batch_tuning.nemotron_parse_workers == 10
    assert extract_params.batch_tuning.nemotron_parse_batch_size == 11
    assert extract_params.batch_tuning.gpu_nemotron_parse == 0.6

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.local_ingest_embed_backend == "hf"
    assert embed_params.batch_tuning.gpu_embed == 0.5
    assert embed_params.batch_tuning.embed_workers_min == 1
    assert embed_params.batch_tuning.embed_workers_initial == 4
    assert embed_params.batch_tuning.embed_workers_max == 8


def test_build_ingest_pipeline_attaches_store_after_embed_with_tuning(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "stored.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    plan = ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(document)], input_type="pdf"),
            image_store=ingest_plan.IngestImageStoreOptions(
                images_uri=str(tmp_path / "stored-images"),
                workers=4,
            ),
        )
    )
    ingestor = ingest_execution.build_ingest_pipeline(plan)

    assert ingestor is fake_ingestor
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "embed",
        "store",
        "vdb_upload",
    ]
    store_params = fake_ingestor.store.call_args.args[0]
    assert isinstance(store_params, StoreParams)
    assert store_params.storage_uri.endswith("/stored-images")
    assert store_params.batch_tuning.store_workers == 4


def test_execute_ingest_plan_returns_structured_execution_data(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "execution-result.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 9)

    plan = ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(document)]),
            runtime=ingest_plan.IngestRuntimeOptions(run_mode="inprocess"),
            storage=ingest_plan.IngestStorageOptions(
                lancedb_uri="/tmp/nemo-test-lancedb",
                table_name="execution_result",
            ),
        )
    )
    execution = ingest_execution.execute_ingest_plan(plan)

    assert execution.documents == [str(document)]
    assert execution.lancedb_uri == "/tmp/nemo-test-lancedb"
    assert execution.table_name == "execution_result"
    assert execution.lancedb_target == "/tmp/nemo-test-lancedb/execution_result"
    assert execution.n_rows == 9
    assert execution.result_n_rows == 1
    assert execution.result == [{"status": "ok"}]
    assert execution.run_metadata["branch_summary"]
    summary = execution.to_summary_dict()
    assert summary == {
        "n_documents": 1,
        "lancedb_uri": "/tmp/nemo-test-lancedb",
        "table_name": "execution_result",
        "n_rows": 9,
        "result_n_rows": 1,
    }


def test_execute_ingest_plan_requires_vdb_stage_for_row_verification(monkeypatch, tmp_path) -> None:
    document = tmp_path / "no-vdb.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called when row verification has no VDB stage")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)
    plan = ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(document)]),
            runtime=ingest_plan.IngestRuntimeOptions(run_mode="inprocess"),
        )
    )
    plan = replace(plan, vdb_params=None)

    with pytest.raises(ValueError, match="Row verification checks the effective VDB upload target"):
        ingest_execution.execute_ingest_plan(plan)


def test_root_ingest_reports_empty_directory_error(tmp_path) -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(tmp_path)])

    assert result.exit_code == 1
    assert "No supported ingest files found under directory" in result.output


def test_root_ingest_reports_unknown_default_input_type(tmp_path) -> None:
    document = tmp_path / "payload.bin"
    document.write_bytes(b"unknown")

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 1
    assert "Unsupported input file type(s) for retriever ingest" in result.output


def test_root_ingest_routes_text_inputs_by_default_to_auto_planner(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "notes.txt"
    document.write_text("not a pdf", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


@pytest.mark.parametrize("suffix", [".md", ".json", ".sh"])
def test_root_ingest_treats_documented_plain_text_extensions_as_text(monkeypatch, tmp_path, suffix) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / f"notes{suffix}"
    document.write_text("plain text content", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_help_defaults_to_local_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer_rich_utils, "MAX_WIDTH", 200)
    monkeypatch.setattr(typer_rich_utils, "FORCE_TERMINAL", False)
    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "--help"],
        prog_name="retriever",
    )

    assert result.exit_code == 0
    assert "Usage: retriever ingest [OPTIONS] {documents}..." in result.output
    assert "input formats, not commands" in result.output
    assert "CPU-only hosts use NVIDIA's hosted embedding endpoint" in result.output
    assert NEMOTRON_3_EMBED_MODEL in result.output
    assert "retriever ingest batch --help" in result.output
    assert "retriever ingest service --help" in result.output
    for option in (
        "--index-mode",
        "--lancedb-uri",
        "--table-name",
        "--embed-invoke-url",
        "--embed-model-name",
        "--append",
    ):
        assert option in result.output
    assert "--input-type" not in result.output
    assert "--run-mode" not in result.output
    assert "Commands" not in result.output
    assert "│ html " not in result.output
    assert "│ txt " not in result.output


def test_root_ingest_mode_overview_hides_legacy_local_alias() -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest"], prog_name="retriever", env={"COLUMNS": "200"})

    assert result.exit_code == 2
    assert "retriever ingest DOCUMENTS" in result.output
    assert "│ batch " in result.output
    assert "│ service " in result.output
    assert "│ local " not in result.output
    assert "retriever ingest local" not in result.output


def test_root_ingest_errors_reference_only_the_public_help_path() -> None:
    for flag in ("-h", "--not-an-ingest-option"):
        result = RUNNER.invoke(cli_main.app, ["ingest", flag], prog_name="retriever")

        assert result.exit_code == 2
        assert "Try 'retriever ingest --help' for help" in result.output
        assert "retriever ingest local" not in result.output


def test_root_ingest_batch_help_remains_mode_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer_rich_utils, "MAX_WIDTH", 200)
    monkeypatch.setattr(typer_rich_utils, "FORCE_TERMINAL", False)
    result = RUNNER.invoke(cli_main.app, ["ingest", "batch", "--help"])

    assert result.exit_code == 0
    assert "Usage: root ingest batch [OPTIONS] {documents}..." in result.output
    assert "--ray-address" in result.output
    assert "--pdf-extract-workers" in result.output
    assert "--lancedb-uri" in result.output
    assert "--dedup" in result.output
    assert "--no-dedup" in result.output
    assert "--service-url" not in result.output
    assert "--input-type" not in result.output


def test_root_ingest_local_help_uses_shared_graph_contract() -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", "local", "--help"], prog_name="retriever")

    assert result.exit_code == 0
    assert "Usage: retriever ingest [OPTIONS] {documents}..." in result.output
    assert "retriever ingest local" not in result.output
    assert "--input-type" not in result.output
    assert "--run-mode" not in result.output
    assert "--service-url" not in result.output
    assert "--ray-address" in result.output
    assert "--profile" in result.output
    assert "<auto|fast-text>" in result.output
    assert "--extract-images" in result.output
    assert "--use-page" not in result.output
    assert "--use-graphic" not in result.output
    assert "--use-table" not in result.output
    assert "--embed-modality" in result.output
    assert "--embed-granular" in result.output
    assert "--text-elements-" in result.output
    assert "--structured-ele" in result.output
    assert "--text-chunk" in result.output
    assert "--store-images-" in result.output
    assert "--api-key" in result.output
    assert "--dedup" in result.output
    assert "--no-dedup" in result.output
    assert "--caption" in result.output
    assert "--index-mode" in result.output
    assert "--hybrid" not in result.output
    assert "--sparse" not in result.output
    assert re.search(r"--no-caption(?!-)", result.output) is None


@pytest.mark.parametrize(
    ("args", "expected_flag"),
    [
        (["--ray-address", "ray://cluster:10001"], "--ray-address"),
        (["--no-ray-log-to-driver"], "--ray-log-to-driver"),
        (["--pdf-extract-workers", "2"], "--pdf-extract-workers"),
    ],
)
def test_root_ingest_local_rejects_batch_only_options(tmp_path, args: list[str], expected_flag: str) -> None:
    document = tmp_path / "local-batch-only.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), *args])

    assert result.exit_code == 1
    assert "Batch-only option(s) require `retriever ingest batch`" in result.output
    assert expected_flag in result.output


def test_root_ingest_default_local_rejects_batch_only_options(tmp_path) -> None:
    document = tmp_path / "default-local-batch-only.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", "--ray-address", "ray://cluster:10001", str(document)])

    assert result.exit_code == 1
    assert "Batch-only option(s) require `retriever ingest batch`" in result.output
    assert "--ray-address" in result.output


def test_root_ingest_service_help_hides_local_only_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer_rich_utils, "MAX_WIDTH", 200)
    monkeypatch.setattr(typer_rich_utils, "FORCE_TERMINAL", False)
    result = RUNNER.invoke(cli_main.app, ["ingest", "service", "--help"])

    assert result.exit_code == 0
    assert "Usage: root ingest service [OPTIONS] {documents}..." in result.output
    assert "--service-url" in result.output
    assert "--dedup" in result.output
    assert "--no-dedup" in result.output
    assert "--extract-images" in result.output
    assert "--extract-nested-images" in result.output
    assert "--embed-granular" in result.output
    assert "--lancedb-uri" not in result.output
    assert "--overwrite" not in result.output
    assert "--append" not in result.output
    assert "--ray-address" not in result.output
    assert "--embed-invoke" not in result.output
    assert "--local-ingest" not in result.output
    assert "--ocr-lang" not in result.output
    assert "--api-key" not in result.output
    assert "--caption-invoke" not in result.output


def test_root_ingest_dry_run_prints_plan_without_creating_ingestor(monkeypatch, tmp_path) -> None:
    document = tmp_path / "fast.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for --dry-run")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", "local", str(document), "--profile", "fast-text", "--dry-run"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["profile"] == "fast-text"
    assert payload["create_ingestor"] == {"run_mode": "inprocess"}
    assert payload["extract"]["method"] == "pdfium"
    assert payload["extract"]["extract_images"] is False
    assert payload["extract"]["use_page_elements"] is False
    assert payload["extract"]["extract_tables"] is False


@pytest.mark.parametrize("mode", ["local", "batch"])
@pytest.mark.parametrize(
    ("cli_args", "expected_dedup"),
    [
        pytest.param([], None, id="unspecified"),
        pytest.param(
            ["--caption"],
            {"content_hash": True, "bbox_iou": True, "iou_threshold": 0.45},
            id="caption-default",
        ),
        pytest.param(
            ["--dedup"],
            {"content_hash": True, "bbox_iou": True, "iou_threshold": 0.45},
            id="enabled",
        ),
        pytest.param(
            ["--no-dedup"],
            {"content_hash": False, "bbox_iou": False, "iou_threshold": 0.45},
            id="disabled",
        ),
    ],
)
def test_root_ingest_dedup_option_preserves_tri_state_in_dry_run(
    tmp_path,
    mode: str,
    cli_args: list[str],
    expected_dedup: dict[str, object] | None,
) -> None:
    document = tmp_path / "dedup-option.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", mode, str(document), *cli_args, "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dedup"] == expected_dedup


def test_root_ingest_dry_run_routes_avi_to_video(monkeypatch, tmp_path) -> None:
    document = tmp_path / "sample.avi"
    document.write_bytes(b"AVI")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for --dry-run")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--dry-run"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["documents"] == [str(document)]
    assert payload["branch_summary"] == "video:1"


def test_root_ingest_dry_run_discovers_avi_in_mixed_directory(monkeypatch, tmp_path) -> None:
    avi_document = tmp_path / "sample.avi"
    avi_document.write_bytes(b"AVI")
    pdf_document = tmp_path / "sample.pdf"
    pdf_document.write_bytes(b"%PDF-1.4\n")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for --dry-run")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["documents"] == sorted((str(avi_document), str(pdf_document)))
    assert payload["branch_summary"] == "pdf:1, video:1"


def test_dry_run_secret_redaction_covers_common_credential_names() -> None:
    payload = {
        "api_key": "nvapi-test",
        "auth_token": "token-test",
        "password": "pw-test",
        "client_secret": "secret-test",
        "bearer_token": "bearer-test",
        "credential_path": "/tmp/credentials",
        "nested": [{"refreshToken": "refresh-test", "plain": "value"}],
        "max_tokens": 1024,
        "num_tokens_per_batch": 256,
        "tokenizer_path": "/tmp/tokenizer",
        "safe": "visible",
    }

    redacted = ingest_workflow._strip_secret_values(payload)

    assert redacted == {
        "api_key": "<redacted>",
        "auth_token": "<redacted>",
        "password": "<redacted>",
        "client_secret": "<redacted>",
        "bearer_token": "<redacted>",
        "credential_path": "<redacted>",
        "nested": [{"refreshToken": "<redacted>", "plain": "value"}],
        "max_tokens": 1024,
        "num_tokens_per_batch": 256,
        "tokenizer_path": "/tmp/tokenizer",
        "safe": "visible",
    }


def test_root_ingest_passes_high_level_extract_overrides(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--method",
            "pdfium",
            "--dpi",
            "250",
            "--no-extract-tables",
            "--no-extract-images",
            "--no-extract-charts",
            "--no-extract-page-as-image",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "pdfium"
    assert extract_params.dpi == 250
    assert extract_params.extract_text is True
    assert extract_params.extract_images is False
    assert extract_params.extract_tables is False
    assert extract_params.extract_charts is False
    assert extract_params.extract_infographics is False
    assert extract_params.extract_page_as_image is False
    assert extract_params.use_page_elements is True


@pytest.mark.parametrize(
    "flag",
    [
        "--use-page-elements",
        "--no-use-page-elements",
        "--use-graphic-elements",
        "--no-use-graphic-elements",
        "--use-table-structure",
        "--no-use-table-structure",
        "--graphic-elements-invoke-url",
    ],
)
def test_root_ingest_rejects_internal_stage_flags(tmp_path, flag: str) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), flag])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_root_ingest_caption_is_optional_and_passes_minimal_caption_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "captioned.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--caption",
            "--caption-invoke-url",
            "http://vlm:8000/v1/chat/completions",
            "--caption-model-name",
            "nvidia/test-vlm",
            "--caption-gpu-memory-utilization",
            "0.8",
            "--caption-context-text-max-chars",
            "512",
            "--caption-infographics",
        ],
    )

    assert result.exit_code == 0
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "dedup",
        "caption",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    fake_ingestor.dedup.assert_called_once()
    dedup_params = fake_ingestor.dedup.call_args.args[0]
    assert dedup_params == DedupParams(content_hash=True, bbox_iou=True, iou_threshold=0.45)
    caption_params = fake_ingestor.caption.call_args.args[0]
    assert isinstance(caption_params, CaptionParams)
    assert caption_params.endpoint_url == "http://vlm:8000/v1/chat/completions"
    assert caption_params.model_name == "nvidia/test-vlm"
    assert caption_params.gpu_memory_utilization == 0.8
    assert caption_params.context_text_max_chars == 512
    assert caption_params.caption_infographics is True


def test_root_ingest_rejects_caption_options_without_caption(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "not-captioned.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--caption-invoke-url",
            "http://vlm:8000/v1/chat/completions",
            "--caption-gpu-memory-utilization",
            "0.8",
        ],
    )

    assert result.exit_code == 1
    assert "Caption options require --caption" in result.output
    fake_ingestor.caption.assert_not_called()
    fake_ingestor.embed.assert_not_called()


def test_root_ingest_auto_passes_audio_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "meeting.wav"
    document.write_bytes(b"audio")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env",
        lambda: ASRParams(segment_audio=False),
    )

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--segment-audio",
            "--audio-split-type",
            "time",
            "--audio-split-interval",
            "42",
        ],
    )

    assert result.exit_code == 0
    kwargs = fake_ingestor.extract.call_args.kwargs
    assert isinstance(kwargs["audio_chunk_params"], AudioChunkParams)
    assert kwargs["audio_chunk_params"].split_type == "time"
    assert kwargs["audio_chunk_params"].split_interval == 42
    assert isinstance(kwargs["asr_params"], ASRParams)
    assert kwargs["asr_params"].segment_audio is True


def test_root_ingest_auto_passes_video_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "demo.mp4"
    document.write_bytes(b"video")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env",
        lambda: ASRParams(segment_audio=False),
    )

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(document),
            "--no-video-extract-audio",
            "--video-frame-fps",
            "0.25",
            "--no-video-frame-dedup",
            "--no-video-frame-text-dedup",
            "--video-frame-text-dedup-max-dropped-frames",
            "5",
            "--no-video-av-fuse",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    kwargs = fake_ingestor.extract.call_args.kwargs
    assert isinstance(kwargs["audio_chunk_params"], AudioChunkParams)
    assert kwargs["audio_chunk_params"].enabled is False
    assert isinstance(kwargs["video_frame_params"], VideoFrameParams)
    assert kwargs["video_frame_params"].fps == 0.25
    assert kwargs["video_frame_params"].dedup is False
    assert isinstance(kwargs["video_text_dedup_params"], VideoFrameTextDedupParams)
    assert kwargs["video_text_dedup_params"].enabled is False
    assert kwargs["video_text_dedup_params"].max_dropped_frames == 5
    assert isinstance(kwargs["av_fuse_params"], AudioVisualFuseParams)
    assert kwargs["av_fuse_params"].enabled is False


def test_root_ingest_rejects_removed_profiles(tmp_path) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), "--profile", "ocr"])

    assert result.exit_code == 2
    assert "is not one of 'auto', 'fast-text'" in result.output


def test_root_ingest_routes_tiff_inputs_by_default_to_auto_planner(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "scan.tiff"
    document.write_bytes(b"tiff")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert fake_ingestor.extract.call_args.kwargs == {}


def test_root_ingest_auto_mixed_directory_uses_auto_extraction(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    dataset = tmp_path / "dataset"
    nested = dataset / "nested"
    nested.mkdir(parents=True)
    pdf = dataset / "manual.pdf"
    text = nested / "notes.txt"
    image = nested / "diagram.png"
    pdf.write_bytes(b"%PDF-1.4\n")
    text.write_text("notes", encoding="utf-8")
    image.write_bytes(b"png")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(dataset)])

    assert result.exit_code == 0
    assert set(fake_ingestor.files.call_args.args[0]) == {
        str(pdf.resolve()),
        str(text.resolve()),
        str(image.resolve()),
    }
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_text_formats_directory_includes_documented_plain_text_extensions(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    dataset = tmp_path / "data"
    dataset.mkdir()
    html = dataset / "architecture.html"
    text = dataset / "api_changelog.txt"
    markdown = dataset / "aurora_README.md"
    json_document = dataset / "metadata.json"
    shell_script = dataset / "setup.sh"
    html.write_text("<h1>Architecture</h1>", encoding="utf-8")
    text.write_text("API changelog", encoding="utf-8")
    markdown.write_text("# Aurora", encoding="utf-8")
    json_document.write_text('{"project": "aurora"}', encoding="utf-8")
    shell_script.write_text("#!/bin/sh\necho aurora\n", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 5)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(dataset)])

    assert result.exit_code == 0, result.output
    assert set(fake_ingestor.files.call_args.args[0]) == {
        str(html.resolve()),
        str(text.resolve()),
        str(markdown.resolve()),
        str(json_document.resolve()),
        str(shell_script.resolve()),
    }
    extract_kwargs = fake_ingestor.extract.call_args.kwargs
    assert isinstance(extract_kwargs["text_params"], TextChunkParams)
    assert isinstance(extract_kwargs["html_params"], HtmlChunkParams)
    assert "Ingested 5 file(s) → 5 row(s)" in result.output


def test_root_ingest_directory_discovers_text_extensions_case_insensitively(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "README.MD"
    document.write_text("# Heading\n", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document.resolve())],)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_expands_documented_plain_text_glob(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    nested = tmp_path / "nested"
    nested.mkdir()
    script = nested / "setup.sh"
    script.write_text("#!/bin/sh\necho hello\n", encoding="utf-8")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(tmp_path / "**" / "*.sh")])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(script)],)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_reports_os_errors(monkeypatch) -> None:
    def fail_resolve_ingest_plan(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(ingest_cli_graph, "resolve_ingest_plan", fail_resolve_ingest_plan)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", "blocked.pdf"])

    assert result.exit_code == 1
    assert "Error: permission denied" in result.output


def test_root_cli_error_handler_includes_pydantic_validation_error() -> None:
    assert ValidationError in cli_shared.ROOT_CLI_ERRORS


def test_resolve_ingest_plan_validates_run_mode_before_creating_ingestor(
    monkeypatch,
) -> None:
    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for an invalid run mode")

    monkeypatch.setattr(ingest_execution, "create_ingestor", fail_create_ingestor)

    with pytest.raises(ValueError, match="run_mode must be one of"):
        ingest_plan.resolve_ingest_plan(
            ingest_plan.IngestPlanRequest(
                source=ingest_plan.IngestSourceOptions(documents=["ignored.pdf"]),
                runtime=ingest_plan.IngestRuntimeOptions(run_mode="parallel"),  # type: ignore[arg-type]
            )
        )


def test_silence_noisy_libraries_sets_env_vars(monkeypatch) -> None:
    for var in (
        "VLLM_LOGGING_LEVEL",
        "TRANSFORMERS_VERBOSITY",
        "HF_HUB_VERBOSITY",
        "TQDM_DISABLE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
    ):
        monkeypatch.delenv(var, raising=False)

    cli_shared.silence_noisy_libraries()

    assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["TQDM_DISABLE"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert logging.getLogger("vllm").level == logging.ERROR
    assert logging.getLogger("transformers").level == logging.ERROR


def test_quiet_capture_swallows_output_on_success(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with cli_shared.quiet_capture():
        sys.stdout.write("noisy stdout\n")
        sys.stdout.flush()
        sys.stderr.write("noisy stderr\n")
        sys.stderr.flush()

    captured = capfd.readouterr()
    assert "noisy stdout" not in captured.out
    assert "noisy stderr" not in captured.err


def test_quiet_capture_flushes_captured_output_to_stderr_on_error(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with cli_shared.quiet_capture():
            sys.stdout.write("about to fail\n")
            sys.stdout.flush()
            sys.stderr.write("diagnostic detail\n")
            sys.stderr.flush()
            raise RuntimeError("boom")

    captured = capfd.readouterr()
    # Both stdout and stderr output from the failing block are surfaced on
    # stderr so an operator/agent can debug the failure.
    assert "about to fail" in captured.err
    assert "diagnostic detail" in captured.err
    assert captured.out == ""


def test_root_ingest_quiet_invokes_silencing_and_capture(monkeypatch, tmp_path) -> None:
    import contextlib

    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "quiet.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 3)

    silenced: list[bool] = []
    monkeypatch.setattr(ingest_cli_shared, "silence_noisy_libraries", lambda: silenced.append(True))

    captured_use: list[bool] = []

    @contextlib.contextmanager
    def fake_quiet_capture() -> Any:
        captured_use.append(True)
        yield

    monkeypatch.setattr(ingest_cli_shared, "quiet_capture", fake_quiet_capture)

    result = RUNNER.invoke(cli_main.app, ["ingest", "local", str(document), "--quiet"])

    assert result.exit_code == 0
    assert silenced == [True]
    assert captured_use == [True]
    assert "Ingested 1 file(s) → 3 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_index_mode_hybrid_passes_hybrid_into_vdb_kwargs(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake_ingestor)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_, **__: 1)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(doc),
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
            "--index-mode",
            "hybrid",
        ],
    )

    assert result.exit_code == 0
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/lancedb",
        "table_name": "docs",
        "overwrite": True,
        "hybrid": True,
        "embedding_model_name": "nvidia/nemotron-3-embed-1b",
    }


@pytest.mark.parametrize("flag", ["--hybrid", "--sparse"])
def test_root_ingest_rejects_deprecated_index_mode_aliases(tmp_path, flag: str) -> None:
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", str(doc), flag])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_root_ingest_no_dedup_passes_all_false_opt_out(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(doc), "--caption", "--no-dedup"])

    assert result.exit_code == 0, result.output
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "dedup",
        "caption",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    fake_ingestor.dedup.assert_called_once()
    dedup_params = fake_ingestor.dedup.call_args.args[0]
    assert dedup_params.content_hash is False
    assert dedup_params.bbox_iou is False
    fake_ingestor.caption.assert_called_once()


def test_root_ingest_default_builds_vector_and_fts_table(monkeypatch, tmp_path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nemo_retriever.common.vdb.lancedb import LanceDB

    fake_ingestor = _make_fake_ingestor()
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")
    db_path = tmp_path / "db"
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [1.0, 0.0],
                    "content": "alpha hybrid manual",
                    "content_metadata": {"id": "alpha", "page_number": 1},
                    "source_metadata": {"source_id": str(doc)},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.0, 1.0],
                    "content": "beta hybrid manual",
                    "content_metadata": {"id": "beta", "page_number": 2},
                    "source_metadata": {"source_id": str(doc)},
                },
            },
        ]
    ]

    def write_with_real_lancedb(params) -> Any:
        LanceDB(
            **params.vdb_kwargs,
            vector_dim=2,
            num_partitions=1,
            num_sub_vectors=1,
        ).run(records)
        return fake_ingestor

    fake_ingestor.vdb_upload.side_effect = write_with_real_lancedb
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake_ingestor)
    monkeypatch.setattr(
        ingest_execution,
        "_count_lancedb_rows",
        lambda uri, table_name: lancedb.connect(uri).open_table(table_name).count_rows(),
    )

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", str(doc), "--lancedb-uri", str(db_path), "--table-name", "hybrid_docs"],
    )

    assert result.exit_code == 0, result.output
    table = lancedb.connect(str(db_path)).open_table("hybrid_docs")
    assert "vector" in table.schema.names
    index_names = {index.name.lower() for index in table.list_indices()}
    assert any("text" in name or "fts" in name for name in index_names)


def test_root_ingest_index_mode_sparse_skips_embedding_and_writes_fts_table(monkeypatch, tmp_path) -> None:
    lancedb = pytest.importorskip("lancedb")
    fake_ingestor = _make_fake_ingestor()
    doc = tmp_path / "a.pdf"
    doc.write_bytes(b"%PDF-1.4\n")
    fake_ingestor.ingest.return_value = [
        {
            "text": "alpha sparse manual",
            "metadata": {
                "content_metadata": {"id": "alpha", "page_number": 1, "type": "text"},
                "source_metadata": {"source_id": str(doc)},
            },
        }
    ]

    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(doc),
            "--lancedb-uri",
            str(tmp_path / "db"),
            "--table-name",
            "sparse_docs",
            "--index-mode",
            "sparse",
        ],
    )

    assert result.exit_code == 0, result.output
    fake_ingestor.embed.assert_not_called()
    fake_ingestor.vdb_upload.assert_not_called()

    table = lancedb.connect(str(tmp_path / "db")).open_table("sparse_docs")
    assert "vector" not in table.schema.names
    assert table.schema.metadata[b"retrieval_mode"] == b"sparse"
    assert table.schema.metadata[b"nemo_retriever.retrieval_mode"] == b"sparse"
    assert b"nemo_retriever.embedding_model_name" not in table.schema.metadata
    index_names = {index.name.lower() for index in table.list_indices()}
    assert any("text" in name or "fts" in name for name in index_names)
