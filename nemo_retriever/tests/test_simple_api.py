# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simple, verb-first API (:mod:`nemo_retriever.simple`).

These tests pin the *shape* of the friendly API: the verbs exist, they read
every supported media type, and each one wires through to the underlying
engine with the right, implementation-free defaults. The heavy engine is
stubbed so the tests stay fast and hardware-independent.
"""

from __future__ import annotations

import pytest

import nemo_retriever
from nemo_retriever import simple
from nemo_retriever.common.input_files import AUTO_INPUT_EXTENSIONS


class _RecordingIngestor:
    """Fluent stand-in for the real ingestor that records the verbs called."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args, **kwargs) -> "_RecordingIngestor":
        self.calls.append((name, args, kwargs))
        return self

    def files(self, *args, **kwargs):
        return self._record("files", *args, **kwargs)

    def extract(self, *args, **kwargs):
        return self._record("extract", *args, **kwargs)

    def extract_txt(self, *args, **kwargs):
        return self._record("extract_txt", *args, **kwargs)

    def extract_html(self, *args, **kwargs):
        return self._record("extract_html", *args, **kwargs)

    def extract_image_files(self, *args, **kwargs):
        return self._record("extract_image_files", *args, **kwargs)

    def extract_audio(self, *args, **kwargs):
        return self._record("extract_audio", *args, **kwargs)

    def extract_video(self, *args, **kwargs):
        return self._record("extract_video", *args, **kwargs)

    def embed(self, *args, **kwargs):
        return self._record("embed", *args, **kwargs)

    def vdb_upload(self, *args, **kwargs):
        return self._record("vdb_upload", *args, **kwargs)

    def ingest(self, *args, **kwargs):
        self._record("ingest", *args, **kwargs)
        return "RESULT"

    def method_names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


@pytest.fixture
def recorder(monkeypatch) -> _RecordingIngestor:
    """Replace ``create_ingestor`` with a call-recording fluent stub."""
    import nemo_retriever.ingestor as ingestor_pkg

    rec = _RecordingIngestor()
    monkeypatch.setattr(ingestor_pkg, "create_ingestor", lambda *a, **k: rec)
    return rec


# ---------------------------------------------------------------------------
# Supported media
# ---------------------------------------------------------------------------


def test_supported_media_covers_exactly_the_engines_file_types() -> None:
    covered: set[str] = set()
    for extensions in simple.supported_media().values():
        covered |= set(extensions)
    assert covered == set(AUTO_INPUT_EXTENSIONS)


def test_supported_media_returns_a_safe_copy() -> None:
    snapshot = simple.supported_media()
    snapshot["documents"] = ()
    assert simple.MEDIA_TYPES["documents"] == (".pdf", ".docx", ".pptx")


def test_media_families_are_the_expected_plain_language_names() -> None:
    assert set(simple.MEDIA_TYPES) == {
        "documents",
        "text",
        "web_pages",
        "images",
        "audio",
        "video",
    }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


_PUBLIC_VERBS = [
    "extract",
    "extract_documents",
    "extract_text",
    "extract_web_pages",
    "extract_images",
    "extract_audio",
    "extract_video",
    "ingest",
    "search",
    "ask",
    "supported_media",
    "MEDIA_TYPES",
]


@pytest.mark.parametrize("name", _PUBLIC_VERBS)
def test_verbs_are_part_of_the_public_surface(name: str) -> None:
    assert name in simple.__all__
    assert hasattr(simple, name)


def test_simple_module_is_reachable_from_the_package() -> None:
    # ``from nemo_retriever import simple`` is the documented entry point.
    assert nemo_retriever.simple is simple


# ---------------------------------------------------------------------------
# extract - reads every media type through the right door
# ---------------------------------------------------------------------------


def test_extract_reads_any_mix_and_returns_content(recorder) -> None:
    result = simple.extract("a.pdf")
    assert result == "RESULT"
    assert recorder.method_names() == ["files", "extract", "ingest"]
    # The generic verb lets the engine detect each file type on its own.
    _, _, extract_kwargs = recorder.calls[1]
    assert extract_kwargs == {}


@pytest.mark.parametrize(
    ("verb", "expected_builder"),
    [
        ("extract_text", "extract_txt"),
        ("extract_web_pages", "extract_html"),
        ("extract_images", "extract_image_files"),
        ("extract_audio", "extract_audio"),
        ("extract_video", "extract_video"),
    ],
)
def test_typed_extract_verbs_route_to_their_media_handler(recorder, verb, expected_builder) -> None:
    result = getattr(simple, verb)("input")
    assert result == "RESULT"
    assert recorder.method_names() == ["files", expected_builder, "ingest"]


def test_extract_documents_uses_the_document_reader(recorder) -> None:
    simple.extract_documents("manual.pdf")
    assert recorder.method_names() == ["files", "extract", "ingest"]
    _, _, extract_kwargs = recorder.calls[1]
    assert extract_kwargs == {"extraction_mode": "pdf"}


# ---------------------------------------------------------------------------
# ingest - makes content searchable
# ---------------------------------------------------------------------------


def test_ingest_prepares_content_without_saving_by_default(recorder) -> None:
    result = simple.ingest("reports/report.pdf")
    assert result == "RESULT"
    assert recorder.method_names() == ["files", "extract", "embed", "ingest"]


def test_ingest_into_saves_to_a_named_library(recorder) -> None:
    simple.ingest("reports/report.pdf", into="my-library")
    assert recorder.method_names() == ["files", "extract", "embed", "vdb_upload", "ingest"]
    _, _, upload_kwargs = recorder.calls[3]
    assert upload_kwargs == {"vdb_kwargs": {"table_name": "my-library"}}


# ---------------------------------------------------------------------------
# Source normalization - files, folders, patterns, and lists
# ---------------------------------------------------------------------------


def test_a_single_file_is_passed_through_unchanged(recorder) -> None:
    simple.extract("data/report.pdf")
    (_, files_args, _) = recorder.calls[0]
    assert files_args[0] == ["data/report.pdf"]


def test_a_folder_is_read_all_the_way_through(recorder, tmp_path) -> None:
    folder = tmp_path / "corpus"
    folder.mkdir()
    simple.extract(folder)
    (_, files_args, _) = recorder.calls[0]
    assert files_args[0] == [str(folder / "**" / "*")]


def test_a_list_of_sources_is_kept_together(recorder, tmp_path) -> None:
    folder = tmp_path / "corpus"
    folder.mkdir()
    simple.extract(["a.pdf", "b.mp3", folder])
    (_, files_args, _) = recorder.calls[0]
    assert files_args[0] == ["a.pdf", "b.mp3", str(folder / "**" / "*")]


# ---------------------------------------------------------------------------
# search / ask - use what was ingested
# ---------------------------------------------------------------------------


def test_search_looks_inside_the_named_library(monkeypatch) -> None:
    captured: dict = {}

    class _FakeRetriever:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def query(self, question, *, top_k):
            captured["query"] = (question, top_k)
            return ["hit-1", "hit-2"]

    monkeypatch.setattr("nemo_retriever.graph.retriever.Retriever", _FakeRetriever)

    hits = simple.search("my-library", "quarterly revenue", limit=3)

    assert hits == ["hit-1", "hit-2"]
    assert captured["init"] == {"vdb_kwargs": {"table_name": "my-library"}}
    assert captured["query"] == ("quarterly revenue", 3)


def test_ask_writes_an_answer_from_the_library(monkeypatch) -> None:
    captured: dict = {}

    class _FakeAnswer:
        answer = "Revenue grew 12%."

    class _FakeRetriever:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def answer(self, question, *, llm):
            captured["answer"] = (question, llm)
            return _FakeAnswer()

    class _FakeWriter:
        @classmethod
        def from_kwargs(cls, *, model):
            captured["model"] = model
            return cls()

    monkeypatch.setattr("nemo_retriever.graph.retriever.Retriever", _FakeRetriever)
    monkeypatch.setattr("nemo_retriever.models.llm.clients.litellm.LiteLLMClient", _FakeWriter, raising=False)

    reply = simple.ask("my-library", "How did revenue change?", model="some-model")

    assert reply == "Revenue grew 12%."
    assert captured["init"] == {"vdb_kwargs": {"table_name": "my-library"}}
    assert captured["model"] == "some-model"
    question, writer = captured["answer"]
    assert question == "How did revenue change?"
    assert isinstance(writer, _FakeWriter)
