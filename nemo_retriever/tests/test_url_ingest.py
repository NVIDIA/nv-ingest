# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading

import httpx
import pandas as pd
import pytest

from nemo_retriever.common.params import UrlFetchParams
from nemo_retriever.common.url_fetch import (
    FetchedUrl,
    UrlFetchFailure,
    cleanup_fetched_urls,
    fetch_urls,
    normalize_urls,
    restore_url_source_value,
)
from nemo_retriever.ingestor.branch_extraction import _driver_local_files_to_ray_dataset, restore_source_urls
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.service.client import FileUpload
from nemo_retriever.service.service_ingestor import ServiceIngestor
from nemo_retriever.service.services.pipeline_executor import _merge_document_metadata


PDF_URL = "https://example.test/document"
PDF_BYTES = b"%PDF-1.7\n"


def test_url_fetch_params_defaults() -> None:
    params = UrlFetchParams()

    assert params.request_timeout_s == 30.0
    assert params.follow_redirects is True
    assert params.max_response_bytes == 10_000_000
    assert params.max_concurrency == 8


@pytest.mark.parametrize("value", ["relative/path.pdf", "file:///tmp/a.pdf", "", "   "])
def test_normalize_urls_rejects_non_http_sources(value: str) -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)|nonempty"):
        normalize_urls(value)


def test_fetch_urls_dispatches_extensionless_pdf_from_content_type(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES)

    # Exercise the classifier without relying on external networking while
    # retaining httpx's real streaming response behavior.
    from nemo_retriever.common import url_fetch

    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer secret"}) as client:
        outcome = url_fetch._fetch_one(
            client, PDF_URL, 0, UrlFetchParams(headers={"Authorization": "Bearer secret"}), tmp_path
        )

    assert isinstance(outcome, FetchedUrl)
    assert outcome.input_type == "pdf"
    assert outcome.classification_filename.endswith(".pdf")
    assert outcome.local_path.read_bytes() == PDF_BYTES
    cleanup_fetched_urls([outcome])


def test_fetch_urls_collects_http_and_size_failures(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing":
            return httpx.Response(404, content=b"missing")
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 11)

    real_client = httpx.Client

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    fetched, failures = fetch_urls(
        ["https://example.test/missing", "https://example.test/large"],
        UrlFetchParams(max_response_bytes=10),
    )

    assert fetched == []
    assert [failure.url for failure in failures] == [
        "https://example.test/missing",
        "https://example.test/large",
    ]
    assert "404" in failures[0].message
    assert "max_response_bytes=10" in failures[1].message


def test_graph_ingestor_returns_url_fetch_failures(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 404")
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    result, failures = GraphIngestor().urls(PDF_URL).extract().ingest(return_failures=True)

    assert result.empty
    assert failures == [(PDF_URL, "HTTP 404")]


@pytest.fixture(params=[False, True], ids=["system-temp", "symlinked-temp"])
def url_spool_temp_root(request, monkeypatch, tmp_path):
    if not request.param:
        yield
        return
    # Exercise real mkdtemp/NamedTemporaryFile through a symlink, as on macOS.
    with tempfile.TemporaryDirectory(prefix="nrl-url-test-") as directory:
        alias = tmp_path / "temp-alias"
        alias.symlink_to(directory, target_is_directory=True)
        monkeypatch.setattr(tempfile, "tempdir", str(alias))
        yield


def test_graph_ingestor_fetches_extensionless_html_through_markitdown(monkeypatch, url_spool_temp_root) -> None:
    url = "https://example.test/article"
    html = b"<html><body><h1>URL acceptance</h1><p>Converted by MarkItDown.</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html)

    real_client = httpx.Client

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(
        "nemo_retriever.common.modality.html.convert._get_txt_tokenizer",
        lambda *args, **kwargs: _TinyTokenizer(),
    )

    result, failures = GraphIngestor(run_mode="inprocess").urls(url).extract().ingest(return_failures=True)

    assert failures == []
    assert result["path"].tolist() == [url]
    assert result.iloc[0]["metadata"]["source_path"] == url
    markdown = result.iloc[0]["text"]
    assert "# URL acceptance" in markdown
    assert "Converted by MarkItDown." in markdown


def test_real_url_spool_paths_match_batch_provenance(monkeypatch, url_spool_temp_root) -> None:
    real_client = httpx.Client

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"URL text")
        )
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    ingestor = GraphIngestor(run_mode="batch").urls(PDF_URL)
    try:
        ingestor._prepare_url_inputs()
        path = ingestor._url_documents()[0]
        assert path == str(Path(path).resolve())
        assert ingestor._url_source_map == {path: PDF_URL}
        frame = pd.DataFrame([{"path": str(Path(path).resolve()), "metadata": {"source_path": path}}])
        restored = restore_source_urls(frame, source_map=ingestor._url_source_map)
        assert restored.iloc[0]["path"] == PDF_URL
        assert restored.iloc[0]["metadata"]["source_path"] == PDF_URL
    finally:
        ingestor._cleanup_url_inputs()
    assert not Path(path).exists()


def test_graph_ingestor_raises_url_fetch_failures_by_default(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 500")
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    with pytest.raises(GraphIngestionError, match="HTTP 500"):
        GraphIngestor().urls(PDF_URL).extract().ingest()


def test_service_collect_inputs_builds_url_upload(monkeypatch, tmp_path) -> None:
    local_path = tmp_path / "url.pdf"
    local_path.write_bytes(PDF_BYTES)
    fetched = FetchedUrl(
        url=PDF_URL,
        local_path=local_path,
        content_type="application/pdf",
        classification_filename="url-00000000.pdf",
        input_type="pdf",
        transport_path="url-source://00000000/url-00000000.pdf",
    )
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    inputs = ServiceIngestor().urls(PDF_URL)._collect_inputs()

    assert len(inputs) == 1
    upload = inputs[0]
    assert isinstance(upload, FileUpload)
    assert upload.path == local_path
    assert upload.classification_filename == "url-00000000.pdf"
    assert upload.metadata == {"_nrl_source_url": PDF_URL}


def test_service_metadata_restores_original_url() -> None:
    result = pd.DataFrame(
        [
            {
                "path": "url-source://00000000/url-00000000.pdf",
                "metadata": {
                    "source_path": "url-source://00000000/url-00000000.pdf",
                    "content_metadata": {"type": "text"},
                },
            }
        ]
    )

    _merge_document_metadata(
        result,
        {"_nrl_source_url": PDF_URL, "tenant": "test"},
        source_identifier="url-source://00000000/url-00000000.pdf",
    )

    assert result.iloc[0]["path"] == PDF_URL
    assert result.iloc[0]["metadata"]["source_path"] == PDF_URL
    assert result.iloc[0]["metadata"]["content_metadata"]["tenant"] == "test"
    assert "_nrl_source_url" not in result.iloc[0]["metadata"]["content_metadata"]


def test_service_ingest_returns_fetch_failure_without_contacting_service(monkeypatch) -> None:
    failure = UrlFetchFailure(PDF_URL, "HTTPStatusError", "HTTP 401")
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([], [failure]),
    )

    ingestor = ServiceIngestor(base_url="https://service.invalid").urls(PDF_URL)
    ingestor._document_ids = ["old-document"]
    result, failures = ingestor.ingest(return_failures=True)

    assert result.document_ids == []
    assert failures[0][0] == PDF_URL
    assert "HTTP 401" in failures[0][1]


class _TinyTokenizer:
    def __init__(self) -> None:
        self._text = ""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self._text = text
        return list(range(len(text)))

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(self._text[index] for index in ids)


def _fetched_text(tmp_path, *, url: str = PDF_URL) -> FetchedUrl:
    local_path = tmp_path / "url.txt"
    local_path.write_text("URL ingestion works", encoding="utf-8")
    return FetchedUrl(
        url=url,
        local_path=local_path,
        content_type="text/plain",
        classification_filename="url-00000000.txt",
        input_type="txt",
        transport_path="url-source://00000000/url-00000000.txt",
    )


def test_redirect_final_url_supplies_supported_suffix() -> None:
    from nemo_retriever.common import url_fetch

    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://cdn.example.test/report.pdf"),
        headers={"content-type": "application/octet-stream"},
    )

    _, filename, input_type, _ = url_fetch._classify_response(PDF_URL, response, 0)

    assert filename.endswith(".pdf")
    assert input_type == "pdf"


def test_unexpected_fetch_error_is_logged_and_reraised(monkeypatch, tmp_path, caplog) -> None:
    from nemo_retriever.common import url_fetch

    def raise_bug(*args):
        raise RuntimeError("bug")

    monkeypatch.setattr(url_fetch, "_classify_response", raise_bug)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"body"))) as client:
        with pytest.raises(RuntimeError, match="bug"), caplog.at_level("ERROR"):
            url_fetch._fetch_one(client, PDF_URL, 0, UrlFetchParams(), tmp_path)

    assert "Unexpected URL fetch failure at input position 0" in caplog.text


def test_fetch_urls_cleans_spool_on_unexpected_failure(monkeypatch, tmp_path) -> None:
    from nemo_retriever.common import url_fetch

    spool_dir = tmp_path / "url-spool"

    def make_spool_dir(*, prefix):
        assert prefix == "nrl-url-fetch-"
        spool_dir.mkdir()
        return str(spool_dir)

    def raise_bug(*args):
        raise RuntimeError("bug")

    monkeypatch.setattr(url_fetch.tempfile, "mkdtemp", make_spool_dir)
    monkeypatch.setattr(url_fetch, "_fetch_one", raise_bug)

    with pytest.raises(RuntimeError, match="bug"):
        fetch_urls([PDF_URL], UrlFetchParams())

    assert not spool_dir.exists()


@pytest.mark.parametrize("ingestor_type", [GraphIngestor, ServiceIngestor])
def test_parameterless_url_append_retains_fetch_settings(ingestor_type) -> None:
    ingestor = ingestor_type().urls(PDF_URL, headers={"Authorization": "Bearer token"}, max_concurrency=2)

    ingestor.urls("https://example.test/second.pdf")

    assert ingestor._url_fetch_params.headers == {"Authorization": "Bearer token"}
    assert ingestor._url_fetch_params.max_concurrency == 2


def test_url_page_provenance_preserves_suffix() -> None:
    transport = "url-source://00000000/url-00000000.pdf"

    assert restore_url_source_value(f"{transport}_1", {transport: PDF_URL}) == f"{PDF_URL}_1"


def test_graph_ingestor_successfully_ingests_fetched_text(monkeypatch, tmp_path) -> None:
    fetched = _fetched_text(tmp_path)
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer",
        lambda *args, **kwargs: _TinyTokenizer(),
    )

    result, failures = GraphIngestor(run_mode="inprocess").urls(PDF_URL).extract().ingest(return_failures=True)

    assert failures == []
    assert result["path"].tolist() == [PDF_URL]
    assert result.iloc[0]["metadata"]["source_path"] == PDF_URL
    assert not fetched.local_path.exists()


def test_batch_url_path_restores_provenance_before_post_stages(monkeypatch, tmp_path) -> None:
    fetched = _fetched_text(tmp_path)
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    def execute(executor):
        assert executor.driver_local_paths == {str(fetched.local_path)}
        frame = pd.DataFrame([{"path": str(fetched.local_path), "metadata": {"source_path": str(fetched.local_path)}}])
        return restore_source_urls(frame, source_map=executor.source_map)

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.ExtractionBranchExecutor.execute", execute)

    result = GraphIngestor(run_mode="batch").urls(PDF_URL).extract().ingest()

    assert result.iloc[0]["path"] == PDF_URL
    assert result.iloc[0]["metadata"]["source_path"] == PDF_URL


def test_batch_local_and_url_inputs_share_one_extraction_executor(monkeypatch, tmp_path) -> None:
    local_path = tmp_path / "local.txt"
    local_path.write_text("local", encoding="utf-8")
    fetched = _fetched_text(tmp_path)
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    class FakeDataset:
        def __init__(self) -> None:
            self.unioned = []

        def union(self, other):
            self.unioned.append(other)
            return self

        def schema(self, *, fetch_if_missing=False):
            return None

        def map_batches(self, *args, **kwargs):
            return self

    class FakeData:
        def read_binary_files(self, paths, *, include_paths):
            assert paths == [str(local_path)]
            assert include_paths is True
            return FakeDataset()

        def from_pandas_refs(self, refs):
            assert len(refs) == 1
            return FakeDataset()

    class FakeRay:
        def __init__(self) -> None:
            self.data = FakeData()

        def put(self, frame):
            assert frame.iloc[0]["path"] == str(fetched.local_path)
            return object()

    class FakeCluster:
        def available_cpu_count(self):
            return 4

        def available_gpu_count(self):
            return 1

        def total_cpu_count(self):
            return 4

        def total_gpu_count(self):
            return 1

    executor_reservations = []
    build_inputs = []

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self._source_cpu_reservation = kwargs["source_cpu_reservation"]
            executor_reservations.append(self._source_cpu_reservation)

        def build_dataset(self, data):
            build_inputs.append(data)
            return FakeDataset()

        def ingest(self, data):
            return pd.DataFrame({"done": [True]})

    def fake_preflight(executors, resources, *, reserved_cpus=0):
        assert len(executors) == 2
        assert reserved_cpus == 0

    monkeypatch.setattr(GraphIngestor, "_ensure_batch_runtime", lambda self: (FakeRay(), FakeCluster()))
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.RayDataExecutor", FakeExecutor)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.preflight_executors", fake_preflight)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_graph", lambda **kwargs: object())
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_post_extract_graph", lambda **kwargs: object())

    GraphIngestor(run_mode="batch").files(str(local_path)).urls(PDF_URL).extract().ingest()

    assert executor_reservations == [1, 0]
    assert len(build_inputs) == 1
    assert len(build_inputs[0].unioned) == 1


def test_explicit_pdf_mode_rejects_fetched_text(monkeypatch, tmp_path) -> None:
    fetched = _fetched_text(tmp_path)
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    with pytest.raises(ValueError, match="extraction_mode='pdf'"):
        GraphIngestor(run_mode="inprocess").urls(PDF_URL).extract(extraction_mode="pdf").ingest()


def test_service_ingest_upload_flow_restores_url_and_cleans_spool(monkeypatch, tmp_path) -> None:
    fetched = _fetched_text(tmp_path)
    observed = {}
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    async def stream(self, files, **kwargs):
        observed["upload"] = files[0]
        assert files[0].path.read_text(encoding="utf-8") == "URL ingestion works"
        yield {"event": "job_created", "job_id": "job-1"}
        yield {"event": "upload_complete", "filename": files[0].filename, "document_id": "doc-1"}
        yield {"event": "document_complete", "document_id": "doc-1", "status": "completed"}
        yield {"event": "job_finalized", "job_id": "job-1"}

    monkeypatch.setattr(
        "nemo_retriever.service.client.RetrieverServiceClient.aingest_documents_stream",
        stream,
    )

    result, failures = ServiceIngestor().urls(PDF_URL).ingest(return_results=False, return_failures=True)

    assert isinstance(observed["upload"], FileUpload)
    assert result.document_ids == ["doc-1"]
    assert result.document_filenames == {"doc-1": PDF_URL}
    assert failures == []
    assert not fetched.local_path.exists()


def test_service_upload_failure_uses_original_url(monkeypatch, tmp_path) -> None:
    fetched = _fetched_text(tmp_path)
    monkeypatch.setattr(
        "nemo_retriever.service.service_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    async def stream(self, files, **kwargs):
        yield {"event": "job_created", "job_id": "job-1"}
        yield {"event": "upload_failed", "filename": files[0].filename, "error": "rejected"}
        yield {"event": "job_failed", "job_id": "job-1"}

    monkeypatch.setattr(
        "nemo_retriever.service.client.RetrieverServiceClient.aingest_documents_stream",
        stream,
    )

    _, failures = ServiceIngestor().urls(PDF_URL).ingest(return_results=False, return_failures=True)

    assert failures == [(PDF_URL, "upload failed: rejected")]


def test_adding_url_does_not_bypass_local_explicit_mode_validation(monkeypatch, tmp_path) -> None:
    local_text = tmp_path / "local.txt"
    local_text.write_text("local", encoding="utf-8")
    fetched = _fetched_text(tmp_path, url="https://example.test/remote.txt")
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.fetch_urls",
        lambda urls, params: ([fetched], []),
    )

    with pytest.raises(ValueError, match="local.txt"):
        (
            GraphIngestor(run_mode="inprocess")
            .files(str(local_text))
            .urls("https://example.test/remote.txt")
            .extract(extraction_mode="pdf")
            .ingest()
        )


def test_malformed_content_disposition_falls_back_to_response_mime(tmp_path) -> None:
    from nemo_retriever.common import url_fetch

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
                "content-disposition": 'attachment; filename="//[report.pdf"',
            },
            content=PDF_BYTES,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        outcome = url_fetch._fetch_one(client, PDF_URL, 0, UrlFetchParams(), tmp_path)

    assert isinstance(outcome, FetchedUrl)
    assert outcome.input_type == "pdf"
    assert outcome.local_path.read_bytes() == PDF_BYTES
    cleanup_fetched_urls([outcome])


def test_driver_local_url_files_move_into_ray_object_store(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class _FakeData:
        def __init__(self) -> None:
            self.refs = []

        def from_pandas_refs(self, refs):
            self.refs = refs
            return "ray-dataset"

    class _FakeRay:
        def __init__(self) -> None:
            self.data = _FakeData()
            self.frames = []

        def put(self, frame):
            self.frames.append(frame.copy())
            return f"ref-{len(self.frames)}"

    ray_module = _FakeRay()

    dataset = _driver_local_files_to_ray_dataset(ray_module, [str(first), str(second)])

    assert dataset == "ray-dataset"
    assert ray_module.data.refs == ["ref-1", "ref-2"]
    assert [frame.iloc[0]["bytes"] for frame in ray_module.frames] == [b"first", b"second"]
    assert [frame.iloc[0]["path"] for frame in ray_module.frames] == [str(first), str(second)]


def test_async_service_cancellation_waits_for_fetch_cleanup(monkeypatch, tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    local_path = tmp_path / "cancelled.txt"

    def blocking_fetch(urls, params):
        started.set()
        assert release.wait(timeout=5)
        local_path.write_text("download completed after cancellation", encoding="utf-8")
        return (
            [
                FetchedUrl(
                    url=PDF_URL,
                    local_path=local_path,
                    content_type="text/plain",
                    classification_filename="url-00000000.txt",
                    input_type="txt",
                    transport_path="url-source://00000000/url-00000000.txt",
                )
            ],
            [],
        )

    monkeypatch.setattr("nemo_retriever.service.service_ingestor.fetch_urls", blocking_fetch)
    ingestor = ServiceIngestor().urls(PDF_URL)

    async def cancel_during_fetch() -> None:
        event_task = asyncio.create_task(anext(ingestor.aingest_stream()))
        assert await asyncio.to_thread(started.wait, 2)
        event_task.cancel()
        await asyncio.sleep(0.05)
        assert not event_task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await event_task

    asyncio.run(cancel_during_fetch())

    assert not local_path.exists()
    assert ingestor._fetched_urls == []


def test_async_service_cancellation_logs_input_collection_failure(monkeypatch, caplog) -> None:
    started = threading.Event()
    release = threading.Event()

    def failing_fetch(urls, params):
        started.set()
        assert release.wait(timeout=5)
        raise OSError("could not write URL spool")

    monkeypatch.setattr("nemo_retriever.service.service_ingestor.fetch_urls", failing_fetch)
    ingestor = ServiceIngestor().urls(PDF_URL)

    async def cancel_during_fetch() -> None:
        event_task = asyncio.create_task(anext(ingestor.aingest_stream()))
        assert await asyncio.to_thread(started.wait, 2)
        event_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await event_task

    with caplog.at_level("ERROR", logger="nemo_retriever.service.service_ingestor"):
        asyncio.run(cancel_during_fetch())

    records = [
        record
        for record in caplog.records
        if record.message == "Service input collection failed while cancellation was being handled"
    ]
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], OSError)
