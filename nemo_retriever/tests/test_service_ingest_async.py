# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`ServiceIngestor.ingest_async` artifact-flag wiring.

The async wrapper accepts ``return_failures`` / ``return_traces`` from
the public ingestor interface; these tests pin the contract that the
resolved future returns the same tuple/list shape the synchronous
:meth:`ServiceIngestor.ingest` produces with the same flags.

The flags themselves are also re-checked at the sync layer so that
regressions show up on the closer surface (``ingest()``) before they
hit the ``ingest_async()`` glue.
"""

from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import patch

import pytest

from nemo_retriever.common.params import IngestExecuteParams
from nemo_retriever.service.service_ingestor import ServiceIngestor, ServiceIngestResult


# ----------------------------------------------------------------------
# Fixture: a stub ingest_stream that exercises both success and failure
# event paths without touching the HTTP transport.
# ----------------------------------------------------------------------


def _stub_event_sequence() -> list[dict[str, Any]]:
    return [
        {
            "event": "job_created",
            "job_id": "JOB-1",
            "expected_documents": 2,
            "trace_id": "trace-123",
        },
        {"event": "upload_complete", "filename": "a.pdf", "document_id": "doc-a"},
        {"event": "upload_complete", "filename": "b.pdf", "document_id": "doc-b"},
        {
            "event": "document_complete",
            "document_id": "doc-a",
            "status": "completed",
        },
        {
            "event": "document_complete",
            "document_id": "doc-b",
            "status": "failed",
            "error": "boom",
        },
        {"event": "job_partial", "job_id": "JOB-1"},
    ]


def _fake_materialize_completed_document(
    self: ServiceIngestor,
    document_id: str,
    *,
    return_results: bool,
    client: Any = None,
) -> list[dict[str, Any]] | None:
    if not return_results and self._save_to_disk_dir is None:
        return None
    rows = [
        {
            "path": f"/uploads/{document_id}.pdf",
            "page_number": 1,
            "text": f"content-{document_id}",
            "metadata": {"source_id": document_id},
        }
    ]
    if self._save_to_disk_dir is not None:
        self._write_result_data_to_disk(document_id, rows)
    return rows if return_results else None


@pytest.fixture
def stub_ingestor() -> Iterator[ServiceIngestor]:
    """A ``ServiceIngestor`` whose stream yields a fixed event sequence."""
    ing = ServiceIngestor(base_url="http://example:7670").files(["a.pdf", "b.pdf"])
    events = _stub_event_sequence()
    stream_calls: list[dict[str, Any]] = []
    setattr(ing, "_stream_calls", stream_calls)

    def _fake_stream(
        self: ServiceIngestor,
        *,
        retain_results: bool = False,
        result_schema: str = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> Iterator[dict[str, Any]]:
        stream_calls.append(
            {
                "retain_results": retain_results,
                "result_schema": result_schema,
                "return_embeddings": return_embeddings,
                "return_images": return_images,
            }
        )
        return iter(events)

    with (
        patch.object(ServiceIngestor, "ingest_stream", _fake_stream),
        patch.object(ServiceIngestor, "_materialize_completed_document", _fake_materialize_completed_document),
    ):
        yield ing


# ----------------------------------------------------------------------
# Sync surface
# ----------------------------------------------------------------------


def test_ingest_default_returns_service_ingest_result(stub_ingestor: ServiceIngestor) -> None:
    """Default flags preserve completion metadata without fetching row payloads."""
    result = stub_ingestor.ingest()
    assert isinstance(result, ServiceIngestResult)
    assert not isinstance(result, tuple)
    assert result.job_id == "JOB-1"
    assert result.job_status == "partial_success"
    # Both completed and failed document_complete events land in the
    # ServiceIngestResult list; only the failed ones are mirrored on
    # ``.failures``.
    assert len(result) == 2
    assert result.failures == [("doc-b", "boom")]
    assert result.document_filenames == {"doc-a": "a.pdf", "doc-b": "b.pdf"}
    assert result.dataframe is None
    assert getattr(stub_ingestor, "_stream_calls")[-1]["retain_results"] is False


def test_ingest_explicit_results_preserve_dataframe(stub_ingestor: ServiceIngestor) -> None:
    with pytest.warns(DeprecationWarning, match="legacy result rows are deprecated"):
        result = stub_ingestor.ingest(return_results=True)
    assert result.dataframe is not None
    assert len(result.dataframe) == 1
    assert "document_id" not in result.dataframe.columns
    assert result.dataframe.iloc[0]["text"] == "content-doc-a"


def test_ingest_default_exposes_trace_id_from_job_created(stub_ingestor: ServiceIngestor) -> None:
    result = stub_ingestor.ingest(result_schema="compact")
    assert isinstance(result, ServiceIngestResult)
    assert result.trace_id == "trace-123"


def test_ingest_return_failures_returns_tuple(stub_ingestor: ServiceIngestor) -> None:
    result, failures = stub_ingestor.ingest(return_failures=True, result_schema="compact")
    assert isinstance(result, ServiceIngestResult)
    assert isinstance(failures, list)
    assert failures == [("doc-b", "boom")]
    assert failures == list(result.failures)


def test_ingest_return_traces_returns_tuple_with_all_events(stub_ingestor: ServiceIngestor) -> None:
    result, traces = stub_ingestor.ingest(return_traces=True, result_schema="compact")
    assert isinstance(result, ServiceIngestResult)
    assert isinstance(traces, list)
    assert traces == _stub_event_sequence()
    assert traces[0]["trace_id"] == "trace-123"


def test_ingest_both_flags_returns_three_tuple(stub_ingestor: ServiceIngestor) -> None:
    out = stub_ingestor.ingest(return_failures=True, return_traces=True, result_schema="compact")
    assert isinstance(out, tuple)
    assert len(out) == 3
    result, failures, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]
    assert traces == _stub_event_sequence()


def test_ingest_reads_flags_from_params_model(stub_ingestor: ServiceIngestor) -> None:
    """``IngestExecuteParams`` is the public params object for this method."""
    params = IngestExecuteParams(return_failures=True, return_traces=True, result_schema="compact")
    out = stub_ingestor.ingest(params=params)
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_ingest_kwargs_take_precedence_over_params(stub_ingestor: ServiceIngestor) -> None:
    """Explicit kwargs win over fields on the ``params`` model."""
    params = IngestExecuteParams(return_failures=True, return_traces=True)
    out = stub_ingestor.ingest(params=params, return_failures=False, return_traces=False, result_schema="compact")
    assert isinstance(out, ServiceIngestResult)
    assert not isinstance(out, tuple)


def test_ingest_ignores_unrelated_kwargs(stub_ingestor: ServiceIngestor) -> None:
    """Service run_mode silently drops execute-time knobs it cannot honour."""
    out = stub_ingestor.ingest(show_progress=True, parallel=True, max_workers=4, result_schema="compact")
    assert isinstance(out, ServiceIngestResult)


def test_ingest_return_results_false_skips_dataframe(stub_ingestor: ServiceIngestor) -> None:
    result = stub_ingestor.ingest(return_results=False)
    assert isinstance(result, ServiceIngestResult)
    assert result.dataframe is None


def test_ingest_return_results_reads_from_params_model(stub_ingestor: ServiceIngestor) -> None:
    params = IngestExecuteParams(return_results=False)
    result = stub_ingestor.ingest(params=params)
    assert result.dataframe is None


def test_ingest_return_results_kwargs_override_params(stub_ingestor: ServiceIngestor) -> None:
    params = IngestExecuteParams(return_results=False)
    result = stub_ingestor.ingest(params=params, return_results=True, result_schema="compact")
    assert result.dataframe is not None
    assert len(result.dataframe) == 1


def test_ingest_forwards_bulk_result_flags_from_kwargs(stub_ingestor: ServiceIngestor) -> None:
    with pytest.warns(DeprecationWarning, match="legacy result rows are deprecated"):
        stub_ingestor.ingest(return_results=True, return_embeddings=True, return_images=True)

    stream_calls = getattr(stub_ingestor, "_stream_calls")
    assert stream_calls[-1]["return_embeddings"] is True
    assert stream_calls[-1]["return_images"] is True


def test_ingest_forwards_bulk_result_flags_from_params_model(stub_ingestor: ServiceIngestor) -> None:
    params = IngestExecuteParams(return_results=True, return_embeddings=True, return_images=True)
    with pytest.warns(DeprecationWarning, match="legacy result rows are deprecated"):
        stub_ingestor.ingest(params=params)

    stream_calls = getattr(stub_ingestor, "_stream_calls")
    assert stream_calls[-1]["return_embeddings"] is True
    assert stream_calls[-1]["return_images"] is True


# ----------------------------------------------------------------------
# Async-future surface (the originally reported defect)
# ----------------------------------------------------------------------


def test_ingest_async_forwards_return_failures(stub_ingestor: ServiceIngestor) -> None:
    future = stub_ingestor.ingest_async(return_failures=True, result_schema="compact")
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    result, failures = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]


def test_ingest_async_forwards_return_traces(stub_ingestor: ServiceIngestor) -> None:
    future = stub_ingestor.ingest_async(return_traces=True, result_schema="compact")
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    result, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert traces == _stub_event_sequence()


def test_ingest_async_forwards_both_flags(stub_ingestor: ServiceIngestor) -> None:
    """Regression for the reported F105 defect.

    Prior to the fix, ``ingest_async`` deleted the flags before
    submitting ``self.ingest``, so the future always resolved to a
    plain :class:`ServiceIngestResult`.
    """
    future = stub_ingestor.ingest_async(return_failures=True, return_traces=True, result_schema="compact")
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    assert len(out) == 3
    result, failures, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]
    assert traces == _stub_event_sequence()


def test_ingest_async_default_matches_ingest_default(stub_ingestor: ServiceIngestor) -> None:
    """No flags ⇒ future resolves to a plain :class:`ServiceIngestResult`."""
    future = stub_ingestor.ingest_async(result_schema="compact")
    out = future.result(timeout=5.0)
    assert isinstance(out, ServiceIngestResult)
    assert not isinstance(out, tuple)
    assert out.dataframe is None


@pytest.mark.parametrize("return_results", [False, True, None])
def test_ingest_async_forwards_return_results(stub_ingestor: ServiceIngestor, return_results) -> None:
    future = stub_ingestor.ingest_async(return_results=return_results, result_schema="compact")
    out = future.result(timeout=5.0)
    assert isinstance(out, ServiceIngestResult)
    assert (out.dataframe is not None) is bool(return_results)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"return_results": None},
        {"return_results": False},
        {"params": {}},
        {"params": IngestExecuteParams()},
        {"params": IngestExecuteParams().model_dump()},
        {"params": IngestExecuteParams(return_results=True), "return_results": None},
        {"params": {"return_results": True}, "return_results": False},
    ],
)
def test_ingest_default_creates_no_result_client(stub_ingestor: ServiceIngestor, options) -> None:
    with patch.object(
        stub_ingestor,
        "_new_result_fetch_client",
        side_effect=AssertionError("result client must not be created"),
    ):
        result = stub_ingestor.ingest(**options)

    assert result.dataframe is None


def test_ingest_default_still_fetches_disk_output(stub_ingestor: ServiceIngestor, tmp_path) -> None:
    import json

    stub_ingestor.save_to_disk(output_directory=str(tmp_path), compression=None)
    result = stub_ingestor.ingest(result_schema="compact")

    assert result.dataframe is None
    assert getattr(stub_ingestor, "_stream_calls")[-1]["retain_results"] is True
    saved = json.loads((tmp_path / "doc-a.json").read_text())
    assert saved["rows"][0]["text"] == "content-doc-a"
