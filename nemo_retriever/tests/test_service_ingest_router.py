# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Phase 1 tests for the ``/v1/ingest`` router.

We use a FastAPI ``TestClient`` against a standalone service whose pipeline
pool is initialised with a stub work-fn that records ``WorkItem`` instances.
This lets us assert:

* the validated ``PipelineSpec`` is attached to the work-item;
* policy denials produce HTTP 403 / 501 responses;
* ``/v1/ingest/pipeline-config`` exposes the ``allowed_overrides`` block.

The stub work-fn never imports ``nemo_retriever.ingestor.graph_ingestor`` so the test
runs without any GPU / Ray dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.common.schemas.collections import IngestOperation
from nemo_retriever.common.schemas.requests import IngestRequest
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    AuthConfig,
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
    VectorDbConfig,
)
from nemo_retriever.service import tracing
from nemo_retriever.service.routers.dashboard import VdbQueryRequest, vdb_query
from nemo_retriever.service.routers.ingest import _route_by_page_count
from nemo_retriever.service.services.job_tracker import get_job_tracker
from nemo_retriever.service.services.pipeline_pool import PoolType, WorkItem
from nemo_retriever.service.utils.file_type import FileCategory
from .conftest import create_test_job


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


class _CollectingExporter:
    def __init__(self, exported: list[Any]) -> None:
        self._exported = exported

    def export(self, spans: Any) -> SpanExportResult:
        self._exported.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


@pytest.fixture
def app_with_stub_pool(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Build a standalone-mode app whose pools record items instead of running pipelines."""

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, [{"id": item.id, "stub": True}]

    def _stub_realtime(_config: ServiceConfig):
        return _stub_work

    def _stub_batch(_config: ServiceConfig):
        return _stub_work

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        _stub_realtime,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        _stub_batch,
    )

    cfg = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def traced_app_with_stub_pool(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Build a traced standalone app with in-memory span export."""
    from nemo_retriever.service.tracing import _reset_tracing_for_tests

    exported: list[Any] = []
    _reset_tracing_for_tests()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, [{"id": item.id, "stub": True}]

    def _stub_realtime(_config: ServiceConfig):
        return _stub_work

    def _stub_batch(_config: ServiceConfig):
        return _stub_work

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        _stub_realtime,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        _stub_batch,
    )

    cfg = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    try:
        app = create_app(cfg)
        with TestClient(app) as client:
            yield client, exported
    finally:
        _reset_tracing_for_tests()


@pytest.fixture
def traced_gateway_app(monkeypatch: pytest.MonkeyPatch):
    """Build a traced gateway app so dashboard job APIs are mounted."""
    from nemo_retriever.service.tracing import _reset_tracing_for_tests

    exported: list[Any] = []
    _reset_tracing_for_tests()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)

    cfg = ServiceConfig(
        mode="gateway",
        auth=AuthConfig(allow_unscoped_dev=True),
    )
    try:
        app = create_app(cfg)
        with TestClient(app) as client:
            yield client, exported
    finally:
        _reset_tracing_for_tests()


def _make_pdf_bytes() -> bytes:
    """Return a 1-byte non-PDF payload — the worker is stubbed so content doesn't matter."""
    return b"%PDF-1.4\n%stub\n"


@pytest.mark.parametrize("category", [FileCategory.TEXT, FileCategory.HTML, FileCategory.IMAGE])
def test_non_pdf_categories_skip_pdf_page_count(monkeypatch: pytest.MonkeyPatch, category: FileCategory) -> None:
    monkeypatch.setattr(
        "nemo_retriever.service.routers.ingest._count_pdf_pages",
        lambda _: pytest.fail("non-PDF uploads must not be parsed as PDFs"),
    )

    assert _route_by_page_count(b"not a PDF", IngestRequest(), category) is PoolType.REALTIME


def _wait_for_items(captured_items: list[WorkItem], count: int) -> None:
    import time as _time

    deadline = _time.monotonic() + 5.0
    while len(captured_items) < count and _time.monotonic() < deadline:
        _time.sleep(0.05)


def test_ingest_without_spec_falls_back_to_legacy_pipeline(
    app_with_stub_pool: TestClient, captured_items: list[WorkItem]
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "document_id" in body
    assert body["document_id"] == body["attempt_id"]
    assert body["job_id"] == job_id

    status = app_with_stub_pool.get(f"/v1/ingest/job/{job_id}/document/{body['document_id']}")
    assert status.status_code in (200, 202), status.text

    # Wait briefly for the async worker loop to consume the queued item.
    import time as _time

    deadline = _time.monotonic() + 5.0
    while not captured_items and _time.monotonic() < deadline:
        _time.sleep(0.05)

    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is None  # legacy path
    assert item.job_id == job_id


def test_ingest_with_valid_spec_attaches_to_work_item(
    app_with_stub_pool: TestClient, captured_items: list[WorkItem]
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    metadata = {
        "pipeline": {
            "extraction_mode": "pdf",
            "extract_params": {"extract_text": False, "dpi": 300},
            "stage_order": ["extract"],
        }
    }
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 202, resp.text

    import time as _time

    deadline = _time.monotonic() + 5.0
    while not captured_items and _time.monotonic() < deadline:
        _time.sleep(0.05)

    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is not None
    assert item.pipeline_spec["extraction_mode"] == "pdf"
    assert item.pipeline_spec["extract_params"]["dpi"] == 300
    assert item.pipeline_spec["stage_order"] == ["extract"]


def test_ingest_rejects_trust_sensitive_override(
    app_with_stub_pool: TestClient,
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"extract_params": {"page_elements_invoke_url": "http://attacker/"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "trust-sensitive" in resp.json()["detail"]


def test_ingest_rejects_caption_when_endpoint_not_configured(
    app_with_stub_pool: TestClient,
) -> None:
    """Without ``nim_endpoints.caption_invoke_url``, caption overrides are 403."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"caption_params": {"prompt": "Describe"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "caption" in resp.json()["detail"].lower()


def test_ingest_rejects_webhook_when_sinks_disabled(
    app_with_stub_pool: TestClient,
) -> None:
    """Without ``sinks.webhook_url_prefixes`` set, the ``webhook`` stage is not allowed."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {
        "pipeline": {
            "webhook_params": {"endpoint_url": "http://x/"},
            "stage_order": ["webhook"],
        }
    }
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"].lower()
    assert "webhook" in detail
    assert "allowed_stages" in detail or "not in" in detail


def test_ingest_rejects_webhook_params_without_stage_when_sinks_disabled(
    app_with_stub_pool: TestClient,
) -> None:
    """Bare ``webhook_params`` (no stage_order entry) still fails the sink allowlist check."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"webhook_params": {"endpoint_url": "http://x/"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "disabled" in resp.json()["detail"].lower()


def test_create_job_returns_201_and_aggregate_fields(
    app_with_stub_pool: TestClient,
) -> None:
    """POST /v1/ingest/job opens a fresh aggregate with status=pending."""
    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 3, "label": "smoke"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["expected_documents"] == 3
    assert body["status"] == "pending"
    assert body["label"] == "smoke"
    assert body["job_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_documents": 1, "operation": "replace", "target_document_id": "doc"},
        {
            "expected_documents": 2,
            "collection_name": "research",
            "operation": "replace",
            "target_document_id": "doc",
        },
        {"expected_documents": 1, "collection_name": "research", "operation": "replace"},
        {"expected_documents": 1, "operation": "append", "target_document_id": "doc"},
    ],
)
def test_invalid_collection_job_is_rejected_before_registration(
    app_with_stub_pool: TestClient,
    payload: dict[str, Any],
) -> None:
    tracker = get_job_tracker()
    assert tracker is not None
    initial_jobs = len(tracker.all_jobs())

    response = app_with_stub_pool.post("/v1/ingest/job", json=payload)

    assert response.status_code == 422
    assert len(tracker.all_jobs()) == initial_jobs


def test_create_job_succeeds_when_tracing_span_setup_fails(
    app_with_stub_pool: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("tracer unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.get_tracer", _raise)

    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "label": "trace-failure"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["job_id"]
    assert body.get("trace_id") is None
    assert "x-trace-id" not in resp.headers


def test_create_job_succeeds_when_tracing_span_enter_fails(
    app_with_stub_pool: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingSpanContext:
        def __enter__(self) -> Any:
            raise RuntimeError("span enter failed")

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            raise AssertionError("span context should not be entered")

    class _Tracer:
        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            return _FailingSpanContext()

    monkeypatch.setattr("nemo_retriever.service.tracing.get_tracer", lambda: _Tracer())

    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "label": "trace-enter-failure"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["job_id"]
    assert body.get("trace_id") is None
    assert "x-trace-id" not in resp.headers


def test_create_job_succeeds_when_trace_context_injection_fails(
    app_with_stub_pool: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise() -> Any:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.inject_trace_context", _raise)

    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "label": "inject-failure"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["job_id"]
    assert body.get("trace_id") is None
    assert "x-trace-id" not in resp.headers


def test_document_upload_succeeds_when_accept_trace_context_extraction_fails(
    app_with_stub_pool: TestClient,
    captured_items: list[WorkItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = create_test_job(app_with_stub_pool)

    def _raise(_carrier: dict[str, str]) -> Any:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.extract_trace_context", _raise)

    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        headers={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )

    assert resp.status_code == 202, resp.text
    _wait_for_items(captured_items, 1)
    assert captured_items[0].job_id == job_id


def test_document_upload_succeeds_when_enqueue_trace_context_injection_fails(
    app_with_stub_pool: TestClient,
    captured_items: list[WorkItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = create_test_job(app_with_stub_pool)

    def _raise() -> Any:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.inject_trace_context", _raise)

    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )

    assert resp.status_code == 202, resp.text
    _wait_for_items(captured_items, 1)
    assert captured_items[0].job_id == job_id
    assert captured_items[0].trace_context == {}


def test_create_job_with_tracing_returns_trace_id_body_header_and_snapshot(
    traced_app_with_stub_pool: tuple[TestClient, list[Any]],
) -> None:
    client, exported_spans = traced_app_with_stub_pool

    resp = client.post(
        "/v1/ingest/job",
        json={"expected_documents": 2, "label": "trace-smoke"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    trace_id = body["trace_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)
    assert resp.headers["x-trace-id"] == trace_id

    snapshot = client.get(f"/v1/ingest/job/{body['job_id']}")
    assert snapshot.status_code == 200, snapshot.text
    assert snapshot.json()["trace_id"] == trace_id
    assert exported_spans


def test_create_job_with_tracing_uses_inbound_traceparent(
    traced_app_with_stub_pool: tuple[TestClient, list[Any]],
) -> None:
    client, exported_spans = traced_app_with_stub_pool

    with tracing.start_span("client.parent"):
        inbound_trace_id = tracing.current_trace_id_hex()
        carrier = dict(tracing.inject_trace_context())
    assert inbound_trace_id is not None

    resp = client.post(
        "/v1/ingest/job",
        headers=carrier,
        json={"expected_documents": 1, "label": "trace-parent"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["trace_id"] == inbound_trace_id
    assert resp.headers["x-trace-id"] == inbound_trace_id

    ingest_span = next(span for span in exported_spans if span.name == "ingest.job")
    assert f"{ingest_span.context.trace_id:032x}" == inbound_trace_id

    snapshot = client.get(f"/v1/ingest/job/{body['job_id']}")
    assert snapshot.status_code == 200, snapshot.text
    assert snapshot.json()["trace_id"] == inbound_trace_id


def test_job_upload_routes_emit_accept_spans(
    traced_app_with_stub_pool: tuple[TestClient, list[Any]],
    captured_items: list[WorkItem],
) -> None:
    client, exported_spans = traced_app_with_stub_pool
    job_id = create_test_job(client, expected_documents=3)

    document_resp = client.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    page_resp = client.post(
        f"/v1/ingest/job/{job_id}/page",
        files={"file": ("page.png", b"page", "image/png")},
        data={
            "document_id": "source-doc",
            "page_number": "1",
            "filename": "source.pdf",
        },
    )
    whole_resp = client.post(
        f"/v1/ingest/job/{job_id}/whole",
        files={"file": ("whole.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )

    assert document_resp.status_code == 202, document_resp.text
    assert page_resp.status_code == 202, page_resp.text
    assert whole_resp.status_code == 202, whole_resp.text
    assert document_resp.json()["document_id"] == document_resp.json()["attempt_id"]
    assert whole_resp.json()["document_id"] == whole_resp.json()["attempt_id"]
    _wait_for_items(captured_items, 3)

    names = {span.name for span in exported_spans}
    assert "ingest.document.accept" in names
    assert "ingest.page.accept" in names
    assert "ingest.whole.accept" in names

    accept_spans = {
        span.name: span
        for span in exported_spans
        if span.name in {"ingest.document.accept", "ingest.page.accept", "ingest.whole.accept"}
    }
    for span in accept_spans.values():
        attrs = dict(span.attributes)
        assert attrs["service.role"] == "standalone"
        assert attrs["job.id"] == job_id
        assert attrs["route"].startswith("/v1/ingest/job/")


def test_job_upload_accept_span_uses_job_trace_when_request_has_no_traceparent(
    traced_app_with_stub_pool: tuple[TestClient, list[Any]],
    captured_items: list[WorkItem],
) -> None:
    client, exported_spans = traced_app_with_stub_pool
    create_resp = client.post("/v1/ingest/job", json={"expected_documents": 1})
    assert create_resp.status_code == 201, create_resp.text
    job_id = create_resp.json()["job_id"]
    job_trace_id = create_resp.json()["trace_id"]

    upload_resp = client.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )

    assert upload_resp.status_code == 202, upload_resp.text
    _wait_for_items(captured_items, 1)

    accept_span = next(span for span in exported_spans if span.name == "ingest.document.accept")
    assert f"{accept_span.context.trace_id:032x}" == job_trace_id
    assert captured_items[0].trace_context["traceparent"].split("-")[1] == job_trace_id


def test_job_upload_accept_span_prefers_inbound_traceparent_over_job_trace(
    traced_app_with_stub_pool: tuple[TestClient, list[Any]],
    captured_items: list[WorkItem],
) -> None:
    client, exported_spans = traced_app_with_stub_pool
    create_resp = client.post("/v1/ingest/job", json={"expected_documents": 1})
    assert create_resp.status_code == 201, create_resp.text
    job_id = create_resp.json()["job_id"]
    job_trace_id = create_resp.json()["trace_id"]

    with tracing.start_span("client.parent"):
        inbound_trace_id = tracing.current_trace_id_hex()
        carrier = dict(tracing.inject_trace_context())
    assert inbound_trace_id is not None
    assert inbound_trace_id != job_trace_id

    upload_resp = client.post(
        f"/v1/ingest/job/{job_id}/document",
        headers=carrier,
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )

    assert upload_resp.status_code == 202, upload_resp.text
    _wait_for_items(captured_items, 1)

    accept_span = next(span for span in exported_spans if span.name == "ingest.document.accept")
    assert f"{accept_span.context.trace_id:032x}" == inbound_trace_id
    assert f"{accept_span.context.trace_id:032x}" != job_trace_id
    assert captured_items[0].trace_context["traceparent"].split("-")[1] == inbound_trace_id


def test_dashboard_job_views_include_trace_id(
    traced_gateway_app: tuple[TestClient, list[Any]],
) -> None:
    client, exported_spans = traced_gateway_app

    resp = client.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "label": "dashboard-trace"},
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    job_id = body["job_id"]
    trace_id = body["trace_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)

    snapshot = client.get("/v1/dashboard/api/jobs/snapshot")
    assert snapshot.status_code == 200, snapshot.text
    snapshot_jobs = {job["job_id"]: job for job in snapshot.json()["jobs"]}
    assert snapshot_jobs[job_id]["trace_id"] == trace_id

    listing = client.get("/v1/dashboard/api/jobs/list")
    assert listing.status_code == 200, listing.text
    listed_jobs = {job["job_id"]: job for job in listing.json()["jobs"]}
    assert listed_jobs[job_id]["trace_id"] == trace_id

    detail = client.get(f"/v1/dashboard/api/jobs/{job_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["trace_id"] == trace_id
    assert exported_spans


def test_dashboard_vdb_query_forwards_internal_token(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url: str, **kwargs):
            captured.update(url=url, **kwargs)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"results": []},
            )

    monkeypatch.setattr(
        "nemo_retriever.service.routers.dashboard.httpx.AsyncClient",
        FakeClient,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=ServiceConfig(
                    vectordb=VectorDbConfig(
                        enabled=True,
                        vectordb_url="http://vectordb",
                        internal_api_token="internal-secret",
                    )
                )
            )
        )
    )
    asyncio.run(
        vdb_query(
            VdbQueryRequest(query="retrieval"),
            request,
        )
    )

    assert captured["headers"] == {"X-NRL-Internal-Token": "internal-secret"}


def test_create_job_retain_results_persisted_on_aggregate(
    app_with_stub_pool: TestClient,
) -> None:
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "retain_results": True},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]
    tracker = get_job_tracker()
    assert tracker is not None
    agg = tracker.get_job(job_id)
    assert agg is not None
    assert agg.retain_results is True


def test_create_job_idempotency_includes_behavior_but_not_label(
    app_with_stub_pool: TestClient,
) -> None:
    payload = {
        "expected_documents": 1,
        "idempotency_key": "request-key",
        "retain_results": False,
        "label": "first label",
    }
    created = app_with_stub_pool.post("/v1/ingest/job", json=payload)
    assert created.status_code == 201, created.text

    replay = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={**payload, "label": "updated label"},
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["job_id"] == created.json()["job_id"]

    conflict = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={**payload, "retain_results": True},
    )
    assert conflict.status_code == 409, conflict.text


def test_create_job_rejects_invalid_collection_name_before_registration(
    app_with_stub_pool: TestClient,
) -> None:
    tracker = get_job_tracker()
    assert tracker is not None
    response = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "collection_name": "foo/documents"},
    )

    assert response.status_code == 422
    assert tracker.all_jobs() == []


def test_get_job_returns_aggregate_snapshot(app_with_stub_pool: TestClient) -> None:
    job_id = create_test_job(app_with_stub_pool, expected_documents=2)
    resp = app_with_stub_pool.get(f"/v1/ingest/job/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["expected_documents"] == 2
    assert body["status"] == "pending"
    assert body["documents"] is None  # not requested
    assert body["counts"] == {} or "pending" in body["counts"]


def test_get_job_missing_returns_404(app_with_stub_pool: TestClient) -> None:
    resp = app_with_stub_pool.get("/v1/ingest/job/does-not-exist")
    assert resp.status_code == 404


def test_upload_to_missing_job_returns_404(app_with_stub_pool: TestClient) -> None:
    resp = app_with_stub_pool.post(
        "/v1/ingest/job/does-not-exist/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.anyio
async def test_rejected_standalone_submission_unregisters_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.routers import ingest

    unregistered: list[str] = []

    async def _reject(_pool_type: PoolType, _item: WorkItem) -> None:
        raise HTTPException(status_code=429, detail="queue full")

    monkeypatch.setattr(ingest, "_enqueue_or_reject", _reject)
    monkeypatch.setattr(
        ingest,
        "_register_document_under_job",
        lambda **_kwargs: (SimpleNamespace(), True),
    )
    monkeypatch.setattr(
        ingest,
        "get_job_tracker",
        lambda: type(
            "Tracker",
            (),
            {"unregister_pending": staticmethod(unregistered.append)},
        )(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(mode="standalone"))))

    with pytest.raises(HTTPException, match="429"):
        await ingest._submit_job_work_item(
            request,
            PoolType.BATCH,
            WorkItem(id="rejected", job_id="job"),
            manifest_entry_id=None,
        )

    assert unregistered == ["rejected"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    [OSError("disk full"), ValueError("invalid work record")],
)
async def test_gateway_admission_failure_restores_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    from nemo_retriever.service.routers import ingest
    from nemo_retriever.service.services import sidecar_store

    unregistered: list[str] = []
    restored: list[str] = []

    async def _spool_failure(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(ingest, "_gateway_enqueue", _spool_failure)
    monkeypatch.setattr(
        ingest,
        "_register_document_under_job",
        lambda **_kwargs: (SimpleNamespace(), True),
    )
    monkeypatch.setattr(
        ingest,
        "get_job_tracker",
        lambda: SimpleNamespace(unregister_pending=unregistered.append),
    )
    monkeypatch.setattr(
        sidecar_store,
        "get_sidecar_store",
        lambda: SimpleNamespace(restore=restored.append),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(mode="gateway"))))

    with pytest.raises(type(failure), match=str(failure)):
        await ingest._submit_job_work_item(
            request,
            PoolType.BATCH,
            WorkItem(
                id="spooled",
                job_id="job",
                sidecar_attachment=("sidecar-key", SimpleNamespace()),
            ),
            manifest_entry_id=None,
        )

    assert unregistered == ["spooled"]
    assert restored == ["sidecar-key"]


@pytest.mark.anyio
async def test_gateway_enqueue_unregisters_pending_when_broker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from nemo_retriever.service.routers import ingest
    from nemo_retriever.service.services import work_queue

    unregistered: list[str] = []
    monkeypatch.setattr(work_queue, "get_work_broker", lambda: None)
    monkeypatch.setattr(
        ingest,
        "get_job_tracker",
        lambda: type("Tracker", (), {"unregister_pending": staticmethod(unregistered.append)})(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await ingest._gateway_enqueue(
            None,  # type: ignore[arg-type]
            PoolType.BATCH,
            work_id="work-id",
            job_id="job-id",
            payload=b"payload",
            filename="document.pdf",
        )

    assert exc_info.value.status_code == 503
    assert unregistered == ["work-id"]


def test_collection_page_is_rejected_before_registration(
    app_with_stub_pool: TestClient,
    captured_items: list[WorkItem],
) -> None:
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    assert tracker is not None
    tracker.register_job(
        "collection-page",
        expected_documents=1,
        collection_name="research",
        scope="default",
    )
    response = app_with_stub_pool.post(
        "/v1/ingest/job/collection-page/page",
        files={"file": ("page.png", b"page", "image/png")},
        data={"document_id": "source", "page_number": "1", "filename": "source.pdf"},
    )
    assert response.status_code == 422
    assert tracker.job_documents("collection-page") == []
    assert captured_items == []


@pytest.mark.parametrize("endpoint", ["document", "whole"])
def test_collection_upload_propagates_server_storage_context(
    app_with_stub_pool: TestClient,
    captured_items: list[WorkItem],
    endpoint: str,
) -> None:
    """Both upload routes must emit the same collection work envelope."""
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    assert tracker is not None
    job_id = f"collection-{endpoint}"
    tracker.register_job(
        job_id,
        expected_documents=1,
        collection_name="research",
        scope="workspace",
        operation="append",
    )
    response = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/{endpoint}",
        headers={"X-NRL-Scope": "workspace"},
        files={"file": ("report.txt", b"finding", "text/plain")},
        data={"metadata": "{}"},
    )
    assert response.status_code == 202, response.text
    _wait_for_items(captured_items, 1)
    item = captured_items[0]
    body = response.json()
    assert item.write.scope == "workspace"
    assert item.write.collection_name == "research"
    assert item.write.operation is IngestOperation.APPEND
    assert item.write.content_sha256 == hashlib.sha256(b"finding").hexdigest()
    assert item.write.storage_document_id == body["document_id"]
    assert item.id == body["attempt_id"]
    assert body["document_id"] != body["attempt_id"]


def test_upload_beyond_capacity_returns_409(app_with_stub_pool: TestClient, captured_items: list[WorkItem]) -> None:
    """The (expected_documents + 1)th upload must be rejected with 409."""
    job_id = create_test_job(app_with_stub_pool, expected_documents=1)
    first = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("a.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert first.status_code == 202, first.text

    second = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("b.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert second.status_code == 409, second.text


def test_pipeline_config_endpoint_reports_allowed_overrides(
    app_with_stub_pool: TestClient,
) -> None:
    resp = app_with_stub_pool.get("/v1/ingest/pipeline-config")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "allowed_overrides" in body
    assert body["allowed_overrides"]["mode"] == "allow_list"
    assert "dpi" in body["allowed_overrides"]["allowed_extract_keys"]
    assert "ocr_invoke_url" in body["allowed_overrides"]["denied_key_substrings"]


def test_document_registration_returns_503_if_tracker_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.routers import ingest

    monkeypatch.setattr(ingest, "get_job_tracker", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        ingest._register_document_under_job(
            document_id="attempt",
            job_id="job",
            filename="document.pdf",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Job tracker not available"
