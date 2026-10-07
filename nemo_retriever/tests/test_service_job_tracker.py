# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J1: JobTracker + JobAggregate semantics.

These tests pin down the contract that the rest of the job-shaped
ingestion pipeline depends on:

* Documents must be registered against an existing job.
* Counts roll up correctly across PENDING / PROCESSING / COMPLETED /
  FAILED transitions.
* The aggregate auto-finalizes the moment terminal_count == expected.
* Terminal status derivation distinguishes completed / failed /
  partial_success.
* Adding a 101st doc to a 100-doc job is rejected (JobFullError).
* Adding to a finalized job is rejected (JobFinalizedError).
* The event bus is called with the correct ``job_id`` routing key.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from nemo_retriever.service.services.event_bus import EventBus
from nemo_retriever.service.services.job_tracker import (
    DocumentStatus,
    JobAggregateStatus,
    JobFinalizedError,
    JobFullError,
    JobNotFoundError,
    JobTracker,
    JobTrackerAtCapacityError,
    JobTrackerError,
    MarkOutcome,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _RecordingBus:
    """Drop-in EventBus stand-in that records every publish for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str | None, dict[str, Any]]] = []

    def publish_sync(self, event: dict[str, Any], *, job_id: str | None = None) -> None:
        self.events.append((job_id, event))


def _make_tracker_with_bus() -> tuple[JobTracker, _RecordingBus]:
    tracker = JobTracker()
    bus = _RecordingBus()
    tracker.set_event_bus(bus)
    return tracker, bus


def _age_job(tracker: JobTracker, job_id: str, *, seconds: float, terminal: bool = False) -> None:
    with tracker._lock:
        stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
        field = "finalized_at" if terminal else "created_at"
        setattr(tracker._jobs[job_id], field, stamp)


# ----------------------------------------------------------------------
# Job creation
# ----------------------------------------------------------------------


def test_register_job_creates_pending_aggregate() -> None:
    tracker, bus = _make_tracker_with_bus()
    agg = tracker.register_job("job-1", expected_documents=3, label="weekly")
    assert agg.job_id == "job-1"
    assert agg.expected_documents == 3
    assert agg.status == JobAggregateStatus.PENDING
    assert agg.label == "weekly"
    assert agg.document_ids == []
    assert agg.counts == {s.value: 0 for s in DocumentStatus}
    # job_created event emitted with matching routing key.
    assert bus.events
    routing, event = bus.events[0]
    assert routing == "job-1"
    assert event["type"] == "job_created"


def test_register_job_persists_trace_context_and_events_include_trace_id() -> None:
    tracker, bus = _make_tracker_with_bus()
    trace_id = "0123456789abcdef0123456789abcdef"
    trace_context = {
        "traceparent": f"00-{trace_id}-0123456789abcdef-01",
    }

    agg = tracker.register_job(
        "job-trace",
        expected_documents=1,
        trace_id=trace_id,
        trace_context=trace_context,
    )

    assert agg.trace_id == trace_id
    assert agg.trace_context == trace_context
    trace_context["traceparent"] = "mutated"
    assert agg.trace_context["traceparent"] != "mutated"

    stored = tracker.get_job("job-trace")
    assert stored is not None
    assert stored.trace_id == trace_id
    assert stored.trace_context["traceparent"] != "mutated"

    routing, event = bus.events[0]
    assert routing == "job-trace"
    assert event["type"] == "job_created"
    assert event["trace_id"] == trace_id


def test_register_job_rejects_nonpositive_expected_count() -> None:
    tracker = JobTracker()
    with pytest.raises(ValueError, match="positive"):
        tracker.register_job("x", expected_documents=0)


def test_register_job_rejects_duplicate_id() -> None:
    tracker = JobTracker()
    tracker.register_job("dup", expected_documents=1)
    with pytest.raises(JobTrackerError, match="already exists"):
        tracker.register_job("dup", expected_documents=2)


def test_register_job_rejects_at_max_jobs() -> None:
    tracker = JobTracker(max_jobs=2)
    tracker.register_job("a", expected_documents=10)
    tracker.register_job("b", expected_documents=10)
    with pytest.raises(JobTrackerAtCapacityError, match="capacity"):
        tracker.register_job("c", expected_documents=1)


def test_stale_pending_job_evicted_to_make_room() -> None:
    tracker = JobTracker(max_jobs=1, stale_job_ttl_s=60.0)
    tracker.register_job("old", expected_documents=5)
    with tracker._lock:
        tracker._jobs["old"].created_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    agg = tracker.register_job("new", expected_documents=1)
    assert agg.job_id == "new"
    assert tracker.get_job("old") is None


def test_expired_job_id_can_be_reused() -> None:
    tracker = JobTracker(max_jobs=1, stale_job_ttl_s=60.0)
    tracker.register_job("reusable", expected_documents=1)
    _age_job(tracker, "reusable", seconds=61)

    replacement = tracker.register_job("reusable", expected_documents=2)

    assert replacement.expected_documents == 2


def test_expired_job_releases_idempotency_key() -> None:
    tracker = JobTracker(max_jobs=1, stale_job_ttl_s=60.0)
    tracker.register_job(
        "expired",
        expected_documents=1,
        scope="workspace-a",
        idempotency_key="request-key",
        idempotency_fingerprint="old-request",
    )
    _age_job(tracker, "expired", seconds=61)

    replacement = tracker.register_job(
        "replacement",
        expected_documents=2,
        scope="workspace-a",
        idempotency_key="request-key",
        idempotency_fingerprint="new-request",
    )

    assert replacement.job_id == "replacement"


def test_dropping_a_keyed_job_leaves_the_current_key_owner_replayable() -> None:
    tracker = JobTracker(stale_job_ttl_s=60.0)
    tracker.register_job(
        "unfingerprinted",
        expected_documents=1,
        scope="workspace-a",
        idempotency_key="request-key",
    )
    owner = tracker.register_job(
        "owner",
        expected_documents=1,
        scope="workspace-a",
        idempotency_key="request-key",
        idempotency_fingerprint="request",
    )
    _age_job(tracker, "unfingerprinted", seconds=61)
    tracker.get_job("unfingerprinted")

    replay = tracker.register_job(
        "replay",
        expected_documents=1,
        scope="workspace-a",
        idempotency_key="request-key",
        idempotency_fingerprint="request",
    )

    assert replay.job_id == owner.job_id


@pytest.mark.parametrize(
    "reader",
    [
        "get_job",
        "all_jobs",
        "should_retain_results",
        "job_documents",
        "all_documents",
        "get_document",
        "get_result_data",
        "summary",
    ],
)
def test_stale_job_is_evicted_on_every_read_path(reader: str) -> None:
    tracker = JobTracker(stale_job_ttl_s=60.0)
    tracker.register_job("stale", expected_documents=2, retain_results=True)
    tracker.register_document("doc", job_id="stale")
    _age_job(tracker, "stale", seconds=61)

    if reader == "get_job":
        result = tracker.get_job("stale")
    elif reader == "all_jobs":
        result = tracker.all_jobs()
    elif reader == "should_retain_results":
        result = tracker.should_retain_results("stale")
    elif reader == "job_documents":
        result = tracker.job_documents("stale")
    elif reader == "all_documents":
        result = tracker.all_documents()
    elif reader == "get_document":
        result = tracker.get_document("doc")
    elif reader == "get_result_data":
        result = tracker.get_result_data("doc")
    else:
        result = tracker.summary()["total_jobs"]

    assert result in (None, False, 0, [])
    with tracker._lock:
        assert "stale" not in tracker._jobs
        assert "doc" not in tracker._documents


def test_stale_job_rejects_document_registration() -> None:
    tracker = JobTracker(stale_job_ttl_s=60.0)
    tracker.register_job("stale", expected_documents=1)
    _age_job(tracker, "stale", seconds=61)

    with pytest.raises(JobNotFoundError, match="not found"):
        tracker.register_document("late", job_id="stale")


@pytest.mark.parametrize("transition", ["processing", "completed", "failed"])
def test_stale_job_cannot_be_revived_by_late_transition(transition: str) -> None:
    tracker = JobTracker(stale_job_ttl_s=60.0)
    tracker.register_job("stale", expected_documents=1)
    tracker.register_document("doc", job_id="stale")
    _age_job(tracker, "stale", seconds=61)

    if transition == "processing":
        tracker.mark_processing("doc")
    elif transition == "completed":
        assert tracker.mark_completed("doc") == MarkOutcome.UNKNOWN_DOCUMENT
    else:
        assert tracker.mark_failed("doc", "late") == MarkOutcome.UNKNOWN_DOCUMENT

    with tracker._lock:
        assert "stale" not in tracker._jobs
        assert "doc" not in tracker._documents


def test_terminal_job_results_remain_idempotent_until_terminal_ttl() -> None:
    tracker = JobTracker(ttl_s=60.0, stale_job_ttl_s=60.0)
    tracker.register_job("terminal", expected_documents=1, retain_results=True)
    tracker.register_document("doc", job_id="terminal")
    tracker.mark_completed("doc", result_data=[{"x": 1}])
    _age_job(tracker, "terminal", seconds=59, terminal=True)

    assert tracker.get_result_data("doc") == [{"x": 1}]
    assert tracker.get_result_data("doc") == [{"x": 1}]

    _age_job(tracker, "terminal", seconds=61, terminal=True)
    assert tracker.get_result_data("doc") is None
    assert tracker.get_job("terminal") is None


# ----------------------------------------------------------------------
# Document registration
# ----------------------------------------------------------------------


def test_register_document_requires_existing_job() -> None:
    tracker = JobTracker()
    with pytest.raises(JobNotFoundError):
        tracker.register_document("doc-1", job_id="missing")


def test_register_document_appends_to_aggregate() -> None:
    tracker, _bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2)
    rec = tracker.register_document("doc-1", job_id="j", filename="a.pdf")
    assert rec.job_id == "j"
    assert rec.status == DocumentStatus.PENDING
    assert rec.filename == "a.pdf"

    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.document_ids == ["doc-1"]
    assert agg.counts[DocumentStatus.PENDING.value] == 1


def test_unregister_pending_releases_count_and_manifest_reservation() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document(
        "rejected-attempt",
        job_id="j",
        manifest_entry_id="entry",
    )

    assert tracker.unregister_pending("rejected-attempt") is True
    job = tracker.get_job("j")
    assert job is not None
    assert job.document_ids == []
    assert job.counts[DocumentStatus.PENDING.value] == 0

    replay, created = tracker.register_document_idempotent(
        "accepted-attempt",
        job_id="j",
        manifest_entry_id="entry",
    )
    assert created is True
    assert replay.id == "accepted-attempt"


def test_register_document_rejects_capacity_overflow() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=2)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    with pytest.raises(JobFullError, match="capacity"):
        tracker.register_document("c", job_id="j")


def test_register_document_rejects_duplicate_id() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=3)
    tracker.register_document("dup", job_id="j")
    with pytest.raises(JobTrackerError, match="already registered"):
        tracker.register_document("dup", job_id="j")


def test_register_document_rejects_finalized_job() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_completed("a", result_rows=1)
    with pytest.raises(JobFinalizedError):
        tracker.register_document("b", job_id="j")


# ----------------------------------------------------------------------
# Lifecycle: counts + status transitions
# ----------------------------------------------------------------------


def test_mark_processing_promotes_job_to_processing() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    bus.events.clear()
    tracker.mark_processing("a")

    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.status == JobAggregateStatus.PROCESSING
    assert agg.counts[DocumentStatus.PROCESSING.value] == 1
    assert agg.counts[DocumentStatus.PENDING.value] == 1

    # Document event + a single job_started event.
    job_started = [e for _r, e in bus.events if e["type"] == "job_started"]
    assert len(job_started) == 1
    assert job_started[0]["job_id"] == "j"


def test_mark_processing_is_idempotent() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_processing("a")  # no-op
    rec = tracker.get_document("a")
    assert rec is not None
    assert rec.status == DocumentStatus.PROCESSING


def test_mark_completed_updates_counts_and_doc_record() -> None:
    tracker, _bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2, retain_results=True)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_completed("a", result_rows=42, result_data=[{"k": "v"}])

    rec = tracker.get_document("a")
    assert rec is not None
    assert rec.status == DocumentStatus.COMPLETED
    assert rec.result_rows == 42
    assert rec.result_data == [{"k": "v"}]
    assert rec.completed_at is not None

    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.counts[DocumentStatus.COMPLETED.value] == 1
    assert agg.counts[DocumentStatus.PROCESSING.value] == 0
    # Not finalized yet — second doc still pending.
    assert agg.status == JobAggregateStatus.PROCESSING


# ----------------------------------------------------------------------
# Auto-finalization + terminal status derivation
# ----------------------------------------------------------------------


def test_job_finalizes_completed_when_all_succeed() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_completed("a")
    tracker.mark_processing("b")
    tracker.mark_completed("b")

    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.status == JobAggregateStatus.COMPLETED
    assert agg.finalized_at is not None
    assert agg.elapsed_s is not None and agg.elapsed_s >= 0

    finalized = [e for _r, e in bus.events if e["type"] == "job_finalized"]
    assert len(finalized) == 1
    evt = finalized[0]
    assert evt["status"] == "completed"
    # J5 enriched payload — dashboard subscribers should see these
    # without having to call /v1/ingest/job/{id}.
    assert evt["completed"] == 2
    assert evt["failed"] == 0
    assert evt["remaining"] == 0
    assert evt["progress_pct"] == 100.0
    assert evt["expected_documents"] == 2
    assert evt["started_at"] is not None
    assert evt["finalized_at"] is not None


def test_job_finalizes_failed_when_all_fail() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_failed("a", "boom")
    tracker.mark_failed("b", "kapow")

    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.status == JobAggregateStatus.FAILED
    failed = [e for _r, e in bus.events if e["type"] == "job_failed"]
    assert len(failed) == 1
    evt = failed[0]
    assert evt["status"] == "failed"
    assert evt["completed"] == 0
    assert evt["failed"] == 2
    assert evt["progress_pct"] == 100.0


def test_job_finalizes_partial_when_mixed() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=3)
    for did in ("a", "b", "c"):
        tracker.register_document(did, job_id="j")
    tracker.mark_processing("a")
    tracker.mark_completed("a")
    tracker.mark_failed("b", "boom")
    tracker.mark_completed("c")
    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.status == JobAggregateStatus.PARTIAL_SUCCESS

    # A partial finalization emits ``job_partial`` (not ``job_finalized``)
    # so dashboard UIs can render a distinct status colour without
    # re-fetching the aggregate.
    partial = [e for _r, e in bus.events if e["type"] == "job_partial"]
    finalized = [e for _r, e in bus.events if e["type"] == "job_finalized"]
    failed = [e for _r, e in bus.events if e["type"] == "job_failed"]
    assert len(partial) == 1, [e["type"] for _r, e in bus.events]
    assert finalized == []
    assert failed == []
    assert partial[0]["status"] == "partial_success"
    assert partial[0]["progress_pct"] == 100.0
    assert partial[0]["completed"] == 2
    assert partial[0]["failed"] == 1
    assert partial[0]["remaining"] == 0


def test_terminal_status_is_idempotent() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")
    tracker.mark_completed("a")
    # Calling mark_completed again on the same doc is a no-op.
    tracker.mark_completed("a")
    agg = tracker.get_job("j")
    assert agg is not None
    assert agg.counts[DocumentStatus.COMPLETED.value] == 1


# ----------------------------------------------------------------------
# job_progress event cadence
# ----------------------------------------------------------------------


def test_progress_events_at_default_10_doc_cadence() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=25)
    doc_ids = [f"d-{i}" for i in range(25)]
    for did in doc_ids:
        tracker.register_document(did, job_id="j")

    for did in doc_ids:
        tracker.mark_processing(did)
        tracker.mark_completed(did)

    progress = [e for _r, e in bus.events if e["type"] == "job_progress"]
    finalized = [e for _r, e in bus.events if e["type"] == "job_finalized"]
    # Progress: at 10, 20. (25 hits finalize, no separate progress.)
    assert len(progress) == 2
    assert len(finalized) == 1
    # J5 enriched payload — verify the cadence positions are
    # surfaced with sensible counts + percentages.
    assert progress[0]["completed"] == 10
    assert progress[0]["remaining"] == 15
    assert progress[0]["progress_pct"] == 40.0
    assert progress[1]["completed"] == 20
    assert progress[1]["remaining"] == 5
    assert progress[1]["progress_pct"] == 80.0


def test_progress_events_respect_custom_step() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.set_progress_step(5)
    tracker.register_job("j", expected_documents=10)
    for i in range(10):
        did = f"d-{i}"
        tracker.register_document(did, job_id="j")
        tracker.mark_processing(did)
        tracker.mark_completed(did)
    progress = [e for _r, e in bus.events if e["type"] == "job_progress"]
    # Step=5 over 10 docs → progress at 5 (10 is finalize).
    assert len(progress) == 1


# ----------------------------------------------------------------------
# Bus routing key
# ----------------------------------------------------------------------


def test_all_events_carry_job_id_routing_key() -> None:
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("alpha", expected_documents=1)
    tracker.register_document("doc-x", job_id="alpha")
    tracker.mark_processing("doc-x")
    tracker.mark_completed("doc-x")
    for routing, _event in bus.events:
        assert routing == "alpha"


def test_full_lifecycle_event_ordering_for_completed_job() -> None:
    """A successful 2-doc job emits exactly: created → started → finalized.

    Plus one document-level event per doc (``processing`` and
    ``completed`` are the two terminal/transient states a doc passes
    through). Verifying the order is what dashboard subscribers rely
    on to drive their state machine.
    """
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2, label="full")
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")

    tracker.mark_processing("a")
    tracker.mark_completed("a")
    tracker.mark_processing("b")
    tracker.mark_completed("b")

    types = [e["type"] for _r, e in bus.events]
    # First event is the aggregate-level ``job_created``.
    assert types[0] == "job_created"
    # First processing call also promotes the job to processing.
    assert "job_started" in types
    assert types.index("job_started") > types.index("job_created")
    # Final job-level event is ``job_finalized``.
    assert types[-1] == "job_finalized"
    # ``job_failed`` and ``job_partial`` should not appear for a clean job.
    assert "job_failed" not in types
    assert "job_partial" not in types
    # Document-level events for both docs.
    doc_events = [e for _r, e in bus.events if e["type"] == "completed"]
    assert {e["document_id"] for e in doc_events} == {"a", "b"}


def test_full_lifecycle_event_ordering_for_failed_job() -> None:
    """A 2-doc job where both docs fail emits ``job_failed`` (not finalized)."""
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=2)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    tracker.mark_processing("a")
    tracker.mark_failed("a", "boom")
    tracker.mark_failed("b", "kapow")

    types = [e["type"] for _r, e in bus.events]
    assert types[0] == "job_created"
    assert types[-1] == "job_failed"
    assert "job_finalized" not in types
    assert "job_partial" not in types


def test_register_document_does_not_emit_progress_event() -> None:
    """``register_document`` is silent at the aggregate level — it only
    appends to the aggregate's ``document_ids``. Subscribers learn of
    new documents through document-level events (``processing``,
    ``completed``, ``failed``) so we don't spam ``job_progress`` for
    registrations.
    """
    tracker, bus = _make_tracker_with_bus()
    tracker.register_job("j", expected_documents=3)
    tracker.register_document("a", job_id="j")
    tracker.register_document("b", job_id="j")
    tracker.register_document("c", job_id="j")
    types = [e["type"] for _r, e in bus.events]
    assert types.count("job_created") == 1
    assert "job_progress" not in types
    assert "job_started" not in types


def test_real_event_bus_filters_by_job_id() -> None:
    """End-to-end with the real EventBus: subscriber for job A never sees B's events."""
    bus = EventBus()

    async def _run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sub_a_id, q_a = bus.subscribe(job_id="A")
        sub_b_id, q_b = bus.subscribe(job_id="B")
        try:
            bus.publish_sync({"type": "doc", "id": "1"}, job_id="A")
            bus.publish_sync({"type": "doc", "id": "2"}, job_id="B")
            bus.publish_sync({"type": "doc", "id": "3"}, job_id="A")

            received_a: list[dict[str, Any]] = []
            received_b: list[dict[str, Any]] = []
            while not q_a.empty():
                received_a.append(q_a.get_nowait())
            while not q_b.empty():
                received_b.append(q_b.get_nowait())
            return received_a, received_b
        finally:
            bus.unsubscribe(sub_a_id)
            bus.unsubscribe(sub_b_id)

    a, b = asyncio.run(_run())
    assert [e["id"] for e in a] == ["1", "3"]
    assert [e["id"] for e in b] == ["2"]


def test_real_event_bus_firehose_subscriber_sees_everything() -> None:
    bus = EventBus()

    async def _run() -> list[dict[str, Any]]:
        sub_id, q = bus.subscribe(job_id=None)
        try:
            bus.publish_sync({"type": "x"}, job_id="A")
            bus.publish_sync({"type": "y"}, job_id="B")
            received: list[dict[str, Any]] = []
            while not q.empty():
                received.append(q.get_nowait())
            return received
        finally:
            bus.unsubscribe(sub_id)

    events = asyncio.run(_run())
    assert {e["type"] for e in events} == {"x", "y"}


# ----------------------------------------------------------------------
# Reads + job_documents
# ----------------------------------------------------------------------


def test_job_documents_returns_records_in_arrival_order() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=3)
    for did in ("first", "second", "third"):
        tracker.register_document(did, job_id="j")
    docs = tracker.job_documents("j")
    assert [d.id for d in docs] == ["first", "second", "third"]


def test_summary_groups_by_job_aggregate_status() -> None:
    tracker = JobTracker()
    tracker.register_job("a", expected_documents=1)
    tracker.register_document("a-1", job_id="a")
    tracker.mark_completed("a-1")
    tracker.register_job("b", expected_documents=1)
    tracker.register_document("b-1", job_id="b")
    tracker.mark_failed("b-1", "boom")

    summary = tracker.summary()
    assert summary["total_jobs"] == 2
    assert summary["total_documents"] == 2
    assert summary[JobAggregateStatus.COMPLETED.value] == 1
    assert summary[JobAggregateStatus.FAILED.value] == 1


def test_mark_completed_drops_result_data_when_retain_false() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1, retain_results=False)
    tracker.register_document("d", job_id="j")
    tracker.mark_completed("d", result_rows=3, result_data=[{"x": 1}])
    rec = tracker.get_document("d")
    assert rec is not None
    assert rec.result_rows == 3
    assert rec.result_data is None
    assert tracker.get_result_data("d") is None


def test_get_result_data_is_idempotent_until_job_eviction() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1, retain_results=True)
    tracker.register_document("d", job_id="j")
    rows = [{"metadata": {"tags": ["original"]}}]
    expected = [{"metadata": {"tags": ["original"]}}]

    tracker.mark_completed("d", result_data=rows)
    rows[0]["metadata"]["tags"].append("producer-mutation")

    first_read = tracker.get_result_data("d")
    assert first_read == expected
    assert first_read is not None
    first_read[0]["metadata"]["tags"].append("reader-mutation")

    assert tracker.get_result_data("d") == expected
