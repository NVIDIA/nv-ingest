# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory job + document status tracker for pipeline work items.

Service run_mode is "job-shaped": clients :func:`POST /v1/ingest/job` to
create a :class:`JobAggregate` with a known ``expected_documents`` count,
then upload each document into that job. The tracker keeps both layers:

* :class:`DocumentRecord` — one per uploaded file (page or document). Was
  formerly called ``JobRecord``; the rename clarifies intent.
* :class:`JobAggregate` — one per client-issued job, carries the
  ``document_ids`` it owns plus rolled-up ``counts`` (pending /
  processing / completed / failed) and a derived ``status`` that
  transitions to ``completed`` / ``failed`` / ``partial_success``
  only when every document has reached a terminal state.

Per-job auto-finalization: as documents transition through
:meth:`JobTracker.mark_completed` / :meth:`JobTracker.mark_failed`,
the tracker checks ``len(document_ids_in_terminal_state) == expected_documents``
and, if so, computes the job's terminal status and emits a
``job_finalized`` event. No timeout / TTL based finalization — the
caller declares the expected count at create time, and the job waits
for that many documents to resolve.

Abandoned jobs (never finalized) are bounded by ``max_jobs`` (hard
reject on create) and ``stale_job_ttl_s`` (evicted from memory).

Threading model: all writes are guarded by a single ``threading.Lock``.
Reads are best-effort consistent (we return defensive copies in
:meth:`get_job` / :meth:`get_document`).

Singleton access mirrors the other service singletons::

    if (tracker := get_job_tracker()) is not None:
        tracker.register_job(job_id, expected_documents=N)
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from pydantic import Field

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.common.schemas.collections import IngestOperation

logger = logging.getLogger(__name__)


# ── status enums ──────────────────────────────────────────────────────


class DocumentStatus(str, Enum):
    """Lifecycle state of one uploaded document.

    Mirrors the previous ``JobStatus`` — renamed to reflect the fact
    that the ``Job`` concept now refers to the aggregate, not the
    per-file work item.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Back-compat alias for in-flight callers that still import ``JobStatus``
# from this module. New code should reach for :class:`DocumentStatus` or
# :class:`JobAggregateStatus` directly.
JobStatus = DocumentStatus


class JobAggregateStatus(str, Enum):
    """Roll-up state of a :class:`JobAggregate`.

    ``pending`` and ``processing`` are non-terminal. The three terminal
    states distinguish "all completed" from "all failed" from "mixed".
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


_DOC_TERMINAL: frozenset[DocumentStatus] = frozenset({DocumentStatus.COMPLETED, DocumentStatus.FAILED})
_JOB_TERMINAL: frozenset[JobAggregateStatus] = frozenset(
    {
        JobAggregateStatus.COMPLETED,
        JobAggregateStatus.FAILED,
        JobAggregateStatus.PARTIAL_SUCCESS,
    }
)


class MarkOutcome(str, Enum):
    """Result of a :meth:`JobTracker.mark_completed` / :meth:`mark_failed` call.

    Routers use this to log accurately when a worker-pod callback fires:

    * ``transitioned`` — the document moved into the requested terminal
      state and a per-document SSE event was published. This is the
      common path.
    * ``idempotent`` — the document was already in a terminal state, so
      the call was a no-op. Surfaces duplicate callbacks (worker retry,
      bulk poll racing SSE, …).
    * ``unknown_document`` — the tracker has no record of the supplied
      document id. The most common cause is a gateway-pod restart
      between accepting an upload and the worker firing its callback,
      which silently strands the doc on the client. Treated as a
      warning so it stands out in gateway logs during hang triage.
    """

    TRANSITIONED = "transitioned"
    IDEMPOTENT = "idempotent"
    UNKNOWN_DOCUMENT = "unknown_document"


# ── data models ───────────────────────────────────────────────────────


class DocumentRecord(RichModel):
    """Per-document tracker entry.

    Each document belongs to exactly one :class:`JobAggregate`. The
    ``job_id`` back-reference lets the SSE event router route events
    to the correct subscriber.
    """

    id: str
    stable_document_id: str
    job_id: str
    status: DocumentStatus = DocumentStatus.PENDING
    submitted_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_s: float | None = None
    result_rows: int | None = None
    result_data: list[dict[str, Any]] | None = None
    error: str | None = None
    filename: str | None = Field(default=None, description="Original upload filename surfaced in the dashboard")
    collection_name: str | None = None
    content_sha256: str | None = None
    manifest_entry_id: str | None = None


class JobAggregate(RichModel):
    """Aggregate state for a client-issued job.

    The lifecycle is fully driven by document transitions:

    * Created at :func:`POST /v1/ingest/job` with
      ``status=pending``, ``expected_documents=N``.
    * Transitions to ``processing`` the moment any document begins
      processing.
    * Auto-finalizes to ``completed`` / ``failed`` / ``partial_success``
      once exactly ``expected_documents`` records reach terminal state.

    ``document_ids`` is kept in arrival order so the UI can render a
    consistent timeline.
    """

    job_id: str
    expected_documents: int
    document_ids: list[str] = []
    counts: dict[str, int] = {}
    status: JobAggregateStatus = JobAggregateStatus.PENDING
    created_at: str = ""
    started_at: str | None = None
    finalized_at: str | None = None
    elapsed_s: float | None = None
    label: str | None = Field(default=None, description="Optional client-supplied job tag")
    metadata: dict[str, Any] = {}
    trace_id: str | None = None
    trace_context: dict[str, str] = Field(default_factory=dict)
    retain_results: bool = Field(default=False, description="Keep completed document result payloads in memory")
    collection_name: str | None = None
    scope: str = "default"
    operation: IngestOperation = IngestOperation.APPEND
    target_document_id: str | None = None
    idempotency_key: str | None = None
    idempotency_fingerprint: str | None = None
    document_manifest: list[dict[str, str]] = Field(default_factory=list)


# ── eviction tunables (apply to terminal aggregates) ──────────────────

DEFAULT_TTL_S: float = 4 * 3600  # 4 hours after finalize
DEFAULT_STALE_JOB_TTL_S: float = 4 * 3600  # non-terminal jobs with no finalize
DEFAULT_MAX_JOBS: int = 200_000
_EVICTION_INTERVAL: int = 50  # periodic eviction on document registration


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_counts() -> dict[str, int]:
    return {s.value: 0 for s in DocumentStatus}


# ── exceptions raised by JobTracker mutators ──────────────────────────


class JobTrackerError(RuntimeError):
    """Base for tracker-level errors that the router maps to HTTP responses."""

    status_code: int = 500


class JobNotFoundError(JobTrackerError):
    status_code = 404


class JobFullError(JobTrackerError):
    """Raised when adding a document to an over-capacity job."""

    status_code = 409


class JobFinalizedError(JobTrackerError):
    """Raised when mutating a job that has already reached terminal state."""

    status_code = 409


class JobTrackerAtCapacityError(JobTrackerError):
    """Raised when the in-memory job store is at ``max_jobs``."""

    status_code = 503


# ── tracker singleton ────────────────────────────────────────────────


class JobTracker:
    """Thread-safe in-memory store of jobs + documents."""

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        stale_job_ttl_s: float = DEFAULT_STALE_JOB_TTL_S,
        max_jobs: int = DEFAULT_MAX_JOBS,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobAggregate] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._started_mono: dict[str, float] = {}  # per-document elapsed timing
        self._job_started_mono: dict[str, float] = {}  # per-job elapsed timing
        self._event_bus: Any = None
        # Observers receive defensive snapshots after a genuine terminal
        # document transition. This keeps all completion paths consistent.
        self._terminal_observers: list[Callable[[DocumentRecord, JobAggregate | None], None]] = []
        self._ttl_s = ttl_s
        self._stale_job_ttl_s = stale_job_ttl_s
        self._max_jobs = max_jobs
        self._reg_count = 0
        # Track which integer progress milestone (in completed+failed doc
        # counts) we last published, so we don't emit duplicate progress
        # events on every doc transition.
        self._progress_published: dict[str, int] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str]] = {}
        self._accepted_manifest_entries: dict[tuple[str, str], str] = {}
        self._progress_step: int = 10

    # ── wiring ───────────────────────────────────────────────────────

    def set_event_bus(self, bus: Any) -> None:
        """Attach an :class:`EventBus` so state transitions publish SSE events."""
        self._event_bus = bus

    def add_terminal_observer(self, observer: Callable[[DocumentRecord, JobAggregate | None], None]) -> None:
        """Register a callback for real document terminal transitions.

        Args:
            observer: Callback receiving defensive snapshots of the transitioned
                document and its aggregate job, when available.

        Returns:
            None. The callback runs outside the tracker lock after tracker
            state is updated; duplicate and unknown callbacks do not invoke it.
        """
        self._terminal_observers.append(observer)

    def set_progress_step(self, step: int) -> None:
        """Override the progress-event cadence (default: every 10 docs)."""
        if step <= 0:
            raise ValueError("progress step must be positive")
        self._progress_step = step

    # ── job lifecycle ────────────────────────────────────────────────

    def register_job(
        self,
        job_id: str,
        *,
        expected_documents: int,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        retain_results: bool = False,
        trace_id: str | None = None,
        trace_context: dict[str, str] | None = None,
        collection_name: str | None = None,
        scope: str = "default",
        operation: IngestOperation = IngestOperation.APPEND,
        target_document_id: str | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
        document_manifest: list[dict[str, str]] | None = None,
    ) -> JobAggregate:
        """Create a new :class:`JobAggregate` in ``pending`` state."""
        if expected_documents <= 0:
            raise ValueError(f"expected_documents must be positive; got {expected_documents}")
        with self._lock:
            now = datetime.now(timezone.utc)
            self._evict_locked(now=now)
            if idempotency_key and idempotency_fingerprint:
                entry = self._idempotency.get((scope, idempotency_key))
                if entry:
                    existing_job_id, previous_fingerprint = entry
                    if previous_fingerprint != idempotency_fingerprint:
                        raise JobFullError("Idempotency key was already used with a different request payload")
                    existing = self._get_live_job_locked(existing_job_id, now=now)
                    if existing is not None:
                        return existing.model_copy(deep=True)
            if job_id in self._jobs:
                raise JobTrackerError(f"Job {job_id!r} already exists")
            if len(self._jobs) >= self._max_jobs:
                raise JobTrackerAtCapacityError(
                    f"Job tracker is at capacity ({self._max_jobs} jobs); "
                    "retry after in-flight jobs finalize or abandon stale jobs."
                )
            agg = JobAggregate(
                job_id=job_id,
                expected_documents=expected_documents,
                document_ids=[],
                counts=_empty_counts(),
                status=JobAggregateStatus.PENDING,
                created_at=_utcnow_iso(),
                label=label,
                metadata=dict(metadata or {}),
                trace_id=trace_id,
                trace_context=dict(trace_context or {}),
                retain_results=retain_results,
                collection_name=collection_name,
                scope=scope,
                operation=operation,
                target_document_id=target_document_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=idempotency_fingerprint,
                document_manifest=list(document_manifest or []),
            )
            agg.counts[DocumentStatus.PENDING.value] = 0
            self._jobs[job_id] = agg
            if idempotency_key and idempotency_fingerprint:
                self._idempotency[(scope, idempotency_key)] = (
                    job_id,
                    idempotency_fingerprint,
                )
        logger.info(
            "Job registered: %s (expected_documents=%d, label=%r)",
            job_id,
            expected_documents,
            label,
        )
        self._publish_job_event("job_created", agg)
        return agg.model_copy(deep=True)

    def get_job(self, job_id: str) -> JobAggregate | None:
        with self._lock:
            agg = self._get_live_job_locked(job_id)
            return agg.model_copy(deep=True) if agg is not None else None

    def all_jobs(self) -> list[JobAggregate]:
        """Return a snapshot of every aggregate (defensive deep copies)."""
        with self._lock:
            self._evict_locked()
            return [a.model_copy(deep=True) for a in self._jobs.values()]

    def should_retain_results(self, job_id: str | None) -> bool:
        """Return whether completed row payloads should be kept for *job_id*."""
        if not job_id:
            return False
        with self._lock:
            agg = self._get_live_job_locked(job_id)
            return bool(agg.retain_results) if agg is not None else False

    def job_documents(self, job_id: str) -> list[DocumentRecord]:
        """Return every document record belonging to *job_id* in arrival order."""
        with self._lock:
            agg = self._get_live_job_locked(job_id)
            if agg is None:
                return []
            out: list[DocumentRecord] = []
            for did in agg.document_ids:
                rec = self._documents.get(did)
                if rec is not None:
                    out.append(rec.model_copy(deep=True))
            return out

    def all_documents(self) -> list[DocumentRecord]:
        """Return a defensive snapshot of every document across all jobs.

        Insertion order matches the order documents were registered, which
        keeps SSE "catch-up" snapshots deterministic for clients.
        """
        with self._lock:
            self._evict_locked()
            return [rec.model_copy(deep=True) for rec in self._documents.values()]

    # ── document lifecycle ───────────────────────────────────────────

    def register_document(
        self,
        document_id: str,
        *,
        job_id: str,
        filename: str | None = None,
        content_sha256: str | None = None,
        stable_document_id: str | None = None,
        manifest_entry_id: str | None = None,
    ) -> DocumentRecord:
        """Attach a new :class:`DocumentRecord` to *job_id*.

        Raises :class:`JobNotFoundError` if the job does not exist,
        :class:`JobFinalizedError` if the job has already reached a
        terminal state, or :class:`JobFullError` if the job is at
        capacity (``len(document_ids) == expected_documents``).
        """
        rec, _ = self.register_document_idempotent(
            document_id,
            job_id=job_id,
            filename=filename,
            content_sha256=content_sha256,
            stable_document_id=stable_document_id,
            manifest_entry_id=manifest_entry_id,
        )
        return rec

    def register_document_idempotent(
        self,
        attempt_id: str,
        *,
        job_id: str,
        filename: str | None = None,
        content_sha256: str | None = None,
        stable_document_id: str | None = None,
        manifest_entry_id: str | None = None,
    ) -> tuple[DocumentRecord, bool]:
        """Register an upload or return its original accepted record.

        Manifest-entry replay is checked before terminal/capacity validation so a
        client can safely replay every file after losing any upload response.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            agg = self._get_live_job_locked(job_id, now=now)
            if agg is None:
                raise JobNotFoundError(f"Job {job_id!r} not found")
            if manifest_entry_id:
                existing_attempt = self._accepted_manifest_entries.get((job_id, manifest_entry_id))
                if existing_attempt:
                    existing = self._documents[existing_attempt]
                    if existing.filename != filename or existing.content_sha256 != content_sha256:
                        raise JobFullError("Manifest entry was already accepted with different filename or content")
                    return existing.model_copy(deep=True), False
            if agg.status in _JOB_TERMINAL:
                raise JobFinalizedError(
                    f"Job {job_id!r} has already finalized with status "
                    f"{agg.status.value!r}; cannot add more documents."
                )
            if len(agg.document_ids) >= agg.expected_documents:
                raise JobFullError(
                    f"Job {job_id!r} is at capacity "
                    f"({agg.expected_documents} documents); rejected document "
                    f"#{len(agg.document_ids) + 1}."
                )
            if attempt_id in self._documents:
                raise JobTrackerError(f"Document attempt {attempt_id!r} already registered.")
            rec = DocumentRecord(
                id=attempt_id,
                stable_document_id=stable_document_id or attempt_id,
                job_id=job_id,
                status=DocumentStatus.PENDING,
                submitted_at=_utcnow_iso(),
                filename=filename,
                collection_name=agg.collection_name,
                content_sha256=content_sha256,
                manifest_entry_id=manifest_entry_id,
            )
            self._documents[attempt_id] = rec
            agg.document_ids.append(attempt_id)
            if manifest_entry_id:
                self._accepted_manifest_entries[(job_id, manifest_entry_id)] = attempt_id
            agg.counts[DocumentStatus.PENDING.value] = agg.counts.get(DocumentStatus.PENDING.value, 0) + 1
            self._reg_count += 1
            if self._reg_count % _EVICTION_INTERVAL == 0:
                self._evict_locked(now=now)
        return rec.model_copy(deep=True), True

    def mark_processing(self, document_id: str) -> None:
        """Transition a document from ``pending`` → ``processing``.

        Also promotes the parent job from ``pending`` to ``processing``
        the first time any of its documents starts.
        """
        with self._lock:
            rec = self._get_live_document_locked(document_id)
            if rec is None:
                return
            if rec.status != DocumentStatus.PENDING:
                return  # idempotent: only PENDING → PROCESSING is meaningful
            rec.status = DocumentStatus.PROCESSING
            rec.started_at = _utcnow_iso()
            self._started_mono[document_id] = time.monotonic()
            self._adjust_counts_locked(rec.job_id, DocumentStatus.PENDING, DocumentStatus.PROCESSING)
            agg = self._jobs.get(rec.job_id)
            if agg is not None and agg.status == JobAggregateStatus.PENDING:
                agg.status = JobAggregateStatus.PROCESSING
                agg.started_at = _utcnow_iso()
                self._job_started_mono[rec.job_id] = time.monotonic()
                # Publish a 'job_started' so the UI can switch state.
                bus_agg = agg.model_copy(deep=True)
            else:
                bus_agg = None
        if bus_agg is not None:
            self._publish_job_event("job_started", bus_agg)

    def unregister_pending(self, document_id: str) -> bool:
        """Roll back an admission rejected before a worker can claim it."""
        with self._lock:
            rec = self._documents.get(document_id)
            if rec is None or rec.status != DocumentStatus.PENDING:
                return False
            agg = self._jobs.get(rec.job_id)
            self._documents.pop(document_id, None)
            if rec.manifest_entry_id:
                manifest_key = (rec.job_id, rec.manifest_entry_id)
                if self._accepted_manifest_entries.get(manifest_key) == document_id:
                    self._accepted_manifest_entries.pop(manifest_key, None)
            if agg is not None:
                try:
                    agg.document_ids.remove(document_id)
                except ValueError:
                    pass
                agg.counts[DocumentStatus.PENDING.value] = max(0, agg.counts.get(DocumentStatus.PENDING.value, 0) - 1)
            return True

    def mark_completed(
        self,
        document_id: str,
        *,
        result_rows: int = 0,
        result_data: list[dict[str, Any]] | None = None,
        elapsed_s: float | None = None,
    ) -> MarkOutcome:
        """Transition a document to ``completed``; maybe finalize the job.

        Returns a :class:`MarkOutcome` so the gateway callback handler
        can surface duplicate / orphaned callbacks in logs (the common
        symptom of a hung client whose docs were stranded by a gateway
        pod restart).
        """
        return self._mark_terminal(
            document_id,
            new_status=DocumentStatus.COMPLETED,
            result_rows=result_rows,
            result_data=result_data,
            elapsed_s=elapsed_s,
        )

    def mark_failed(
        self,
        document_id: str,
        error: str,
        *,
        elapsed_s: float | None = None,
    ) -> MarkOutcome:
        """Transition a document to ``failed``; maybe finalize the job.

        See :meth:`mark_completed` for the meaning of the return value.
        """
        return self._mark_terminal(
            document_id,
            new_status=DocumentStatus.FAILED,
            error=error,
            elapsed_s=elapsed_s,
        )

    def _mark_terminal(
        self,
        document_id: str,
        *,
        new_status: DocumentStatus,
        result_rows: int = 0,
        result_data: list[dict[str, Any]] | None = None,
        error: str | None = None,
        elapsed_s: float | None = None,
    ) -> MarkOutcome:
        # Phase 1: under lock, mutate state and gather snapshots.
        with self._lock:
            rec = self._get_live_document_locked(document_id)
            if rec is None:
                # A worker callback for a doc the tracker has never seen
                # is the classic symptom of a gateway-pod restart
                # between upload acceptance and worker completion: the
                # doc lives on the client (and on the worker that
                # eventually finishes it) but no longer on this
                # gateway, so no SSE event will be published. Surface
                # this loudly so hung clients are diagnosable from
                # gateway logs alone.
                logger.warning(
                    "JobTracker.%s: no record of document %r — callback dropped (likely "
                    "gateway-pod restart between upload acceptance and worker callback); "
                    "client may hang waiting for an SSE event that will never arrive",
                    ("mark_failed" if new_status == DocumentStatus.FAILED else "mark_completed"),
                    document_id,
                )
                return MarkOutcome.UNKNOWN_DOCUMENT
            if rec.status in _DOC_TERMINAL:
                return MarkOutcome.IDEMPOTENT  # duplicate callback / poll race
            old_status = rec.status
            rec.status = new_status
            rec.completed_at = _utcnow_iso()
            rec.result_rows = result_rows
            agg_for_retain = self._jobs.get(rec.job_id)
            retain_results = bool(agg_for_retain.retain_results) if agg_for_retain is not None else False
            rec.result_data = copy.deepcopy(result_data) if retain_results else None
            rec.error = error
            if elapsed_s is not None:
                rec.elapsed_s = elapsed_s
            else:
                t0 = self._started_mono.pop(document_id, None)
                rec.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None
            self._adjust_counts_locked(rec.job_id, old_status, new_status)
            doc_snapshot = rec.model_copy(deep=True)

            # Maybe-finalize the aggregate.
            agg = self._jobs.get(rec.job_id)
            finalized_snapshot: JobAggregate | None = None
            progress_snapshot: JobAggregate | None = None
            aggregate_snapshot: JobAggregate | None = None
            if agg is not None:
                terminal_count = agg.counts.get(DocumentStatus.COMPLETED.value, 0) + agg.counts.get(
                    DocumentStatus.FAILED.value, 0
                )
                if terminal_count == agg.expected_documents and agg.status not in _JOB_TERMINAL:
                    agg.status = self._derive_terminal_status_locked(agg)
                    agg.finalized_at = _utcnow_iso()
                    t0 = self._job_started_mono.pop(rec.job_id, None)
                    agg.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None
                    finalized_snapshot = agg.model_copy(deep=True)
                elif terminal_count > 0:
                    last_published = self._progress_published.get(rec.job_id, 0)
                    if terminal_count - last_published >= self._progress_step:
                        self._progress_published[rec.job_id] = terminal_count
                        progress_snapshot = agg.model_copy(deep=True)
                aggregate_snapshot = agg.model_copy(deep=True)

        # Phase 2: project and publish with the lock released.
        for observer in tuple(self._terminal_observers):
            try:
                observer(doc_snapshot, aggregate_snapshot)
            except Exception:
                logger.exception("JobTracker terminal observer failed for document %s", document_id)

        # Events are published after observers see finalized state.
        self._publish_document_event(doc_snapshot)
        if progress_snapshot is not None:
            self._publish_job_event("job_progress", progress_snapshot)
        if finalized_snapshot is not None:
            # Three terminal event names so dashboard subscribers can
            # render correct status colours without re-fetching the
            # aggregate. ``COMPLETED`` → all docs succeeded;
            # ``PARTIAL_SUCCESS`` → at least one succeeded and one
            # failed; ``FAILED`` → every doc failed.
            if finalized_snapshot.status == JobAggregateStatus.FAILED:
                event_name = "job_failed"
            elif finalized_snapshot.status == JobAggregateStatus.PARTIAL_SUCCESS:
                event_name = "job_partial"
            else:
                event_name = "job_finalized"
            self._publish_job_event(event_name, finalized_snapshot)

        return MarkOutcome.TRANSITIONED

    # ── internal helpers ─────────────────────────────────────────────

    def _adjust_counts_locked(
        self,
        job_id: str,
        old: DocumentStatus,
        new: DocumentStatus,
    ) -> None:
        agg = self._jobs.get(job_id)
        if agg is None:
            return
        agg.counts[old.value] = max(0, agg.counts.get(old.value, 0) - 1)
        agg.counts[new.value] = agg.counts.get(new.value, 0) + 1

    @staticmethod
    def _derive_terminal_status_locked(agg: JobAggregate) -> JobAggregateStatus:
        completed = agg.counts.get(DocumentStatus.COMPLETED.value, 0)
        failed = agg.counts.get(DocumentStatus.FAILED.value, 0)
        if failed == 0 and completed > 0:
            return JobAggregateStatus.COMPLETED
        if completed == 0 and failed > 0:
            return JobAggregateStatus.FAILED
        if completed > 0 and failed > 0:
            return JobAggregateStatus.PARTIAL_SUCCESS
        # No docs registered (shouldn't happen — guarded earlier).
        return JobAggregateStatus.FAILED

    def _job_age_s_locked(self, agg: JobAggregate, *, now: datetime) -> float | None:
        stamp = agg.finalized_at if agg.status in _JOB_TERMINAL else agg.created_at
        if not stamp:
            return None
        try:
            return (now - datetime.fromisoformat(stamp)).total_seconds()
        except (ValueError, TypeError):
            return None

    def _expiry_reason_locked(self, agg: JobAggregate, *, now: datetime) -> str | None:
        age_s = self._job_age_s_locked(agg, now=now)
        if age_s is None:
            return None
        if agg.status in _JOB_TERMINAL:
            return "terminal" if age_s > self._ttl_s else None
        return "stale" if age_s > self._stale_job_ttl_s else None

    def _drop_if_expired_locked(self, job_id: str, *, now: datetime) -> bool:
        agg = self._jobs.get(job_id)
        if agg is None:
            return False
        reason = self._expiry_reason_locked(agg, now=now)
        if reason is None:
            return False
        self._drop_job_locked(job_id)
        logger.debug("Job tracker eviction: removed %s job %s", reason, job_id)
        return True

    def _get_live_job_locked(self, job_id: str, *, now: datetime | None = None) -> JobAggregate | None:
        checked_at = now or datetime.now(timezone.utc)
        if self._drop_if_expired_locked(job_id, now=checked_at):
            return None
        return self._jobs.get(job_id)

    def _get_live_document_locked(
        self,
        document_id: str,
        *,
        now: datetime | None = None,
    ) -> DocumentRecord | None:
        rec = self._documents.get(document_id)
        if rec is None:
            return None
        checked_at = now or datetime.now(timezone.utc)
        if self._drop_if_expired_locked(rec.job_id, now=checked_at):
            return None
        return rec

    def _evict_locked(self, *, now: datetime | None = None) -> None:
        """Drop expired terminal jobs, stale non-terminal jobs, then bound count."""
        checked_at = now or datetime.now(timezone.utc)
        expired: list[tuple[str, str]] = []
        for jid, agg in self._jobs.items():
            reason = self._expiry_reason_locked(agg, now=checked_at)
            if reason is not None:
                expired.append((jid, reason))

        for jid, _ in expired:
            self._drop_job_locked(jid)

        if len(self._jobs) > self._max_jobs:
            terminal = [(jid, agg.finalized_at or "") for jid, agg in self._jobs.items() if agg.status in _JOB_TERMINAL]
            terminal.sort(key=lambda t: t[1])
            excess = len(self._jobs) - self._max_jobs
            for jid, _ in terminal[:excess]:
                self._drop_job_locked(jid)

        if expired:
            stale_count = sum(reason == "stale" for _, reason in expired)
            terminal_count = len(expired) - stale_count
            logger.debug(
                "Job tracker eviction: removed %d stale and %d terminal job aggregate(s); %d remaining",
                stale_count,
                terminal_count,
                len(self._jobs),
            )

    def _drop_job_locked(self, job_id: str) -> None:
        agg = self._jobs.pop(job_id, None)
        if agg is None:
            return
        if agg.idempotency_key:
            # Only release the key if this job still owns it; the map is written
            # solely for jobs that carry a fingerprint, so a keyed job may point
            # at an entry that now belongs to a later job.
            entry_key = (agg.scope, agg.idempotency_key)
            owner = self._idempotency.get(entry_key)
            if owner is not None and owner[0] == job_id:
                del self._idempotency[entry_key]
        for did in agg.document_ids:
            rec = self._documents.pop(did, None)
            if rec and rec.manifest_entry_id:
                self._accepted_manifest_entries.pop((job_id, rec.manifest_entry_id), None)
            self._started_mono.pop(did, None)
        self._job_started_mono.pop(job_id, None)
        self._progress_published.pop(job_id, None)

    # ── event-bus plumbing ───────────────────────────────────────────

    def _publish_document_event(self, rec: DocumentRecord) -> None:
        if self._event_bus is None:
            return
        event: dict[str, Any] = {
            "type": rec.status.value,
            "id": rec.id,
            "document_id": rec.stable_document_id,
            "attempt_id": rec.id,
            "job_id": rec.job_id,
            "status": rec.status.value,
            "result_rows": rec.result_rows,
            "elapsed_s": rec.elapsed_s,
            "error": rec.error,
            "filename": rec.filename,
        }
        self._event_bus.publish_sync(event, job_id=rec.job_id)

    def _publish_job_event(self, event_type: str, agg: JobAggregate) -> None:
        if self._event_bus is None:
            return
        completed = agg.counts.get(DocumentStatus.COMPLETED.value, 0)
        failed = agg.counts.get(DocumentStatus.FAILED.value, 0)
        terminal = completed + failed
        remaining = max(0, agg.expected_documents - terminal)
        progress_pct = round(terminal * 100.0 / agg.expected_documents, 2) if agg.expected_documents else 0.0
        event: dict[str, Any] = {
            "type": event_type,
            "id": agg.job_id,
            "job_id": agg.job_id,
            "status": agg.status.value,
            "expected_documents": agg.expected_documents,
            "counts": dict(agg.counts),
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "progress_pct": progress_pct,
            "elapsed_s": agg.elapsed_s,
            "started_at": agg.started_at,
            "finalized_at": agg.finalized_at,
            "label": agg.label,
        }
        if agg.trace_id:
            event["trace_id"] = agg.trace_id
        self._event_bus.publish_sync(event, job_id=agg.job_id)

    # ── reads ────────────────────────────────────────────────────────

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            rec = self._get_live_document_locked(document_id)
            return rec.model_copy(deep=True) if rec is not None else None

    def get_result_data(self, document_id: str) -> list[dict[str, Any]] | None:
        """Return retained ``result_data`` for *document_id* without consuming it."""
        with self._lock:
            rec = self._get_live_document_locked(document_id)
            return copy.deepcopy(rec.result_data) if rec is not None else None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            self._evict_locked()
            total = len(self._jobs)
            doc_total = len(self._documents)
            by_status = {s.value: 0 for s in JobAggregateStatus}
            for agg in self._jobs.values():
                by_status[agg.status.value] += 1
            return {
                "total_jobs": total,
                "total_documents": doc_total,
                **by_status,
            }


# ── module-level singleton, same shape as the other services ────────


_instance: JobTracker | None = None


def init_job_tracker() -> JobTracker:
    global _instance
    _instance = JobTracker()
    logger.info("Job tracker initialised")
    return _instance


def get_job_tracker() -> JobTracker | None:
    return _instance


def shutdown_job_tracker() -> None:
    global _instance
    if _instance is not None:
        summary = _instance.summary()
        logger.info("Job tracker shut down: %s", summary)
        _instance = None
