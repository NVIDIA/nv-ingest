# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent VectorDB writes against a real LanceDB table.

These tests exercise the ingest path the VectorDB service uses (``hybrid``
index mode, so both a vector and an FTS index exist) with real index builds.
``write_to_index`` is never replaced with a no-op: a mocked index phase cannot
observe a writer blocked behind another writer's index build, which is exactly
what regressed durability and queryability for concurrent ingests.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

lancedb = pytest.importorskip("lancedb")

from lancedb.table import LanceTable  # noqa: E402

from nemo_retriever.common.vdb import lancedb as lancedb_module  # noqa: E402
from nemo_retriever.common.vdb import lancedb_capabilities  # noqa: E402
from nemo_retriever.common.vdb.lancedb import LanceDB  # noqa: E402

_DIM = 4
_TABLE = "nemo_retriever"
_WAIT_TIMEOUT_S = 30.0
_INDEX_READY_TIMEOUT_S = 3.0


@pytest.fixture(autouse=True)
def _bounded_index_ready_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap index-readiness waits so a regression fails fast instead of stalling."""
    monkeypatch.setattr(
        lancedb_capabilities,
        "INDEX_READY_TIMEOUT",
        timedelta(seconds=_INDEX_READY_TIMEOUT_S),
    )


def _record(text: str, vector: list[float]) -> dict:
    return {
        "document_type": "text",
        "embedding": vector,
        "metadata": {
            "content": text,
            "content_metadata": {"page_number": 1, "type": "text"},
            "source_metadata": {"source_id": f"/data/{text}.txt", "source_name": f"{text}.txt"},
        },
    }


def _backend(tmp_path) -> LanceDB:
    """Build the backend the VectorDB service uses for ``index_mode='hybrid'``."""
    return LanceDB(
        uri=str(tmp_path),
        table_name=_TABLE,
        vector_dim=_DIM,
        overwrite=False,
        hybrid=True,
        build_index=True,
        num_partitions=2,
        num_sub_vectors=2,
    )


def _seed(backend: LanceDB) -> None:
    backend.run(
        [
            [
                _record("seed_alpha", [1.0, 0.0, 0.0, 0.0]),
                _record("seed_beta", [0.0, 1.0, 0.0, 0.0]),
            ]
        ]
    )


def _open_table(backend: LanceDB) -> LanceTable:
    return lancedb.connect(backend.uri).open_table(backend.table_name)


def _row_count(backend: LanceDB) -> int:
    return int(_open_table(backend).count_rows())


def _unindexed_rows(backend: LanceDB, column: str) -> int:
    stubs = [
        stub for stub in _open_table(backend).list_indices() if column in [str(entry) for entry in (stub.columns or [])]
    ]
    assert stubs, f"no LanceDB index on column {column!r}"
    return sum(int(stub.num_unindexed_rows) for stub in stubs)


def _append_unindexed_row(backend: LanceDB, text: str, vector: list[float]) -> None:
    """Commit a row without rebuilding indexes, leaving every index a tail."""
    backend.create_index(records=[[_record(text, vector)]], table_name=backend.table_name)


def _await(event: threading.Event) -> None:
    assert event.wait(timeout=_WAIT_TIMEOUT_S), "timed out waiting for the write under test"


def _record_index_waits(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which column each index-readiness wait was scoped to.

    Asserting on the columns rather than on elapsed time states the actual
    requirement, and does not turn a slow machine into a failure.
    """
    waited: list[str] = []
    original = lancedb_module.wait_for_column_index

    def wait_for_column_index(table, column, **kwargs):
        waited.append(column)
        return original(table, column, **kwargs)

    monkeypatch.setattr(lancedb_module, "wait_for_column_index", wait_for_column_index)
    return waited


def _await_index_requests(backend: LanceDB, expected: int) -> None:
    """Block until ``expected`` writers have asked for index maintenance.

    Rows are committed just before a writer registers its index generation.
    Waiting for the registration instead of the rows makes coalescing
    deterministic, because every writer is then known to be queued.
    """
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        with backend._index_generation_lock:
            if backend._index_requested_generation >= expected:
                return
        time.sleep(0.01)
    raise AssertionError(f"only {backend._index_requested_generation} of {expected} writers requested a rebuild")


def _await_committed(backend: LanceDB, expected: int) -> None:
    """Block until ``expected`` rows are committed to the table."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if _row_count(backend) >= expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"table never reached {expected} committed rows")


def _gate_first_index_build(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Make the first vector-index build block, emulating a slow index build.

    Returns ``(entered, release)``: ``entered`` is set once a writer is parked
    in its index phase, and setting ``release`` lets that build proceed.
    """
    original = LanceTable.create_index
    entered = threading.Event()
    release = threading.Event()
    gated = threading.Event()

    def create_index(self, *args, **kwargs):
        if not gated.is_set():
            gated.set()
            entered.set()
            assert release.wait(timeout=_WAIT_TIMEOUT_S), "gated index build was never released"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LanceTable, "create_index", create_index)
    return entered, release


def test_vector_index_phase_does_not_wait_on_an_fts_tail(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vector phase must not block on rows another writer left unindexed."""
    backend = _backend(tmp_path)
    _seed(backend)
    _append_unindexed_row(backend, "gamma_comet", [0.0, 0.0, 1.0, 0.0])
    assert _unindexed_rows(backend, "text") == 1
    waited = _record_index_waits(monkeypatch)

    backend.write_to_index([], table=_open_table(backend), num_partitions=2, hybrid=False)

    assert waited == ["vector"]
    # The FTS tail is untouched: the vector phase neither waited for it nor
    # rebuilt it.
    assert _unindexed_rows(backend, "text") == 1
    assert _unindexed_rows(backend, "vector") == 0


def test_fts_index_phase_does_not_wait_on_a_vector_tail(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(tmp_path)
    _seed(backend)
    _append_unindexed_row(backend, "gamma_comet", [0.0, 0.0, 1.0, 0.0])
    waited = _record_index_waits(monkeypatch)

    backend.write_to_index([], table=_open_table(backend), sparse=True)

    assert waited == ["text"]
    assert _unindexed_rows(backend, "text") == 0
    assert _unindexed_rows(backend, "vector") == 1


def test_concurrent_writes_are_durable_and_queryable(tmp_path) -> None:
    """Two concurrent service writes must both land and both be retrievable."""
    backend = _backend(tmp_path)
    _seed(backend)
    assert _row_count(backend) == 2

    batches = [
        [_record("gamma_comet", [0.0, 0.0, 1.0, 0.0])],
        [_record("delta_pulsar", [0.0, 0.0, 0.0, 1.0])],
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(backend.run, [batch]) for batch in batches]:
            future.result(timeout=_WAIT_TIMEOUT_S)

    assert _row_count(backend) == 4
    table = _open_table(backend)
    stored = {row["text"] for row in table.to_arrow().to_pylist()}
    assert {"gamma_comet", "delta_pulsar"} <= stored
    for sentinel in ("gamma_comet", "delta_pulsar"):
        hits = table.search(sentinel, query_type="fts").limit(10).to_list()
        assert sentinel in {hit["text"] for hit in hits}


def test_rows_are_committed_while_another_writer_builds_its_index(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later write must not be held behind an in-flight index build."""
    backend = _backend(tmp_path)
    _seed(backend)
    entered, release = _gate_first_index_build(monkeypatch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        blocked = pool.submit(backend.run, [[_record("gamma_comet", [0.0, 0.0, 1.0, 0.0])]])
        _await(entered)

        later = pool.submit(backend.run, [[_record("delta_pulsar", [0.0, 0.0, 0.0, 1.0])]])
        _await_committed(backend, 4)
        # The gated writer is still parked in its index build, yet every row is durable.
        assert not blocked.done()

        release.set()
        blocked.result(timeout=_WAIT_TIMEOUT_S)
        later.result(timeout=_WAIT_TIMEOUT_S)

    stored = {row["text"] for row in _open_table(backend).to_arrow().to_pylist()}
    assert {"gamma_comet", "delta_pulsar"} <= stored


def test_index_rebuilds_stay_serialized_and_coalesce(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writers queued behind one index build share a single follow-up rebuild."""
    backend = _backend(tmp_path)
    _seed(backend)

    builds: list[str] = []
    active = 0
    overlapped = False
    state_lock = threading.Lock()
    real_write_to_index = backend.write_to_index

    def write_to_index(*args, **kwargs):
        nonlocal active, overlapped
        with state_lock:
            active += 1
            overlapped = overlapped or active > 1
            builds.append(threading.current_thread().name)
        try:
            return real_write_to_index(*args, **kwargs)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(backend, "write_to_index", write_to_index)
    entered, release = _gate_first_index_build(monkeypatch)

    followers = ["delta_pulsar", "epsilon_quasar", "zeta_nebula"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(backend.run, [[_record("gamma_comet", [0.0, 0.0, 1.0, 0.0])]])
        _await(entered)

        rest = [pool.submit(backend.run, [[_record(name, [1.0, 1.0, 0.0, 0.0])]]) for name in followers]
        _await_committed(backend, 2 + 1 + len(followers))
        _await_index_requests(backend, 1 + len(followers))

        release.set()
        first.result(timeout=_WAIT_TIMEOUT_S)
        for future in rest:
            future.result(timeout=_WAIT_TIMEOUT_S)

    assert not overlapped, "index builds must never run concurrently"
    # Without coalescing each of the four writers rebuilds: the gated build plus
    # one per follower. Coalescing collapses the followers into one rebuild.
    assert len(builds) == 2
    stored = {row["text"] for row in _open_table(backend).to_arrow().to_pylist()}
    assert {"gamma_comet", *followers} <= stored
