# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb._lancedb_stream import (
    DataCommittedFinalizationError,
    OversizedVdbRowError,
    VdbWriteNotFinalized,
)
from nemo_retriever.common.vdb._lancedb_stream_state import (
    CommitOutcomeUnknown,
    VdbOperationConflict,
)
from nemo_retriever.common.vdb.adt_vdb import UnsupportedVDBOperation
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import VdbUploadError
from nemo_retriever.operators.vdb import IngestVdbOperator


def _record(
    row_id: int,
    *,
    vector_dim: int = 2,
    embedding: list[float] | None = None,
    text_prefix: str = "chunk",
    padding: int = 0,
) -> dict[str, Any]:
    vector = embedding if embedding is not None else [float(row_id), *[1.0] * (vector_dim - 1)]
    return {
        "document_type": "text",
        "metadata": {
            "embedding": vector,
            "content": f"{text_prefix}-{row_id} " + ("x" * padding),
            "content_metadata": {
                "type": "text",
                "id": f"row-{row_id}",
                "page_number": row_id,
            },
            "source_metadata": {
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "source_name": f"doc-{row_id}.pdf",
            },
        },
    }


def _records(
    start: int,
    stop: int,
    *,
    vector_dim: int = 2,
    text_prefix: str = "chunk",
    padding: int = 0,
) -> list[dict[str, Any]]:
    return [
        _record(
            row_id,
            vector_dim=vector_dim,
            text_prefix=text_prefix,
            padding=padding,
        )
        for row_id in range(start, stop)
    ]


def _writer_process(
    uri: str,
    records: list[dict[str, Any]],
    *,
    operation: str = "stream",
    operation_id: str | None = "shared-append",
    block_in_add: bool,
    add_entered: Any | None = None,
    add_done: Any | None = None,
    build_index: bool = False,
    index_entered: Any | None = None,
    release_index: Any | None = None,
    started: Any | None = None,
    lock_attempted: Any | None = None,
    lock_acquired: Any | None = None,
    lock_phase: str = "table",
) -> None:
    """Run one real LanceDB write in a spawned process."""

    if lock_attempted is not None:
        assert lock_acquired is not None
        from nemo_retriever.common.vdb import lancedb as lancedb_module

        original_acquire = lancedb_module.FileLock.acquire

        def observed_acquire(self, *args, **kwargs):
            if not Path(self.lock_file).name.startswith(f"{lock_phase}-"):
                return original_acquire(self, *args, **kwargs)
            lock_attempted.set()
            result = original_acquire(self, *args, **kwargs)
            lock_acquired.set()
            return result

        lancedb_module.FileLock.acquire = observed_acquire

    table_type = type(lancedb.connect(uri).open_table("chunks"))
    original_add = table_type.add

    def observed_add(self, *args, **kwargs):
        if add_entered is not None:
            add_entered.set()
        if block_in_add and not threading.Event().wait(timeout=30):
            raise TimeoutError("timed out waiting to release the first LanceDB add")
        result = original_add(self, *args, **kwargs)
        if add_done is not None:
            add_done.set()
        return result

    table_type.add = observed_add
    if index_entered is not None:
        original_create_index = table_type.create_index

        def observed_create_index(self, *args, **kwargs):
            index_entered.set()
            if release_index is not None:
                assert release_index.wait(timeout=30)
            return original_create_index(self, *args, **kwargs)

        table_type.create_index = observed_create_index
    backend = LanceDB(
        uri=uri,
        table_name="chunks",
        vector_dim=2,
        overwrite=False,
        build_index=build_index,
        index_type="IVF_FLAT",
        num_partitions=2,
        stream_operation_id=operation_id,
    )
    if started is not None:
        started.set()
    if operation == "stream":
        backend.stream_ingest(records)
    elif operation == "run":
        backend.run([records])
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _backend(uri: Path, **overrides: Any) -> LanceDB:
    kwargs: dict[str, Any] = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": 2,
        "overwrite": True,
        "build_index": False,
    }
    kwargs.update(overrides)
    return LanceDB(**kwargs)


def _table(uri: Path):
    table = lancedb.connect(str(uri)).open_table("chunks")
    table.checkout_latest()
    return table


def _state(uri: Path) -> tuple[list[str], list[int]]:
    table = _table(uri)
    ids = sorted(table.to_arrow().column("id").to_pylist())
    versions = [int(version["version"]) for version in table.list_versions()]
    return ids, versions


def _product_metadata(schema: pa.Schema) -> dict[bytes, bytes]:
    return {key: value for key, value in (schema.metadata or {}).items() if not key.startswith(b"nemo_retriever.sink_")}


def test_stream_ingest_is_lazy_and_byte_bounded_with_legacy_query_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical records become bounded Arrow batches in one indexed lifecycle."""

    common = {
        "build_index": True,
        "hybrid": True,
        "index_type": "IVF_FLAT",
        "num_partitions": 2,
    }
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_batch_bytes=1024,
        **common,
    )
    records = _records(0, 16, padding=200)
    legacy.run([records])

    pulls: list[int] = []
    observed_batch_bytes: list[int] = []

    connection_type = type(lancedb.connect(str(tmp_path / "streaming")))
    original_create_table = connection_type.create_table

    def observed_create_table(self, name, data=None, *args, **kwargs):
        if name == "chunks":
            assert isinstance(data, pa.RecordBatchReader)
            assert 0 < len(pulls) < len(records)
            original_reader = data

            def observed_batches() -> Iterator[pa.RecordBatch]:
                for batch in original_reader:
                    assert isinstance(batch, pa.RecordBatch)
                    observed_batch_bytes.append(int(batch.get_total_buffer_size()))
                    yield batch

            data = pa.RecordBatchReader.from_batches(original_reader.schema, observed_batches())
        return original_create_table(self, name, *args, data=data, **kwargs)

    monkeypatch.setattr(connection_type, "create_table", observed_create_table)

    def record_stream() -> Iterator[dict[str, Any]]:
        for row_index, record in enumerate(records):
            pulls.append(row_index)
            yield record

    streaming.stream_ingest(record_stream())

    assert pulls == list(range(len(records)))
    assert len(observed_batch_bytes) > 1
    assert max(observed_batch_bytes) <= 1024

    legacy_table = _table(tmp_path / "legacy")
    streaming_table = _table(tmp_path / "streaming")
    assert legacy_table.schema.remove_metadata() == streaming_table.schema.remove_metadata()
    assert _product_metadata(legacy_table.schema) == _product_metadata(streaming_table.schema)
    assert legacy_table.to_arrow().sort_by("id").to_pylist() == streaming_table.to_arrow().sort_by("id").to_pylist()
    assert {tuple(index.columns) for index in streaming_table.list_indices()} == {
        ("vector",),
        ("text",),
    }

    vectors = [[1.0, 1.0], [13.0, 1.0]]
    query_texts = ["chunk-1", "chunk-13"]
    legacy_hits = legacy.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    streaming_hits = streaming.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    assert [[hit["id"] for hit in hits] for hits in streaming_hits] == [
        [hit["id"] for hit in hits] for hits in legacy_hits
    ]


def test_conversion_failure_does_not_commit_a_partial_stream(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    rows = pd.DataFrame(
        [
            {"text": "embedded", "embedding": [1.0, 1.0]},
            {"text": "missing", "embedding": None, "embedding_error": "embedding failed"},
        ]
    )

    with pytest.raises(VdbUploadError, match="refusing a partial write"):
        IngestVdbOperator(vdb=backend)._stream_ingest([rows])

    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


def test_oversized_row_fails_without_table_mutation(tmp_path: Path) -> None:
    backend = _backend(
        tmp_path,
        stream_batch_bytes=256,
        stream_operation_id="oversized",
    )

    with pytest.raises(OversizedVdbRowError, match="max_batch_bytes=256"):
        backend.stream_ingest([_record(0, padding=4096)])

    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


@pytest.mark.parametrize(
    "control",
    [
        {"stream_batch_bytes": 1024},
        {"stream_optimize": True},
        {"stream_operation_id": "fixed-operation"},
    ],
)
def test_stream_controls_are_rejected_by_legacy_mutations(tmp_path: Path, control: dict[str, Any]) -> None:
    backend = _backend(tmp_path, **control)
    message = r"require LanceDB\.stream_ingest\(\)"

    with pytest.raises(ValueError, match=message):
        backend.run([])


def test_stream_optimize_runs_after_the_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _backend(tmp_path, stream_operation_id="seed").stream_ingest([_record(0)])
    table_type = type(_table(tmp_path))
    original_optimize = table_type.optimize
    optimized_row_counts: list[int] = []

    def observed_optimize(table, *args, **kwargs):
        optimized_row_counts.append(int(table.count_rows()))
        return original_optimize(table, *args, **kwargs)

    monkeypatch.setattr(table_type, "optimize", observed_optimize)
    _backend(
        tmp_path,
        overwrite=False,
        stream_optimize=True,
        stream_operation_id="optimized-append",
    ).stream_ingest([_record(1)])

    assert optimized_row_counts == [2]
    assert _table(tmp_path).count_rows() == 2


def test_stream_refreshes_cached_table_for_legacy_mutations(tmp_path: Path) -> None:
    backend = _backend(tmp_path, overwrite=False)
    backend.run([_records(0, 1)])
    assert backend._open_table("chunks").count_rows() == 1

    backend.stream_ingest(_records(1, 2))
    assert backend._open_table("chunks").count_rows() == 2

    backend.run([_records(2, 3)])

    rows = {row["id"]: row for row in backend._open_table("chunks").to_arrow().to_pylist()}
    assert set(rows) == {"row-0", "row-1", "row-2"}


@pytest.mark.parametrize(
    ("policy", "embedding", "expected_vectors"),
    [
        pytest.param("drop", [1.0], [], id="drop-wrong-width"),
        pytest.param("fill", [1.0], [[-3.5, -3.5]], id="fill-wrong-width"),
        pytest.param("fill", ["not-a-number"], [[-3.5, -3.5]], id="fill-wrong-width-nonnumeric"),
        pytest.param("fill", [1.0, float("nan")], [[-3.5, -3.5]], id="fill-nan"),
        pytest.param("null", [1.0], [None], id="null-wrong-width"),
    ],
)
def test_bad_vector_policies_match_legacy(
    tmp_path: Path,
    policy: str,
    embedding: list[Any],
    expected_vectors: list[list[float] | None],
) -> None:
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": policy,
        "fill_value": -3.5,
    }
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_operation_id=f"bad-vector-{policy}",
        **common,
    )

    legacy.run([[_record(0, embedding=embedding)]])
    streaming.stream_ingest([_record(0, embedding=embedding)])

    legacy_vectors = _table(tmp_path / "legacy").to_arrow()["vector"].to_pylist()
    streaming_vectors = _table(tmp_path / "streaming").to_arrow()["vector"].to_pylist()
    assert streaming_vectors == legacy_vectors == expected_vectors


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        pytest.param([1.0], "expected dimension 2", id="wrong-width"),
        pytest.param([1.0, float("nan")], "contains NaN", id="nan"),
    ],
)
def test_bad_vector_error_is_adapter_owned(tmp_path: Path, embedding: list[float], message: str) -> None:
    backend = _backend(tmp_path, validate_vector_length=False, on_bad_vectors="error")

    with pytest.raises(ValueError, match=message):
        backend.stream_ingest([_record(0, embedding=embedding)])


def test_deferred_dimension_inference_after_invalid_prefix_matches_legacy(tmp_path: Path) -> None:
    records = [
        _record(0, embedding=[]),
        _record(1, embedding=[float("nan"), 1.0, 2.0]),
        _record(2, embedding=[2.0, 1.0, 2.0]),
    ]
    common = {"vector_dim": None, "on_bad_vectors": "drop"}
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_operation_id="infer-after-invalid-prefix",
        **common,
    )

    legacy.run([records])
    streaming.stream_ingest(iter(records))

    legacy_table = _table(tmp_path / "legacy")
    streaming_table = _table(tmp_path / "streaming")
    assert streaming_table.schema.field("vector").type.list_size == 3
    assert streaming_table.to_arrow().to_pylist() == legacy_table.to_arrow().to_pylist()
    assert streaming_table.to_arrow().column("id").to_pylist() == ["row-2"]


def test_append_preserves_compatible_existing_schema(tmp_path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
            pa.field("legacy_extra", pa.string()),
        ],
        metadata={
            b"retrieval_mode": b"dense",
            b"nemo_retriever.retrieval_mode": b"dense",
        },
    )
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "chunks",
        data=[
            {
                "vector": [1.0, 0.0],
                "text": "seed",
                "metadata": '{"type":"text","id":"seed"}',
                "source": '{"source_id":"/tmp/seed.pdf"}',
                "id": "seed",
                "legacy_extra": "preserved",
            }
        ],
        schema=schema,
    )

    _backend(
        tmp_path,
        overwrite=False,
        stream_operation_id="append-schema-superset",
    ).stream_ingest([_record(1)])

    table = _table(tmp_path)
    rows = sorted(table.to_arrow().to_pylist(), key=lambda row: row["id"])
    assert table.schema == schema
    assert [(row["id"], row["legacy_extra"]) for row in rows] == [
        ("row-1", None),
        ("seed", "preserved"),
    ]


def _fail_after_first_arrow_batch() -> Iterator[dict[str, Any]]:
    yield _record(10, padding=64)
    yield _record(11, padding=64)
    raise RuntimeError("injected source failure")


@pytest.mark.parametrize(
    ("overwrite", "operation_id"),
    [
        pytest.param(False, None, id="default-append"),
        pytest.param(False, "durable-append", id="durable-append"),
        pytest.param(True, "durable-overwrite", id="durable-overwrite"),
    ],
)
def test_midstream_precommit_failure_preserves_target_and_is_retryable(
    tmp_path: Path,
    overwrite: bool,
    operation_id: str | None,
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    before = _state(tmp_path)
    backend = _backend(
        tmp_path,
        overwrite=overwrite,
        stream_batch_bytes=512,
        stream_operation_id=operation_id,
    )

    with pytest.raises(RuntimeError, match="injected source failure"):
        backend.stream_ingest(_fail_after_first_arrow_batch())

    assert _state(tmp_path) == before
    backend.stream_ingest(_records(10, 14, padding=64))

    ids, versions = _state(tmp_path)
    expected = [f"row-{row_id}" for row_id in range(10, 14)]
    if not overwrite:
        expected.extend(["row-0", "row-1"])
    assert ids == sorted(expected)
    assert len(ids) == len(set(ids))
    assert len(versions) == len(before[1]) + 1


def test_acknowledged_append_retry_is_a_noop_and_changed_content_conflicts(tmp_path: Path) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        stream_batch_bytes=512,
        stream_operation_id="append-10-14",
    )
    records = _records(10, 14, padding=64)
    backend.stream_ingest(records)
    after_first = _state(tmp_path)
    success_tags = sorted(name for name in _table(tmp_path).tags.list() if name.startswith("nemo_sink_success_"))
    assert len(success_tags) == 2

    retry_backend = _backend(
        tmp_path,
        overwrite=False,
        stream_batch_bytes=2048,
        stream_operation_id="append-10-14",
    )
    retry_backend.stream_ingest(iter(records))
    assert _state(tmp_path) == after_first
    assert (
        sorted(name for name in _table(tmp_path).tags.list() if name.startswith("nemo_sink_success_")) == success_tags
    )

    with pytest.raises(VdbOperationConflict, match="different stored content"):
        retry_backend.stream_ingest(_records(10, 14, text_prefix="changed", padding=64))
    assert _state(tmp_path) == after_first


def test_default_finalization_failure_reports_committed_version_without_durable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=2,
        stream_batch_bytes=512,
    )
    retry_records = _records(10, 14, padding=64)

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(backend, "write_to_index", fail_index)
    with pytest.raises(DataCommittedFinalizationError, match="Do not replay these records") as failure:
        backend.stream_ingest(retry_records)
    committed_ids = [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert failure.value.data_version == _table(tmp_path).version
    assert _state(tmp_path)[0] == committed_ids
    assert not [name for name in _table(tmp_path).tags.list() if name.startswith("nemo_sink_")]

    reconstructed = _backend(tmp_path, overwrite=False)
    assert reconstructed.retrieval([[0.0, 1.0]], top_k=10)
    reconstructed.stream_ingest(_records(20, 22))
    assert _state(tmp_path)[0] == committed_ids + ["row-20", "row-21"]


@pytest.mark.parametrize(
    ("num_partitions", "error_type"),
    [(None, TypeError), (float("inf"), OverflowError)],
)
def test_default_post_commit_index_config_error_reports_committed_version(
    tmp_path: Path,
    num_partitions: Any,
    error_type: type[Exception],
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=num_partitions,
    )

    with pytest.raises(DataCommittedFinalizationError, match="Do not replay these records") as failure:
        backend.stream_ingest(_records(10, 12))

    assert isinstance(failure.value.__cause__, error_type)
    assert failure.value.data_version == _table(tmp_path).version
    assert _state(tmp_path)[0] == ["row-0", "row-1", "row-10", "row-11"]


def test_explicit_operation_id_resumes_finalization_after_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    records = _records(10, 14, padding=64)
    backend = _backend(
        tmp_path,
        overwrite=False,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=2,
        stream_batch_bytes=512,
        stream_operation_id="durable-finalization",
    )

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(backend, "write_to_index", fail_index)
    with pytest.raises(DataCommittedFinalizationError, match="durable-finalization") as failure:
        backend.stream_ingest(records)
    assert failure.value.data_version == _table(tmp_path).version
    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        backend.retrieval([[0.0, 1.0]], top_k=10)

    reconstructed = _backend(
        tmp_path,
        overwrite=False,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=2,
        stream_batch_bytes=2048,
        stream_operation_id="durable-finalization",
    )
    reconstructed.stream_ingest(iter(records))

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert reconstructed.retrieval([[0.0, 1.0]], top_k=10)
    tags = _table(tmp_path).tags.list()
    assert len([name for name in tags if name.startswith("nemo_sink_success_")]) == 1
    assert not [name for name in tags if name.startswith(("nemo_sink_pending_", "nemo_sink_data_"))]


def test_same_operation_is_exactly_once_across_concurrent_backend_instances(
    tmp_path: Path,
) -> None:
    _backend(tmp_path, stream_operation_id="seed").stream_ingest(_records(0, 2))
    context = mp.get_context("spawn")
    first_add_entered = context.Event()
    second_add_entered = context.Event()
    second_lock_attempted = context.Event()
    second_lock_acquired = context.Event()
    first = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(10, 14)),
        kwargs={
            "block_in_add": True,
            "add_entered": first_add_entered,
        },
    )
    second = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(10, 14)),
        kwargs={
            "block_in_add": False,
            "add_entered": second_add_entered,
            "lock_attempted": second_lock_attempted,
            "lock_acquired": second_lock_acquired,
        },
    )
    first.start()
    try:
        assert first_add_entered.wait(timeout=30)
        second.start()
        assert second_lock_attempted.wait(timeout=30)
        assert not second_lock_acquired.wait(timeout=1)
        assert not second_add_entered.is_set()
        first.terminate()
        first.join(timeout=10)
        assert not first.is_alive()
        assert second_lock_acquired.wait(timeout=30)
        assert second_add_entered.wait(timeout=30)
        second.join(timeout=30)
        assert not second.is_alive()
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]


def test_stream_and_legacy_writes_are_serialized_across_processes(tmp_path: Path) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    context = mp.get_context("spawn")
    stream_add_entered = context.Event()
    legacy_started = context.Event()
    legacy_add_entered = context.Event()
    streaming = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(10, 14)),
        kwargs={
            "operation_id": None,
            "block_in_add": True,
            "add_entered": stream_add_entered,
        },
    )
    legacy = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(20, 24)),
        kwargs={
            "operation": "run",
            "operation_id": None,
            "block_in_add": False,
            "add_entered": legacy_add_entered,
            "started": legacy_started,
        },
    )
    streaming.start()
    try:
        assert stream_add_entered.wait(timeout=30)
        legacy.start()
        assert legacy_started.wait(timeout=30)
        assert not legacy_add_entered.wait(timeout=1)
        streaming.terminate()
        streaming.join(timeout=10)
        assert not streaming.is_alive()
        assert legacy_add_entered.wait(timeout=30)
        legacy.join(timeout=30)
        assert not legacy.is_alive()
        assert legacy.exitcode == 0
    finally:
        for process in (streaming, legacy):
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-20",
        "row-21",
        "row-22",
        "row-23",
    ]


def test_stream_waits_for_legacy_index_finalization_across_processes(tmp_path: Path) -> None:
    _backend(
        tmp_path,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=2,
    ).stream_ingest(_records(0, 4))
    context = mp.get_context("spawn")
    legacy_mutation_done = context.Event()
    legacy_index_entered = context.Event()
    release_legacy_index = context.Event()
    stream_mutation_done = context.Event()
    stream_index_entered = context.Event()
    stream_index_lock_attempted = context.Event()
    stream_index_lock_acquired = context.Event()
    legacy = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(10, 14)),
        kwargs={
            "operation": "run",
            "operation_id": None,
            "block_in_add": False,
            "add_done": legacy_mutation_done,
            "build_index": True,
            "index_entered": legacy_index_entered,
            "release_index": release_legacy_index,
        },
    )
    streaming = context.Process(
        target=_writer_process,
        args=(str(tmp_path), _records(20, 24)),
        kwargs={
            "operation": "stream",
            "operation_id": None,
            "block_in_add": False,
            "add_done": stream_mutation_done,
            "build_index": True,
            "index_entered": stream_index_entered,
            "lock_attempted": stream_index_lock_attempted,
            "lock_acquired": stream_index_lock_acquired,
            "lock_phase": "index",
        },
    )

    legacy.start()
    try:
        assert legacy_mutation_done.wait(timeout=30)
        assert legacy_index_entered.wait(timeout=30)
        streaming.start()
        assert stream_mutation_done.wait(timeout=30)
        assert stream_index_lock_attempted.wait(timeout=30)
        assert not stream_index_lock_acquired.wait(timeout=1)
        assert not stream_index_entered.is_set()
        release_legacy_index.set()
        assert stream_index_lock_acquired.wait(timeout=30)
        assert stream_index_entered.wait(timeout=30)
        legacy.join(timeout=30)
        streaming.join(timeout=30)
        assert not legacy.is_alive()
        assert not streaming.is_alive()
        assert legacy.exitcode == 0
        assert streaming.exitcode == 0
    finally:
        release_legacy_index.set()
        for process in (legacy, streaming):
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    ids, _versions = _state(tmp_path)
    expected_ids = [f"row-{row_id}" for row_id in [*range(4), *range(10, 14), *range(20, 24)]]
    assert ids == sorted(expected_ids)
    table = _table(tmp_path)
    vector_index = next(index for index in table.list_indices() if tuple(index.columns) == ("vector",))
    stats = table.index_stats(vector_index.name)
    assert stats.num_indexed_rows == len(ids)
    assert stats.num_unindexed_rows == 0


@pytest.mark.parametrize(
    "uri",
    ["s3://example/retriever", "file://localhost/tmp/retriever"],
)
def test_nonfilesystem_lancedb_retains_fallback_without_consuming_stream(uri: str) -> None:
    backend = LanceDB(uri=uri, table_name="chunks")
    assert not backend.supports_stream_ingest
    pulled = False

    def records() -> Iterator[dict[str, Any]]:
        nonlocal pulled
        pulled = True
        yield _record(0)

    with pytest.raises(UnsupportedVDBOperation, match="unavailable"):
        backend.stream_ingest(records())
    assert not pulled


def test_file_uri_uses_the_bounded_streaming_path(tmp_path: Path) -> None:
    uri = tmp_path.as_uri()
    backend = LanceDB(
        uri=uri,
        table_name="chunks",
        vector_dim=2,
        build_index=False,
        stream_operation_id="file-uri",
    )

    assert backend.supports_stream_ingest
    service_backend = LanceDB(uri=uri, table_name="service", _service_table_schema=True)
    assert not service_backend.supports_stream_ingest
    backend.stream_ingest(_records(0, 2))
    assert sorted(lancedb.connect(uri).open_table("chunks").to_arrow().column("id").to_pylist()) == [
        "row-0",
        "row-1",
    ]


def test_lost_append_commit_acknowledgement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        stream_operation_id="append-unknown",
    )
    tags_type = type(_table(tmp_path).tags)
    original_create = tags_type.create

    def fail_data_marker(self, tag: str, version: int) -> None:
        if tag.startswith("nemo_sink_data_"):
            raise RuntimeError("injected lost acknowledgement")
        original_create(self, tag, version)

    monkeypatch.setattr(tags_type, "create", fail_data_marker)
    with pytest.raises(DataCommittedFinalizationError, match="delete pending Lance tag") as failure:
        backend.stream_ingest(_records(10, 14))
    assert failure.value.data_version == _table(tmp_path).version
    assert "nemo_sink_pending_" in str(failure.value)
    after_commit = _state(tmp_path)
    monkeypatch.setattr(tags_type, "create", original_create)

    with pytest.raises(CommitOutcomeUnknown, match="Do not replay these records"):
        backend.stream_ingest(_records(10, 14))
    assert _state(tmp_path) == after_commit


def test_explicit_overwrite_retries_after_data_marker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=True,
        stream_operation_id="overwrite-marker-failure",
    )
    tags_type = type(_table(tmp_path).tags)
    original_create = tags_type.create

    def fail_data_marker(self, tag: str, version: int) -> None:
        if tag.startswith("nemo_sink_data_"):
            raise RuntimeError("injected data marker failure")
        original_create(self, tag, version)

    monkeypatch.setattr(tags_type, "create", fail_data_marker)
    with pytest.raises(
        DataCommittedFinalizationError,
        match="Retry with stream_operation_id='overwrite-marker-failure'",
    ):
        backend.stream_ingest(_records(10, 14))
    monkeypatch.setattr(tags_type, "create", original_create)

    reconstructed = _backend(
        tmp_path,
        overwrite=True,
        stream_operation_id="overwrite-marker-failure",
    )
    reconstructed.stream_ingest(_records(10, 14))

    assert _state(tmp_path)[0] == ["row-10", "row-11", "row-12", "row-13"]
    tags = _table(tmp_path).tags.list()
    assert len([name for name in tags if name.startswith("nemo_sink_success_")]) == 1
    assert not [name for name in tags if name.startswith(("nemo_sink_pending_", "nemo_sink_data_"))]


def test_default_lost_append_acknowledgement_fails_closed_without_durable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(tmp_path).stream_ingest(_records(0, 2))
    backend = _backend(tmp_path, overwrite=False)
    table_type = type(_table(tmp_path))
    original_add = table_type.add

    def add_then_lose_acknowledgement(self, *args, **kwargs):
        original_add(self, *args, **kwargs)
        raise RuntimeError("injected lost append acknowledgement")

    monkeypatch.setattr(table_type, "add", add_then_lose_acknowledgement)
    with pytest.raises(CommitOutcomeUnknown, match="Do not replay these records"):
        backend.stream_ingest(_records(10, 14))

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert not [name for name in _table(tmp_path).tags.list() if name.startswith("nemo_sink_")]
    assert _backend(tmp_path, overwrite=False).retrieval([[0.0, 1.0]], top_k=10)


def test_default_lost_create_acknowledgement_fails_closed_without_durable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(tmp_path, overwrite=False)
    connection_type = type(lancedb.connect(str(tmp_path)))
    original_create_table = connection_type.create_table

    def create_then_lose_acknowledgement(self, *args, **kwargs):
        original_create_table(self, *args, **kwargs)
        raise RuntimeError("injected lost create acknowledgement")

    monkeypatch.setattr(connection_type, "create_table", create_then_lose_acknowledgement)
    with pytest.raises(CommitOutcomeUnknown, match="Do not replay these records"):
        backend.stream_ingest(_records(0, 2))

    assert _state(tmp_path)[0] == ["row-0", "row-1"]
    assert not [name for name in _table(tmp_path).tags.list() if name.startswith("nemo_sink_")]
