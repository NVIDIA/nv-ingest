# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The opt-in worker-native fragment sink stores exactly what the columnar driver stream stores."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")
lance = pytest.importorskip("lance")

from nemo_retriever.common.vdb._lancedb_stream import DataCommittedFinalizationError
from nemo_retriever.common.vdb._lancedb_stream_state import VdbOperationConflict
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import VdbUploadError
from nemo_retriever.graph.executor import arrow_table_to_pandas
from nemo_retriever.operators.vdb import IngestVdbOperator

DIM = 4


def _row(row_id: int, *, text: str | None = None) -> dict[str, Any]:
    return {
        "text": f"chunk {row_id}" if text is None else text,
        "embedding": [float(row_id) + 0.25 * index for index in range(DIM)],
        "document_type": "text",
        "source_id": f"/tmp/doc-{row_id % 3}.pdf",
        "metadata": {
            "content_metadata": {"type": "text", "id": f"row-{row_id}", "page_number": row_id},
            "source_metadata": {"source_id": f"/tmp/doc-{row_id % 3}.pdf", "source_name": f"doc-{row_id % 3}.pdf"},
        },
    }


def _frames(rows: list[dict[str, Any]], *, batch_size: int) -> list[pd.DataFrame]:
    table = pa.Table.from_pylist(rows)
    return [arrow_table_to_pandas(table.slice(start, batch_size)) for start in range(0, table.num_rows, batch_size)]


def _backend(uri: Path, **overrides: Any) -> LanceDB:
    kwargs: dict[str, Any] = {"uri": str(uri), "table_name": "chunks", "vector_dim": DIM, "build_index": False}
    kwargs.update(overrides)
    return LanceDB(**kwargs)


def _rows(uri: Path) -> pa.Table:
    table = lancedb.connect(str(uri)).open_table("chunks").to_arrow()
    return table.sort_by("id")


def _worker_write(backend: LanceDB, frames: list[pd.DataFrame]) -> int:
    """Drive the two halves the executor runs: one receipt per Ray block, then one driver commit."""
    operator = IngestVdbOperator(vdb=backend)
    receipts = [operator._write_partition(frame) for frame in frames]
    assert all(len(receipt["fragments"]) <= 1 for receipt in receipts)
    return operator._commit_partitions(receipts)


def test_worker_write_matches_columnar_stream(tmp_path: Path) -> None:
    rows = [_row(0), _row(1, text=""), *(_row(index) for index in range(2, 7))]  # row 1 is dropped: no text
    frames = _frames(rows, batch_size=2)

    stream_backend = _backend(tmp_path / "stream")
    assert stream_backend.supports_stream_ingest_batches
    stream_submitted = IngestVdbOperator(vdb=stream_backend)._stream_ingest(iter(frames))

    worker_backend = _backend(tmp_path / "worker")
    worker_submitted = _worker_write(worker_backend, frames)

    assert worker_submitted == stream_submitted == 6
    expected, actual = _rows(tmp_path / "stream"), _rows(tmp_path / "worker")
    assert actual.schema.equals(expected.schema, check_metadata=True)
    assert actual.schema.field("vector").type == pa.list_(pa.float32(), DIM)
    assert actual.num_rows == 6
    assert actual.equals(expected), f"{actual.to_pylist()} != {expected.to_pylist()}"
    worker_dataset = lance.dataset(str(tmp_path / "worker" / "chunks.lance"))
    assert len(worker_dataset.get_fragments()) == 4
    # Same data-file version as the LanceDB writer, so fragment-written tables scan like stream-written ones.
    stream_dataset = lance.dataset(str(tmp_path / "stream" / "chunks.lance"))
    assert worker_dataset.data_storage_version == stream_dataset.data_storage_version


def test_worker_write_overwrite_then_append(tmp_path: Path) -> None:
    frames = _frames([_row(index) for index in range(5)], batch_size=3)
    uri = tmp_path / "db"

    assert _worker_write(_backend(uri), frames) == 5
    assert _worker_write(_backend(uri), frames) == 5
    assert _rows(uri).num_rows == 5

    assert _worker_write(_backend(uri, overwrite=False), frames) == 5
    assert _rows(uri).num_rows == 10


def test_worker_write_without_partitions_creates_no_table(tmp_path: Path) -> None:
    backend = _backend(tmp_path / "db")
    assert IngestVdbOperator(vdb=backend)._commit_partitions([]) == 0
    assert "chunks" not in lancedb.connect(str(tmp_path / "db")).list_tables().tables


def test_worker_write_is_the_default_for_local_tables(tmp_path: Path) -> None:
    assert _backend(tmp_path / "db").supports_worker_write
    assert _backend(tmp_path / "db", sparse=True).supports_worker_write
    assert _backend(tmp_path / "db", stream_operation_id="job-1").supports_worker_write


def _vector_pages(uri: Path) -> list[tuple[int, int]]:
    """``(rows, vector pages)`` per data file: Lance writes one vector page per Arrow batch it received."""
    dataset = lancedb.connect(str(uri)).open_table("chunks").to_lance()
    vector_index = dataset.schema.get_field_index("vector")
    layout: list[tuple[int, int]] = []
    for fragment in dataset.get_fragments():
        for data_file in fragment.data_files():
            metadata = lance.file.LanceFileReader(f"{dataset.uri}/data/{data_file.path()}").metadata()
            layout.append((int(metadata.num_rows), len(metadata.columns[vector_index].pages)))
    return layout


def _max_rows_per_page(layout: list[tuple[int, int]]) -> int:
    return max(-(-rows // pages) for rows, pages in layout)


def test_worker_write_hybrid_builds_fts_index(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(40)]
    backend = _backend(tmp_path / "db", hybrid=True, build_index=True)
    assert _worker_write(backend, _frames(rows, batch_size=16)) == 40

    table = lancedb.connect(str(tmp_path / "db")).open_table("chunks")
    assert table.schema.metadata[b"retrieval_mode"] == b"hybrid"
    index_types = {str(index.index_type).lower() for index in table.list_indices()}
    assert any("fts" in index_type for index_type in index_types), index_types
    assert backend.get_index_metadata("retrieval_mode") == "hybrid"


def _wide_frames(count: int, *, dim: int, batch_size: int) -> list[pd.DataFrame]:
    """Frames with ``dim``-wide vectors; Lance only flushes vector pages once a batch reaches ~8 MiB."""
    import numpy as np

    table = pa.Table.from_pylist([_row(index) for index in range(count)])
    values = pa.array(np.random.default_rng(0).standard_normal(count * dim, dtype=np.float32))
    offsets = pa.array(np.arange(0, (count + 1) * dim, dim, dtype=np.int32))
    embeddings = pa.ListArray.from_arrays(offsets, values)
    table = table.set_column(table.schema.get_field_index("embedding"), "embedding", embeddings)
    return [arrow_table_to_pandas(table.slice(start, batch_size)) for start in range(0, count, batch_size)]


def test_optimize_keeps_vector_page_size(tmp_path: Path) -> None:
    from nemo_retriever.common.vdb.lancedb import _LANCE_ROWS_PER_PAGE, _optimize_lancedb_table

    dim, count = 2048, 3 * _LANCE_ROWS_PER_PAGE
    uri = tmp_path / "db"
    frames = _wide_frames(count, dim=dim, batch_size=count // 2)
    assert _worker_write(_backend(uri, vector_dim=dim), frames) == count
    before = _vector_pages(uri)
    assert len(before) == 2 and _max_rows_per_page(before) <= _LANCE_ROWS_PER_PAGE, before

    table = lancedb.connect(str(uri)).open_table("chunks")
    _optimize_lancedb_table(table)
    after = _vector_pages(uri)
    assert len(after) == 1 and sum(rows for rows, _ in after) == count, after
    assert _max_rows_per_page(after) <= _LANCE_ROWS_PER_PAGE, after
    assert _rows(uri).num_rows == count


def _unsearchable_row(row_id: int) -> dict[str, Any]:
    """A status-style row with no text and no embedding; the stream path skips it."""
    row = _row(row_id, text="")
    row["embedding"] = None
    return row


def test_worker_write_infers_vector_dim_and_rejects_disagreeing_partitions(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    assert (
        _worker_write(_backend(uri, vector_dim=None), _frames([_row(index) for index in range(4)], batch_size=2)) == 4
    )
    assert _rows(uri).schema.field("vector").type.list_size == DIM

    short = _row(9)
    short["embedding"] = short["embedding"][:-1]
    with pytest.raises(ValueError, match="disagree on vector_dim"):
        _worker_write(
            _backend(tmp_path / "other", vector_dim=None),
            _frames([_row(1)], batch_size=1) + _frames([short], batch_size=1),
        )
    assert not list((tmp_path / "other" / "chunks.lance" / "data").glob("*"))


def test_worker_write_finalization_failure_reports_committed_data(tmp_path: Path, monkeypatch) -> None:
    uri = tmp_path / "db"
    backend = _backend(uri)

    def failing_finalize(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("index build failed")

    monkeypatch.setattr(backend, "_finalize_stream_write", failing_finalize)
    with pytest.raises(DataCommittedFinalizationError, match="version 1"):
        _worker_write(backend, _frames([_row(index) for index in range(4)], batch_size=2))
    assert _rows(uri).num_rows == 4


def test_worker_write_skips_unsearchable_partition_when_others_have_rows(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    frames = _frames([_unsearchable_row(0), _unsearchable_row(1), _row(2), _row(3)], batch_size=2)
    assert _worker_write(_backend(uri), frames) == 2
    assert _rows(uri)["text"].to_pylist() == ["chunk 2", "chunk 3"]


def test_worker_write_applies_fail_closed_rules_over_the_whole_upload(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    only_unsearchable = _frames([_unsearchable_row(0), _unsearchable_row(1)], batch_size=1)
    with pytest.raises(VdbUploadError):
        _worker_write(_backend(uri), only_unsearchable)

    missing = _row(0)
    missing["embedding"] = None
    partial = _frames([missing, _row(1)], batch_size=1)
    with pytest.raises(VdbUploadError, match="missing embeddings"):
        _worker_write(_backend(uri), partial)
    # Fragments may be on disk, but nothing was committed: no manifest exists.
    assert not list((uri / "chunks.lance").glob("_versions/*.manifest"))


def _data_files(uri: Path) -> list[Path]:
    return sorted((uri / "chunks.lance" / "data").glob("*.lance"))


def test_worker_write_rows_must_be_positive(tmp_path: Path) -> None:
    for rows in (0, -1):
        with pytest.raises(ValueError, match="worker_write_rows"):
            _backend(tmp_path / "db", worker_write_rows=rows)


def test_rejected_upload_removes_its_fragment_files(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    missing = _row(0)
    missing["embedding"] = None
    with pytest.raises(VdbUploadError, match="missing embeddings"):
        _worker_write(_backend(uri), _frames([_row(1), _row(2), missing], batch_size=2))
    assert _data_files(uri) == []


def test_rejected_append_keeps_committed_files_and_removes_its_own(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    first = _backend(uri, embedding_model_name="model-a")
    _worker_write(first, _frames([_row(index) for index in range(4)], batch_size=2))
    committed = _data_files(uri)
    append = _backend(uri, overwrite=False, embedding_model_name="another-model")
    with pytest.raises(ValueError, match="model"):
        _worker_write(append, _frames([_row(index) for index in range(4, 8)], batch_size=2))
    assert _data_files(uri) == committed
    assert _rows(uri).num_rows == 4


class _RecordingDataset:
    """Stands in for a Ray dataset: records ``map_batches`` options and runs the writer on pandas frames."""

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = frames
        self.options: dict = {}

    def map_batches(self, fn, **options):
        self.options = options
        fn = fn() if isinstance(fn, type) else fn
        outputs = [
            fn(frame if options["batch_format"] == "pandas" else pa.Table.from_pandas(frame)) for frame in self.frames
        ]
        return SimpleNamespace(iter_rows=lambda: iter([row for out in outputs for row in out.to_dict("records")]))


@pytest.mark.parametrize("preserve, batch_format", [(True, "pandas"), (False, "pyarrow")])
def test_executor_worker_write_batch_format_and_driver_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, batch_format: str
) -> None:
    ray = pytest.importorskip("ray")
    from ray.data import ActorPoolStrategy

    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.pipeline_graph import Node

    vdb_kwargs = {"uri": str(tmp_path / "db"), "table_name": "chunks", "vector_dim": DIM, "build_index": False}
    vdb_kwargs.update(worker_write_rows=3)
    node = Node(IngestVdbOperator(vdb_op="lancedb", vdb_kwargs=vdb_kwargs))
    dataset = _RecordingDataset([pd.DataFrame([_row(index) for index in range(3)])])
    driver_node = "a" * 56
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: driver_node))
    result = RayDataExecutor.__new__(RayDataExecutor)._worker_write_ingest(
        dataset, node, preserve_pandas_blocks=preserve
    )
    assert dataset.options["batch_format"] == batch_format
    assert dataset.options["batch_size"] == 3
    compute = dataset.options["compute"]
    assert isinstance(compute, ActorPoolStrategy)
    assert (compute.min_size, compute.max_size) == (2, 2)
    assert dataset.options["num_cpus"] == 0
    strategy = dataset.options["scheduling_strategy"]
    assert (strategy.node_id, strategy.soft) == (driver_node, False)
    assert result["submitted_records"].tolist() == [3]


def test_executor_removes_finished_fragments_when_a_write_actor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ray = pytest.importorskip("ray")
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.pipeline_graph import Node

    class _FailingDataset(_RecordingDataset):
        def map_batches(self, fn, **options):
            finished = super().map_batches(fn, **options)

            def rows():
                yield from finished.iter_rows()
                raise RuntimeError("write actor failed")

            return SimpleNamespace(iter_rows=rows)

    uri = tmp_path / "db"
    vdb_kwargs = {"uri": str(uri), "table_name": "chunks", "vector_dim": DIM, "build_index": False}
    node = Node(IngestVdbOperator(vdb_op="lancedb", vdb_kwargs=vdb_kwargs))
    dataset = _FailingDataset(_frames([_row(index) for index in range(4)], batch_size=2))
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "a" * 56))
    with pytest.raises(RuntimeError, match="write actor failed"):
        RayDataExecutor.__new__(RayDataExecutor)._worker_write_ingest(dataset, node)
    assert not list((uri / "chunks.lance" / "data").glob("*"))


def test_worker_write_operation_id_retry_does_not_append_again(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    rows = [_row(index) for index in range(6)]
    assert _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-1"), _frames(rows, batch_size=2)) == 6
    committed = _data_files(uri)
    # Different partitioning, same rows: the digest is order- and partition-independent.
    _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-1"), _frames(rows[::-1], batch_size=4))
    assert _rows(uri).num_rows == 6
    assert _data_files(uri) == committed


def test_worker_write_operation_id_rejects_different_rows(tmp_path: Path) -> None:
    uri = tmp_path / "db"
    _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-1"), _frames([_row(1)], batch_size=1))
    committed = _data_files(uri)
    with pytest.raises(VdbOperationConflict):
        _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-1"), _frames([_row(2)], batch_size=1))
    assert _data_files(uri) == committed


def test_worker_write_operation_id_resumes_after_finalization_failure(tmp_path: Path, monkeypatch) -> None:
    uri = tmp_path / "db"
    _worker_write(_backend(uri, overwrite=False), _frames([_row(100)], batch_size=1))
    frames = _frames([_row(index) for index in range(4)], batch_size=2)
    original = LanceDB._finalize_stream_write
    monkeypatch.setattr(LanceDB, "_finalize_stream_write", lambda self, **kwargs: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(DataCommittedFinalizationError):
        _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-2"), frames)
    monkeypatch.setattr(LanceDB, "_finalize_stream_write", original)
    _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-2"), frames)
    assert _rows(uri).num_rows == 5


def test_worker_write_applies_sidecar_metadata_like_the_stream(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.csv"
    pd.DataFrame({"source": ["/tmp/doc-0.pdf", "/tmp/doc-1.pdf"], "team": ["alpha", "beta"]}).to_csv(
        sidecar, index=False
    )
    frames = _frames([_row(index) for index in range(5)], batch_size=2)

    def vdb_kwargs(name: str) -> dict[str, Any]:
        return {
            "uri": str(tmp_path / name),
            "table_name": "chunks",
            "vector_dim": DIM,
            "build_index": False,
            "meta_dataframe": str(sidecar),
            "meta_source_field": "source",
            "meta_fields": ["team"],
        }

    IngestVdbOperator(vdb_op="lancedb", vdb_kwargs=vdb_kwargs("stream"))._stream_ingest(iter(frames))
    kwargs = {"vdb_op": "lancedb", "vdb_kwargs": vdb_kwargs("worker")}
    operator = IngestVdbOperator(**kwargs)
    assert operator._supports_worker_write()
    write = operator._partition_writer()
    sidecar.unlink()  # The write actors use the driver's lookup, not the file.
    assert operator._commit_partitions([write(frame) for frame in frames]) == 5

    expected, actual = _rows(tmp_path / "stream"), _rows(tmp_path / "worker")
    assert actual.equals(expected)
    assert any('"team":"alpha"' in value for value in actual.column("metadata").to_pylist())


def test_worker_write_sparse_table_matches_the_stream(tmp_path: Path) -> None:
    frames = _frames([_row(0), _row(1, text=""), *(_row(index) for index in range(2, 6))], batch_size=2)
    stream_backend = _backend(tmp_path / "stream", sparse=True, vector_dim=None)
    assert not stream_backend.supports_stream_ingest_batches
    stream_submitted = IngestVdbOperator(vdb=stream_backend)._stream_ingest(iter(frames))

    worker_backend = _backend(tmp_path / "worker", sparse=True, vector_dim=None)
    assert worker_backend.supports_worker_write
    assert _worker_write(worker_backend, frames) == stream_submitted

    expected, actual = _rows(tmp_path / "stream"), _rows(tmp_path / "worker")
    assert "vector" not in actual.schema.names
    assert actual.schema.equals(expected.schema, check_metadata=True)
    assert actual.equals(expected)
    assert actual.num_rows == 5


def test_return_results_writes_from_workers_and_returns_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.graph import executor as executor_module
    from nemo_retriever.graph.executor import RayDataExecutor

    sink = SimpleNamespace(_supports_worker_write=lambda: True)
    calls: list[str] = []
    materialized = SimpleNamespace(name="materialized")
    dataset = SimpleNamespace(materialize=lambda: calls.append("materialize") or materialized)
    executor = RayDataExecutor.__new__(RayDataExecutor)
    executor.graph = None
    monkeypatch.setattr(executor, "_linearize", lambda graph: [SimpleNamespace(operator=sink)])
    monkeypatch.setattr(executor, "_stream_ingest_index", lambda nodes, require_terminal: 0)
    monkeypatch.setattr(executor, "_build_dataset", lambda data, segment: dataset)
    monkeypatch.setattr(executor_module, "_requires_stable_pandas_blocks", lambda nodes: False)

    def worker_write(ds, node, **kwargs):
        calls.append("write")
        assert ds is materialized
        return pd.DataFrame([{"input_rows": 2, "submitted_records": 2}])

    monkeypatch.setattr(executor, "_worker_write_ingest", worker_write)
    monkeypatch.setattr(executor_module, "ray_dataset_to_pandas", lambda ds: pd.DataFrame({"read": [ds.name]}))

    result = executor._ingest(None, return_results=True)

    assert calls == ["materialize", "write"]
    assert result["read"].tolist() == ["materialized"]


def test_worker_write_operation_id_empty_retry_cannot_skip_finalization(tmp_path: Path, monkeypatch) -> None:
    uri = tmp_path / "db"
    _worker_write(_backend(uri, overwrite=False), _frames([_row(100)], batch_size=1))
    monkeypatch.setattr(LanceDB, "_finalize_stream_write", lambda self, **kwargs: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(DataCommittedFinalizationError):
        _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-3"), _frames([_row(0)], batch_size=1))
    monkeypatch.undo()
    with pytest.raises(VdbOperationConflict):
        _worker_write(_backend(uri, overwrite=False, stream_operation_id="job-3"), [])


def test_worker_write_ships_a_supplied_backend_to_the_tasks(tmp_path: Path) -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    operator = IngestVdbOperator(vdb=_backend(tmp_path / "db"))
    write = cloudpickle.loads(cloudpickle.dumps(operator._partition_writer()))
    frames = _frames([_row(index) for index in range(5)], batch_size=2)
    assert operator._commit_partitions([write(frame) for frame in frames]) == 5
    assert _rows(tmp_path / "db").num_rows == 5


def test_worker_append_without_vector_dim_takes_the_table_width(tmp_path: Path) -> None:
    short = {**_row(0), "embedding": [0.5, 0.5, 0.5]}
    frames = _frames([short, _row(1), _row(2)], batch_size=3)
    for name in ("stream", "worker"):
        _worker_write(_backend(tmp_path / name), _frames([_row(100)], batch_size=1))

    stream = IngestVdbOperator(vdb=_backend(tmp_path / "stream", overwrite=False, vector_dim=None))
    stream_submitted = stream._stream_ingest(iter(frames))
    operator = IngestVdbOperator(vdb=_backend(tmp_path / "worker", overwrite=False, vector_dim=None))
    write = operator._partition_writer()
    assert operator._commit_partitions([write(frame) for frame in frames]) == stream_submitted

    assert _rows(tmp_path / "worker").equals(_rows(tmp_path / "stream"))


def test_worker_append_with_every_row_dropped_keeps_the_table_version(tmp_path: Path) -> None:
    bad = {**_row(5), "embedding": [float("nan")] * DIM}
    frames = _frames([bad], batch_size=1)
    versions = {}
    for name in ("stream", "worker"):
        _worker_write(_backend(tmp_path / name), _frames([_row(100)], batch_size=1))
        backend = _backend(tmp_path / name, overwrite=False, on_bad_vectors="drop", validate_vector_length=False)
        before = lancedb.connect(str(tmp_path / name)).open_table("chunks").version
        files_before = _data_files(tmp_path / name)
        if name == "stream":
            IngestVdbOperator(vdb=backend)._stream_ingest(iter(frames))
        else:
            _worker_write(backend, frames)
        versions[name] = lancedb.connect(str(tmp_path / name)).open_table("chunks").version - before
        assert _data_files(tmp_path / name) == files_before

    assert versions == {"stream": 0, "worker": 0}
    assert _rows(tmp_path / "worker").equals(_rows(tmp_path / "stream"))
