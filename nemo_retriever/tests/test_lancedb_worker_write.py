# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The opt-in worker-native fragment sink stores exactly what the columnar driver stream stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")
lance = pytest.importorskip("lance")

from nemo_retriever.common.vdb.lancedb import LanceDB
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

    worker_backend = _backend(tmp_path / "worker", worker_write=True)
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
    assert worker_dataset.data_storage_version == stream_dataset.data_storage_version == "2.1"


def test_worker_write_overwrite_then_append(tmp_path: Path) -> None:
    frames = _frames([_row(index) for index in range(5)], batch_size=3)
    uri = tmp_path / "db"

    assert _worker_write(_backend(uri, worker_write=True), frames) == 5
    assert _worker_write(_backend(uri, worker_write=True), frames) == 5
    assert _rows(uri).num_rows == 5

    assert _worker_write(_backend(uri, worker_write=True, overwrite=False), frames) == 5
    assert _rows(uri).num_rows == 10


def test_worker_write_without_partitions_creates_no_table(tmp_path: Path) -> None:
    backend = _backend(tmp_path / "db", worker_write=True)
    assert IngestVdbOperator(vdb=backend)._commit_partitions([]) == 0
    assert "chunks" not in lancedb.connect(str(tmp_path / "db")).list_tables().tables


def test_worker_write_requires_dense_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="worker_write"):
        _backend(tmp_path / "db", worker_write=True, sparse=True)


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
    backend = _backend(tmp_path / "db", worker_write=True, hybrid=True, build_index=True)
    assert _worker_write(backend, _frames(rows, batch_size=16)) == 40

    table = lancedb.connect(str(tmp_path / "db")).open_table("chunks")
    assert table.schema.metadata[b"retrieval_mode"] == b"hybrid"
    index_types = {str(index.index_type).lower() for index in table.list_indices()}
    assert any("fts" in index_type for index_type in index_types), index_types
    assert backend.get_index_metadata("retrieval_mode") == "hybrid"


def test_sdk_vdb_kwargs_keep_worker_write_and_default_to_hybrid(tmp_path: Path) -> None:
    from nemo_retriever.common.params.models import VdbUploadParams
    from nemo_retriever.ingest.index_mode import resolve_vdb_upload_kwargs

    params = VdbUploadParams(vdb_kwargs={"uri": str(tmp_path / "db"), "worker_write": True})
    kwargs = resolve_vdb_upload_kwargs(params)
    assert kwargs["worker_write"] is True
    assert kwargs["hybrid"] is True


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
    assert _worker_write(_backend(uri, worker_write=True, vector_dim=dim), frames) == count
    before = _vector_pages(uri)
    assert len(before) == 2 and _max_rows_per_page(before) <= _LANCE_ROWS_PER_PAGE, before

    table = lancedb.connect(str(uri)).open_table("chunks")
    _optimize_lancedb_table(table)
    after = _vector_pages(uri)
    assert len(after) == 1 and sum(rows for rows, _ in after) == count, after
    assert _max_rows_per_page(after) <= _LANCE_ROWS_PER_PAGE, after
    assert _rows(uri).num_rows == count
