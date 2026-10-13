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
    assert len(lance.dataset(str(tmp_path / "worker" / "chunks.lance")).get_fragments()) == 4


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
