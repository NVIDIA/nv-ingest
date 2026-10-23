# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Arrow and recovery helpers for LanceDB streaming ingestion.

The ``LanceDB`` adapter owns write orchestration. This module contains the
backend-private mechanics that do not need adapter state.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nemo_retriever.common.vdb._lancedb_stream_state import (
    VdbOperationConflict,
    _incomplete_marker_names,
    _operation_marker_names,
    _state_marker_prefix,
)
from nemo_retriever.common.vdb.lancedb_capabilities import _table_schema

_CREATE_OPERATION_KEY = b"nemo_retriever.sink_create_operation_sha256"
_CREATE_REQUEST_KEY = b"nemo_retriever.sink_create_request_sha256"
# Estimates can undercount Python-object overhead. This cap bounds the number
# of canonical dictionaries retained before an exact Arrow allocation decides
# whether to emit or bisect the candidate batch.
_MAX_PENDING_CANONICAL_ROWS = 256


class OversizedVdbRowError(ValueError):
    """One canonical stored row cannot fit in the configured Arrow budget."""


class VdbWriteNotFinalized(RuntimeError):
    """A reader reached a table whose coordinated sink lifecycle is incomplete."""


class DataCommittedFinalizationError(RuntimeError):
    """LanceDB committed data, but its durable marker or finalization failed."""

    def __init__(
        self,
        table_name: str,
        data_version: int,
        *,
        retry_operation_id: str | None = None,
        recovery_action: str | None = None,
    ) -> None:
        if recovery_action is not None:
            action = recovery_action
        elif retry_operation_id is None:
            action = (
                "Do not replay these records. Inspect the committed table version and complete "
                "any required index or optimization maintenance separately."
            )
        else:
            action = (
                f"Retry with stream_operation_id={retry_operation_id!r} to resume finalization; "
                "do not replay with another operation ID."
            )
        super().__init__(
            f"LanceDB data committed to table {table_name!r} at version {data_version}, "
            f"but finalization failed. {action}"
        )
        self.data_version = int(data_version)


def _assert_lancedb_table_ready(table: Any) -> None:
    """Reject reads while a bounded sink operation is pending finalization."""

    tags = table.tags.list()
    incomplete = sorted(_incomplete_marker_names(tags))
    if incomplete:
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has a data write that is not finalized; "
            "retry the original operation with its original stream_operation_id before reading. "
            f"If the retry reports an indeterminate append, inspect the committed version and reconcile {incomplete!r} "
            "before deleting any pending tag."
        )

    if not _bounded_create_is_finalized(table):
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} was created by a sink operation that is not finalized; "
            "retry the original create with its original stream_operation_id. If that ID is unavailable, "
            "inspect and remove or rebuild the table before reading."
        )


def _bounded_create_is_finalized(table: Any) -> bool:
    """Return whether namespaced bounded-create metadata has a success marker."""

    metadata = table.schema.metadata or {}
    create_operation = metadata.get(_CREATE_OPERATION_KEY)
    create_request = metadata.get(_CREATE_REQUEST_KEY)
    if create_operation is None and create_request is None:
        return True
    if create_operation is None or create_request is None:
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has incomplete sink creation metadata and cannot be recovered "
            "automatically; inspect and remove or rebuild the table before reading."
        )
    success_prefix = (
        _state_marker_prefix(
            "success",
            create_operation.decode("ascii")[:24],
            create_request.decode("ascii")[:24],
        )
        + "_"
    )
    return any(name.startswith(success_prefix) for name in table.tags.list())


@dataclass(slots=True)
class _StreamStats:
    client_records: int = 0
    rows_written: int = 0
    stored_hash_sum: int = 0
    stored_hash_xor: int = 0
    vector_dim: int | None = None

    @property
    def digest(self) -> str:
        width = 32
        modulus = 1 << (width * 8)
        payload = (
            int(self.rows_written).to_bytes(8, "big")
            + int(self.stored_hash_sum % modulus).to_bytes(width, "big")
            + int(self.stored_hash_xor).to_bytes(width, "big")
        )
        return hashlib.sha256(payload).hexdigest()


def _estimated_arrow_row_bytes(row: dict[str, Any], schema: pa.Schema) -> int:
    """Return a conservative buffer-size estimate for one canonical row."""

    # Offset/validity/alignment overhead is deliberately over-counted.  The
    # emitted RecordBatch is still measured exactly before it crosses into
    # LanceDB.
    total = 64
    for field in schema:
        value = row.get(field.name)
        if pa.types.is_fixed_size_list(field.type):
            # Arrow materializes the fixed-width child buffer even when the
            # parent list is null.
            total += 32 + int(field.type.list_size) * 4
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            total += 32 + (len(str(value).encode("utf-8")) if value is not None else 0)
        elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            total += 16
        elif value is None:
            total += 8
        else:
            total += 64 + len(str(value).encode("utf-8"))
    return total


def _exact_single_arrow_row_bytes(row: dict[str, Any], schema: pa.Schema) -> int | None:
    """Compute one canonical row's Arrow buffers without allocating them."""

    total = 0
    for field in schema:
        value = row.get(field.name)
        field_type = field.type
        if pa.types.is_fixed_size_list(field_type) and pa.types.is_float32(field_type.value_type):
            list_size = int(field_type.list_size)
            total += list_size * 4
            if value is None:
                total += 1 + ((list_size + 7) // 8)
            elif any(item is None for item in value):
                total += (list_size + 7) // 8
        elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
            total += 16 if pa.types.is_large_string(field_type) else 8
            if value is None:
                total += 1
            else:
                total += len(str(value).encode("utf-8"))
        elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
            total += int(field_type.bit_width) // 8
            if value is None:
                total += 1
        else:
            return None
    return total


def _record_batch(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=schema)


_STORED_ROW_DIGEST_DOMAIN = b"nemo-retriever-vdb-row-v1\0"


def _record_stream_batch(stats: _StreamStats, batch: pa.RecordBatch, *, include_digest: bool) -> None:
    """Record emitted rows and, when requested, their batch-independent identity."""

    stats.rows_written += batch.num_rows
    if not include_digest:
        return

    for row_index in range(batch.num_rows):
        hasher = hashlib.sha256(_STORED_ROW_DIGEST_DOMAIN)
        # ``take`` compacts the row; ``slice`` can retain unrelated buffers
        # from the source batch and make the identity batch-boundary dependent.
        row = batch.take(pa.array([row_index], type=pa.int32()))
        hasher.update(row.serialize())
        row_hash = int.from_bytes(hasher.digest(), "big")
        stats.stored_hash_sum += row_hash
        stats.stored_hash_xor ^= row_hash


def _checked_batches(
    rows: Iterable[dict[str, Any]],
    *,
    schema: pa.Schema,
    max_batch_bytes: int,
    stats: _StreamStats,
    include_digest: bool,
) -> Iterator[pa.RecordBatch]:
    """Pack rows into owned Arrow batches under ``max_batch_bytes``."""

    pending: list[dict[str, Any]] = []
    estimated_bytes = 0

    def emit(candidate: list[dict[str, Any]]) -> Iterator[pa.RecordBatch]:
        batch = _record_batch(candidate, schema)
        retained_bytes = int(batch.get_total_buffer_size())
        if retained_bytes > max_batch_bytes:
            if len(candidate) == 1:
                raise OversizedVdbRowError(
                    "One canonical VDB row requires "
                    f"{retained_bytes} Arrow buffer bytes, exceeding max_batch_bytes={max_batch_bytes}."
                )
            midpoint = len(candidate) // 2
            yield from emit(candidate[:midpoint])
            yield from emit(candidate[midpoint:])
            return
        _record_stream_batch(stats, batch, include_digest=include_digest)
        yield batch

    for row in rows:
        exact_row_bytes = _exact_single_arrow_row_bytes(row, schema)
        if exact_row_bytes is not None and exact_row_bytes > max_batch_bytes:
            raise OversizedVdbRowError(
                "One canonical VDB row requires "
                f"{exact_row_bytes} Arrow buffer bytes, exceeding max_batch_bytes={max_batch_bytes}."
            )
        row_estimate = _estimated_arrow_row_bytes(row, schema)
        if pending and (
            len(pending) >= _MAX_PENDING_CANONICAL_ROWS or estimated_bytes + row_estimate > max_batch_bytes
        ):
            yield from emit(pending)
            pending = []
            estimated_bytes = 0
        pending.append(row)
        estimated_bytes += row_estimate
    if pending:
        yield from emit(pending)


@dataclass(frozen=True, slots=True)
class _PreparedColumns:
    """One projected batch: fixed-width vectors plus the string columns, row aligned."""

    vector: pa.Array
    text: pa.Array
    metadata: pa.Array
    source: pa.Array
    id: pa.Array

    def record_batch(self, schema: pa.Schema) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [self.vector, self.text, self.metadata, self.source, self.id],
            schema=schema,
        )


def _bounded_record_batches(
    prepared: Iterable[_PreparedColumns],
    *,
    schema: pa.Schema,
    max_batch_bytes: int,
    stats: _StreamStats,
    include_digest: bool,
) -> Iterator[pa.RecordBatch]:
    """Emit owned Arrow batches under ``max_batch_bytes``, bisecting oversized input."""

    def emit(batch: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
        retained_bytes = int(batch.get_total_buffer_size())
        if retained_bytes > max_batch_bytes:
            if batch.num_rows == 1:
                raise OversizedVdbRowError(
                    "One canonical VDB row requires "
                    f"{retained_bytes} Arrow buffer bytes, exceeding max_batch_bytes={max_batch_bytes}."
                )
            midpoint = batch.num_rows // 2
            # ``take`` compacts; a slice keeps the whole parent buffers.
            yield from emit(batch.take(pa.array(np.arange(0, midpoint), type=pa.int64())))
            yield from emit(batch.take(pa.array(np.arange(midpoint, batch.num_rows), type=pa.int64())))
            return
        _record_stream_batch(stats, batch, include_digest=include_digest)
        yield batch

    for columns in prepared:
        if len(columns.vector) == 0:
            continue
        yield from emit(columns.record_batch(schema))


def _estimated_prepared_row_bytes(fields: dict[str, Any], *, vector_dim: int) -> int:
    """Conservative Arrow byte estimate for one projected row (vector plus four strings)."""

    return (
        64
        + 32
        + int(vector_dim) * 4
        + sum(32 + len(str(fields[name]).encode("utf-8")) for name in ("text", "metadata", "source", "id"))
    )


def _row_chunks(positions: np.ndarray, estimates: list[int], *, max_batch_bytes: int) -> Iterator[np.ndarray]:
    """Split row positions into consecutive chunks whose estimated bytes stay under the limit."""

    start = 0
    chunk_bytes = 0
    for index, estimate in enumerate(estimates):
        if index > start and chunk_bytes + estimate > max_batch_bytes:
            yield positions[start:index]
            start = index
            chunk_bytes = 0
        chunk_bytes += estimate
    if start < len(estimates):
        yield positions[start:]


def _vector_lengths(vectors: pa.Array) -> np.ndarray:
    return pc.fill_null(pc.list_value_length(vectors), 0).to_numpy(zero_copy_only=False)


def _python_float_vectors(vectors: pa.Array) -> pa.Array:
    """Coerce non-numeric list elements with Python ``float()``, exactly as :func:`_inspect_vector` does.

    Only reached for string-typed embedding columns, which the embed stage never
    produces; ordinary float columns skip this.
    """

    return pa.array(
        [
            None if vector is None else [None if item is None else float(item) for item in vector]
            for vector in vectors.to_pylist()
        ],
        type=pa.list_(pa.float64()),
    )


def _apply_bad_vector_policy(
    vectors: pa.Array,
    *,
    vector_dim: int,
    on_bad_vectors: str,
    fill_value: float,
) -> tuple[np.ndarray, pa.Array]:
    """Columnar twin of :func:`_apply_deferred_bad_vector_policy`.

    ``vectors`` holds non-null ``list`` values. Returns the kept row
    positions and their vectors as ``fixed_size_list<float32, vector_dim>``,
    with wrong-width and NaN vectors dropped, filled, nulled, or rejected the
    same way the row path does. As on the row path, the width check runs
    first; only correctly sized vectors are then coerced with ``float()``, so a
    non-numeric value in one of them raises ``ValueError``.
    """

    fixed_type = pa.list_(pa.float32(), int(vector_dim))
    row_count = len(vectors)
    wrong_dim = _vector_lengths(vectors) != vector_dim
    sized = np.flatnonzero(~wrong_dim)
    sized_vectors = vectors.take(pa.array(sized, type=pa.int64()))
    child_type = sized_vectors.type.value_type
    if not (pa.types.is_floating(child_type) or pa.types.is_integer(child_type)):
        sized_vectors = _python_float_vectors(sized_vectors)
    has_nan = np.zeros(row_count, dtype=bool)
    if sized.size:
        nan_flags = pc.fill_null(pc.is_nan(pc.list_flatten(sized_vectors)), False).to_numpy(zero_copy_only=False)
        has_nan[sized] = nan_flags.reshape(sized.size, vector_dim).any(axis=1)
    bad = wrong_dim | has_nan
    if not bad.any():
        return np.arange(row_count), sized_vectors.cast(fixed_type)

    if on_bad_vectors == "error":
        first_bad = int(np.flatnonzero(bad)[0])
        detail = f"expected dimension {vector_dim}" if wrong_dim[first_bad] else "contains NaN"
        raise ValueError(
            f"Invalid LanceDB vector: {detail}. Set on_bad_vectors to 'drop', 'fill', or 'null' to handle it."
        )
    good = np.flatnonzero(~bad)
    good_vectors = sized_vectors.take(pa.array(np.searchsorted(sized, good), type=pa.int64())).cast(fixed_type)
    if on_bad_vectors == "drop":
        return good, good_vectors

    bad_count = int(bad.sum())
    if on_bad_vectors == "fill":
        # LanceDB 0.34 replaces the complete vector when its width is wrong or
        # any element is NaN.
        replacement = pa.array([[float(fill_value)] * vector_dim] * bad_count, type=fixed_type)
    else:
        replacement = pa.nulls(bad_count, type=fixed_type)
    merged = pa.concat_arrays([good_vectors, replacement])
    order = np.empty(row_count, dtype=np.int64)
    order[good] = np.arange(good.size)
    order[np.flatnonzero(bad)] = good.size + np.arange(bad_count)
    return np.arange(row_count), merged.take(pa.array(order, type=pa.int64()))


def _infer_vector_dim_with_spooled_prefix(
    rows: Iterator[dict[str, Any]],
    *,
    validate_vector_length: bool,
    on_bad_vectors: str,
) -> tuple[int, Iterator[dict[str, Any]]]:
    """Infer from the first nonempty list without retaining an unbounded prefix.

    ``infer_vector_dim`` historically scans until it finds a nonempty Python
    list. Canonical rows before that point still need to be replayed after the
    width is known (a tuple, for example, can then pass length validation).
    Store that rare lookahead prefix on disk so input order and legacy policy
    stay exact without making driver memory depend on prefix length.
    """

    prefix = None
    first_inferable: dict[str, Any] | None = None
    vector_dim = 0
    complete = False
    try:
        for row in rows:
            vector = row.get("vector")
            if isinstance(vector, list) and vector:
                vector_dim = len(vector)
                first_inferable = row
                break
            if prefix is None:
                # Ownership transfers to ``replay`` so the file stays open
                # until the prefix has been consumed.
                prefix = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
            pickle.dump(row, prefix, protocol=pickle.HIGHEST_PROTOCOL)
        complete = True
    finally:
        if not complete and prefix is not None:
            prefix.close()

    if first_inferable is None:
        if prefix is not None:
            prefix.close()
        raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")

    enforce_dim = validate_vector_length and on_bad_vectors != "error"

    def replay() -> Iterator[dict[str, Any]]:
        try:
            if prefix is not None:
                prefix.seek(0)
                while True:
                    try:
                        row = pickle.load(prefix)
                    except EOFError:
                        break
                    vector = row.get("vector")
                    if not enforce_dim or (isinstance(vector, (list, tuple)) and len(vector) == vector_dim):
                        yield row
            yield first_inferable
            yield from rows
        finally:
            if prefix is not None:
                prefix.close()

    return vector_dim, replay()


_VectorStatus = Literal["valid", "wrong_dim", "nan", "uncoercible"]


def _inspect_vector(vector: Any, vector_dim: int) -> tuple[_VectorStatus, list[float | None]]:
    """Classify one vector and normalize values that Arrow can represent."""

    try:
        if len(vector) != vector_dim:
            return "wrong_dim", []
    except TypeError:
        return "wrong_dim", []

    normalized: list[float | None] = []
    has_nan = False
    for value in vector:
        if value is None:
            normalized_value = None
        else:
            try:
                normalized_value = float(value)
            except (TypeError, ValueError):
                return "uncoercible", []
        if normalized_value is not None and math.isnan(normalized_value):
            has_nan = True
        normalized.append(normalized_value)
    return ("nan" if has_nan else "valid"), normalized


def _apply_deferred_bad_vector_policy(
    rows: Iterable[dict[str, Any]],
    *,
    vector_dim: int,
    sparse: bool,
    on_bad_vectors: str,
    fill_value: float,
) -> Iterator[dict[str, Any]]:
    """Apply the LanceDB writer policy before Arrow fixes the vector width.

    The legacy writer gives LanceDB Python rows, so LanceDB can drop, fill, or
    reject variable-length vectors while converting them to its fixed-width
    schema.  A native RecordBatch must already satisfy that schema.  Reproduce
    the pinned LanceDB policy here only when the legacy wrapper intentionally
    defers validation to the writer.
    """

    if sparse:
        yield from rows
        return

    for row in rows:
        vector = row.get("vector")
        if isinstance(vector, (list, tuple)):
            vector_values = vector
        elif isinstance(vector, np.ndarray) and vector.ndim == 1:
            # Pinned LanceDB accepts one-dimensional NumPy arrays as nested
            # Python vectors. Other merely iterable containers fail its Arrow
            # conversion, so do not broaden this normalization arbitrarily.
            vector_values = vector
        else:
            vector_values = None
        status, normalized_vector = _inspect_vector(vector_values, vector_dim)
        if status == "uncoercible":
            # Let Arrow report non-coercible values exactly as the legacy
            # LanceDB Python-row conversion does. They are not shape/NaN cases
            # governed by ``on_bad_vectors``.
            yield row
            continue
        if status == "valid":
            yield {**row, "vector": normalized_vector}
            continue

        if on_bad_vectors == "drop":
            continue
        if on_bad_vectors == "fill":
            # LanceDB 0.34 replaces the complete vector when its width is
            # wrong or any element is NaN. Preserve that supported runtime
            # contract instead of retaining a prefix that legacy ingestion
            # discards.
            yield {**row, "vector": [float(fill_value)] * vector_dim}
            continue
        if on_bad_vectors == "null":
            yield {**row, "vector": None}
            continue

        detail = f"expected dimension {vector_dim}" if status == "wrong_dim" else "contains NaN"
        raise ValueError(
            f"Invalid LanceDB vector: {detail}. Set on_bad_vectors to 'drop', 'fill', or 'null' to handle it."
        )


def _reject_empty_operation_bypass(table: Any | None, *, operation_id: str) -> None:
    """Keep a true empty no-op from concealing durable operation state."""

    if table is None:
        return
    tags = table.tags.list()
    has_operation_state = bool(_operation_marker_names(tags, operation_id))
    has_incomplete_state = bool(_incomplete_marker_names(tags))
    if has_operation_state or has_incomplete_state:
        raise VdbOperationConflict(
            f"VDB sink operation_id {operation_id!r} has durable state; "
            "retry the original operation with its original stream_operation_id; "
            "empty input cannot verify that operation."
        )
    if not _bounded_create_is_finalized(table):
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has an unfinished create operation; empty input cannot finalize it."
        )


def _identity_token(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().encode("ascii")


def _with_create_identity(schema: pa.Schema, *, operation_id: str, request_fingerprint: str) -> pa.Schema:
    """Bind a newly created table to the operation in its data commit."""

    metadata = dict(schema.metadata or {})
    metadata[_CREATE_OPERATION_KEY] = _identity_token(operation_id)
    metadata[_CREATE_REQUEST_KEY] = _identity_token(request_fingerprint)
    return schema.with_metadata(metadata)


def _matches_create_identity(table: Any, *, operation_id: str, request_fingerprint: str) -> bool:
    metadata = _table_schema(table).metadata or {}
    return metadata.get(_CREATE_OPERATION_KEY) == _identity_token(operation_id) and metadata.get(
        _CREATE_REQUEST_KEY
    ) == _identity_token(request_fingerprint)


def _table_content_identity(table: Any, schema: pa.Schema) -> tuple[int, str]:
    """Read a table through bounded one-row batches for rare create recovery."""

    stats = _StreamStats()
    reader = table.search().select(schema.names).to_batches(batch_size=1)
    for batch in reader:
        _record_stream_batch(stats, batch, include_digest=True)
    return stats.rows_written, stats.digest


def _schemas_have_same_fields(actual: pa.Schema, expected: pa.Schema) -> bool:
    """Ignore only durable sink identity keys while validating the schema."""

    def product_schema(schema: pa.Schema) -> pa.Schema:
        metadata = {
            key: value
            for key, value in (schema.metadata or {}).items()
            if key not in {_CREATE_OPERATION_KEY, _CREATE_REQUEST_KEY}
        }
        return schema.with_metadata(metadata or None)

    return product_schema(actual).equals(product_schema(expected), check_metadata=True)


def _validate_index_coverage(
    table: Any,
    *,
    rows: int,
    vector_index_expected: bool,
    text_index_expected: bool,
) -> None:
    """Require every configured index to cover the finalized table version."""

    expected_columns = {("vector",)} if vector_index_expected else set()
    if text_index_expected:
        expected_columns.add(("text",))

    indices = {tuple(index.columns): index for index in table.list_indices()}
    missing = sorted(expected_columns - set(indices))
    if missing:
        raise RuntimeError(f"LanceDB index validation failed; missing index columns: {missing!r}.")

    for columns in sorted(expected_columns):
        index = indices[columns]
        stats = table.index_stats(index.name)
        if stats is None or int(stats.num_indexed_rows) != int(rows) or int(stats.num_unindexed_rows) != 0:
            raise RuntimeError(
                f"LanceDB index validation failed for {columns!r}: "
                f"expected {rows} indexed and 0 unindexed rows, got {stats!r}."
            )


def _rows_at_version(uri: str, table_name: str, version: int | None) -> int:
    if version is None:
        return 0
    import lancedb

    table = lancedb.connect(uri=uri).open_table(table_name)
    table.checkout(int(version))
    return int(table.count_rows())
