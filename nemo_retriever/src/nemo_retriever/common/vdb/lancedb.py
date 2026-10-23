# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, FrozenSet
from urllib.parse import urlparse
from urllib.request import url2pathname

import lancedb
import pyarrow as pa
from filelock import FileLock

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
)
from nemo_retriever.common.schemas.embedding import EMBEDDING_SPLIT_METADATA_KEY, embedding_split_content
from nemo_retriever.common.vdb._lancedb_stream import (
    DataCommittedFinalizationError,
    VdbWriteNotFinalized,
    _apply_deferred_bad_vector_policy,
    _assert_lancedb_table_ready,
    _bounded_create_is_finalized,
    _checked_batches,
    _infer_vector_dim_with_spooled_prefix,
    _inspect_vector,
    _matches_create_identity,
    _reject_empty_operation_bypass,
    _rows_at_version,
    _schemas_have_same_fields,
    _StreamStats,
    _table_content_identity,
    _validate_index_coverage,
    _with_create_identity,
)
from nemo_retriever.common.vdb._lancedb_stream_state import (
    CommitOutcomeUnknown,
    SinkOperationMarkers,
    VdbOperationConflict,
)
from nemo_retriever.common.vdb.adt_vdb import (
    VDB,
    CollectionWriteContext,
    CollectionWriteResult,
    UnsupportedVDBOperation,
)
from nemo_retriever.common.vdb.hybrid_fusion import (
    HybridFusionPolicy,
    WeightedRRFReranker,
)
from nemo_retriever.common.vdb.lancedb_capabilities import (
    _table_schema,
    inspect_lancedb_table_object,
    wait_for_column_index,
)
from nemo_retriever.common.vdb.lancedb_schema import (
    build_lancedb_row,
    infer_vector_dim,
    lancedb_schema,
    normalize_content_type,
)

logger = logging.getLogger(__name__)


_DEFAULT_VECTOR_DIM: Final[int] = 2048
_DEFAULT_STREAM_BATCH_BYTES: Final[int] = 256 << 20
_VALID_ON_BAD_VECTORS: Final[FrozenSet[str]] = frozenset({"drop", "fill", "null", "error"})
_RETRIEVAL_MODE_METADATA_KEY: Final[bytes] = b"retrieval_mode"
_NEMO_RETRIEVER_RETRIEVAL_MODE_METADATA_KEY: Final[bytes] = b"nemo_retriever.retrieval_mode"
_EMBEDDING_MODEL_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_name"
_EMBEDDING_MODEL_REVISION_METADATA_KEY: Final[bytes] = b"nemo_retriever.embedding_model_revision"
_MISSING_FTS_POSITIONS_ERROR: Final[str] = "position is not found but required for phrase queries"
# Appended rows remain searchable through LanceDB's unindexed-tail scan until
# optimize() folds them into FTS. These thresholds follow its recommended cadence.
_SERVICE_OPTIMIZE_WRITE_THRESHOLD: Final[int] = 20
_SERVICE_OPTIMIZE_ROW_THRESHOLD: Final[int] = 100_000


def _filesystem_lancedb_path(uri: str) -> Path | None:
    """Resolve an embedded LanceDB path, including a local ``file://`` URI."""

    uri_text = os.fspath(uri)
    parsed = urlparse(uri_text)
    if parsed.scheme == "file":
        if parsed.netloc:
            return None
        return Path(url2pathname(parsed.path))
    if "://" not in uri_text:
        return Path(uri_text)
    return None


def _is_filesystem_lancedb_uri(uri: str) -> bool:
    """Return whether ``uri`` names the embedded filesystem implementation."""

    return _filesystem_lancedb_path(uri) is not None


def _local_table_lock(uri: str, table_name: str, *, phase: str) -> AbstractContextManager[Any]:
    """Return a process-shared phase lock for a local LanceDB table."""

    root = _filesystem_lancedb_path(uri)
    if root is None:
        return nullcontext()
    root = root.expanduser().resolve()
    lock_dir = root / ".nemo-retriever-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    table_token = hashlib.sha256(table_name.encode("utf-8")).hexdigest()[:24]
    return FileLock(lock_dir / f"{phase}-{table_token}.lock")


def _table_mutation_lock(uri: str, table_name: str) -> AbstractContextManager[Any]:
    """Return the process-shared mutation lock for a local LanceDB table."""

    return _local_table_lock(uri, table_name, phase="table")


def _table_index_lock(uri: str, table_name: str) -> AbstractContextManager[Any]:
    """Return the process-shared index lock for a local LanceDB table."""

    return _local_table_lock(uri, table_name, phase="index")


def _without_fts_phrase_syntax(query_text: str) -> str:
    """Convert a quoted FTS phrase into an unquoted term query."""

    return " ".join(query_text.replace('"', " ").split())


def _run_fts_query_with_legacy_index_fallback(build_query, query_text: str) -> list[dict[str, Any]]:
    """Run FTS, retrying quoted searches on indexes built without positions."""

    try:
        return build_query(query_text).to_list()
    except RuntimeError as exc:
        fallback_text = _without_fts_phrase_syntax(query_text)
        if _MISSING_FTS_POSITIONS_ERROR not in str(exc) or fallback_text == query_text or not fallback_text:
            raise
        logger.info("LanceDB FTS index does not contain positions; retrying quoted query without phrase syntax.")
        return build_query(fallback_text).to_list()


def _normalize_on_bad_vectors(value: str) -> str:
    """Validate and normalize an ``on_bad_vectors`` policy string.

    LanceDB's ``Table.create`` accepts a fixed set of policies for handling rows
    whose vector column does not match the declared fixed-size schema. We
    surface the same vocabulary on this wrapper for direct SDK configuration.

    Args:
        value: User-supplied policy name. Whitespace and case are ignored.

    Returns:
        The normalized lower-case policy string.

    Raises:
        ValueError: If ``value`` is not one of ``drop``, ``fill``, ``null``,
            or ``error``.
    """
    normalized = (value or "drop").strip().lower()
    if normalized not in _VALID_ON_BAD_VECTORS:
        raise ValueError(f"on_bad_vectors must be one of {sorted(_VALID_ON_BAD_VECTORS)}; got {value!r}")
    return normalized


def _stabilize_fill_vectors(
    rows: list[dict[str, Any]],
    *,
    vector_dim: int,
    fill_value: float,
) -> list[dict[str, Any]]:
    """Make ``fill`` independent of the installed LanceDB release.

    LanceDB 0.34 replaces the complete vector when its width is wrong or any
    element is NaN. Newer releases preserve valid elements and fill only the
    invalid positions. NeMo Retriever supports both installation paths, so
    normalize the historical public behavior before handing rows to LanceDB.
    Values that LanceDB cannot coerce remain untouched so its normal validation
    and error reporting still apply.
    """

    replacement = [float(fill_value)] * int(vector_dim)
    stabilized: list[dict[str, Any]] = []
    for row in rows:
        status, _ = _inspect_vector(row.get("vector"), vector_dim)
        if status == "uncoercible":
            stabilized.append(row)
        else:
            stabilized.append({**row, "vector": list(replacement)} if status != "valid" else row)
    return stabilized


def _json_str(value) -> str:
    """
    Convert Python objects (dict/list/etc.) to a compact JSON string.

    LanceDB table schema stores `metadata` and `source` as strings, so we must
    serialize nested structures before ingestion.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(value)


def _maybe_parse_json(value):
    """Best-effort parse for JSON-serialized string columns."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return {}
    # Avoid accidental parsing of plain strings that are not JSON objects/arrays.
    if not (s.startswith("{") or s.startswith("[")):
        return value
    try:
        return json.loads(s)
    except Exception:
        return value


def _is_ivf_vector_index(index_type: object) -> bool:
    """Return True if ``index_type`` names an IVF-style index (K-means partitions)."""
    s = str(index_type or "").upper()
    return s.startswith("IVF") or "IVF_" in s


def _effective_ivf_num_partitions(num_rows: int, requested: int) -> int | None:
    """Compute a valid ``num_partitions`` for Lance IVF training.

    K-means centroids must be strictly fewer than the number of training vectors
    (``num_partitions < num_rows``). For empty or single-row tables, IVF
    training is skipped (return ``None``).

    Args:
        num_rows: Row count in the table.
        requested: Caller-configured ``num_partitions``.

    Returns:
        Clamped partition count, or ``None`` if vector index build should be skipped.
    """
    if num_rows <= 0:
        return None
    if num_rows == 1:
        return None
    cap = num_rows - 1
    return min(int(requested), max(1, cap))


def _with_retrieval_mode_metadata(
    schema: pa.Schema,
    retrieval_mode: str | None,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
) -> pa.Schema:
    if retrieval_mode is None:
        return schema
    metadata = dict(schema.metadata or {})
    encoded_mode = str(retrieval_mode).encode("utf-8")
    metadata[_RETRIEVAL_MODE_METADATA_KEY] = encoded_mode
    metadata[_NEMO_RETRIEVER_RETRIEVAL_MODE_METADATA_KEY] = encoded_mode
    if embedding_model_name:
        metadata[_EMBEDDING_MODEL_METADATA_KEY] = embedding_model_name.encode("utf-8")
    if embedding_model_revision:
        metadata[_EMBEDDING_MODEL_REVISION_METADATA_KEY] = embedding_model_revision.encode("utf-8")
    return schema.with_metadata(metadata)


def _lancedb_arrow_schema(
    vector_dim: int,
    *,
    retrieval_mode: str | None = None,
    embedding_model_name: str | None = None,
    embedding_model_revision: str | None = None,
) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), int(vector_dim))),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    return _with_retrieval_mode_metadata(
        schema,
        retrieval_mode,
        embedding_model_name,
        embedding_model_revision,
    )


def _sparse_lancedb_arrow_schema(*, retrieval_mode: str | None = "sparse") -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    return _with_retrieval_mode_metadata(schema, retrieval_mode)


def _schema_vector_dim(schema: pa.Schema) -> int | None:
    """Return a fixed vector width from a LanceDB table schema when present."""
    try:
        vector_type = schema.field("vector").type
    except KeyError:
        return None
    if pa.types.is_fixed_size_list(vector_type):
        return int(vector_type.list_size)
    return None


def lancedb_row_count(uri: str, table_name: str) -> int:
    """Return the number of rows in a LanceDB table."""
    table = lancedb.connect(uri).open_table(table_name)
    return int(table.count_rows())


def _validate_append_schema(table: Any, expected_schema: pa.Schema, *, table_name: str, uri: str) -> None:
    """Fail before append when an existing table cannot accept this writer's rows."""
    existing_schema = _table_schema(table)
    existing_fields = {field.name: field for field in existing_schema}

    for expected_field in expected_schema:
        existing_field = existing_fields.get(expected_field.name)
        if existing_field is None:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} is missing required field "
                f"{expected_field.name!r}; use overwrite=True to replace the table."
            )
        if existing_field.type != expected_field.type:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} has incompatible field "
                f"{expected_field.name!r}: got {existing_field.type}, expected {expected_field.type}; "
                "use overwrite=True to replace the table."
            )


def _validate_append_embedding_model(
    table: Any,
    embedding_model_name: str | None,
    embedding_model_revision: str | None,
    *,
    table_name: str,
    uri: str,
) -> None:
    """Reject appends that would mix known embedding models in one table."""
    if not embedding_model_name:
        return

    metadata = _table_schema(table).metadata or {}
    stored_value = metadata.get(_EMBEDDING_MODEL_METADATA_KEY)
    if stored_value is None:
        return

    stored_model = stored_value.decode("utf-8", errors="replace").strip()
    if stored_model and stored_model != embedding_model_name:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model {stored_model!r}; "
            f"cannot append vectors from {embedding_model_name!r}. Use the table model or overwrite the table."
        )

    stored_revision_value = metadata.get(_EMBEDDING_MODEL_REVISION_METADATA_KEY)
    if stored_revision_value is None:
        return
    stored_revision = stored_revision_value.decode("utf-8", errors="replace").strip()
    if stored_revision and not embedding_model_revision:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model revision {stored_revision!r}; "
            "cannot append vectors without a known revision. Use the table revision or overwrite the table."
        )
    if stored_revision and stored_revision != embedding_model_revision:
        raise ValueError(
            f"LanceDB table {table_name!r} at {uri!r} uses embedding model revision {stored_revision!r}; "
            f"cannot append vectors from revision {embedding_model_revision!r}. "
            "Use the table revision or overwrite the table."
        )


def _is_missing_lancedb_table_error(exc: ValueError) -> bool:
    return "was not found" in str(exc)


def _record_timing(event: str, duration_s: float, extra: dict | None = None):
    timing_path = os.getenv("NV_INGEST_LANCEDB_TIMING_PATH")
    if not timing_path:
        return
    payload = {
        "event": event,
        "duration_s": duration_s,
        "timestamp_s": time.time(),
    }
    if extra:
        payload.update(extra)
    timing_dir = os.path.dirname(timing_path)
    if timing_dir:
        os.makedirs(timing_dir, exist_ok=True)
    with open(timing_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _get_text_for_element(element):
    """
    Extract searchable text from an element based on document_type.

    This prevents base64-encoded images from being stored in the text field.
    """
    doc_type = element.get("document_type")
    metadata = element.get("metadata", {})

    if doc_type == "text":
        return metadata.get("content")
    elif doc_type == "structured":
        # Tables, charts, infographics
        table_meta = metadata.get("table_metadata", {})
        return table_meta.get("table_content")
    elif doc_type == "image":
        # Use caption/OCR text, not raw base64 image data
        image_meta = metadata.get("image_metadata", {})
        content_meta = metadata.get("content_metadata", {})
        if content_meta.get("subtype") == "page_image":
            return image_meta.get("text")
        else:
            return image_meta.get("caption")
    elif doc_type == "audio":
        audio_meta = metadata.get("audio_metadata", {})
        return audio_meta.get("audio_transcript")
    else:
        # Fallback for unknown types
        return metadata.get("content")


def _create_lancedb_result(
    element: dict[str, Any],
    *,
    expected_dim: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Project one canonical NRL record, returning its drop reason if rejected."""

    metadata = element.get("metadata", {})
    doc_type = element.get("document_type")
    embedding = metadata.get("embedding")
    if embedding is None:
        return None, "dropped_no_embedding"

    if expected_dim is not None and (not isinstance(embedding, (list, tuple)) or len(embedding) != expected_dim):
        got_len: Any = len(embedding) if hasattr(embedding, "__len__") else "n/a"
        logger.debug(
            "Dropping row with bad embedding (got_len=%s, expected=%d, doc_type=%s)",
            got_len,
            expected_dim,
            doc_type,
        )
        return None, "dropped_bad_length"

    content_meta = metadata.get("content_metadata", {})
    split_content = embedding_split_content(metadata)
    text = split_content if split_content is not None else _get_text_for_element(element)
    if split_content is not None:
        content_meta = {**content_meta, EMBEDDING_SPLIT_METADATA_KEY: metadata[EMBEDDING_SPLIT_METADATA_KEY]}

    if split_content is None and (not isinstance(text, str) or not text.strip()):
        is_canonical_image = (
            doc_type == "image" and isinstance(content_meta, dict) and content_meta.get("type") == "image"
        )
        if not is_canonical_image:
            source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
            page_number = content_meta.get("page_number") if isinstance(content_meta, dict) else None
            logger.debug(
                "No text found for entity: %s page: %s type: %s",
                source_name,
                page_number,
                doc_type,
            )
            return None, "dropped_no_text"
        text = ""

    row_id = content_meta.get("id") if isinstance(content_meta, dict) else None
    if row_id is None and isinstance(metadata, dict):
        row_id = metadata.get("id")
    return (
        {
            "vector": embedding,
            "text": text,
            "metadata": _json_str(content_meta),
            "source": _json_str(metadata.get("source_metadata", {})),
            "id": str(row_id) if row_id is not None else "",
        },
        None,
    )


def _log_lancedb_result_counts(counts: dict[str, int], *, expected_dim: int | None) -> None:
    if not any(counts[key] for key in ("dropped_no_embedding", "dropped_bad_length", "dropped_no_text")):
        return
    logger.warning(
        "_create_lancedb_results: accepted=%d dropped_no_embedding=%d "
        "dropped_bad_length=%d dropped_no_text=%d expected_dim=%s",
        counts["accepted"],
        counts["dropped_no_embedding"],
        counts["dropped_bad_length"],
        counts["dropped_no_text"],
        expected_dim,
    )


def _create_lancedb_results(
    results,
    *,
    expected_dim: int | None = _DEFAULT_VECTOR_DIM,
) -> tuple[list, dict[str, int]]:
    """Transform Nemo Retriever Library (NRL) pipeline results into LanceDB ingestible rows.

    Extracts the appropriate searchable text per ``document_type`` and, when
    ``expected_dim`` is set, validates that each row's embedding is shaped
    consistently with the LanceDB fixed-size-list schema before forwarding it
    to the writer. Canonical image records may use ``text=""`` when both
    ``document_type`` and ``content_metadata.type`` are ``"image"``; all other
    dense records still require text. The graph adapter owns image-backing
    validation and emits that normalized record shape.
    Rows whose embedding is missing, of the wrong type, or of the wrong length
    are dropped and counted; per-row reasons are emitted at ``DEBUG`` and a
    single structured ``WARNING`` summary is emitted at the end of the call
    when any drops occurred.

    Passing ``expected_dim=None`` disables the length check entirely. Callers
    that prefer to defer to LanceDB's ``on_bad_vectors`` policy on the writer
    side (e.g. ``LanceDB(on_bad_vectors="error")``) should use this mode so
    bad rows reach LanceDB rather than being silently dropped at the wrapper.

    Args:
        results: Iterable of pipeline output result lists, where each element
            is a per-document list of NRL record dicts.
        expected_dim: Required vector length, or ``None`` to skip the length
            check. Defaults to :data:`_DEFAULT_VECTOR_DIM`.

    Returns:
        ``(rows, counts)`` where ``rows`` is the list of dicts shaped for
        LanceDB ingestion (``vector``, ``text``, ``metadata``, ``source``)
        and ``counts`` is a dict containing ``accepted``,
        ``dropped_no_embedding``, ``dropped_bad_length``, and
        ``dropped_no_text`` keys.
    """
    lancedb_rows: list[dict[str, Any]] = []
    expected_dim_int = int(expected_dim) if expected_dim is not None else None
    counts = {
        "accepted": 0,
        "dropped_no_embedding": 0,
        "dropped_bad_length": 0,
        "dropped_no_text": 0,
    }
    for result in results:
        for element in result:
            row, drop_reason = _create_lancedb_result(element, expected_dim=expected_dim_int)
            if drop_reason is not None:
                counts[drop_reason] += 1
                continue
            if row is not None:
                lancedb_rows.append(row)
                counts["accepted"] += 1

    _log_lancedb_result_counts(counts, expected_dim=expected_dim_int)
    return lancedb_rows, counts


def _to_service_lancedb_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Adapt one canonical dense row to the established service table schema."""

    content_metadata = _maybe_parse_json(row.get("metadata"))
    if not isinstance(content_metadata, dict):
        content_metadata = {}
    source_metadata = _maybe_parse_json(row.get("source"))
    if not isinstance(source_metadata, dict):
        source_metadata = {}
    source_id = next(
        (
            str(value).strip()
            for value in (
                source_metadata.get("source_id"),
                source_metadata.get("source_name"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    content_type = normalize_content_type(content_metadata.get("type") or content_metadata.get("_content_type"))
    if content_type:
        content_metadata = dict(content_metadata)
        content_metadata["type"] = content_type
        content_metadata["_content_type"] = content_type
    wide_row = build_lancedb_row(
        SimpleNamespace(
            metadata={
                "source_path": source_id,
                "content_metadata": content_metadata,
            },
            path=source_id,
            page_number=content_metadata.get("page_number"),
            text=row.get("text") or "",
            _stored_image_uri=content_metadata.get("stored_image_uri"),
            _content_type=content_type,
            _bbox_xyxy_norm=content_metadata.get("bbox_xyxy_norm"),
        ),
        embedding=row.get("vector"),
    )
    if wide_row is None:
        return None
    wide_row["metadata"] = _json_str(content_metadata)
    wide_row["source"] = _json_str(source_metadata)
    wide_row["content_type"] = content_type or ""
    return wide_row


def _to_service_lancedb_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt canonical dense rows to the established service table schema."""

    wide_rows = []
    for row in rows:
        wide_row = _to_service_lancedb_row(row)
        if wide_row is not None:
            wide_rows.append(wide_row)
    return wide_rows


def _create_sparse_lancedb_result(element: dict[str, Any]) -> dict[str, Any] | None:
    metadata = element.get("metadata", {})
    content_meta = metadata.get("content_metadata", {})
    text = _get_text_for_element(element)
    if not isinstance(text, str) or not text.strip():
        source_name = metadata.get("source_metadata", {}).get("source_name", "unknown")
        page_number = content_meta.get("page_number") if isinstance(content_meta, dict) else None
        logger.debug("No text found for sparse entity: %s page: %s", source_name, page_number)
        return None

    row_id = content_meta.get("id") if isinstance(content_meta, dict) else None
    if row_id is None and isinstance(metadata, dict):
        row_id = metadata.get("id")
    return {
        "text": text,
        "metadata": _json_str(content_meta),
        "source": _json_str(metadata.get("source_metadata", {})),
        "id": str(row_id) if row_id is not None else "",
    }


def _log_sparse_result_counts(counts: dict[str, int]) -> None:
    if counts["dropped_no_text"]:
        logger.warning(
            "_create_sparse_lancedb_results: accepted=%d dropped_no_text=%d",
            counts["accepted"],
            counts["dropped_no_text"],
        )


def _create_sparse_lancedb_results(results) -> tuple[list, dict[str, int]]:
    """Transform NRL records into LanceDB rows for FTS-only sparse retrieval."""

    lancedb_rows: list[dict[str, Any]] = []
    counts = {"accepted": 0, "dropped_no_text": 0}
    for result in results:
        for element in result:
            row = _create_sparse_lancedb_result(element)
            if row is None:
                counts["dropped_no_text"] += 1
                continue
            lancedb_rows.append(row)
            counts["accepted"] += 1

    _log_sparse_result_counts(counts)
    return lancedb_rows, counts


class LanceDB(VDB):
    """LanceDB operator implementing the VDB interface."""

    def __init__(
        self,
        uri: str | None = None,
        overwrite: bool = True,
        table_name: str = "nemo-retriever",
        index_type: str = "IVF_HNSW_SQ",
        metric: str = "l2",
        num_partitions: int = 16,
        num_sub_vectors: int = 256,
        hybrid: bool = False,
        sparse: bool = False,
        fts_language: str = "English",
        embedding_model_name: str | None = None,
        vector_dim: int | None = _DEFAULT_VECTOR_DIM,
        on_bad_vectors: str = "drop",
        fill_value: float = 0.0,
        validate_vector_length: bool = True,
        build_index: bool | None = None,
        expiration_cleanup_enabled: bool = True,
        embedding_model_revision: str | None = None,
        stream_batch_bytes: int = _DEFAULT_STREAM_BATCH_BYTES,
        stream_optimize: bool = False,
        stream_operation_id: str | None = None,
        **kwargs,
    ):
        create_index = kwargs.pop("create_index", None)
        service_table_schema = bool(kwargs.pop("_service_table_schema", False))
        service_index_mode = kwargs.pop("_service_index_mode", None)
        if build_index is None:
            build_index = True if create_index is None else bool(create_index)
        elif create_index is not None and bool(create_index) != bool(build_index):
            raise ValueError("Pass only one index toggle: build_index or create_index.")

        if vector_dim is not None and int(vector_dim) <= 0:
            raise ValueError(f"vector_dim must be positive; got {vector_dim}")
        if sparse and hybrid:
            raise ValueError("LanceDB sparse ingest cannot also be hybrid; pass only one retrieval mode.")
        if isinstance(stream_batch_bytes, bool) or not isinstance(stream_batch_bytes, int) or stream_batch_bytes <= 0:
            raise ValueError("stream_batch_bytes must be a positive integer")
        if not isinstance(stream_optimize, bool):
            raise TypeError("stream_optimize must be a boolean")
        if stream_operation_id is not None:
            if not isinstance(stream_operation_id, str) or not stream_operation_id.strip():
                raise ValueError("stream_operation_id must be a non-empty string or None")
            stream_operation_id = stream_operation_id.strip()
        self.uri = uri or "lancedb"
        self.overwrite = bool(overwrite)
        self.table_name = table_name
        self.build_index = bool(build_index)
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.hybrid = hybrid
        self.sparse = bool(sparse)
        self.fts_language = fts_language
        self.embedding_model_name = embedding_model_name
        self.embedding_model_revision = embedding_model_revision
        self.vector_dim = int(vector_dim) if vector_dim is not None else None
        self.on_bad_vectors = _normalize_on_bad_vectors(on_bad_vectors)
        self.fill_value = float(fill_value)
        self.validate_vector_length = bool(validate_vector_length)
        self.expiration_cleanup_enabled = bool(expiration_cleanup_enabled)
        self.stream_batch_bytes = stream_batch_bytes
        self.stream_optimize = stream_optimize
        self.stream_operation_id = stream_operation_id
        # Remote stores and private service schemas retain the legacy path.
        self.supports_stream_ingest = (
            _is_filesystem_lancedb_uri(self.uri) and not service_table_schema and service_index_mode is None
        )
        self._service_table_schema = service_table_schema
        self._service_index_mode = str(service_index_mode) if service_index_mode is not None else None
        self._writes_since_optimize = 0
        # Process-local status is diagnostic; persisted index statistics are authoritative.
        self._last_optimization: dict[str, Any] = {
            "status": "never",
            "completed_at": None,
            "error": None,
        }
        self._collection_store: Any | None = None
        self._collection_store_init_failed = False
        self._collection_store_lock = threading.Lock()
        # Reuse native LanceDB handles for the lifetime of this backend. Opening
        # a connection and table for every service health check and query leaves
        # native file descriptors pending collection long enough for sustained
        # workloads to exhaust the process limit.
        self._connection_lock = threading.RLock()
        self._connections: dict[str, Any] = {}
        self._opened_tables: dict[tuple[str, str], Any] = {}
        # Row admission is serialized on its own short-lived lock so a caller
        # never waits on index maintenance to get its rows committed.
        self._write_lock = threading.Lock()
        # Serialize the complete streaming lifecycle within this instance;
        # the filesystem lock coordinates other local instances and processes.
        self._stream_lock = threading.Lock()
        # LanceDB treats competing index commits as a conflict, so only one
        # rebuild may run at a time. Rebuilds are coalesced by generation:
        # a rebuild that starts after a batch was committed also covers it.
        self._index_lock = threading.Lock()
        self._index_generation_lock = threading.Lock()
        self._index_requested_generation = 0
        self._index_completed_generation = 0
        super().__init__(**kwargs)
        if self._service_index_mode is not None and self.hybrid:
            db = self._connect()
            if self.table_name in db.list_tables().tables:
                self._ensure_fts_index(self._open_table(self.table_name))

    @staticmethod
    def _is_fts_index(index: Any) -> bool:
        index_type = str(getattr(index, "index_type", "") or "").lower()
        index_name = str(getattr(index, "name", "") or "").lower()
        return index_type == "fts" or "text" in index_name or "fts" in index_name

    def _ensure_fts_index(self, table: Any) -> None:
        if inspect_lancedb_table_object(table).has_fts:
            return
        started = time.perf_counter()
        num_rows = int(table.count_rows())
        table.create_fts_index("text", language=self.fts_language, replace=True)
        wait_for_column_index(table, "text", covered_rows=num_rows)
        _record_timing("lancedb.fts_index_ready", time.perf_counter() - started)

    def _fts_unindexed_rows(self, table: Any) -> int | None:
        values: list[int] = []
        try:
            indices = list(table.list_indices())
        except Exception:
            logger.debug(
                "Unable to enumerate LanceDB indexes for %s",
                getattr(self, "table_name", "<unknown>"),
                exc_info=True,
            )
            return None
        for index in indices:
            if not self._is_fts_index(index):
                continue
            try:
                stats = table.index_stats(index.name)
            except Exception:
                logger.debug("Unable to read LanceDB index stats for %s", index.name, exc_info=True)
                continue
            value = (
                stats.get("num_unindexed_rows")
                if isinstance(stats, dict)
                else getattr(stats, "num_unindexed_rows", None)
            )
            if value is not None:
                values.append(int(value))
        return sum(values) if values else None

    def _optimize_service_table_if_due(self, table: Any) -> None:
        # Persisted FTS statistics keep the row threshold valid across restarts.
        unindexed_rows = self._fts_unindexed_rows(table)
        if self._writes_since_optimize < _SERVICE_OPTIMIZE_WRITE_THRESHOLD and (
            unindexed_rows is None or unindexed_rows < _SERVICE_OPTIMIZE_ROW_THRESHOLD
        ):
            return
        try:
            table.optimize()
        except Exception as exc:
            self._last_optimization = {
                "status": "error",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            logger.exception("LanceDB optimization failed for table %r", self.table_name)
            # Preserve the trigger state so a later write retries transient failures.
            return
        self._writes_since_optimize = 0
        self._last_optimization = {
            "status": "ok",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }

    def _connect(self, uri: str | None = None) -> Any:
        """Return a backend-owned LanceDB connection for ``uri``."""

        resolved_uri = str(uri or self.uri)
        with self._connection_lock:
            connection = self._connections.get(resolved_uri)
            if connection is None:
                connection = lancedb.connect(uri=resolved_uri)
                self._connections[resolved_uri] = connection
            return connection

    def _open_table(self, table_name: str, *, uri: str | None = None) -> Any:
        """Return a backend-owned table handle instead of reopening per query."""

        resolved_uri = str(uri or self.uri)
        key = (resolved_uri, table_name)
        with self._connection_lock:
            table = self._opened_tables.get(key)
            if table is None:
                table = self._connect(resolved_uri).open_table(table_name)
                self._opened_tables[key] = table
            return table

    def _remember_table(self, table_name: str, table: Any, *, uri: str | None = None) -> Any:
        """Cache a table returned by a create or append operation."""

        resolved_uri = str(uri or self.uri)
        with self._connection_lock:
            self._opened_tables[(resolved_uri, table_name)] = table
        return table

    def _get_collection_store(self) -> Any:
        """Lazily initialize collection catalogs only when a collection API is used."""

        store = self._collection_store
        if store is None:
            with self._collection_store_lock:
                store = self._collection_store
                if store is None:
                    from nemo_retriever.common.vdb.lancedb_collections import (
                        LanceDBCollectionStore,
                    )

                    try:
                        store = LanceDBCollectionStore(
                            self,
                            expiration_cleanup_enabled=self.expiration_cleanup_enabled,
                        )
                    except Exception:
                        self._collection_store_init_failed = True
                        raise
                    self._collection_store_init_failed = False
                    self._collection_store = store
        return store

    def create_collection(
        self,
        *,
        scope: str,
        request: CollectionCreateRequest,
    ) -> CollectionInfo:
        """Create a logical collection through the LanceDB collection store."""

        return self._get_collection_store().create_collection(scope, request)

    def get_collection(
        self,
        *,
        scope: str,
        collection_name: str,
    ) -> CollectionInfo:
        """Return a logical collection from the LanceDB collection store."""

        return self._get_collection_store().get_collection(scope, collection_name)

    def list_collections(
        self,
        *,
        scope: str,
        limit: int,
        continuation_token: str | None,
    ) -> CollectionPage:
        """List logical collections through the LanceDB collection store."""

        return self._get_collection_store().list_collections(
            scope,
            limit,
            continuation_token,
        )

    def update_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        """Update a logical collection through the LanceDB collection store."""

        return self._get_collection_store().update_collection(
            scope,
            collection_name,
            request,
        )

    def delete_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        if_exists: bool,
    ) -> CollectionDeleteResult:
        """Delete a logical collection through the LanceDB collection store."""

        return self._get_collection_store().delete_collection(
            scope,
            collection_name,
            if_exists,
        )

    def get_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> DocumentInfo:
        """Return one collection document through the LanceDB collection store."""

        return self._get_collection_store().get_document(
            scope,
            collection_name,
            document_id,
        )

    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        """List collection documents through the LanceDB collection store."""

        return self._get_collection_store().list_documents(
            scope,
            collection_name,
            limit,
            continuation_token,
        )

    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        """Delete one collection document through the LanceDB collection store."""

        return self._get_collection_store().delete_document(
            scope,
            collection_name,
            document_id,
            if_exists,
        )

    def write_collection(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Write canonical records using the collection lifecycle contract."""

        return self._get_collection_store().write_collection(records, context=context)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Retrieve scoped collection hits using LanceDB's collection contract."""

        return self._get_collection_store().retrieve_collection(
            vectors,
            scope=scope,
            collection_name=collection_name,
            query_texts=query_texts,
            top_k=top_k,
            **kwargs,
        )

    def reconcile_collections(self) -> dict[str, int]:
        """Resume interrupted collection and document lifecycle operations."""

        return self._get_collection_store().reconcile_collections()

    def health(self) -> dict[str, Any]:
        """Return legacy table and optional collection-store health."""

        from nemo_retriever.common.vdb.lancedb_collections import LanceDBCollectionStore

        db = self._connect()
        table_exists = self.table_name in db.list_tables().tables
        total_rows = 0
        effective_mode: str | None = None
        retrieval_strategies: list[str] = []
        table: Any | None = None
        capabilities = None
        if table_exists:
            table = self._open_table(self.table_name)
            self._checkout_latest(table)
            try:
                total_rows = int(table.count_rows())
            except Exception:
                logger.warning(
                    "Failed to count rows in the default LanceDB table",
                    exc_info=True,
                )
            try:
                capabilities = inspect_lancedb_table_object(table)
                mode = capabilities.retrieval_mode
                if mode in {"dense", "hybrid"}:
                    effective_mode = str(mode)
                    retrieval_strategies = [str(mode)]
                else:
                    effective_mode = "unknown"
            except Exception:
                effective_mode = "unknown"
                logger.warning(
                    "Failed to resolve the default LanceDB retrieval mode",
                    exc_info=True,
                )

        if self._collection_store_init_failed:
            raise RuntimeError("Collection catalog initialization failed")
        store = self._collection_store
        collection_health = store.health() if store is not None else LanceDBCollectionStore.empty_health()
        service_health: dict[str, Any] = {}
        if self._service_index_mode is not None:
            service_health = {
                "configured_index_mode": self._service_index_mode,
                "effective_index_mode": effective_mode,
                "fts_present": bool(capabilities and capabilities.has_fts),
                "fts_unindexed_rows": (
                    self._fts_unindexed_rows(table)
                    if table is not None and capabilities is not None and capabilities.has_fts
                    else None
                ),
                "last_optimization": dict(self._last_optimization),
            }
        return {
            **collection_health,
            **service_health,
            "total_rows": total_rows,
            "table_exists": table_exists,
            "effective_retrieval_mode": effective_mode,
            "retrieval_strategies": retrieval_strategies,
        }

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Read one NeMo Retriever metadata value from the selected table."""
        uri = str(kwargs.get("table_path") or kwargs.get("uri") or kwargs.get("lancedb_uri") or self.uri)
        table_name = str(kwargs.get("table_name") or kwargs.get("lancedb_table") or self.table_name)
        table = self._open_table(table_name, uri=uri)
        self._checkout_latest(table)
        metadata = _table_schema(table).metadata or {}
        value = metadata.get(f"nemo_retriever.{key}".encode("utf-8"))
        if value is None and key == "retrieval_mode":
            value = metadata.get(_RETRIEVAL_MODE_METADATA_KEY)
        if value is None:
            return None
        return value.decode("utf-8", errors="replace").strip() or None

    def create_index(self, records=None, table_name: str = "nemo-retriever", **kwargs):
        """Create or update a LanceDB table and populate it with transformed records.

        Validates per-row vector shape (when ``validate_vector_length`` is set
        on the instance and ``on_bad_vectors`` is not ``"error"``) and forwards
        LanceDB's ``on_bad_vectors`` policy as defense-in-depth so that any
        rows escaping the row-builder check are still handled by the LanceDB
        writer instead of aborting the run. When ``on_bad_vectors == "error"``
        the wrapper deliberately skips its own length check so that LanceDB
        itself raises on the bad row, matching the documented strict-fail
        semantics of that policy.
        """
        connect_start = time.perf_counter()
        db = self._connect()
        _record_timing("lancedb.connect", time.perf_counter() - connect_start)
        record_batches = list(records or [])

        if self.sparse:
            results, counts = _create_sparse_lancedb_results(record_batches)
            schema = _sparse_lancedb_arrow_schema()
            write_kwargs: dict[str, Any] = {}
        else:
            enforce_dim = self.validate_vector_length and self.on_bad_vectors != "error"
            vector_dim = self.vector_dim
            if vector_dim is None and not self.overwrite:
                try:
                    existing_table = self._open_table(self.table_name)
                except ValueError as exc:
                    if not _is_missing_lancedb_table_error(exc):
                        raise
                else:
                    vector_dim = _schema_vector_dim(_table_schema(existing_table))

            if vector_dim is None:
                results, counts = _create_lancedb_results(record_batches, expected_dim=None)
                vector_dim = infer_vector_dim(results)
                if vector_dim <= 0:
                    raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")
                if enforce_dim:
                    results, counts = _create_lancedb_results(record_batches, expected_dim=vector_dim)
            else:
                results, counts = _create_lancedb_results(
                    record_batches, expected_dim=vector_dim if enforce_dim else None
                )

            if self.on_bad_vectors == "fill":
                results = _stabilize_fill_vectors(
                    results,
                    vector_dim=vector_dim,
                    fill_value=self.fill_value,
                )

            if self._service_table_schema:
                results = _to_service_lancedb_rows(results)
                schema = _with_retrieval_mode_metadata(
                    lancedb_schema(vector_dim),
                    "hybrid" if self.hybrid else "dense",
                    embedding_model_name=self.embedding_model_name,
                    embedding_model_revision=self.embedding_model_revision,
                )
            else:
                schema = _lancedb_arrow_schema(
                    vector_dim,
                    retrieval_mode="hybrid" if self.hybrid else "dense",
                    embedding_model_name=self.embedding_model_name,
                    embedding_model_revision=self.embedding_model_revision,
                )

            write_kwargs = {
                "on_bad_vectors": self.on_bad_vectors,
            }
            if self.on_bad_vectors == "fill":
                write_kwargs["fill_value"] = self.fill_value

        create_kwargs: dict[str, Any] = {
            "schema": schema,
            **write_kwargs,
        }

        create_start = time.perf_counter()

        if self.overwrite:
            table = db.create_table(
                table_name,
                data=results,
                mode="overwrite",
                **create_kwargs,
            )
            event = "lancedb.create_table"
        else:
            try:
                table = self._open_table(table_name)
            except ValueError as exc:
                if not _is_missing_lancedb_table_error(exc):
                    raise
                table = db.create_table(
                    table_name,
                    data=results,
                    mode="create",
                    **create_kwargs,
                )
                event = "lancedb.create_table"
            else:
                _validate_append_schema(table, schema, table_name=table_name, uri=self.uri)
                if results:
                    _validate_append_embedding_model(
                        table,
                        self.embedding_model_name,
                        self.embedding_model_revision,
                        table_name=table_name,
                        uri=self.uri,
                    )
                    existing_rows = int(table.count_rows())
                    logger.warning(
                        "Appending %d row(s) to existing LanceDB table %r at %s "
                        "(existing_rows=%d). Append mode does not deduplicate; rerunning the same inputs "
                        "will duplicate rows.",
                        len(results),
                        table_name,
                        self.uri,
                        existing_rows,
                    )
                    table.add(
                        results,
                        mode="append",
                        **write_kwargs,
                    )
                event = "lancedb.add_rows"

        self._remember_table(table_name, table)
        _record_timing(
            event,
            time.perf_counter() - create_start,
            {"rows": len(results), **counts},
        )
        return table

    def write_to_index(
        self,
        records,
        table=None,
        index_type="IVF_HNSW_SQ",
        metric="l2",
        num_partitions=16,
        num_sub_vectors=256,
        hybrid: bool | None = None,
        sparse: bool | None = None,
        fts_language: str | None = None,
        **kwargs,
    ):
        """Create vector and optionally FTS indexes on the LanceDB table.

        For IVF index types, ``num_partitions`` is clamped so that
        ``num_partitions < row_count`` (Lance K-means requirement). Empty or
        single-row tables skip the vector index; hybrid FTS may still be built.
        """
        hybrid = hybrid if hybrid is not None else self.hybrid
        sparse = sparse if sparse is not None else self.sparse
        fts_language = fts_language or self.fts_language

        if sparse:
            fts_index_start = time.perf_counter()
            sparse_rows = int(table.count_rows())
            table.create_fts_index("text", language=fts_language, replace=True)
            wait_for_column_index(table, "text", covered_rows=sparse_rows)
            _record_timing("lancedb.fts_index_ready", time.perf_counter() - fts_index_start)
            return

        num_rows = int(table.count_rows())
        requested_partitions = int(num_partitions)
        use_ivf = _is_ivf_vector_index(index_type)
        effective_partitions: int | None
        if use_ivf:
            effective_partitions = _effective_ivf_num_partitions(num_rows, requested_partitions)
        else:
            effective_partitions = requested_partitions

        vector_index_start = time.perf_counter()
        if use_ivf and effective_partitions is None:
            if num_rows == 0:
                logger.warning(
                    "Skipping LanceDB vector index: empty table (index_type=%s).",
                    index_type,
                )
            else:
                logger.info(
                    "Skipping LanceDB vector index: IVF needs at least two rows (got %d; index_type=%s).",
                    num_rows,
                    index_type,
                )
        else:
            partitions_for_index = (
                int(effective_partitions) if effective_partitions is not None else requested_partitions
            )
            if use_ivf and partitions_for_index != requested_partitions:
                logger.info(
                    "Clamping num_partitions from %d to %d (table has %d rows; IVF requires partitions < row count).",
                    requested_partitions,
                    partitions_for_index,
                    num_rows,
                )
            table.create_index(
                index_type=index_type,
                metric=metric,
                num_partitions=partitions_for_index,
                num_sub_vectors=num_sub_vectors,
                vector_column_name="vector",
                replace=True,
            )
            wait_for_column_index(table, "vector", covered_rows=num_rows)
            _record_timing("lancedb.vector_index_ready", time.perf_counter() - vector_index_start)

        if hybrid:
            fts_index_start = time.perf_counter()
            table.create_fts_index("text", language=fts_language, replace=True)
            wait_for_column_index(table, "text", covered_rows=num_rows)
            _record_timing("lancedb.fts_index_ready", time.perf_counter() - fts_index_start)

    def _iter_stream_rows(
        self,
        records: Iterable[dict[str, Any]],
        stats: _StreamStats,
    ) -> Iterator[dict[str, Any]]:
        """Project canonical NRL records one at a time for LanceDB storage."""

        dense_counts = {
            "accepted": 0,
            "dropped_no_embedding": 0,
            "dropped_bad_length": 0,
            "dropped_no_text": 0,
        }
        sparse_counts = {"accepted": 0, "dropped_no_text": 0}
        for record in records:
            stats.client_records += 1

            if self.sparse:
                row = _create_sparse_lancedb_result(record)
                if row is None:
                    sparse_counts["dropped_no_text"] += 1
                    continue
                sparse_counts["accepted"] += 1
                yield row
                continue

            enforce_dim = self.validate_vector_length and self.on_bad_vectors != "error"
            expected_dim = stats.vector_dim if enforce_dim else None
            row, drop_reason = _create_lancedb_result(record, expected_dim=expected_dim)
            if drop_reason is not None:
                dense_counts[drop_reason] += 1
                continue
            if row is None:
                continue
            dense_counts["accepted"] += 1
            yield row

        if self.sparse:
            _log_sparse_result_counts(sparse_counts)
        else:
            _log_lancedb_result_counts(dense_counts, expected_dim=stats.vector_dim)

    def _stream_schema(self, vector_dim: int | None) -> pa.Schema:
        if self.sparse:
            return _sparse_lancedb_arrow_schema()
        if vector_dim is None:
            raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")

        retrieval_mode = "hybrid" if self.hybrid else "dense"
        return _lancedb_arrow_schema(
            vector_dim,
            retrieval_mode=retrieval_mode,
            embedding_model_name=self.embedding_model_name,
            embedding_model_revision=self.embedding_model_revision,
        )

    def _stream_request_fingerprint(self, schema: pa.Schema, *, mode: str) -> str:
        """Identify settings that affect the durable table result."""

        return json.dumps(
            {
                "table": self.table_name,
                "mode": mode,
                "schema": schema.to_string(show_schema_metadata=True),
                "build_index": self.build_index,
                "index_type": str(self.index_type),
                "metric": str(self.metric),
                "num_partitions": int(self.num_partitions),
                "num_sub_vectors": int(self.num_sub_vectors),
                "fts_language": str(self.fts_language),
                "hybrid": self.hybrid,
                "sparse": self.sparse,
                "on_bad_vectors": self.on_bad_vectors,
                "fill_value": self.fill_value,
                "validate_vector_length": self.validate_vector_length,
                "optimize": self.stream_optimize,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _mutate_stream_data(
        self,
        *,
        db: Any,
        existing_table: Any | None,
        markers: SinkOperationMarkers | None,
        mode: str,
        base_version: int | None,
        schema: pa.Schema,
        batches: Iterable[pa.RecordBatch],
        stats: _StreamStats,
    ) -> int:
        """Perform or recover the single Lance data mutation."""

        if markers is not None and markers.state == "data":
            return int(markers.recorded_version)

        if existing_table is not None and mode == "append" and stats.rows_written == 0:
            # Legacy append does not create a table version when every client
            # record is dropped, but still finalizes the operation.
            existing_table.checkout_latest()
            data_version = int(existing_table.version)
            if markers is not None:
                markers.mark_data(existing_table, version=data_version, rows=0, digest=stats.digest)
            return data_version

        write_kwargs: dict[str, Any] = {"on_bad_vectors": self.on_bad_vectors}
        if self.on_bad_vectors == "fill":
            write_kwargs["fill_value"] = self.fill_value
        reader = pa.RecordBatchReader.from_batches(schema, batches)
        try:
            if existing_table is None:
                table = db.create_table(
                    self.table_name,
                    data=reader,
                    schema=schema,
                    mode="create",
                    **write_kwargs,
                )
                data_version = int(table.version)
            else:
                add_result = existing_table.add(reader, mode=mode, **write_kwargs)
                table = existing_table
                table.checkout_latest()
                data_version = int(add_result.version)
        except (OSError, RuntimeError, ValueError, pa.ArrowException) as exc:
            if markers is not None:
                markers.abort_if_unchanged(existing_table)
            if existing_table is None:
                created_table = None
                try:
                    created_table = lancedb.connect(uri=self.uri).open_table(self.table_name)
                except ValueError as open_exc:
                    if not _is_missing_lancedb_table_error(open_exc):
                        raise
                if created_table is not None:
                    created_table.checkout_latest()
                    created_version = int(created_table.version)
                    if markers is not None and _matches_create_identity(
                        created_table,
                        operation_id=markers.operation_id,
                        request_fingerprint=markers.request_fingerprint,
                    ):
                        raise DataCommittedFinalizationError(
                            self.table_name,
                            created_version,
                            retry_operation_id=markers.operation_id,
                        ) from exc
                    raise CommitOutcomeUnknown(
                        f"LanceDB table creation may have committed at version {created_version}. "
                        "Do not replay these records; reconcile the table before another write."
                    ) from exc
            elif base_version is not None and mode == "append":
                latest = lancedb.connect(uri=self.uri).open_table(self.table_name)
                latest.checkout_latest()
                latest_version = int(latest.version)
                if latest_version != base_version:
                    operation = f" operation_id {markers.operation_id!r}" if markers is not None else ""
                    raise CommitOutcomeUnknown(
                        f"LanceDB append{operation} may have committed while the table advanced "
                        f"from version {base_version} to {latest_version}. Do not replay these records; "
                        "reconcile the table before another write."
                    ) from exc
            raise

        if markers is not None:
            try:
                markers.mark_data(
                    table,
                    version=data_version,
                    rows=stats.rows_written,
                    digest=stats.digest,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                retry_operation_id = markers.operation_id if existing_table is None or mode == "overwrite" else None
                recovery_action = None
                if retry_operation_id is None:
                    recovery_action = (
                        "The durable data marker was not recorded. Do not replay this append. "
                        f"Inspect committed version {data_version}, complete the configured index and optimization "
                        f"maintenance, then delete pending Lance tag {markers.pending_tag!r} only after confirming "
                        "that the version contains the intended rows."
                    )
                raise DataCommittedFinalizationError(
                    self.table_name,
                    data_version,
                    retry_operation_id=retry_operation_id,
                    recovery_action=recovery_action,
                ) from exc
        return data_version

    def _finalize_stream_write(
        self,
        *,
        markers: SinkOperationMarkers | None,
        data_version: int,
        expected_schema: pa.Schema,
        expected_rows: int,
        stats: _StreamStats,
    ) -> None:
        """Validate data and indexes before recording durable success."""

        fresh_table = lancedb.connect(uri=self.uri).open_table(self.table_name)
        if self.build_index:
            self._maintain_indexes(None, fresh_table)
        if self.stream_optimize:
            with self._index_lock, _table_index_lock(self.uri, self.table_name):
                fresh_table.checkout_latest()
                fresh_table.optimize()

        fresh_table.checkout_latest()
        if not _schemas_have_same_fields(_table_schema(fresh_table), expected_schema):
            raise RuntimeError(f"LanceDB schema validation failed for table {self.table_name!r}")
        final_rows = int(fresh_table.count_rows())
        if final_rows != expected_rows:
            raise RuntimeError(
                f"LanceDB row-count validation failed for table {self.table_name!r}: "
                f"expected {expected_rows}, got {final_rows}."
            )
        if self.build_index:
            vector_index_expected = not self.sparse and not (_is_ivf_vector_index(self.index_type) and final_rows < 2)
            _validate_index_coverage(
                fresh_table,
                rows=final_rows,
                vector_index_expected=vector_index_expected,
                text_index_expected=self.sparse or self.hybrid,
            )

        final_version = int(fresh_table.version)
        if final_version < data_version:
            raise RuntimeError(
                f"LanceDB version validation failed for table {self.table_name!r}: "
                f"data_version={data_version}, final_version={final_version}."
            )
        if markers is not None:
            markers.mark_success(
                fresh_table,
                version=final_version,
                rows=stats.rows_written,
                digest=stats.digest,
            )
        self._remember_table(self.table_name, fresh_table)

    def _durable_stream_schema(
        self,
        *,
        existing_table: Any | None,
        base_schema: pa.Schema,
        operation_id: str,
        mode: str,
    ) -> tuple[pa.Schema, str, bool]:
        """Bind a caller-owned operation ID to the requested table result."""

        request_fingerprint = self._stream_request_fingerprint(base_schema, mode=mode)
        recovered_create = bool(
            existing_table is not None
            and _matches_create_identity(
                existing_table,
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
            )
        )
        if existing_table is not None and not recovered_create and not _bounded_create_is_finalized(existing_table):
            raise VdbWriteNotFinalized(
                f"LanceDB table {self.table_name!r} has an unfinished create operation; "
                "retry the original operation with its original stream_operation_id before starting another write."
            )
        schema = (
            _with_create_identity(
                base_schema,
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
            )
            if existing_table is None
            else base_schema
        )
        return schema, request_fingerprint, recovered_create

    def _prepare_durable_stream_recovery(
        self,
        *,
        existing_table: Any | None,
        operation_id: str,
        request_fingerprint: str,
        mode: str,
        recovered_create: bool,
        arrow_batches: Iterator[pa.RecordBatch],
        base_schema: pa.Schema,
        stats: _StreamStats,
    ) -> SinkOperationMarkers | None:
        """Prepare or resume the explicit durable-retry state machine."""

        markers = SinkOperationMarkers.prepare(
            existing_table,
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            mode=mode,
        )
        if recovered_create or markers.state in {"data", "success"}:
            try:
                for _ in arrow_batches:
                    pass
            except (OSError, RuntimeError, TypeError, ValueError, pa.ArrowException):
                if markers.state == "write":
                    markers.abort_if_unchanged(existing_table)
                raise

        if recovered_create and markers.state == "write":
            stored_rows, stored_digest = _table_content_identity(existing_table, base_schema)
            if stored_rows != stats.rows_written or stored_digest != stats.digest:
                raise VdbOperationConflict(
                    f"VDB sink operation_id {operation_id!r} found a created table with different stored content."
                )
            existing_table.checkout_latest()
            markers.mark_data(
                existing_table,
                version=int(existing_table.version),
                rows=stats.rows_written,
                digest=stats.digest,
            )
        elif markers.state in {"data", "success"}:
            markers.verify_input(rows=stats.rows_written, digest=stats.digest)

        if markers.state != "success":
            return markers

        markers.cleanup_after_success(existing_table)
        _assert_lancedb_table_ready(existing_table)
        self._remember_table(self.table_name, existing_table)
        return None

    def _write_stream_records(self, records: Iterable[dict[str, Any]], *, operation_id: str | None) -> None:
        """Consume canonical NRL records through one coordinated LanceDB mutation."""

        stats = _StreamStats(vector_dim=self.vector_dim)
        with self._write_lock:
            db = lancedb.connect(uri=self.uri)
            try:
                existing_table = db.open_table(self.table_name)
            except ValueError as exc:
                if not _is_missing_lancedb_table_error(exc):
                    raise
                existing_table = None
            table_exists = existing_table is not None
            base_version: int | None = None
            base_rows = 0
            if existing_table is not None:
                existing_table.checkout_latest()
                base_version = int(existing_table.version)
                if not self.overwrite:
                    base_rows = int(existing_table.count_rows())
                if operation_id is None:
                    _assert_lancedb_table_ready(existing_table)

            if stats.vector_dim is None and existing_table is not None and not self.overwrite and not self.sparse:
                stats.vector_dim = _schema_vector_dim(_table_schema(existing_table))

            canonical_rows = self._iter_stream_rows(records, stats)
            missing = object()
            first_canonical_row = next(canonical_rows, missing)
            if first_canonical_row is missing:
                if stats.client_records == 0:
                    if operation_id is not None:
                        _reject_empty_operation_bypass(existing_table, operation_id=operation_id)
                    if existing_table is not None:
                        self._remember_table(self.table_name, existing_table)
                    return

            canonical_row_stream = chain(
                () if first_canonical_row is missing else (first_canonical_row,),
                canonical_rows,
            )
            if stats.vector_dim is None and not self.sparse:
                stats.vector_dim, canonical_row_stream = _infer_vector_dim_with_spooled_prefix(
                    canonical_row_stream,
                    validate_vector_length=self.validate_vector_length,
                    on_bad_vectors=self.on_bad_vectors,
                )

            policy_rows = _apply_deferred_bad_vector_policy(
                canonical_row_stream,
                vector_dim=int(stats.vector_dim or 0),
                sparse=self.sparse,
                on_bad_vectors=self.on_bad_vectors,
                fill_value=self.fill_value,
            )

            base_schema = self._stream_schema(stats.vector_dim)
            mode = "overwrite" if self.overwrite else "append"
            if operation_id is None:
                schema = base_schema
                request_fingerprint = None
                recovered_create = False
            else:
                schema, request_fingerprint, recovered_create = self._durable_stream_schema(
                    existing_table=existing_table,
                    base_schema=base_schema,
                    operation_id=operation_id,
                    mode=mode,
                )

            expected_schema = base_schema
            if existing_table is not None and not self.overwrite:
                _validate_append_schema(existing_table, schema, table_name=self.table_name, uri=self.uri)
                _validate_append_embedding_model(
                    existing_table,
                    self.embedding_model_name,
                    self.embedding_model_revision,
                    table_name=self.table_name,
                    uri=self.uri,
                )
                expected_schema = _table_schema(existing_table)

            arrow_batches = _checked_batches(
                policy_rows,
                schema=schema,
                max_batch_bytes=self.stream_batch_bytes,
                stats=stats,
                include_digest=operation_id is not None,
            )
            first_batch = next(arrow_batches, None)
            if operation_id is None:
                markers = None
                mutation_base_version = base_version
            else:
                markers = self._prepare_durable_stream_recovery(
                    existing_table=existing_table,
                    operation_id=operation_id,
                    request_fingerprint=request_fingerprint,
                    mode=mode,
                    recovered_create=recovered_create,
                    arrow_batches=arrow_batches,
                    base_schema=base_schema,
                    stats=stats,
                )
                if markers is None:
                    return
                mutation_base_version = markers.base_version

            initial_batches = (
                (pa.RecordBatch.from_pylist([], schema=schema),) if first_batch is None else (first_batch,)
            )

            data_version = self._mutate_stream_data(
                db=db,
                existing_table=existing_table,
                markers=markers,
                mode=mode,
                base_version=mutation_base_version,
                schema=schema,
                batches=chain(initial_batches, arrow_batches),
                stats=stats,
            )
            try:
                created_from_absent = not table_exists or recovered_create
                if self.overwrite or created_from_absent:
                    expected_rows = stats.rows_written
                else:
                    append_base_rows = (
                        base_rows
                        if markers is None
                        else _rows_at_version(self.uri, self.table_name, mutation_base_version)
                    )
                    expected_rows = append_base_rows + stats.rows_written
                self._finalize_stream_write(
                    markers=markers,
                    data_version=data_version,
                    expected_schema=expected_schema,
                    expected_rows=expected_rows,
                    stats=stats,
                )
            except Exception as exc:  # noqa: BLE001 - terminal post-commit boundary preserves replay safety.
                logger.error(
                    "LanceDB data committed at version %s but finalization failed for table %r.",
                    data_version,
                    self.table_name,
                    exc_info=True,
                )
                raise DataCommittedFinalizationError(
                    self.table_name,
                    data_version,
                    retry_operation_id=operation_id,
                ) from exc

    def stream_ingest(self, records: Iterable[dict[str, Any]]) -> None:
        """Ingest canonical records through one bounded LanceDB lifecycle.

        Parameters
        ----------
        records
            A single-pass iterable of canonical NeMo Retriever record
            dictionaries. It is consumed synchronously and to exhaustion.

        Returns
        -------
        None

        Raises
        ------
        UnsupportedVDBOperation
            If the LanceDB URI is not backed by the local filesystem.
        CommitOutcomeUnknown
            If LanceDB may have committed an append but did not acknowledge it.
        DataCommittedFinalizationError
            If data committed but validation, indexing, or optimization failed.
        ValueError
            If records or configured streaming controls are invalid.
        RuntimeError
            If mutation, recovery, validation, indexing, or optimization fails.
        """

        if not self.supports_stream_ingest:
            raise UnsupportedVDBOperation("LanceDB.stream_ingest() is unavailable; use the legacy global-batch path.")

        with self._stream_lock, _table_mutation_lock(self.uri, self.table_name):
            self._write_stream_records(records, operation_id=self.stream_operation_id)

    def _reject_stream_controls_for_legacy_operation(self, operation: str) -> None:
        active_controls = []
        if self.stream_batch_bytes != _DEFAULT_STREAM_BATCH_BYTES:
            active_controls.append(f"stream_batch_bytes={self.stream_batch_bytes}")
        if self.stream_optimize:
            active_controls.append("stream_optimize=True")
        if self.stream_operation_id is not None:
            active_controls.append("stream_operation_id")
        if active_controls:
            controls = ", ".join(active_controls)
            raise ValueError(
                f"LanceDB.{operation}() cannot use {controls}; these controls require "
                "LanceDB.stream_ingest() through the streaming-ingest path."
            )

    def _assert_legacy_table_ready(self, table_name: str) -> None:
        """Refresh and reject a target with durable unfinished stream state."""

        try:
            table = self._connect().open_table(table_name)
        except ValueError as exc:
            if _is_missing_lancedb_table_error(exc):
                return
            raise
        _assert_lancedb_table_ready(table)
        self._remember_table(table_name, table)

    def run(self, records):
        """Commit rows, then bring the table indexes up to date.

        Row admission and index maintenance use separate locks. This keeps
        concurrent appends durable while serializing LanceDB index commits.
        """
        self._reject_stream_controls_for_legacy_operation("run")
        service_write = self._service_index_mode is not None
        with _table_mutation_lock(self.uri, self.table_name), self._write_lock:
            self._assert_legacy_table_ready(self.table_name)
            table_existed = False
            if service_write:
                db = self._connect()
                table_existed = self.table_name in db.list_tables().tables

            table = self.create_index(records=records, table_name=self.table_name)

        if self.build_index:
            self._maintain_indexes(records, table)
        elif service_write and self.hybrid:
            self._maintain_service_fts(table, table_existed=table_existed)
        else:
            logger.info(
                "Skipping LanceDB index creation for table %r because build_index=False.",
                self.table_name,
            )

        return records

    def _maintain_service_fts(self, table: Any, *, table_existed: bool) -> None:
        """Create or incrementally maintain the service's FTS index."""
        with self._index_lock, _table_index_lock(self.uri, self.table_name):
            self._checkout_latest(table)
            self._ensure_fts_index(table)
            if table_existed:
                self._writes_since_optimize += 1
                self._optimize_service_table_if_due(table)

    def _maintain_indexes(self, records, table) -> None:
        """Rebuild table indexes so they cover the rows committed by this call.

        Returns once an index build that started after this caller's rows were
        committed has finished. Callers that arrive while such a build is
        already running wait for it rather than queueing another one.
        """
        with self._index_generation_lock:
            self._index_requested_generation += 1
            required_generation = self._index_requested_generation

        while True:
            with self._index_generation_lock:
                if self._index_completed_generation >= required_generation:
                    return

            with self._index_lock, _table_index_lock(self.uri, self.table_name):
                with self._index_generation_lock:
                    if self._index_completed_generation >= required_generation:
                        return
                    building_generation = self._index_requested_generation

                # Index the newest committed version, not the snapshot this
                # caller happened to open, so one rebuild can cover the rows
                # of every writer it has coalesced.
                self._checkout_latest(table)
                self.write_to_index(
                    records,
                    table=table,
                    index_type=self.index_type,
                    metric=self.metric,
                    num_partitions=self.num_partitions,
                    num_sub_vectors=self.num_sub_vectors,
                    hybrid=self.hybrid,
                    sparse=self.sparse,
                    fts_language=self.fts_language,
                )

                with self._index_generation_lock:
                    self._index_completed_generation = max(
                        self._index_completed_generation,
                        building_generation,
                    )

    @staticmethod
    def _checkout_latest(table) -> None:
        """Advance ``table`` to the latest committed version when supported."""
        checkout_latest = getattr(table, "checkout_latest", None)
        if not callable(checkout_latest):
            return
        try:
            checkout_latest()
        except Exception as exc:  # noqa: BLE001 - version refresh is advisory.
            logger.debug("Could not advance LanceDB table handle to the latest version: %s", exc)

    def sparse_retrieval(self, query_texts: Iterable[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search a LanceDB FTS-only table without query embeddings."""
        table_path = kwargs.pop("table_path", self.uri)
        table_name = kwargs.pop("table_name", self.table_name)
        result_fields = kwargs.pop("result_fields", None)
        top_k = int(kwargs.pop("top_k", 10))
        text_column_value = kwargs.pop("text_column_name", kwargs.pop("fts_columns", "text"))
        if isinstance(text_column_value, (list, tuple)):
            text_column_name = str(text_column_value[0] if text_column_value else "text")
        else:
            text_column_name = str(text_column_value)

        search_kwargs_raw = kwargs.pop("search_kwargs", None)
        if search_kwargs_raw is None:
            search_kwargs: dict[str, Any] = {}
        elif not isinstance(search_kwargs_raw, dict):
            raise TypeError(f"search_kwargs must be a dict or None; got {type(search_kwargs_raw).__name__}")
        else:
            search_kwargs = dict(search_kwargs_raw)

        query_type = search_kwargs.get("query_type")
        if query_type is not None:
            query_type_value = getattr(query_type, "value", query_type)
            if str(query_type_value).lower() != "fts":
                raise ValueError(
                    "LanceDB sparse retrieval requires search_kwargs['query_type']='fts'; " f"got {query_type!r}."
                )
        search_kwargs["query_type"] = "fts"
        search_kwargs.setdefault("fts_columns", text_column_name)

        where_clause = kwargs.pop("where", None)
        _filter_fallback = kwargs.pop("_filter", None)
        if where_clause is None:
            where_clause = _filter_fallback
        if where_clause is not None:
            where_clause = str(where_clause).strip() or None

        table = self._open_table(table_name, uri=table_path)
        self._checkout_latest(table)
        _assert_lancedb_table_ready(table)

        search_results = []
        for query_text in query_texts:

            def build_query(text: str):
                query = table.search(text, **search_kwargs)
                if where_clause is not None:
                    query = query.where(where_clause)
                query = query.limit(top_k)
                if result_fields is not None:
                    query = query.select(result_fields)
                return query

            search_results.append(_run_fts_query_with_legacy_index_fallback(build_query, str(query_text)))

        return search_results

    def retrieval(self, vectors: Iterable[Sequence[float]], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Search LanceDB with precomputed query vectors.

        Keyword arguments
        -----------------
        where:
            Optional SQL predicate (Lance / DataFusion) applied on the vector
            query builder via ``.where(...)`` before ``limit``. Filter against
            table columns: ``vector``, ``text``, ``metadata``, ``source``.
            Note: ``metadata`` and ``source`` are JSON strings at rest.
        _filter:
            Alias for ``where`` when ``where`` is omitted (call-site parity).
        search_kwargs:
            Optional dict of extra keyword arguments forwarded to
            ``table.search`` (e.g. ``query_type``, ``fts_columns``). Do not
            pass ``vector_column_name`` here; use the top-level
            ``vector_column_name`` retrieval argument instead.
        query_texts:
            Raw query strings aligned with ``vectors``. Required for
            ``hybrid=True`` and ignored for dense-only retrieval.
        hybrid_fusion:
            Optional :class:`HybridFusionPolicy`. When present, each hybrid leg
            retrieves at least ``candidate_depth`` rows, applies weighted RRF,
            and returns only the requested ``top_k`` rows.
        """
        hybrid = kwargs.pop("hybrid", self.hybrid)
        hybrid_fusion = kwargs.pop("hybrid_fusion", None)
        query_texts = kwargs.pop("query_texts", None)
        table_path = kwargs.pop("table_path", self.uri)
        table_name = kwargs.pop("table_name", self.table_name)

        result_fields = kwargs.pop("result_fields", None)
        top_k = int(kwargs.pop("top_k", 10))
        refine_factor = int(kwargs.pop("refine_factor", 50))
        n_probe = int(kwargs.pop("n_probe", kwargs.pop("nprobes", 64)))
        vector_column_name = str(kwargs.pop("vector_column_name", "vector"))

        search_kwargs_raw = kwargs.pop("search_kwargs", None)
        if search_kwargs_raw is None:
            search_kwargs: dict[str, Any] = {}
        elif not isinstance(search_kwargs_raw, dict):
            raise TypeError(f"search_kwargs must be a dict or None; got {type(search_kwargs_raw).__name__}")
        else:
            search_kwargs = dict(search_kwargs_raw)

        if hybrid:
            if query_texts is None:
                raise ValueError(
                    "LanceDB hybrid retrieval requires query_texts. Pass query_texts=your_queries "
                    "alongside vectors when calling retrieval() with hybrid=True."
                )
            query_type = search_kwargs.get("query_type")
            if query_type is not None:
                query_type_value = getattr(query_type, "value", query_type)
                if str(query_type_value).lower() != "hybrid":
                    raise ValueError(
                        "LanceDB hybrid retrieval requires search_kwargs['query_type']='hybrid'; "
                        f"got {query_type!r}."
                    )
            search_kwargs["query_type"] = "hybrid"
            search_kwargs.setdefault("fts_columns", "text")
        elif hybrid_fusion is not None:
            raise ValueError("hybrid_fusion requires hybrid=True")

        if hybrid_fusion is not None and not isinstance(hybrid_fusion, HybridFusionPolicy):
            raise TypeError(
                "hybrid_fusion must be a HybridFusionPolicy or None; " f"got {type(hybrid_fusion).__name__}"
            )

        where_clause = kwargs.pop("where", None)
        _filter_fallback = kwargs.pop("_filter", None)
        if where_clause is None:
            where_clause = _filter_fallback
        if where_clause is not None:
            where_clause = str(where_clause).strip() or None

        table = self._open_table(table_name, uri=table_path)
        self._checkout_latest(table)
        _assert_lancedb_table_ready(table)

        if hybrid:
            vectors_for_search = list(vectors)
            query_texts_list = [query_texts] if isinstance(query_texts, str) else list(query_texts)
            if len(query_texts_list) != len(vectors_for_search):
                raise ValueError(
                    "LanceDB hybrid retrieval requires query_texts length to match vectors length; "
                    f"got query_texts={len(query_texts_list)} vectors={len(vectors_for_search)}."
                )
        else:
            vectors_for_search = vectors
            query_texts_list = []

        search_results = []
        for idx, vector in enumerate(vectors_for_search):
            if hybrid:

                def build_query(text: str, *, query_vector=vector):
                    query = (
                        table.search(vector_column_name=vector_column_name, **search_kwargs)
                        .vector(query_vector)
                        .text(text)
                    )
                    if where_clause is not None:
                        query = query.where(where_clause)
                    query_limit = max(top_k, hybrid_fusion.candidate_depth) if hybrid_fusion is not None else top_k
                    query = query.limit(query_limit).refine_factor(refine_factor).nprobes(n_probe)
                    if hybrid_fusion is not None:
                        query = query.rerank(WeightedRRFReranker(hybrid_fusion))
                    if result_fields is not None:
                        query = query.select(result_fields)
                    return query

                results = _run_fts_query_with_legacy_index_fallback(
                    build_query,
                    str(query_texts_list[idx]),
                )
                if hybrid_fusion is not None:
                    results = results[:top_k]
            else:
                query = table.search([vector], vector_column_name=vector_column_name, **search_kwargs)
                if where_clause is not None:
                    query = query.where(where_clause)
                query = query.limit(top_k).refine_factor(refine_factor).nprobes(n_probe)
                if result_fields is not None:
                    query = query.select(result_fields)
                results = query.to_list()
            search_results.append(results)

        return search_results
