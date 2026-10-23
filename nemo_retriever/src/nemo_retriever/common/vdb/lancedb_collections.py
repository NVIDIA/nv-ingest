# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private LanceDB persistence for collection-managed service data."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
    IngestOperation,
)
from nemo_retriever.common.schemas.embedding import EMBEDDING_SPLIT_METADATA_KEY, embedding_split_content
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    UnsupportedVDBOperation,
    VDBInvalidRequest,
    VDBResourceConflict,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.lancedb_capabilities import (
    LanceRetrievalMode,
    LanceTableCapabilities,
    inspect_lancedb_table_object,
)
from nemo_retriever.common.vdb.lancedb_schema import (
    create_or_append_lancedb_table,
    infer_vector_dim,
    lancedb_schema,
)
from nemo_retriever.common.vdb.records import (
    RetrievalContractError,
    normalize_content_type,
    normalize_retrieval_results,
)

logger = logging.getLogger(__name__)

_COLLECTIONS_TABLE = "_nrl_collections"
_DOCUMENTS_TABLE = "_nrl_documents"
_CATALOG_SCHEMA_VERSION = 2
_CATALOG_SCAN_LIMIT = 100_000
_NATIVE_SCORE_FIELDS = frozenset({"_distance", "_score"})
_EMBEDDING_MODEL_METADATA_KEY = b"nemo_retriever.embedding_model_name"


def _with_embedding_model_metadata(schema: pa.Schema, model_name: str | None) -> pa.Schema:
    """Attach the configured embedding identity to a collection table schema."""
    if not model_name:
        return schema
    metadata = dict(schema.metadata or {})
    metadata[_EMBEDDING_MODEL_METADATA_KEY] = model_name.encode("utf-8")
    return schema.with_metadata(metadata)


def _validate_embedding_model(table: Any, configured_model: str | None, *, table_name: str) -> None:
    """Reject collection access when its vector space cannot be verified."""
    if not configured_model:
        return
    schema = table.schema() if callable(table.schema) else table.schema
    stored_value = (schema.metadata or {}).get(_EMBEDDING_MODEL_METADATA_KEY)
    if not stored_value:
        raise VDBInvalidRequest(
            f"Existing LanceDB collection table {table_name!r} does not record its embedding model, "
            "so query compatibility cannot be verified. Rebuild and re-ingest the collection with "
            "the configured embedding model."
        )

    from nemo_retriever.models import resolve_embed_model

    stored_model = resolve_embed_model(stored_value.decode("utf-8", errors="replace").strip())
    expected_model = resolve_embed_model(configured_model)
    if stored_model != expected_model:
        raise VDBInvalidRequest(
            f"Existing LanceDB collection table {table_name!r} uses embedding model {stored_model!r}, "
            f"but the VectorDB service is configured for {expected_model!r}. Use the index model or "
            "rebuild and re-ingest the collection with the configured model."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_uncommitted_initial_append(row: Mapping[str, Any]) -> bool:
    return row.get("recovery_state") == "appending" and not row.get("current_document_version")


def _physical_table(scope: str, collection_name: str) -> str:
    digest = hashlib.sha256(f"{scope}\0{collection_name}".encode()).hexdigest()
    return f"nrl_{digest[:40]}"


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _encode_cursor(resource: str, scope: str, collection: str | None, last: list[str]) -> str:
    """Encode a pagination position bound to its logical resource context."""

    payload = {
        "v": 1,
        "resource": resource,
        "scope": scope,
        "collection": collection,
        "last": last,
    }
    return (
        base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def _decode_cursor(
    token: str | None,
    *,
    resource: str,
    scope: str,
    collection: str | None,
) -> list[str] | None:
    """Decode a cursor after validating its resource, scope, and collection context."""

    if not token:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode())
    except Exception as exc:
        raise VDBInvalidRequest("Invalid continuation token") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or payload.get("resource") != resource
        or payload.get("scope") != scope
        or payload.get("collection") != collection
        or not isinstance(payload.get("last"), list)
    ):
        raise VDBInvalidRequest("Continuation token does not match this resource context")
    return [str(value) for value in payload["last"]]


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"), default=str)


def _content_text(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    content = metadata.get("content")
    if isinstance(content, str):
        return content
    document_type = str(record.get("document_type") or "")
    if document_type == "structured":
        table_metadata = metadata.get("table_metadata")
        if isinstance(table_metadata, dict):
            return str(table_metadata.get("table_content") or "")
    if document_type == "image":
        image_metadata = metadata.get("image_metadata")
        if isinstance(image_metadata, dict):
            content_metadata = metadata.get("content_metadata")
            is_page = isinstance(content_metadata, dict) and content_metadata.get("subtype") == "page_image"
            return str(image_metadata.get("text" if is_page else "caption") or "")
    if document_type == "audio":
        audio_metadata = metadata.get("audio_metadata")
        if isinstance(audio_metadata, dict):
            return str(audio_metadata.get("audio_transcript") or "")
    return ""


def _positive_or_unknown_page(value: Any) -> int:
    if isinstance(value, bool):
        return -1
    try:
        page = int(value)
    except (TypeError, ValueError):
        return -1
    return page if page > 0 else -1


def _collection_rows(
    records: list,
    *,
    context: CollectionWriteContext,
) -> list[dict[str, Any]]:
    """Convert canonical NRL record batches into collection-managed LanceDB rows."""
    rows: list[dict[str, Any]] = []
    created_at = _now()
    row_index = 0

    for batch in records or []:
        if not isinstance(batch, list):
            continue
        for record in batch:
            if not isinstance(record, dict):
                continue
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            vector = record.get("embedding")
            if not isinstance(vector, (list, tuple)) or not vector:
                continue
            content_metadata = metadata.get("content_metadata")
            if not isinstance(content_metadata, dict):
                content_metadata = {}
            source_metadata = metadata.get("source_metadata")
            if not isinstance(source_metadata, dict):
                source_metadata = {}

            split_content = embedding_split_content(metadata)
            text = split_content if split_content is not None else _content_text(record, metadata)
            if split_content is not None:
                content_metadata = {
                    **content_metadata,
                    EMBEDDING_SPLIT_METADATA_KEY: metadata[EMBEDDING_SPLIT_METADATA_KEY],
                }
            content_type = normalize_content_type(content_metadata.get("type") or record.get("document_type"))
            content_type = content_type or ""
            if content_type:
                content_metadata = dict(content_metadata)
                content_metadata["type"] = content_type
                content_metadata["_content_type"] = content_type
            if not text.strip() and content_type != "image" and split_content is None:
                continue

            source_id = str(
                source_metadata.get("source_id") or source_metadata.get("source_name") or context.filename or ""
            )
            source_path = Path(source_id) if source_id else None
            filename = context.filename or (source_path.name if source_path else "")
            pdf_basename = source_path.stem if source_path else Path(filename).stem
            page_number = _positive_or_unknown_page(content_metadata.get("page_number"))
            pdf_page = f"{pdf_basename}_{page_number}" if pdf_basename and page_number > 0 else ""
            stored_image_uri = str(content_metadata.get("stored_image_uri") or "")
            bbox = content_metadata.get("bbox_xyxy_norm")

            rows.append(
                {
                    "vector": list(vector),
                    "pdf_page": pdf_page,
                    "filename": filename,
                    "pdf_basename": pdf_basename,
                    "page_number": page_number,
                    "source": _json_string(source_metadata),
                    "source_id": source_id,
                    "path": source_id,
                    "text": text,
                    "metadata": _json_string(content_metadata),
                    "stored_image_uri": stored_image_uri,
                    "content_type": content_type,
                    "bbox_xyxy_norm": _json_string(bbox) if bbox else "",
                    "chunk_id": hashlib.sha256(
                        f"{context.document_id}\0{context.document_version}\0{row_index}".encode()
                    ).hexdigest(),
                    "document_id": context.document_id,
                    "document_version": context.document_version,
                    "content_sha256": context.content_sha256,
                    "created_at": created_at,
                }
            )
            row_index += 1
    return rows


def _public_collection_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Expose a finite native distance without leaking LanceDB score fields."""
    raw = hit.get("_distance")
    if isinstance(raw, bool):
        raise RetrievalContractError("Dense collection hit is missing a numeric _distance")
    try:
        distance = float(raw)
    except (TypeError, ValueError) as exc:
        raise RetrievalContractError("Dense collection hit is missing a numeric _distance") from exc
    if not math.isfinite(distance):
        raise RetrievalContractError("Dense collection hit has a non-finite _distance")

    public_hit = {key: value for key, value in hit.items() if key not in _NATIVE_SCORE_FIELDS}
    content_type = str(public_hit.get("content_type") or "").lower()
    page_number = _positive_or_unknown_page(public_hit.get("page_number"))
    if content_type.startswith(("audio", "video")) or page_number < 0:
        public_hit["page_number"] = None
        public_hit["pdf_page"] = ""
    else:
        public_hit["page_number"] = page_number
    public_hit["distance"] = distance
    return public_hit


def _normalize_collection_results(
    raw_results: Any,
    *,
    expected_queries: int,
) -> list[list[dict[str, Any]]]:
    """Strictly validate collection result cardinality and hit shape."""
    if not isinstance(raw_results, list) or len(raw_results) != expected_queries:
        raise RetrievalContractError("Collection retrieval returned an invalid query-result cardinality")
    for query_index, hits in enumerate(raw_results):
        if not isinstance(hits, list):
            raise RetrievalContractError(f"Collection retrieval result {query_index} is not a hit list")
        for hit_index, hit in enumerate(hits):
            if not isinstance(hit, Mapping):
                raise RetrievalContractError(f"Collection retrieval hit {query_index}:{hit_index} is not a mapping")
    return normalize_retrieval_results(raw_results)


class LanceDBCollectionStore:
    """Implement optional VDB collection capabilities for LanceDB.

    The store owns private catalogs and maps logical ``(scope, collection)``
    identities to physical tables. Table-user leases keep deletion and recovery
    from racing with active reads or writes.
    """

    def __init__(self, backend: Any, *, expiration_cleanup_enabled: bool = True) -> None:
        self._backend = backend
        self._uri = backend.uri
        self.expiration_cleanup_enabled = expiration_cleanup_enabled
        self.reconciliation_successes = 0
        self.reconciliation_failures = 0
        self._write_lock = threading.Lock()
        self._collection_write_lock = threading.Lock()
        self._table_user_condition = threading.Condition(self._write_lock)
        self._active_table_users: dict[str, int] = {}
        self._db = lancedb.connect(uri=self._uri)
        self._opened_tables: dict[str, Any] = {}
        self._ensure_catalogs()

    def _ensure_catalogs(self) -> None:
        collection_schema = pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
                pa.field("deletion_phase", pa.string()),
                pa.field("retry_count", pa.int64()),
                pa.field("next_retry_at", pa.string()),
                pa.field("last_error", pa.string()),
                pa.field("delete_started_at", pa.string()),
            ]
        )
        document_schema = pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("collection_name", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("job_id", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("content_sha256", pa.string()),
                pa.field("document_version", pa.string()),
                pa.field("status", pa.string()),
                pa.field("chunk_count", pa.int64()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("error", pa.string()),
                pa.field("current_document_version", pa.string()),
                pa.field("pending_document_version", pa.string()),
                pa.field("recovery_state", pa.string()),
            ]
        )
        for name, schema in (
            (_COLLECTIONS_TABLE, collection_schema),
            (_DOCUMENTS_TABLE, document_schema),
        ):
            if self._has_table(name):
                table = self._db.open_table(name)
            else:
                table = self._db.create_table(name, schema=schema, mode="create")
            existing = {field.name: field for field in table.schema}
            for field in schema:
                if field.name in existing and existing[field.name].type != field.type:
                    raise RuntimeError(
                        f"Incompatible {name} catalog column {field.name!r}: "
                        f"expected {field.type}, found {existing[field.name].type}"
                    )
            missing = [field for field in schema if field.name not in existing]
            if missing:
                missing_names = ", ".join(sorted(field.name for field in missing))
                raise RuntimeError(f"Incompatible {name} catalog: missing required columns: {missing_names}")
            index_columns = (
                ("scope", "name", "status", "expires_at")
                if name == _COLLECTIONS_TABLE
                else ("scope", "collection_name", "document_id", "status")
            )
            for column in index_columns:
                table.create_scalar_index(column, replace=True)

    def _rows(
        self,
        table_name: str,
        where: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        table = self._open_table(table_name)
        checkout_latest = getattr(table, "checkout_latest", None)
        if callable(checkout_latest):
            checkout_latest()
        query = table.search()
        if where:
            query = query.where(where)
        if columns:
            query = query.select(columns)
        return query.limit(_CATALOG_SCAN_LIMIT).to_list()

    def _has_table(self, table_name: str) -> bool:
        return table_name in self._db.list_tables().tables

    def _open_table(self, table_name: str) -> Any:
        table = self._opened_tables.get(table_name)
        if table is None:
            table = self._db.open_table(table_name)
            self._opened_tables[table_name] = table
        return table

    def _acquire_table_user_locked(self, table_name: str) -> None:
        self._active_table_users[table_name] = self._active_table_users.get(table_name, 0) + 1

    def _release_table_user(self, table_name: str) -> None:
        with self._table_user_condition:
            remaining = self._active_table_users.get(table_name, 0) - 1
            if remaining > 0:
                self._active_table_users[table_name] = remaining
            else:
                self._active_table_users.pop(table_name, None)
                self._table_user_condition.notify_all()

    def _wait_for_table_users_locked(self, table_name: str) -> None:
        """Wait under the state lock before destructively mutating an active table."""

        while self._active_table_users.get(table_name, 0):
            self._table_user_condition.wait()

    def _collection_row(self, scope: str, name: str, *, active: bool = False) -> dict[str, Any] | None:
        rows = self._rows(
            _COLLECTIONS_TABLE,
            f"scope = {_quoted(scope)} AND name = {_quoted(name)}",
        )
        row = rows[0] if rows else None
        if row and active and row["status"] != "active":
            raise VDBInvalidRequest(f"Collection {name!r} is {row['status']}")
        if row and active and row.get("expires_at"):
            expires = datetime.fromisoformat(str(row["expires_at"]))
            if expires <= datetime.now(timezone.utc):
                raise VDBInvalidRequest(f"Collection {name!r} is expired")
        return row

    @staticmethod
    def _collection_info(row: dict[str, Any]) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            scope=row["scope"],
            status=row["status"],
            description=row.get("description") or None,
            metadata=json.loads(row.get("metadata_json") or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row.get("expires_at") or None,
        )

    @staticmethod
    def _document_info(row: dict[str, Any]) -> DocumentInfo:
        return DocumentInfo(**{key: row.get(key) for key in DocumentInfo.model_fields})

    def create_collection(self, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        """Create a scoped logical collection without exposing its physical table."""

        with self._write_lock:
            if self._collection_row(scope, request.name):
                raise VDBResourceConflict(f"Collection {request.name!r} already exists")
            now = _now()
            row = {
                "scope": scope,
                "name": request.name,
                "physical_table": _physical_table(scope, request.name),
                "status": "active",
                "description": request.description or "",
                "metadata_json": json.dumps(request.metadata, sort_keys=True),
                "created_at": now,
                "updated_at": now,
                "expires_at": request.expires_at or "",
                "deletion_phase": "",
                "retry_count": 0,
                "next_retry_at": "",
                "last_error": "",
                "delete_started_at": "",
            }
            self._db.open_table(_COLLECTIONS_TABLE).add([row])
            return self._collection_info(row)

    def get_collection(self, scope: str, name: str) -> CollectionInfo:
        """Return one scoped collection or raise when it does not exist."""

        row = self._collection_row(scope, name)
        if not row:
            raise VDBResourceNotFound("Collection not found")
        return self._collection_info(row)

    def list_collections(
        self,
        scope: str,
        limit: int,
        continuation_token: str | None,
    ) -> CollectionPage:
        """List collections in one scope using a context-bound cursor."""

        rows = sorted(
            self._rows(_COLLECTIONS_TABLE, f"scope = {_quoted(scope)}"),
            key=lambda row: row["name"],
        )
        last = _decode_cursor(
            continuation_token,
            resource="collections",
            scope=scope,
            collection=None,
        )
        if last is not None:
            if len(last) != 1:
                raise VDBInvalidRequest("Invalid collection continuation token")
            rows = [row for row in rows if row["name"] > last[0]]
        page = rows[:limit]
        next_token = (
            _encode_cursor("collections", scope, None, [page[-1]["name"]]) if len(rows) > limit and page else None
        )
        return CollectionPage(items=[self._collection_info(row) for row in page], next_token=next_token)

    def update_collection(
        self,
        scope: str,
        name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        """Update mutable metadata for an active scoped collection."""

        with self._write_lock:
            row = self._collection_row(scope, name, active=True)
            if not row:
                raise VDBResourceNotFound("Collection not found")
            update = request.model_dump(exclude_unset=True)
            row["description"] = update.get("description", row["description"]) or ""
            if "metadata" in update:
                row["metadata_json"] = json.dumps(update["metadata"] or {}, sort_keys=True)
            now = _now()
            if "expires_at" in update:
                row["expires_at"] = update["expires_at"] or ""
                row["updated_at"] = now
            else:
                self._refresh_collection_activity_row(row, activity_at=now)
            (
                self._db.open_table(_COLLECTIONS_TABLE)
                .merge_insert(["scope", "name"])
                .when_matched_update_all()
                .execute([row])
            )
            return self._collection_info(row)

    def _persist_collection_row(self, row: dict[str, Any]) -> None:
        (
            self._db.open_table(_COLLECTIONS_TABLE)
            .merge_insert(["scope", "name"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    @staticmethod
    def _refresh_collection_activity_row(row: dict[str, Any], *, activity_at: str) -> None:
        """Advance collection activity while preserving its expiration window."""

        expires_at = str(row.get("expires_at") or "")
        updated_at = str(row.get("updated_at") or "")
        row["updated_at"] = activity_at
        if not expires_at or not updated_at:
            return
        ttl = datetime.fromisoformat(expires_at) - datetime.fromisoformat(updated_at)
        if ttl.total_seconds() > 0:
            row["expires_at"] = (datetime.fromisoformat(activity_at) + ttl).isoformat()

    def _refresh_collection_activity_locked(self, scope: str, name: str, *, activity_at: str) -> None:
        """Persist successful indexing activity for an active collection."""

        row = self._collection_row(scope, name)
        if not row:
            raise VDBResourceNotFound("Collection not found")
        if row["status"] != "active":
            return
        updated_at = str(row.get("updated_at") or "")
        if updated_at and datetime.fromisoformat(updated_at) >= datetime.fromisoformat(activity_at):
            return
        self._refresh_collection_activity_row(row, activity_at=activity_at)
        self._persist_collection_row(row)

    @staticmethod
    def _retry_at(retry_count: int) -> str:
        delay = min(3600, 2 ** min(max(retry_count, 1), 12))
        return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()

    def _schedule_collection_retry(self, row: dict[str, Any], phase: str, exc: Exception) -> None:
        retries = int(row.get("retry_count") or 0) + 1
        row.update(
            {
                "status": "deleting",
                "deletion_phase": phase,
                "retry_count": retries,
                "next_retry_at": self._retry_at(retries),
                "last_error": str(exc)[:2000],
                "updated_at": _now(),
            }
        )
        self._persist_collection_row(row)

    def _mark_collection_deleting_locked(self, row: dict[str, Any]) -> None:
        """Enter the first deletion phase with a fresh retry budget.

        Callers must already hold ``_write_lock``; the transition is persisted here
        so an interrupted process resumes from a durable phase.
        """
        now = _now()
        row.update(
            {
                "status": "deleting",
                "deletion_phase": "drop_table",
                "retry_count": 0,
                "next_retry_at": "",
                "last_error": "",
                "delete_started_at": now,
                "updated_at": now,
            }
        )
        self._persist_collection_row(row)

    def _cleanup_collection_locked(self, row: dict[str, Any]) -> bool:
        phase = str(row.get("deletion_phase") or "drop_table")
        try:
            if phase == "drop_table":
                self._wait_for_table_users_locked(row["physical_table"])
                self._db.drop_table(row["physical_table"], ignore_missing=True)
                self._opened_tables.pop(row["physical_table"], None)
                row["deletion_phase"] = phase = "delete_catalog"
                row["updated_at"] = _now()
                self._persist_collection_row(row)
            if phase == "delete_catalog":
                self._db.open_table(_DOCUMENTS_TABLE).delete(
                    f"scope = {_quoted(row['scope'])} AND collection_name = {_quoted(row['name'])}"
                )
                self._db.open_table(_COLLECTIONS_TABLE).delete(
                    f"scope = {_quoted(row['scope'])} AND name = {_quoted(row['name'])}"
                )
            return True
        except Exception as exc:
            logger.exception("Collection cleanup paused at phase %s", phase)
            self._schedule_collection_retry(row, phase, exc)
            return False

    def delete_collection(
        self,
        scope: str,
        name: str,
        if_exists: bool,
    ) -> CollectionDeleteResult:
        """Delete a collection through the retryable table-and-catalog lifecycle."""

        with self._write_lock:
            row = self._collection_row(scope, name)
            if not row:
                if if_exists:
                    return CollectionDeleteResult(
                        name=name,
                        scope=scope,
                        existed=False,
                        deleted=False,
                        status="deleted",
                        cleanup_pending=False,
                    )
                raise VDBResourceNotFound("Collection not found")
            if row["status"] != "deleting":
                self._mark_collection_deleting_locked(row)
            deleted = self._cleanup_collection_locked(row)
            return CollectionDeleteResult(
                name=name,
                scope=scope,
                existed=True,
                deleted=deleted,
                status="deleted" if deleted else "deleting",
                cleanup_pending=not deleted,
            )

    def _resolved_table(self, scope: str, name: str) -> str:
        row = self._collection_row(scope, name, active=True)
        if not row:
            raise VDBResourceNotFound("Collection not found")
        table_name = row["physical_table"]
        return table_name

    def _table_capabilities(self, table_name: str) -> LanceTableCapabilities:
        """Inspect an existing table; callers must have confirmed it exists."""
        return inspect_lancedb_table_object(self._open_table(table_name))

    def _resolve_effective_retrieval_mode(
        self,
        table_name: str,
        capabilities: LanceTableCapabilities | None,
    ) -> LanceRetrievalMode:
        if capabilities is None:
            raise RetrievalContractError(f"Unable to inspect collection table {table_name!r}")
        mode: LanceRetrievalMode = capabilities.retrieval_mode
        if mode == "unknown":
            raise RetrievalContractError("Collection table has no supported vector or FTS search capability")
        if mode != "dense":
            raise UnsupportedVDBOperation(
                f"{mode.capitalize()} collection retrieval is not supported; collection queries require dense vectors"
            )
        return mode

    def _persist_document_row(self, row: dict[str, Any]) -> None:
        (
            self._db.open_table(_DOCUMENTS_TABLE)
            .merge_insert(["scope", "collection_name", "document_id"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    def _document_rows(
        self,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        return self._rows(
            _DOCUMENTS_TABLE,
            f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection_name)} "
            f"AND document_id = {_quoted(document_id)}",
        )

    def write_collection(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Append or replace one document using stable, retry-safe chunk identities."""

        with self._collection_write_lock:
            return self._write_collection_serialized(records, context=context)

    def _write_collection_serialized(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Persist recovery state around LanceDB I/O without blocking unrelated queries."""

        rows = _collection_rows(records, context=context)
        completed_row: dict[str, Any] | None = None
        with self._write_lock:
            table_name = self._resolved_table(context.scope, context.collection_name)
            if records and not rows:
                raise VDBInvalidRequest("Collection records produced no writable vector rows")
            existing = self._document_rows(
                context.scope,
                context.collection_name,
                context.document_id,
            )
            if context.operation is IngestOperation.REPLACE:
                if not existing:
                    raise VDBResourceNotFound("Document not found")
            elif existing:
                document = existing[0]
                known_versions = {
                    str(document.get(field) or "")
                    for field in (
                        "document_version",
                        "current_document_version",
                        "pending_document_version",
                    )
                    if document.get(field)
                }
                if document.get("recovery_state") not in {
                    "",
                    "appending",
                } or known_versions != {context.document_version}:
                    raise VDBResourceConflict("append cannot change an existing document; use replace")
                stored_hash = str(document.get("content_sha256") or "")
                if stored_hash and stored_hash != context.content_sha256:
                    raise VDBResourceConflict("append content does not match the existing document; use replace")

            if rows:
                now = _now()
                created_at = existing[0]["created_at"] if existing else now
                if context.operation is IngestOperation.APPEND:
                    marker = (
                        dict(existing[0])
                        if existing
                        else {
                            "scope": context.scope,
                            "collection_name": context.collection_name,
                            "document_id": context.document_id,
                            "job_id": context.job_id or "",
                            "filename": context.filename,
                            "content_sha256": context.content_sha256,
                            "document_version": "",
                            "status": "appending",
                            "chunk_count": 0,
                            "created_at": created_at,
                            "updated_at": now,
                            "error": "",
                            "current_document_version": "",
                            "pending_document_version": "",
                            "recovery_state": "",
                        }
                    )
                    marker.update(
                        {
                            "job_id": context.job_id or "",
                            "pending_document_version": context.document_version,
                            "recovery_state": "appending",
                            "updated_at": now,
                            "error": "",
                        }
                    )
                    self._persist_document_row(marker)
                elif existing:
                    marker = dict(existing[0])
                    marker.update(
                        {
                            "status": "replacing",
                            "pending_document_version": context.document_version,
                            "recovery_state": "replacing",
                            "updated_at": now,
                            "error": "",
                        }
                    )
                    self._persist_document_row(marker)

                completed_row = {
                    "scope": context.scope,
                    "collection_name": context.collection_name,
                    "document_id": context.document_id,
                    "job_id": context.job_id or "",
                    "filename": context.filename,
                    "content_sha256": context.content_sha256,
                    "document_version": context.document_version,
                    "status": "completed",
                    "chunk_count": len(rows),
                    "created_at": created_at,
                    "updated_at": now,
                    "error": "",
                    "current_document_version": context.document_version,
                    "pending_document_version": "",
                    "recovery_state": "refreshing_collection_activity",
                }
            cached_table = self._opened_tables.get(table_name)
            self._acquire_table_user_locked(table_name)

        try:
            table_exists = self._has_table(table_name)
            table = cached_table
            if rows:
                if not table_exists:
                    vector_dim = infer_vector_dim(rows)
                    if vector_dim == 0:
                        raise VDBInvalidRequest("Cannot infer vector dimension from collection records")
                    schema = _with_embedding_model_metadata(
                        lancedb_schema(vector_dim=vector_dim, collection_managed=True),
                        self._backend.embedding_model_name,
                    )
                    table = create_or_append_lancedb_table(
                        self._db,
                        table_name,
                        rows,
                        schema,
                        overwrite=True,
                    )
                    logger.info(
                        "Created collection LanceDB table %r with %d rows (dim=%d)",
                        table_name,
                        len(rows),
                        vector_dim,
                    )
                else:
                    table = table or self._db.open_table(table_name)
                    _validate_embedding_model(
                        table,
                        self._backend.embedding_model_name,
                        table_name=table_name,
                    )
                    if context.operation is IngestOperation.REPLACE:
                        predicate = f"document_id = {_quoted(context.document_id)}"
                        (
                            table.merge_insert("chunk_id")
                            .when_matched_update_all()
                            .when_not_matched_insert_all()
                            .when_not_matched_by_source_delete(predicate)
                            .execute(rows)
                        )
                    else:
                        (
                            table.merge_insert("chunk_id")
                            .when_matched_update_all()
                            .when_not_matched_insert_all()
                            .execute(rows)
                        )
                    logger.info(
                        "Wrote %d rows to collection table %r operation=%s",
                        len(rows),
                        table_name,
                        context.operation,
                    )
            elif table_exists:
                table = table or self._db.open_table(table_name)

            total_rows = int(table.count_rows()) if table is not None else 0
            with self._write_lock:
                if table is not None:
                    self._opened_tables[table_name] = table
                if completed_row is not None:
                    self._persist_document_row(completed_row)
                    self._refresh_collection_activity_locked(
                        context.scope,
                        context.collection_name,
                        activity_at=completed_row["updated_at"],
                    )
                    completed_row["recovery_state"] = ""
                    self._persist_document_row(completed_row)
            return CollectionWriteResult(written=len(rows), total_rows=total_rows)
        finally:
            self._release_table_user(table_name)

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
        """Run scoped dense retrieval and expose finite native vector distances."""

        if len(query_texts) != len(vectors):
            raise RetrievalContractError("query_texts must contain one entry per query vector")
        with self._write_lock:
            table_name = self._resolved_table(scope, collection_name)
            if not self._has_table(table_name):
                return ([[] for _ in vectors], ["dense"])
            _validate_embedding_model(
                self._open_table(table_name),
                self._backend.embedding_model_name,
                table_name=table_name,
            )
            capabilities = self._table_capabilities(table_name)
            self._resolve_effective_retrieval_mode(table_name, capabilities)
            retrieval_kwargs: dict[str, Any] = {
                **kwargs,
                "table_name": table_name,
                "top_k": top_k,
                "hybrid": False,
            }
            pending_document_ids = sorted(
                str(row["document_id"])
                for row in self._rows(
                    _DOCUMENTS_TABLE,
                    f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection_name)}",
                )
                if _is_uncommitted_initial_append(row)
            )
            if pending_document_ids:
                visibility_filter = " AND ".join(
                    f"document_id != {_quoted(document_id)}" for document_id in pending_document_ids
                )
                requested_filter = retrieval_kwargs.get("where", retrieval_kwargs.get("_filter"))
                if requested_filter is not None and str(requested_filter).strip():
                    visibility_filter = f"({str(requested_filter).strip()}) AND ({visibility_filter})"
                retrieval_kwargs["where"] = visibility_filter

            if capabilities is not None and capabilities.vector_column and capabilities.vector_column != "vector":
                retrieval_kwargs["vector_column_name"] = capabilities.vector_column
            self._acquire_table_user_locked(table_name)

        try:
            raw_results = self._backend.retrieval(vectors, **retrieval_kwargs)
            normalized_results = _normalize_collection_results(raw_results, expected_queries=len(vectors))
            public_results = [[_public_collection_hit(hit) for hit in hits] for hits in normalized_results]
            return public_results, ["dense"]
        finally:
            self._release_table_user(table_name)

    def list_documents(
        self,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        """List committed documents in a collection using a context-bound cursor."""

        self._resolved_table(scope, collection_name)
        rows = self._rows(
            _DOCUMENTS_TABLE,
            f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection_name)}",
        )
        rows = [row for row in rows if not _is_uncommitted_initial_append(row)]
        rows.sort(key=lambda row: (row["created_at"], row["document_id"]))
        last = _decode_cursor(
            continuation_token,
            resource="documents",
            scope=scope,
            collection=collection_name,
        )
        if last is not None:
            if len(last) != 2:
                raise VDBInvalidRequest("Invalid document continuation token")
            rows = [row for row in rows if (row["created_at"], row["document_id"]) > (last[0], last[1])]
        page = rows[:limit]
        return DocumentPage(
            items=[self._document_info(row) for row in page],
            next_token=(
                _encode_cursor(
                    "documents",
                    scope,
                    collection_name,
                    [page[-1]["created_at"], page[-1]["document_id"]],
                )
                if len(rows) > limit and page
                else None
            ),
        )

    def get_document(
        self,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> DocumentInfo:
        """Return one committed document from a scoped collection."""

        self._resolved_table(scope, collection_name)
        rows = self._document_rows(scope, collection_name, document_id)
        if not rows or _is_uncommitted_initial_append(rows[0]):
            raise VDBResourceNotFound("Document not found")
        return self._document_info(rows[0])

    def _reconcile_document_row_locked(self, row: dict[str, Any], table_name: str) -> bool:
        """Complete or roll back one interrupted document lifecycle operation."""

        activity_refresh = "refreshing_collection_activity"
        state = str(row.get("recovery_state") or "")
        scope = str(row["scope"])
        collection_name = str(row["collection_name"])
        document_id = str(row["document_id"])
        try:
            if state in {"appending", "replacing"}:
                pending = str(row.get("pending_document_version") or "")
                pending_chunks: list[dict[str, Any]] = []
                if self._has_table(table_name):
                    chunks = self._rows(
                        table_name,
                        f"document_id = {_quoted(document_id)}",
                        ["document_version", "content_sha256", "filename"],
                    )
                    pending_chunks = [chunk for chunk in chunks if str(chunk.get("document_version") or "") == pending]
                if pending and pending_chunks:
                    pending_chunk = pending_chunks[0]
                    activity_at = _now()
                    row.update(
                        {
                            "document_version": pending,
                            "current_document_version": pending,
                            "content_sha256": str(
                                pending_chunk.get("content_sha256") or row.get("content_sha256") or ""
                            ),
                            "filename": str(pending_chunk.get("filename") or row.get("filename") or ""),
                            "chunk_count": len(pending_chunks),
                            "pending_document_version": "",
                            "status": "completed",
                            "recovery_state": activity_refresh,
                            "updated_at": activity_at,
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    state = activity_refresh
                elif state == "appending" and not row.get("current_document_version"):
                    self._db.open_table(_DOCUMENTS_TABLE).delete(
                        f"scope = {_quoted(scope)} "
                        f"AND collection_name = {_quoted(collection_name)} "
                        f"AND document_id = {_quoted(document_id)}"
                    )
                    return True
                else:
                    row.update(
                        {
                            "pending_document_version": "",
                            "status": "completed",
                            "recovery_state": "",
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    return True
            if state == activity_refresh:
                self._refresh_collection_activity_locked(
                    scope,
                    collection_name,
                    activity_at=row["updated_at"],
                )
                row.update({"recovery_state": "", "error": ""})
                self._persist_document_row(row)
                return True
            if state == "deleting_chunks":
                if self._has_table(table_name):
                    self._wait_for_table_users_locked(table_name)
                    self._open_table(table_name).delete(f"document_id = {_quoted(document_id)}")
                self._db.open_table(_DOCUMENTS_TABLE).delete(
                    f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection_name)} "
                    f"AND document_id = {_quoted(document_id)}"
                )
                return True
            return state == ""
        except Exception as exc:
            row["error"] = str(exc)[:2000]
            if state == activity_refresh:
                row["recovery_state"] = activity_refresh
            else:
                row["updated_at"] = _now()
            self._persist_document_row(row)
            logger.exception("Document reconciliation paused in state %s", state)
            return False

    def delete_document(
        self,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        """Delete a document's chunks and catalog record with recovery state."""

        with self._write_lock:
            table_name = self._resolved_table(scope, collection_name)
            rows = self._document_rows(scope, collection_name, document_id)
            if not rows:
                if if_exists:
                    return DocumentDeleteResult(
                        document_id=document_id,
                        collection_name=collection_name,
                        scope=scope,
                        existed=False,
                        deleted=False,
                        status="deleted",
                        cleanup_pending=False,
                    )
                raise VDBResourceNotFound("Document not found")
            row = rows[0]
            if row.get("recovery_state") != "deleting_chunks":
                row.update(
                    {
                        "status": "deleting",
                        "recovery_state": "deleting_chunks",
                        "updated_at": _now(),
                        "error": "",
                    }
                )
                self._persist_document_row(row)
            deleted = self._reconcile_document_row_locked(row, table_name)
        return DocumentDeleteResult(
            document_id=document_id,
            collection_name=collection_name,
            scope=scope,
            existed=True,
            deleted=deleted,
            status="deleted" if deleted else "deleting",
            cleanup_pending=not deleted,
        )

    def reconcile_collections(self) -> dict[str, int]:
        """Resume recoverable VDB lifecycle work and expire due collections."""
        successes = 0
        failures = 0
        now = datetime.now(timezone.utc)
        now_quoted = _quoted(now.isoformat())

        for candidate in self._rows(_DOCUMENTS_TABLE, "recovery_state != ''"):
            with self._write_lock:
                rows = self._document_rows(
                    candidate["scope"],
                    candidate["collection_name"],
                    candidate["document_id"],
                )
                if not rows or not rows[0].get("recovery_state"):
                    continue
                row = rows[0]
                collection = self._collection_row(row["scope"], row["collection_name"])
                if not collection:
                    continue
                table_name = collection["physical_table"]
                self._wait_for_table_users_locked(table_name)
                rows = self._document_rows(row["scope"], row["collection_name"], row["document_id"])
                if not rows or not rows[0].get("recovery_state"):
                    continue
                if self._reconcile_document_row_locked(rows[0], table_name):
                    successes += 1
                else:
                    failures += 1

        collection_filter = f"status = 'deleting' AND (next_retry_at = '' OR next_retry_at <= {now_quoted})"
        if self.expiration_cleanup_enabled:
            collection_filter += " OR (status = 'active' AND expires_at != '' " f"AND expires_at <= {now_quoted})"
        for candidate in self._rows(_COLLECTIONS_TABLE, collection_filter):
            with self._write_lock:
                row = self._collection_row(candidate["scope"], candidate["name"])
                if not row:
                    continue
                if (
                    self.expiration_cleanup_enabled
                    and row.get("status") == "active"
                    and row.get("expires_at")
                    and datetime.fromisoformat(str(row["expires_at"])) <= now
                ):
                    self._mark_collection_deleting_locked(row)
                if row.get("status") != "deleting":
                    continue
                retry_at = str(row.get("next_retry_at") or "")
                if retry_at and datetime.fromisoformat(retry_at) > now:
                    continue
                if self._cleanup_collection_locked(row):
                    successes += 1
                else:
                    failures += 1

        with self._write_lock:
            self.reconciliation_successes += successes
            self.reconciliation_failures += failures
        return {"successes": successes, "failures": failures}

    @staticmethod
    def empty_health() -> dict[str, Any]:
        """Return collection health before the lazy store is initialized."""
        return {
            "catalog": {
                "healthy": True,
                "initialized": False,
                "schema_version": _CATALOG_SCHEMA_VERSION,
            },
            "collections": {"active": 0, "deleting": 0, "expired": 0},
            "cleanup": {
                "pending": 0,
                "oldest_age_seconds": 0.0,
            },
            "reconciliation": {
                "successes": 0,
                "failures": 0,
            },
            "open_table_cache_count": 0,
        }

    def health(self) -> dict[str, Any]:
        """Summarize catalog, cleanup, and reconciliation state without identifiers."""

        now = datetime.now(timezone.utc)
        collections = self._rows(
            _COLLECTIONS_TABLE,
            columns=["status", "expires_at", "delete_started_at"],
        )
        documents = self._rows(_DOCUMENTS_TABLE, columns=["recovery_state", "updated_at"])
        active = sum(row.get("status") == "active" for row in collections)
        deleting = sum(row.get("status") == "deleting" for row in collections)
        expired = sum(
            bool(row.get("expires_at")) and datetime.fromisoformat(str(row["expires_at"])) <= now for row in collections
        )
        pending_times: list[datetime] = []
        for row in collections:
            if row.get("status") == "deleting" and row.get("delete_started_at"):
                pending_times.append(datetime.fromisoformat(str(row["delete_started_at"])))
        for row in documents:
            if row.get("recovery_state") and row.get("updated_at"):
                pending_times.append(datetime.fromisoformat(str(row["updated_at"])))
        oldest_age = max(((now - started).total_seconds() for started in pending_times), default=0.0)
        return {
            "catalog": {
                "healthy": True,
                "initialized": True,
                "schema_version": _CATALOG_SCHEMA_VERSION,
            },
            "collections": {
                "active": active,
                "deleting": deleting,
                "expired": expired,
            },
            "cleanup": {
                "pending": len(pending_times),
                "oldest_age_seconds": round(oldest_age, 3),
            },
            "reconciliation": {
                "successes": self.reconciliation_successes,
                "failures": self.reconciliation_failures,
            },
            "open_table_cache_count": len(self._opened_tables),
        }
