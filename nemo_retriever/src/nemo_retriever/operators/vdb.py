# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin operators around the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd

from nemo_retriever.common.nvtx import batch_phase
from nemo_retriever.common.vdb.adt_vdb import CollectionWriteContext, UnsupportedVDBOperation, VDB
from nemo_retriever.common.vdb.factory import get_vdb_op_cls
from nemo_retriever.common.vdb.records import (
    _iter_client_vdb_records,
    normalize_retrieval_results,
    to_client_vdb_records,
    validate_collection_retrieval_results,
)
from nemo_retriever.common.vdb.sidecar_metadata import (
    _apply_sidecar_metadata_to_client_record,
    apply_sidecar_metadata_to_client_batches,
    build_sidecar_lookup,
    materialize_sidecar_dataframe,
    split_sidecar_from_vdb_kwargs,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator


def _construct_vdb(
    *,
    vdb: VDB | None = None,
    vdb_op: str | None = None,
    vdb_kwargs: dict[str, Any] | None = None,
) -> VDB:
    if vdb is not None and vdb_op is not None:
        raise ValueError("Pass either vdb or vdb_op, not both.")
    if vdb is None and vdb_op is None:
        raise ValueError("Either vdb or vdb_op is required.")

    return vdb if vdb is not None else get_vdb_op_cls(str(vdb_op))(**dict(vdb_kwargs or {}))


def _iter_batch_rows(batches: Iterable[pd.DataFrame]) -> Iterator[dict[str, Any]]:
    """Yield graph rows from the executor's retained pandas batches."""

    for batch in batches:
        yield from batch.to_dict(orient="records")


def _coerce_embedding_vector(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("embedding")
    if not isinstance(value, list):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
    if isinstance(value, list) and value:
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


def _is_direct_embedding_column(column_name: object) -> bool:
    name = str(column_name).strip().lower()
    return "embedding" in name or name == "vector" or name.endswith("_vector")


def _embedding_error(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if error is None:
        return None
    message = str(error).strip()
    return message or None


def query_vectors_from_embedded_dataframe(df: pd.DataFrame) -> list[list[float]]:
    """Extract one query vector per row from batch-embed output (metadata or payload columns)."""
    vectors: list[list[float]] = []
    for _, row in df.iterrows():
        vec: list[float] | None = None
        embedding_error: str | None = None
        md = row.get("metadata")
        if isinstance(md, dict):
            vec = _coerce_embedding_vector(md)
            embedding_error = _embedding_error(md)
        if vec is None:
            for col in df.columns:
                if col == "metadata":
                    continue
                val = row.get(col)
                if isinstance(val, dict) or _is_direct_embedding_column(col):
                    vec = _coerce_embedding_vector(val)
                    embedding_error = embedding_error or _embedding_error(val)
                if vec is not None:
                    break
        if vec is None:
            error_suffix = f"; embedding error: {embedding_error}" if embedding_error else ""
            raise ValueError(
                "Expected query embeddings in each row's metadata['embedding'] or a payload column "
                f"with key 'embedding'; columns={list(df.columns)}{error_suffix}"
            )
        vectors.append(vec)
    return vectors


class IngestVdbOperator(AbstractOperator):
    """Upload already-embedded graph output through an nv-ingest-client VDB."""

    #: Ray batch mode: repartition to one block and one ``map_batches`` call so
    #: ``VDB.run`` sees the full dataset once (matches historical post-graph upload).
    REQUIRES_GLOBAL_BATCH: bool = True

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, sidecar = split_sidecar_from_vdb_kwargs(merged)
        super().__init__(
            vdb=vdb,
            vdb_op=vdb_op,
            vdb_kwargs=merged,
        )
        self._vdb_kwargs = clean_kwargs
        self._sidecar_spec = sidecar
        self._sidecar_lookup: dict[str, dict[str, Any]] | None = None
        if sidecar is not None:
            _df = materialize_sidecar_dataframe(sidecar)
            self._sidecar_lookup = build_sidecar_lookup(
                _df,
                sidecar["meta_source_field"],
                sidecar["meta_fields"],
            )
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        # Graph ingest emits flat embedded rows, while
        # nv-ingest-client VDB.run still expects nested Nemo Retriever Library (NRL) records.
        records = to_client_vdb_records(data)
        if self._sidecar_spec is not None and self._sidecar_lookup is not None:
            records = apply_sidecar_metadata_to_client_batches(
                records,
                lookup=self._sidecar_lookup,
                meta_fields=self._sidecar_spec["meta_fields"],
                join_key=self._sidecar_spec["meta_join_key"],
            )
        collection_context = kwargs.get("collection_context")
        if collection_context is not None:
            if not isinstance(collection_context, CollectionWriteContext):
                raise TypeError("collection_context must be a CollectionWriteContext")
            if not records or not any(records):
                raise ValueError("Collection writes require at least one canonical VDB record")
            return self._vdb.write_collection(records, context=collection_context)
        if records and any(batch for batch in records):
            self._vdb.run(records)
        return data

    def _supports_stream_ingest(self) -> bool:
        """Return whether the configured VDB opts into canonical record streaming."""

        return bool(getattr(self._vdb, "supports_stream_ingest", False))

    @batch_phase("pipeline.terminal_stream")
    def _stream_ingest(self, batches: Iterable[pd.DataFrame]) -> None:
        """Lazily execute upstream batches and delegate one backend stream."""

        if not self._supports_stream_ingest():
            raise UnsupportedVDBOperation(f"{type(self._vdb).__name__} does not implement stream_ingest()")

        records: Iterable[dict[str, Any]] = _iter_client_vdb_records(_iter_batch_rows(batches))
        if self._sidecar_spec is not None and self._sidecar_lookup is not None:
            undecorated_records = records

            def with_sidecar() -> Iterator[dict[str, Any]]:
                for record in undecorated_records:
                    yield _apply_sidecar_metadata_to_client_record(
                        record,
                        lookup=self._sidecar_lookup,
                        meta_fields=self._sidecar_spec["meta_fields"],
                        join_key=self._sidecar_spec["meta_join_key"],
                    )

            records = with_sidecar()

        record_stream = records
        exhausted = False

        def required_records() -> Iterator[dict[str, Any]]:
            nonlocal exhausted
            yield from record_stream
            exhausted = True

        self._vdb.stream_ingest(required_records())
        if not exhausted:
            raise RuntimeError(
                f"{type(self._vdb).__name__}.stream_ingest() returned before consuming the record stream"
            )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class PutVdbOperator(IngestVdbOperator):
    """Replace existing rows of a VDB table in place on a stable row key.

    Unlike :class:`IngestVdbOperator` (which orchestrates create_index +
    write_to_index, optionally overwriting the whole table), this operator
    calls ``vdb.put(records, ...)`` so that only rows whose ``key`` is in
    ``records`` are touched. Existing rows that match by ``key`` are
    replaced; rows in ``records`` whose ``key`` is not already present in
    the table raise :class:`KeyError` (``put`` never inserts new rows),
    and rows in the table that are not referenced are left untouched.

    The underlying VDB implementation must override
    :meth:`~nemo_retriever.vdb.adt_vdb.VDB.put` with a real
    stable-key in-place replace; currently this is implemented by
    :class:`~nemo_retriever.vdb.lancedb.LanceDB`. ``VDB.put`` itself
    raises :class:`NotImplementedError`, so backends that have not
    overridden it are detected at construction time and fail fast rather
    than silently no-oping at runtime.
    """

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
        key: str = "id",
        table_name: str | None = None,
    ) -> None:
        super().__init__(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=vdb_kwargs)
        # ``put`` is part of the abstract VDB contract, but the base
        # class provides a NotImplementedError stub for backends that
        # cannot support stable-key puts. Treat a not-overridden stub
        # as "unsupported" so misuse surfaces here instead of at the
        # first write.
        if getattr(type(self._vdb), "put", None) is VDB.put:
            raise NotImplementedError(f"VDB backend {type(self._vdb).__name__!r} does not implement put(); ")
        self._key = key
        self._table_name = table_name

    def process(self, data: Any, **kwargs: Any) -> Any:
        records = to_client_vdb_records(data)
        if self._sidecar_spec is not None and self._sidecar_lookup is not None:
            records = apply_sidecar_metadata_to_client_batches(
                records,
                lookup=self._sidecar_lookup,
                meta_fields=self._sidecar_spec["meta_fields"],
                join_key=self._sidecar_spec["meta_join_key"],
            )
        if records and any(batch for batch in records):
            self._vdb.put(records, table_name=self._table_name, key=self._key)
        return data

    def _supports_stream_ingest(self) -> bool:
        """Keep update-only put semantics on the historical global-batch path."""

        return False


class RetrieveVdbOperator(AbstractOperator):
    """Retrieve hits from an nv-ingest-client VDB using precomputed query vectors."""

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
        explode_for_rerank: bool = False,
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, _sidecar = split_sidecar_from_vdb_kwargs(merged)
        clean_kwargs.pop("query_texts", None)
        super().__init__(
            vdb=vdb,
            vdb_op=vdb_op,
            vdb_kwargs=clean_kwargs,
            explode_for_rerank=explode_for_rerank,
        )
        self._vdb_kwargs = clean_kwargs
        self._retrieval_vdb_kwargs = clean_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)
        self._explode_for_rerank = bool(explode_for_rerank)

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Read one index metadata value through the configured VDB."""
        return self._vdb.get_index_metadata(key, **{**self._vdb_kwargs, **kwargs})

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            return query_vectors_from_embedded_dataframe(data)
        return data

    def process(
        self, data: Any, **kwargs: Any
    ) -> list[list[dict[str, Any]]] | tuple[list[list[dict[str, Any]]], list[str]]:
        from nemo_retriever.graph.retriever_utils import filter_retrieval_kwargs

        runtime_kwargs = dict(kwargs)
        scope = runtime_kwargs.pop("scope", None)
        collection_name = runtime_kwargs.pop("collection_name", None)
        if (scope is None) != (collection_name is None):
            raise ValueError("Collection retrieval requires both scope and collection_name")

        retrieval_kwargs = {
            **self._retrieval_vdb_kwargs,
            **filter_retrieval_kwargs(runtime_kwargs),
        }
        if "hybrid" in retrieval_kwargs:
            effective_hybrid = bool(retrieval_kwargs["hybrid"])
        else:
            effective_hybrid = bool(getattr(self._vdb, "hybrid", False))
        if collection_name is not None:
            retrieval_kwargs.pop("collection_name", None)
            top_k = int(retrieval_kwargs.pop("top_k", 10))
            result = self._vdb.retrieve_collection(
                data,
                scope=str(scope),
                collection_name=str(collection_name),
                query_texts=list(kwargs.get("query_texts") or []),
                top_k=top_k,
                **retrieval_kwargs,
            )
            return validate_collection_retrieval_results(result, expected_queries=len(data))

        if effective_hybrid and "query_texts" in kwargs:
            retrieval_kwargs["query_texts"] = kwargs["query_texts"]
        return normalize_retrieval_results(self._vdb.retrieval(data, **retrieval_kwargs))

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        if not self._explode_for_rerank:
            return data
        query_texts = kwargs.get("query_texts")
        if not query_texts:
            return data
        from nemo_retriever.graph.retriever_utils import hits_lists_to_rerank_dataframe

        if not isinstance(data, list):
            return data
        return hits_lists_to_rerank_dataframe(list(query_texts), data)
