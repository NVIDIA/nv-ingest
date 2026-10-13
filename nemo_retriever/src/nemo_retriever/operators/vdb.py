# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin operators around the nv-ingest-client VDB abstraction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import nvtx
import pandas as pd

from nemo_retriever.common.vdb.adt_vdb import CollectionWriteContext, UnsupportedVDBOperation, VDB
from nemo_retriever.common.vdb.factory import get_vdb_op_cls
from nemo_retriever.common.vdb.records import (
    _iter_client_vdb_records,
    embedding_vector,
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

    from nemo_retriever.graph.executor import frame_to_records

    for batch in batches:
        yield from frame_to_records(batch)


def query_vectors_from_embedded_dataframe(
    df: pd.DataFrame,
    *,
    embedding_column: str = "embedding",
) -> list[list[float]]:
    """Return one query vector per row from the embed stage's output column."""
    if embedding_column not in df.columns:
        raise ValueError(f"Expected query embeddings in the '{embedding_column}' column; columns={list(df.columns)}")
    error_column = f"{embedding_column}_error"
    errors = df[error_column].tolist() if error_column in df.columns else [None] * len(df)
    vectors: list[list[float]] = []
    for index, (value, error) in enumerate(zip(df[embedding_column].tolist(), errors)):
        vector = embedding_vector(value)
        if vector is None:
            suffix = f"; embedding error: {error}" if isinstance(error, str) and error.strip() else ""
            raise ValueError(f"Query row {index} has no vector in the '{embedding_column}' column{suffix}")
        vectors.append(vector)
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
        embedding_column: str = "embedding",
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, sidecar = split_sidecar_from_vdb_kwargs(merged)
        super().__init__(
            vdb=vdb,
            vdb_op=vdb_op,
            vdb_kwargs=merged,
            embedding_column=embedding_column,
        )
        self._embedding_column = embedding_column
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
        records = to_client_vdb_records(data, embedding_column=self._embedding_column)
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

    @nvtx.annotate("nrl.batch::pipeline.terminal_stream", color="blue")
    def _stream_ingest(self, batches: Iterable[pd.DataFrame]) -> int:
        """Write one backend stream and return the number of canonical records submitted."""

        if not self._supports_stream_ingest():
            raise UnsupportedVDBOperation(f"{type(self._vdb).__name__} does not implement stream_ingest()")

        records: Iterable[dict[str, Any]] = _iter_client_vdb_records(
            _iter_batch_rows(batches),
            embedding_column=self._embedding_column,
        )
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
        submitted_records = 0

        def required_records() -> Iterator[dict[str, Any]]:
            nonlocal exhausted, submitted_records
            for record in record_stream:
                submitted_records += 1
                yield record
            exhausted = True

        self._vdb.stream_ingest(required_records())
        if not exhausted:
            raise RuntimeError(
                f"{type(self._vdb).__name__}.stream_ingest() returned before consuming the record stream"
            )
        return submitted_records

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class RetrieveVdbOperator(AbstractOperator):
    """Retrieve hits from an nv-ingest-client VDB using precomputed query vectors."""

    def __init__(
        self,
        *,
        vdb: VDB | None = None,
        vdb_op: str | None = None,
        vdb_kwargs: dict[str, Any] | None = None,
        explode_for_rerank: bool = False,
        embedding_column: str = "embedding",
    ) -> None:
        merged = dict(vdb_kwargs or {})
        clean_kwargs, _sidecar = split_sidecar_from_vdb_kwargs(merged)
        clean_kwargs.pop("query_texts", None)
        super().__init__(
            vdb=vdb,
            vdb_op=vdb_op,
            vdb_kwargs=clean_kwargs,
            explode_for_rerank=explode_for_rerank,
            embedding_column=embedding_column,
        )
        self._vdb_kwargs = clean_kwargs
        self._retrieval_vdb_kwargs = clean_kwargs
        self._vdb = _construct_vdb(vdb=vdb, vdb_op=vdb_op, vdb_kwargs=clean_kwargs)
        self._explode_for_rerank = bool(explode_for_rerank)
        self._embedding_column = embedding_column

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Read one index metadata value through the configured VDB."""
        return self._vdb.get_index_metadata(key, **{**self._vdb_kwargs, **kwargs})

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            return query_vectors_from_embedded_dataframe(data, embedding_column=self._embedding_column)
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
