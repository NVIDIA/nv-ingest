# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An explicit indexed/flat pair against one pinned LanceDB table handle."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.vdb.query_options import validate_hnsw_ef

from nemo_retriever.query.comparison import _digest, _vector_hashes, _yaml_digest


def _stat(stats: Any, key: str) -> Any:
    return stats.get(key) if isinstance(stats, dict) else getattr(stats, key, None)


def compare_dense_search(
    backend: Any,
    vectors: list[list[float]],
    *,
    expected_model_name: str | None,
    expected_model_revision: str | None,
    **kwargs: Any,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, Any]]:
    """Search one immutable version twice, verifying the exact shared inputs.

    This function creates no index and never changes a cached table's checkout.
    The caller must already have completed normal query-model admission.
    """
    from nemo_retriever.common.vdb.lancedb import _assert_lancedb_table_ready, _table_schema

    if kwargs.get("hybrid", backend.hybrid) or kwargs.get("hybrid_fusion") is not None:
        raise ValueError("Exact-search comparison supports dense retrieval only.")
    if any(kwargs.get(key) is not None for key in ("scope", "collection_name")):
        raise ValueError("Exact-search comparison does not support collections.")
    if kwargs.get("search_kwargs"):
        raise ValueError("Exact-search comparison does not accept additional search_kwargs.")
    if backend.metric != "l2":
        raise ValueError("Exact-search comparison currently requires the normal l2 metric.")
    uri = str(kwargs.get("table_path") or backend.uri)
    name = str(kwargs.get("table_name") or backend.table_name)
    column = str(kwargs.get("vector_column_name", "vector"))
    top_k = int(kwargs.get("top_k", 10))
    refine = int(kwargs.get("refine_factor", 50))
    nprobes = int(kwargs.get("n_probe", kwargs.get("nprobes", 64)))
    ef = validate_hnsw_ef(kwargs.get("ef"))
    if min(top_k, refine, nprobes) <= 0:
        raise ValueError("Exact-search comparison requires positive candidate/probe/refinement bounds.")
    where = kwargs.get("where")
    if where is None:
        where = kwargs.get("_filter")
    where = str(where).strip() or None if where is not None else None
    fields = kwargs.get("result_fields")

    # Open a private version-pinned handle, separate from backend-owned cached
    # handles that normal queries may advance with checkout_latest().
    connection = backend._connect(uri)
    latest = connection.open_table(name)
    version = int(latest.version)
    table = connection.open_table(name, version=version)
    if int(table.version) != version:
        raise RuntimeError("Could not pin the exact-search comparison table version.")
    _assert_lancedb_table_ready(table)
    schema = _table_schema(table)
    dimension = int(getattr(schema.field(column).type, "list_size", 0))
    before_hashes = _vector_hashes(vectors, dimension)
    metadata = schema.metadata or {}

    def index_metadata(key: str) -> str | None:
        value = metadata.get(f"nemo_retriever.{key}".encode("utf-8"))
        return value.decode("utf-8").strip() or None if value is not None else None

    model = index_metadata("embedding_model_name")
    revision = index_metadata("embedding_model_revision")
    if not model or model != expected_model_name:
        raise ValueError("Pinned table model differs from the admitted query embedding model.")
    if expected_model_revision is not None and revision != expected_model_revision:
        raise ValueError("Pinned table revision differs from the admitted query embedding revision.")
    indices = [index for index in table.list_indices() if list(index.columns) == [column]]
    if len(indices) != 1:
        raise ValueError("Exact-search comparison requires exactly one index on the selected vector column.")
    index = indices[0]
    stats = table.index_stats(index.name)
    if stats is None or _stat(stats, "distance_type") != "l2":
        raise ValueError("Exact-search comparison requires a verified l2 vector index.")
    snapshot = dict(
        table_uri=uri,
        table_name=name,
        table_version=version,
        table_rows=int(table.count_rows()),
        vector_column=column,
        vector_dimension=dimension,
        metric="l2",
        index_name=index.name,
        index_type=str(_stat(stats, "index_type")),
        indexed_rows=int(_stat(stats, "num_indexed_rows")),
        unindexed_rows=int(_stat(stats, "num_unindexed_rows")),
        embedding_model_name=model,
        embedding_model_revision=revision,
        candidate_k=top_k,
        indexed_nprobes=nprobes,
        indexed_refine_factor=refine,
        indexed_ef=ef,
        predicate_sha256=_digest(where) if where is not None else None,
    )

    def search(*, exact: bool) -> list[list[dict[str, Any]]]:
        results = []
        for vector in vectors:
            # Keep the normal retrieval vector container, predicate, limit and
            # projection. Only the explicitly exact leg bypasses the index.
            query = table.search([vector], vector_column_name=column)
            if where is not None:
                query = query.where(where)
            query = query.limit(top_k)
            if exact:
                query = query.bypass_vector_index()
            else:
                query = query.refine_factor(refine).nprobes(nprobes)
                if ef is not None:
                    query = query.ef(ef)
            if fields is not None:
                query = query.select(fields)
            results.append(query.to_list())
        return results

    indexed = search(exact=False)
    indexed_hashes = _vector_hashes(vectors, dimension)
    if indexed_hashes != before_hashes:
        raise RuntimeError("Indexed search changed the shared query-vector inputs.")
    exact = search(exact=True)
    if _vector_hashes(vectors, dimension) != before_hashes or int(table.version) != version:
        raise RuntimeError("Exact-search comparison inputs or pinned table version changed.")
    if len(indexed) != len(vectors) or len(exact) != len(vectors):
        raise RuntimeError("Exact-search comparison returned incomplete expanded-row coverage.")
    snapshot["vector_hashes"] = before_hashes
    snapshot["indexed_vector_fingerprint"] = _yaml_digest(list(indexed_hashes))
    snapshot["exact_vector_fingerprint"] = _yaml_digest(list(before_hashes))
    return indexed, exact, snapshot
