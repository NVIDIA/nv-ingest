# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-only evidence for an opt-in paired dense-search diagnostic.

This API does not persist vectors. Hash metadata uses deterministic safe YAML;
vector payloads use IEEE754 little-endian float64 to preserve the Python float
values produced by the normal query-vector extraction without changing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from nemo_retriever.common.vdb.records import RetrievalHit


def _digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _yaml_digest(value: Any) -> str:
    return _digest(
        yaml.safe_dump(
            value,
            sort_keys=True,
            allow_unicode=True,
            explicit_start=True,
            explicit_end=True,
            default_flow_style=False,
            line_break="\n",
            indent=2,
            width=4096,
        ).encode("utf-8")
    )


def _vector_hashes(vectors: list[list[float]], dimension: int) -> tuple[str, ...]:
    if not vectors or len(vectors) > 256:
        raise ValueError("Exact-search comparison requires between 1 and 256 expanded query rows.")
    if not 0 < dimension <= 65536:
        raise ValueError("Exact-search comparison requires a vector dimension between 1 and 65536.")
    hashes = []
    for vector in vectors:
        if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ValueError("Exact-search comparison requires finite, dimension-compatible query vectors.")
        digest = hashlib.sha256()
        for offset in range(0, dimension, 256):
            chunk = vector[offset : offset + 256]
            digest.update(struct.pack(f"<{len(chunk)}d", *chunk))
        hashes.append(digest.hexdigest())
    return tuple(hashes)


@dataclass(frozen=True)
class DenseSearchRowFingerprint:
    """Hash evidence for one expanded query row without retaining its vector.

    Attributes
    ----------
    expanded_row_index : int
        Position of this row in the ordered input to both search paths.
    original_query_index : int
        Position of the original query before embedding admission split it.
    query_text_sha256 : str
        SHA-256 of this row's query text.
    vector_dimension : int
        Number of finite components in the query vector.
    vector_sha256 : str
        SHA-256 of ordered vector components encoded as little-endian float64.
    """

    expanded_row_index: int
    original_query_index: int
    query_text_sha256: str
    vector_dimension: int
    vector_sha256: str


@dataclass(frozen=True)
class DenseSearchComparisonEvidence:
    """Describe the shared inputs and immutable index used by a paired search.

    A complete capture covers at most 256 expanded rows with vector dimensions
    from 1 through 65,536. It records hashes instead of query text or vectors.
    Diagnostic timings cannot qualify performance or promote a baseline.

    Attributes
    ----------
    schema_version : int
        Version of this evidence format.
    complete : bool
        Whether both search paths and final hit shaping completed.
    timing_qualification : bool
        Always false; diagnostic execution is ineligible for timing comparisons.
    query_text_sha256 : str
        SHA-256 of the original query text.
    table_uri : str
        Filesystem location of the compared LanceDB table.
    table_name : str
        Name of the compared table.
    table_version : int
        Immutable table version used by both paths.
    table_rows : int
        Number of rows in that table version.
    vector_column : str
        Name of the searched vector column.
    vector_dimension : int
        Number of components in each query vector.
    metric : str
        Verified index distance metric.
    index_name : str
        Name of the vector index used by indexed search.
    index_type : str
        Observed index family.
    indexed_rows : int
        Number of rows covered by the vector index.
    unindexed_rows : int
        Number of rows outside the vector index.
    embedding_model_name : str
        Verified embedding model identity.
    embedding_model_revision : str or None
        Verified model revision when available.
    candidate_k : int
        Candidate limit applied to each expanded query row and search path.
    indexed_nprobes : int
        Requested probe count for indexed search.
    indexed_refine_factor : int
        Requested refinement factor for indexed search.
    predicate_sha256 : str or None
        SHA-256 of the shared row predicate, if configured.
    top_k : int
        Final hit limit after shaping each ranking.
    page_dedup : bool
        Whether final rankings deduplicate document pages.
    content_types : tuple of str or None
        Normalized final content-type allowlist, if configured.
    vector_encoding : str
        Encoding used for vector fingerprints.
    expanded_row_count : int
        Number of rows after query embedding admission.
    rows : tuple of DenseSearchRowFingerprint
        Ordered fingerprints mapping expanded rows to original queries.
    indexed_input_sha256 : str
        Fingerprint of indexed search inputs.
    exact_input_sha256 : str
        Fingerprint of exhaustive search inputs.
    shared_input_sha256 : str
        Fingerprint covering the common query, table, and search configuration.
    indexed_ef : int or None
        Explicit HNSW search budget. None requests the native default and does
        not identify its effective value.
    """

    schema_version: int
    complete: bool
    timing_qualification: bool
    query_text_sha256: str
    table_uri: str
    table_name: str
    table_version: int
    table_rows: int
    vector_column: str
    vector_dimension: int
    metric: str
    index_name: str
    index_type: str
    indexed_rows: int
    unindexed_rows: int
    embedding_model_name: str
    embedding_model_revision: str | None
    candidate_k: int
    indexed_nprobes: int
    indexed_refine_factor: int
    predicate_sha256: str | None
    top_k: int
    page_dedup: bool
    content_types: tuple[str, ...] | None
    vector_encoding: str
    expanded_row_count: int
    rows: tuple[DenseSearchRowFingerprint, ...]
    indexed_input_sha256: str
    exact_input_sha256: str
    shared_input_sha256: str
    # Explicit indexed builder budget; None requests the native default, not a resolved value.
    indexed_ef: int | None = None


@dataclass(frozen=True)
class DenseSearchComparisonResult:
    """Return two shaped rankings and evidence of their identical search input.

    Attributes
    ----------
    indexed_hits : list of RetrievalHit
        Ranking produced by the configured vector index.
    exact_hits : list of RetrievalHit
        Ranking produced by exhaustive search on the same table version.
    evidence : DenseSearchComparisonEvidence
        Input and index fingerprints; diagnostic timings cannot qualify
        performance, and an omitted indexed_ef does not resolve the native budget.
    """

    indexed_hits: list[RetrievalHit]
    exact_hits: list[RetrievalHit]
    evidence: DenseSearchComparisonEvidence


@dataclass
class _DenseSearchCapture:
    query_text: str
    expected_model_name: str | None = None
    expected_model_revision: str | None = None
    exact_hits: list[list[dict[str, Any]]] | None = None
    evidence: DenseSearchComparisonEvidence | None = None
    calls: int = 0
    backend_evidence: dict[str, Any] = field(default_factory=dict)
