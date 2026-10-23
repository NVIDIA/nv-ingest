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
    expanded_row_index: int
    original_query_index: int
    query_text_sha256: str
    vector_dimension: int
    vector_sha256: str


@dataclass(frozen=True)
class DenseSearchComparisonEvidence:
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
