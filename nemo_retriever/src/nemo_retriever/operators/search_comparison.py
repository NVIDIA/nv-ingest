# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired dense retrieval after one normal query embedding/admission stage."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.lancedb_comparison import compare_dense_search
from nemo_retriever.common.vdb.records import normalize_retrieval_results
from nemo_retriever.graph.retriever_utils import filter_retrieval_kwargs
from nemo_retriever.operators.vdb import RetrieveVdbOperator
from nemo_retriever.query.comparison import (
    DenseSearchComparisonEvidence,
    DenseSearchRowFingerprint,
    _DenseSearchCapture,
    _digest,
    _yaml_digest,
)
from nemo_retriever.query.expansion import QUERY_INDEX_COLUMN, regroup_query_hits


class _CompareDenseSearchOperator(RetrieveVdbOperator):
    def __init__(self, *, capture: _DenseSearchCapture, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(self._vdb, LanceDB):
            raise ValueError("Exact-search comparison requires the LanceDB backend.")
        self._capture = capture

    def run(self, data: Any, **kwargs: Any) -> Any:
        if self._capture.calls:
            raise RuntimeError("Exact-search comparison expected exactly one embedded query batch.")
        if not isinstance(data, pd.DataFrame) or self._query_text_column is None:
            raise ValueError("Exact-search comparison requires the normal embedded query DataFrame.")
        # The inherited run performs normal extraction, indexed regrouping and
        # validation. process() pairs both searches from that exact vector list.
        indexed = super().run(data, **kwargs)
        self._capture.calls += 1
        snapshot = dict(self._capture.backend_evidence)
        vector_hashes = snapshot.pop("vector_hashes")
        indexed_vector_fingerprint = snapshot.pop("indexed_vector_fingerprint")
        exact_vector_fingerprint = snapshot.pop("exact_vector_fingerprint")
        if indexed_vector_fingerprint != exact_vector_fingerprint:
            raise RuntimeError("Indexed/exact search vector fingerprints differ.")
        rows = tuple(
            DenseSearchRowFingerprint(
                expanded_row_index=ordinal,
                original_query_index=int(index),
                query_text_sha256=_digest(str(text)),
                vector_dimension=snapshot["vector_dimension"],
                vector_sha256=vector_hash,
            )
            for ordinal, (index, text, vector_hash) in enumerate(
                zip(data[QUERY_INDEX_COLUMN], data[self._query_text_column], vector_hashes, strict=True)
            )
        )
        exact = self._capture.exact_hits
        if exact is None:
            raise RuntimeError("Exact-search comparison did not retain the paired hit groups.")
        self._capture.exact_hits = regroup_query_hits(
            exact,
            data[QUERY_INDEX_COLUMN].tolist(),
            query_count=len(kwargs.get("query_texts") or []),
            top_k=int(kwargs.get("top_k", 10)),
        )
        shared = _yaml_digest(dict(rows=[asdict(row) for row in rows], table=snapshot))
        self._capture.evidence = DenseSearchComparisonEvidence(
            schema_version=1,
            complete=False,
            timing_qualification=False,
            query_text_sha256=_digest(self._capture.query_text),
            **snapshot,
            top_k=0,
            page_dedup=False,
            content_types=None,
            vector_encoding="python-float-ieee754-float64-le/v1; metadata=nrl-dense-search-yaml/v1",
            expanded_row_count=len(rows),
            rows=rows,
            indexed_input_sha256=shared,
            exact_input_sha256=shared,
            shared_input_sha256=shared,
        )
        return indexed

    def process(self, data: Any, **kwargs: Any) -> Any:
        options = {**self._retrieval_vdb_kwargs, **filter_retrieval_kwargs(kwargs)}
        indexed, exact, evidence = compare_dense_search(
            self._vdb,
            data,
            expected_model_name=self._capture.expected_model_name,
            expected_model_revision=self._capture.expected_model_revision,
            **options,
        )
        self._capture.exact_hits = normalize_retrieval_results(exact)
        self._capture.backend_evidence = evidence
        return normalize_retrieval_results(indexed)
