# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
System-facing ingestion interface.

This module defines the public API surface for building and executing ingestion.
Concrete implementations are provided by runmodes:

- inprocess: local Python process, no framework assumptions
- batch: large-scale batch execution
- fused: low-latency single-actor GPU model fusion
- online: low-latency, multi-request serving
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from nemo_retriever.application.modes.factory import create_runmode_ingestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import StructuredFetchParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode
from nemo_retriever.params import StructuredDescriptionParams
from nemo_retriever.params import StructuredExtractParams
from nemo_retriever.params import StructuredPIIParams
from nemo_retriever.params import StructuredSemanticLayerParams
from nemo_retriever.params import StructuredUsageWeightsParams
from nemo_retriever.params import VdbUploadParams


def _merge_params[T](params: T | None, kwargs: dict[str, Any]) -> T:
    if params is None:
        return kwargs  # type: ignore[return-value]
    if not kwargs:
        return params
    if hasattr(params, "model_copy"):
        return params.model_copy(update=kwargs)  # type: ignore[return-value]
    return params


def create_ingestor(
    *,
    run_mode: RunMode = "inprocess",
    params: IngestorCreateParams | None = None,
    **kwargs: Any,
) -> "Ingestor":
    """
    Factory for selecting an ingestion runmode implementation.
    """
    merged = _merge_params(params, kwargs)
    if isinstance(merged, IngestorCreateParams):
        parsed = merged
    else:
        parsed = IngestorCreateParams(**merged)
    return create_runmode_ingestor(run_mode=run_mode, params=parsed)


class ingestor:
    """
    Interface base class. All methods intentionally raise NotImplementedError.

    Each runmode should subclass this and eventually provide working behavior.
    """

    RUN_MODE: str = "interface"

    def __init__(self, documents: Optional[List[str]] = None) -> None:
        self._documents: List[str] = list(documents or [])
        self._buffers: List[Tuple[str, BytesIO]] = []

    def _not_implemented(self, method_name: str) -> "None":
        raise NotImplementedError(
            f"{self.__class__.__name__}.{method_name}() is not implemented yet " f"(run_mode={self.RUN_MODE})."
        )

    def files(self, documents: Union[str, List[str]]) -> "ingestor":
        """Add document paths/URIs for processing."""
        self._not_implemented("files")

    def buffers(self, buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]]) -> "ingestor":
        """Add in-memory buffers for processing."""
        self._not_implemented("buffers")

    def load(self) -> "ingestor":
        """
        Placeholder for remote fetch/localization.

        The client-side Ingestor supports downloading remote URIs locally.
        In this system, each runmode may handle remote inputs differently.
        """
        self._not_implemented("load")

    def ingest(
        self,
        params: IngestExecuteParams | None = None,
        **kwargs: Any,
    ) -> Union[List[Any], Tuple[Any, ...]]:
        """
        Execute the configured ingestion pipeline (placeholder).
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("ingest")

    def ingest_async(self, *, return_failures: bool = False, return_traces: bool = False) -> Any:
        """Asynchronously execute ingestion (placeholder)."""
        self._not_implemented("ingest_async")

    def all_tasks(self) -> "ingestor":
        """Record the default task chain (placeholder)."""
        self._not_implemented("all_tasks")

    def dedup(self) -> "ingestor":
        """Record a dedup task configuration."""
        self._not_implemented("dedup")

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "ingestor":
        """Record an embedding task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("embed")

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "ingestor":
        """Record an extract task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("extract")

    def filter(self) -> "ingestor":
        """Record a filter task configuration."""
        self._not_implemented("filter")

    def split(self) -> "ingestor":
        """Record a split task configuration."""
        self._not_implemented("split")

    def store(self) -> "ingestor":
        """Record a store task configuration."""
        self._not_implemented("store")

    def store_embed(self) -> "ingestor":
        """Record a store-embed task configuration."""
        self._not_implemented("store_embed")

    def udf(
        self,
        udf_function: str,
        udf_function_name: Optional[str] = None,
        phase: Optional[Union[int, str]] = None,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> "ingestor":
        """Record a UDF task configuration."""
        self._not_implemented("udf")

    def vdb_upload(self, params: VdbUploadParams | None = None, **kwargs: Any) -> "ingestor":
        """Record a vector DB upload configuration (execution TBD)."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("vdb_upload")

    def save_intermediate_results(self, output_dir: str) -> "ingestor":
        """Record intermediate results persistence configuration."""
        self._not_implemented("save_intermediate_results")

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "ingestor":
        """Record result persistence configuration (execution TBD)."""
        self._not_implemented("save_to_disk")

    def caption(self) -> "ingestor":
        """Record a caption task configuration."""
        self._not_implemented("caption")

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "ingestor":
        """Record PDF split configuration (execution TBD)."""
        self._not_implemented("pdf_split_config")

    def completed_jobs(self) -> int:
        """Return completed job count (placeholder until backend populates job state)."""
        self._not_implemented("completed_jobs")

    def failed_jobs(self) -> int:
        """Return failed job count (placeholder until backend populates job state)."""
        self._not_implemented("failed_jobs")

    def cancelled_jobs(self) -> int:
        """Return cancelled job count (placeholder until backend populates job state)."""
        self._not_implemented("cancelled_jobs")

    def remaining_jobs(self) -> int:
        """Return remaining job count (placeholder until backend populates job state)."""
        self._not_implemented("remaining_jobs")

    def get_status(self) -> Dict[str, str]:
        """
        Return per-document status mapping (placeholder).

        Once Ray execution is wired, this should reflect actual job/task state.
        """
        self._not_implemented("get_status")

    # ------------------------------------------------------------------
    # Structured (database) ingestion — 8-step pipeline
    # ------------------------------------------------------------------

    def extract_structured(
        self,
        params: StructuredExtractParams | None = None,
        **kwargs: Any,
    ) -> "ingestor":
        """Step 1 — Reflect DB schema / parse SQL files → write graph nodes to Neo4j.

        Uses SQLAlchemy reflection and/or SQL file parsing to produce
        Database, Schema, Table, Column, View and Query nodes together with
        their relationships.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("extract_structured")

    def populate_structured_semantic_layer(
        self,
        params: StructuredSemanticLayerParams | None = None,
        **kwargs: Any,
    ) -> "ingestor":
        """Step 2 — Map global business terms/attributes to graph entities.

        Auto-creates Term/Attribute nodes and MAPS_TO_TABLE / MAPS_TO_COLUMN
        relationships for entities that are not already covered by the
        semantic-layer definition.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("populate_structured_semantic_layer")

    def detect_structured_pii(
        self,
        params: StructuredPIIParams | None = None,
        **kwargs: Any,
    ) -> "ingestor":
        """Step 3 — Tag Column nodes with PII type via regex and optional LLM.

        Writes a ``pii_type`` property and a HAS_PII_TYPE relationship onto
        each Column node that matches a known PII pattern.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("detect_structured_pii")

    def populate_structured_usage_weights(
        self,
        params: StructuredUsageWeightsParams | None = None,
        **kwargs: Any,
    ) -> "ingestor":
        """Step 4 — Derive usage weights from query log files.

        Parses SQL query logs, computes Table/Column co-occurrence frequencies,
        and writes ``usage_weight`` float properties back onto the graph nodes.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("populate_structured_usage_weights")

    def generate_structured_descriptions(
        self,
        params: StructuredDescriptionParams | None = None,
        **kwargs: Any,
    ) -> "ingestor":
        """Step 5 — LLM-generate natural-language descriptions for all node types.

        Descriptions are written back to Neo4j as a ``description`` property
        on Database, Schema, Table, Column, View and Query nodes.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("generate_structured_descriptions")

    def fetch_structured(
        self,
        params: StructuredFetchParams | None = None,
        **kwargs: Any,
    ) -> Any:
        """Step 6 — Fetch entity descriptions from Neo4j into a DataFrame.

        Builds a pandas DataFrame with columns ``text`` (the description),
        ``_embed_modality`` = ``"text"``, and ``metadata`` (JSON blob with
        entity_type, entity_name, node_id).  No embedding is computed here;
        the returned DataFrame is passed directly to the embed step.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("fetch_structured")

    def ingest_structured(
        self,
        params: StructuredExtractParams | None = None,
    ) -> Any:
        """Orchestrate the full 8-step structured ingestion pipeline.

        Runs the following steps in order:
        1. extract_structured
        2. populate_structured_semantic_layer
        3. detect_structured_pii
        4. populate_structured_usage_weights
        5. generate_structured_descriptions
        6. fetch_structured
        7. embed
        8. vdb_upload
        """
        self._not_implemented("ingest_structured")


# Backward compatibility alias.
Ingestor = ingestor
