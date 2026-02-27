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
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import RunMode
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


class Ingestor:
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

    def files(self, documents: Union[str, List[str]]) -> "Ingestor":
        """Add document paths/URIs for processing."""
        self._not_implemented("files")

    def buffers(self, buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]]) -> "Ingestor":
        """Add in-memory buffers for processing."""
        self._not_implemented("buffers")

    def load(self) -> "Ingestor":
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

    def all_tasks(self) -> "Ingestor":
        """Record the default task chain (placeholder)."""
        self._not_implemented("all_tasks")

    def dedup(self) -> "Ingestor":
        """Record a dedup task configuration."""
        self._not_implemented("dedup")

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "Ingestor":
        """Record an embedding task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("embed")

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "Ingestor":
        """Record an extract task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("extract")

    def filter(self) -> "Ingestor":
        """Record a filter task configuration."""
        self._not_implemented("filter")

    def split(self) -> "Ingestor":
        """Record a split task configuration."""
        self._not_implemented("split")

    def store(self) -> "Ingestor":
        """Record a store task configuration."""
        self._not_implemented("store")

    def store_embed(self) -> "Ingestor":
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
    ) -> "Ingestor":
        """Record a UDF task configuration."""
        self._not_implemented("udf")

    def vdb_upload(self, params: VdbUploadParams | None = None, **kwargs: Any) -> "Ingestor":
        """Record a vector DB upload configuration (execution TBD)."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("vdb_upload")

    def save_intermediate_results(self, output_dir: str) -> "Ingestor":
        """Record intermediate results persistence configuration."""
        self._not_implemented("save_intermediate_results")

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "Ingestor":
        """Record result persistence configuration (execution TBD)."""
        self._not_implemented("save_to_disk")

    def caption(self) -> "Ingestor":
        """Record a caption task configuration."""
        self._not_implemented("caption")

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "Ingestor":
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
