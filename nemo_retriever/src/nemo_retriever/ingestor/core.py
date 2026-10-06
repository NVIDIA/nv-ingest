# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
System-facing ingestion interface.

This module defines the public API surface for building and executing ingestion.
Concrete implementations are provided by runmodes:

- inprocess: local Python process, no framework assumptions
- batch: large-scale batch execution
- service: remote ingestion service
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Self, Sequence, Tuple, Union

from nemo_retriever.common.params import CaptionParams
from nemo_retriever.common.params import DedupParams
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params import IngestExecuteParams
from nemo_retriever.common.params import IngestorCreateParams
from nemo_retriever.common.params import IngestorRunMode
from nemo_retriever.common.params import StoreParams
from nemo_retriever.common.params import UrlFetchParams
from nemo_retriever.common.params import VdbUploadParams
from nemo_retriever.common.params import WebhookParams


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
    run_mode: IngestorRunMode = "inprocess",
    params: IngestorCreateParams | None = None,
    **kwargs: Any,
) -> "Ingestor":
    """
    Graph-only ingestion factory.
    """
    merged = _merge_params(params, kwargs)
    if isinstance(merged, IngestorCreateParams):
        parsed = merged
    else:
        parsed = IngestorCreateParams(**merged)

    if run_mode == "service":
        from nemo_retriever.service.service_ingestor import ServiceIngestor

        service_kwargs: dict[str, Any] = {
            "base_url": parsed.base_url,
            "documents": parsed.documents,
            "api_token": parsed.api_key,
        }
        if parsed.max_concurrency is not None:
            service_kwargs["max_concurrency"] = parsed.max_concurrency
        return ServiceIngestor(**service_kwargs)

    if run_mode not in {"batch", "inprocess"}:
        raise ValueError(f"create_ingestor supports run modes 'inprocess', 'batch', and 'service'; got {run_mode!r}.")

    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    return GraphIngestor(
        run_mode=run_mode,
        documents=parsed.documents,
        ray_address=parsed.ray_address,
        ray_log_to_driver=parsed.ray_log_to_driver,
        debug=parsed.debug,
        allow_no_gpu=parsed.allow_no_gpu,
        node_overrides=parsed.node_overrides,
        error_policy=parsed.error_policy,
    )


class ingestor:
    """
    Interface base class. All methods intentionally raise NotImplementedError.

    Each runmode should subclass this and eventually provide working behavior.
    """

    RUN_MODE: str = "interface"

    def __init__(self, documents: Optional[List[str]] = None) -> None:
        self._documents: List[str] = list(documents or [])
        self._urls: List[str] = []
        self._url_fetch_params = UrlFetchParams()
        self._buffers: List[Tuple[str, BytesIO]] = []

    def _not_implemented(self, method_name: str) -> "None":
        raise NotImplementedError(
            f"{self.__class__.__name__}.{method_name}() is not implemented yet " f"(run_mode={self.RUN_MODE})."
        )

    def _validate_input_sources(self, inline_texts: Sequence[str] | None) -> None:
        if self._documents or self._buffers or self._urls or inline_texts:
            return
        raise ValueError(
            "No input sources configured. Call files(), urls(), texts(), or buffers() "
            "with at least one source before ingest()."
        )

    def urls(self, urls: Union[str, Sequence[str]], params: UrlFetchParams | None = None, **kwargs: Any) -> "ingestor":
        """Add HTTP(S) URL sources for processing.

        Parameters
        ----------
        urls
            One absolute HTTP(S) URL or a sequence of URLs.
        params
            Shared URL-fetch configuration.
        **kwargs
            Field overrides for :class:`UrlFetchParams`.

        Returns
        -------
        ingestor
            This ingestor for fluent chaining.

        Raises
        ------
        ValueError
            If URL validation or fetch configuration validation fails.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("urls")

    def files(self, documents: Union[str, List[str]]) -> "ingestor":
        """Add document paths/URIs for processing."""
        self._not_implemented("files")

    def texts(self, texts: Union[str, Sequence[str]]) -> Self:
        """Set raw inline text documents for processing."""
        self._not_implemented("texts")

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
        """Execute the configured ingestion pipeline (placeholder).

        In ``run_mode='service'``, ``return_results`` (default ``True``)
        controls whether completed rows are fetched into
        ``ServiceIngestResult.dataframe``.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("ingest")

    def ingest_async(self, *, return_failures: bool = False, return_traces: bool = False) -> Any:
        """Asynchronously execute ingestion (placeholder)."""
        self._not_implemented("ingest_async")

    def all_tasks(self) -> "ingestor":
        """Record the default task chain (placeholder)."""
        self._not_implemented("all_tasks")

    def dedup(self, params: DedupParams | None = None, **kwargs: Any) -> "ingestor":
        """Record image-deduplication settings.

        Passing ``content_hash=False`` and ``bbox_iou=False`` explicitly
        suppresses any automatic deduplication associated with captioning.

        Parameters
        ----------
        params
            Optional image-deduplication parameters.
        **kwargs
            Field overrides merged into ``params`` by concrete implementations.

        Returns
        -------
        ingestor
            The concrete ingestor instance for fluent chaining.

        Raises
        ------
        NotImplementedError
            Always raised by this interface implementation. Concrete run-mode
            implementations override this method.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("dedup")

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "ingestor":
        """Record an embedding task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("embed")

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "ingestor":
        """Record an extract task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("extract")

    def extract_image_files(self, params: ExtractParams | None = None, **kwargs: Any) -> "ingestor":
        """Record an extract-image-files task configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("extract_image_files")

    def filter(self) -> "ingestor":
        """Record a filter task configuration."""
        self._not_implemented("filter")

    def store(self, params: StoreParams | None = None, **kwargs: Any) -> "ingestor":
        """Record a store task configuration for extracted image assets."""
        _ = _merge_params(params, kwargs)
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

    def caption(self, params: "CaptionParams | None" = None, **kwargs: Any) -> "ingestor":
        """Record caption settings.

        Captioning non-image documents automatically enables default image
        deduplication unless it is explicitly disabled through :meth:`dedup`.

        Parameters
        ----------
        params
            Optional image-captioning parameters.
        **kwargs
            Field overrides merged into ``params`` by concrete implementations.

        Returns
        -------
        ingestor
            The concrete ingestor instance for fluent chaining.

        Raises
        ------
        NotImplementedError
            Always raised by this interface implementation. Concrete run-mode
            implementations override this method.
        """
        _ = _merge_params(params, kwargs)
        self._not_implemented("caption")

    def webhook(self, params: "WebhookParams | None" = None, **kwargs: Any) -> "ingestor":
        """Record a webhook notification configuration."""
        _ = _merge_params(params, kwargs)
        self._not_implemented("webhook")

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


# Backward compatibility alias.
Ingestor = ingestor
