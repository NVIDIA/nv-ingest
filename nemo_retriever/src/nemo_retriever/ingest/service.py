# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from nemo_retriever.ingest.plan import (
    IngestInputTypeValue,
    IngestProfileValue,
    build_caption_params,
    build_dedup_params,
    build_store_params,
    build_text_chunk_kwargs,
    expand_ingest_documents,
    profile_extract_defaults,
    validate_ingest_input_type,
    validate_ingest_profile,
)
from nemo_retriever.common.params import (
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    StoreParams,
    TextChunkParams,
)
from nemo_retriever.common.params.models import NO_API_KEY
from nemo_retriever.common.input_files import expand_input_file_patterns, input_type_for_path
from nemo_retriever.ingestor.plans import resolve_effective_dedup_params

logger = logging.getLogger(__name__)

ServiceIngestInputTypeValue = IngestInputTypeValue
ServiceIngestProfileValue = IngestProfileValue


@dataclass(frozen=True)
class ServiceIngestSourceOptions:
    documents: Sequence[str]
    profile: ServiceIngestProfileValue = "auto"
    input_type: ServiceIngestInputTypeValue = "auto"


@dataclass(frozen=True)
class ServiceIngestConnectionOptions:
    service_url: str = "http://localhost:7670"
    service_concurrency: int = 8
    service_api_token: str | None = None


@dataclass(frozen=True)
class ServiceIngestExtractOptions:
    method: str | None = None
    dpi: int | None = None
    extract_text: bool | None = None
    extract_images: bool | None = None
    extract_nested_images: bool | None = None
    extract_tables: bool | None = None
    extract_charts: bool | None = None
    extract_infographics: bool | None = None
    extract_page_as_image: bool | None = None
    use_page_elements: bool | None = None
    use_table_structure: bool | None = None
    table_output_format: str | None = None
    ocr_version: str | None = None


@dataclass(frozen=True)
class ServiceIngestDedupOptions:
    enabled: bool | None = None
    iou_threshold: float | None = None


@dataclass(frozen=True)
class ServiceIngestCaptionOptions:
    enabled: bool = False
    context_text_max_chars: int | None = None
    caption_infographics: bool | None = None


@dataclass(frozen=True)
class ServiceIngestChunkOptions:
    enabled: bool = False
    text_chunk_max_tokens: int | None = None
    text_chunk_overlap_tokens: int | None = None


@dataclass(frozen=True)
class ServiceIngestEmbedOptions:
    embed_modality: str | None = None
    text_elements_modality: str | None = None
    structured_elements_modality: str | None = None
    embed_granularity: str | None = None


@dataclass(frozen=True)
class ServiceIngestImageStoreOptions:
    images_uri: str | None = None


@dataclass(frozen=True)
class ServiceIngestPlanRequest:
    source: ServiceIngestSourceOptions
    connection: ServiceIngestConnectionOptions = field(default_factory=ServiceIngestConnectionOptions)
    extract: ServiceIngestExtractOptions = field(default_factory=ServiceIngestExtractOptions)
    dedup: ServiceIngestDedupOptions = field(default_factory=ServiceIngestDedupOptions)
    caption: ServiceIngestCaptionOptions = field(default_factory=ServiceIngestCaptionOptions)
    chunk: ServiceIngestChunkOptions = field(default_factory=ServiceIngestChunkOptions)
    embed: ServiceIngestEmbedOptions = field(default_factory=ServiceIngestEmbedOptions)
    image_store: ServiceIngestImageStoreOptions = field(default_factory=ServiceIngestImageStoreOptions)


@dataclass(frozen=True)
class ServiceIngestRequest:
    documents: Sequence[str]
    input_type: ServiceIngestInputTypeValue
    extract_params: ExtractParams = field(default_factory=ExtractParams)
    embed_params: EmbedParams = field(default_factory=EmbedParams)
    text_chunk_params: TextChunkParams = field(default_factory=TextChunkParams)
    enable_text_chunk: bool = False
    dedup_params: DedupParams | None = None
    caption_params: CaptionParams | None = None
    store_params: StoreParams | None = None
    connection: ServiceIngestConnectionOptions = field(default_factory=ServiceIngestConnectionOptions)


@dataclass(frozen=True)
class ServiceIngestExecutionResult:
    """Structured result from executing a resolved service ingest request.

    Service mode does not locally verify the remote vector database after
    ingest. ``result_n_rows`` sums the row counts reported by successful
    document-completion events, and ``n_rows`` mirrors that value so root CLI
    summaries keep the same top-level row-count contract as local ingest
    results without downloading retained result payloads.
    """

    request: ServiceIngestRequest
    result: object
    n_rows: int | None
    result_n_rows: int | None
    metadata: dict[str, Any]

    @property
    def documents(self) -> list[str]:
        return list(self.request.documents)

    @property
    def service_url(self) -> str:
        return self.request.connection.service_url

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "run_mode": "service",
            "documents": self.documents,
            "service_url": self.service_url,
            "result": self.result,
            "n_rows": self.n_rows,
            "result_n_rows": self.result_n_rows,
        }


def resolve_service_ingest_request(request: ServiceIngestPlanRequest) -> ServiceIngestRequest:
    """Resolve first-class root service ingest options into service params."""

    source = request.source
    input_type = validate_ingest_input_type(source.input_type)
    profile = validate_ingest_profile(source.profile)
    documents = expand_ingest_documents(source.documents, input_type=input_type)
    _validate_service_profile_documents(profile, documents)

    extract_kwargs = profile_extract_defaults(profile)
    extract_kwargs.update(
        {
            key: value
            for key, value in {
                "method": request.extract.method,
                "dpi": request.extract.dpi,
                "extract_text": request.extract.extract_text,
                "extract_images": request.extract.extract_images,
                "extract_nested_images": request.extract.extract_nested_images,
                "extract_tables": request.extract.extract_tables,
                "extract_charts": request.extract.extract_charts,
                "extract_infographics": request.extract.extract_infographics,
                "extract_page_as_image": request.extract.extract_page_as_image,
                "use_page_elements": request.extract.use_page_elements,
                "use_table_structure": request.extract.use_table_structure,
                "table_output_format": request.extract.table_output_format,
                "ocr_version": request.extract.ocr_version,
            }.items()
            if value is not None
        }
    )
    if request.extract.table_output_format == "markdown":
        extract_kwargs["use_table_structure"] = True

    embed_kwargs = {
        key: value
        for key, value in {
            "embed_modality": request.embed.embed_modality,
            "text_elements_modality": request.embed.text_elements_modality,
            "structured_elements_modality": request.embed.structured_elements_modality,
            "embed_granularity": request.embed.embed_granularity,
        }.items()
        if value is not None
    }
    enable_text_chunk, text_chunk_kwargs = build_text_chunk_kwargs(
        enabled=request.chunk.enabled,
        text_chunk_max_tokens=request.chunk.text_chunk_max_tokens,
        text_chunk_overlap_tokens=request.chunk.text_chunk_overlap_tokens,
    )
    text_chunk_params = TextChunkParams(**text_chunk_kwargs) if enable_text_chunk else TextChunkParams()

    return ServiceIngestRequest(
        documents=documents,
        input_type=input_type,
        extract_params=ExtractParams(**extract_kwargs),
        embed_params=EmbedParams(**embed_kwargs) if embed_kwargs else EmbedParams(),
        text_chunk_params=text_chunk_params,
        enable_text_chunk=enable_text_chunk,
        dedup_params=build_dedup_params(enabled=request.dedup.enabled, iou_threshold=request.dedup.iou_threshold),
        caption_params=build_caption_params(
            enabled=request.caption.enabled,
            caption_context_text_max_chars=request.caption.context_text_max_chars,
            caption_infographics=request.caption.caption_infographics,
        ),
        store_params=build_store_params(images_uri=request.image_store.images_uri),
        connection=request.connection,
    )


def build_service_ingestor(request: ServiceIngestRequest) -> Any:
    """Construct a remote-service ingestor with service-compatible stages."""

    from nemo_retriever.service.service_ingestor import ServiceIngestor

    resolved_files = expand_service_file_patterns(request.documents)
    if not resolved_files:
        raise ValueError("No files matched the input patterns for service mode.")

    ingestor = ServiceIngestor(
        base_url=request.connection.service_url,
        max_concurrency=request.connection.service_concurrency,
        api_token=request.connection.service_api_token,
    ).files(resolved_files)

    ingestor = _attach_service_extract_stage(
        ingestor,
        input_type=request.input_type,
        documents=resolved_files,
        extract_params=request.extract_params,
        enable_text_chunk=request.enable_text_chunk,
        text_chunk_params=request.text_chunk_params,
    )

    if request.dedup_params is not None:
        ingestor = ingestor.dedup(request.dedup_params)

    if request.caption_params is not None:
        ingestor = ingestor.caption(_sanitize_service_caption_params(request.caption_params))

    ingestor = ingestor.embed(_sanitize_service_stage_params(request.embed_params, stage="embed"))

    if request.store_params is not None:
        ingestor = ingestor.store(request.store_params)

    return ingestor


def execute_service_ingest_request(
    request: ServiceIngestRequest,
    *,
    return_results: bool | None = False,
) -> ServiceIngestExecutionResult:
    """Execute a service ingest request and return its structured result.

    ``return_results`` defaults to the user-facing service client behavior.
    Result rows are downloaded only when explicitly requested. Completion
    metadata and event row counts remain available without retained rows.
    """

    result = build_service_ingestor(request).ingest(return_results=return_results)
    failures = list(getattr(result, "failures", ()) or ())
    if failures:
        document, detail = failures[0]
        raise RuntimeError(
            f"Service ingest failed for {len(failures)} document(s); first failure: {document}: {detail}"
        )
    result_n_rows = _count_service_result_rows(result)
    return ServiceIngestExecutionResult(
        request=request,
        result=result,
        n_rows=result_n_rows,
        result_n_rows=result_n_rows,
        metadata={
            "service_url": request.connection.service_url,
            "input_type": request.input_type,
        },
    )


def expand_service_file_patterns(documents: Sequence[str]) -> list[str]:
    """Expand recursive file patterns for service ingest construction."""

    resolved_files: list[str] = []
    for pattern in documents:
        resolved_files.extend(sorted(_glob.glob(str(pattern), recursive=True)))
    return resolved_files


def _service_input_types_for_documents(documents: Sequence[str]) -> set[str]:
    return {
        family
        for document in expand_input_file_patterns(documents)
        for family in [input_type_for_path(document)]
        if family is not None
    }


def service_split_config_for_request(request: ServiceIngestRequest) -> dict[str, Any] | None:
    """Build the service split configuration for a resolved ingest request."""

    chunk_dict = _service_text_chunk_dict(request.text_chunk_params) if request.enable_text_chunk else None
    return _split_config_for_input_type(request.input_type, chunk_dict, documents=request.documents)


def resolve_service_dedup_for_request(
    request: ServiceIngestRequest,
) -> tuple[DedupParams | None, dict[str, Any] | None]:
    """Resolve effective service dedup parameters and mixed-input scope.

    Parameters
    ----------
    request
        Resolved service-ingest request. When deduplication is resolved
        automatically for ``input_type="auto"``, document paths are inspected
        to distinguish standalone images from other inputs.

    Returns
    -------
    tuple[DedupParams | None, dict[str, Any] | None]
        Effective deduplication parameters and an optional mixed-input scope.
        An automatically configured mixed image/document request leaves the
        global parameters unset and reports which input families use
        caption-triggered deduplication.

    Raises
    ------
    FileNotFoundError
        If automatic resolution encounters an auto-typed literal document
        path that does not exist.
    IsADirectoryError
        If automatic resolution encounters an auto-typed literal document
        path that is a directory.
    """

    if request.dedup_params is not None or request.caption_params is None:
        return request.dedup_params, None

    families = (
        _service_input_types_for_documents(request.documents) if request.input_type == "auto" else {request.input_type}
    )
    enabled_families = sorted(family for family in families if family != "image")
    dedup_params = resolve_effective_dedup_params(
        None,
        caption_enabled=True,
        image_only=not enabled_families,
    )
    if dedup_params is not None and "image" in families:
        return None, {
            "mode": "caption_default",
            "enabled_families": enabled_families,
            "exempt_families": ["image"],
        }
    return dedup_params, None


def _validate_service_profile_documents(profile: ServiceIngestProfileValue, documents: Sequence[str]) -> None:
    if profile != "fast-text":
        return
    disallowed = sorted(
        {
            family
            for document in documents
            if not _glob.has_magic(str(document))
            for family in [input_type_for_path(document)]
            if family not in {"pdf", "doc"}
        }
    )
    if disallowed:
        observed = ", ".join(disallowed)
        raise ValueError(f"--profile {profile} only supports PDF/document inputs; observed {observed}.")


def _service_extraction_mode(input_type: str) -> str:
    """Map ingest input type to :class:`PipelineSpec` ``extraction_mode``."""

    return {
        "pdf": "pdf",
        "doc": "pdf",
        "txt": "text",
        "html": "html",
        "audio": "audio",
        "video": "auto",
    }.get(input_type, "auto")


def _service_text_chunk_dict(text_chunk_params: TextChunkParams) -> dict[str, Any]:
    """Serialize text-chunk knobs allowed by the service split_config policy."""

    from nemo_retriever.common.policy import _DEFAULT_ALLOWED_SPLIT_KEYS

    raw = text_chunk_params.model_dump(exclude_none=True)
    return {key: value for key, value in raw.items() if key in _DEFAULT_ALLOWED_SPLIT_KEYS}


def _attach_service_extract_stage(
    ingestor: Any,
    *,
    input_type: str,
    documents: Sequence[str],
    extract_params: ExtractParams,
    enable_text_chunk: bool,
    text_chunk_params: TextChunkParams,
) -> Any:
    """Wire the extraction stage for the remote service ingestor."""

    chunk_dict = _service_text_chunk_dict(text_chunk_params) if enable_text_chunk else None
    service_extract_params = _sanitize_service_stage_params(extract_params, stage="extract")
    if input_type == "image":
        return ingestor.extract_image_files(
            service_extract_params,
            split_config={"image": chunk_dict} if chunk_dict else None,
        )
    return ingestor.extract(
        service_extract_params,
        split_config=_split_config_for_input_type(input_type, chunk_dict, documents=documents),
        extraction_mode=_service_extraction_mode(input_type),
    )


def _sanitize_service_stage_params(params: Any, *, stage: str) -> ExtractParams | EmbedParams:
    """Keep typed client overrides while preventing environment API keys from reaching the wire client."""

    from nemo_retriever.common.policy import _DEFAULT_ALLOWED_EMBED_KEYS, _DEFAULT_ALLOWED_EXTRACT_KEYS

    allowed = {
        "extract": _DEFAULT_ALLOWED_EXTRACT_KEYS,
        "embed": _DEFAULT_ALLOWED_EMBED_KEYS,
    }[stage]
    raw = params.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sanitized = {key: value for key, value in raw.items() if key in allowed}
    if stage == "extract":
        return ExtractParams(
            **sanitized,
            api_key=NO_API_KEY,
            page_elements_api_key=NO_API_KEY,
            ocr_api_key=NO_API_KEY,
        )
    return EmbedParams(**sanitized, api_key=NO_API_KEY)


def _split_config_for_input_type(
    input_type: str,
    chunk_dict: dict[str, Any] | None,
    *,
    documents: Sequence[str] = (),
) -> dict[str, Any] | None:
    if chunk_dict is None:
        return None
    if input_type == "auto":
        return _split_config_for_auto_documents(documents, chunk_dict)
    if input_type in {"pdf", "doc"}:
        return {"pdf": chunk_dict}
    if input_type == "txt":
        return {"text": chunk_dict}
    if input_type == "html":
        return {"html": chunk_dict}
    if input_type == "image":
        return {"image": chunk_dict}
    if input_type == "audio":
        return {"audio": chunk_dict}
    if input_type == "video":
        return {"video": chunk_dict, "audio": chunk_dict}
    return None


def _split_config_for_auto_documents(
    documents: Sequence[str],
    chunk_dict: dict[str, Any],
) -> dict[str, Any] | None:
    input_types = _service_input_types_for_documents(documents)
    split_config: dict[str, Any] = {}
    if input_types & {"pdf", "doc"}:
        split_config["pdf"] = dict(chunk_dict)
    if "txt" in input_types:
        split_config["text"] = dict(chunk_dict)
    if "html" in input_types:
        split_config["html"] = dict(chunk_dict)
    if "image" in input_types:
        split_config["image"] = dict(chunk_dict)
    if "audio" in input_types:
        split_config["audio"] = dict(chunk_dict)
    if "video" in input_types:
        split_config["video"] = dict(chunk_dict)
        split_config["audio"] = dict(chunk_dict)
    return split_config or None


def _sanitize_service_caption_params(caption_params: CaptionParams) -> CaptionParams:
    params_dict = caption_params.model_dump(exclude_none=True, exclude_unset=True)
    return CaptionParams(
        **{
            key: value
            for key, value in params_dict.items()
            if key
            in {
                "prompt",
                "system_prompt",
                "batch_size",
                "context_text_max_chars",
                "caption_infographics",
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
            }
        }
    )


def _count_service_result_rows(result: object) -> int | None:
    dataframe = getattr(result, "dataframe", None)
    if dataframe is not None:
        try:
            return len(dataframe)
        except TypeError:
            return None

    try:
        events = iter(result)
    except TypeError:
        return None

    total = 0
    for event in events:
        if not isinstance(event, dict) or event.get("status") != "completed":
            continue
        result_rows = event.get("result_rows", 0)
        if result_rows is None:
            continue
        try:
            total += int(result_rows)
        except (TypeError, ValueError):
            return None
    return total
