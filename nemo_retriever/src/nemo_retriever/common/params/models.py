# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import warnings
from typing import Any, ClassVar, Literal, Optional, Tuple
from urllib.parse import urlparse


from upath import UPath

from nemo_retriever.tabular_data.sql_database import SQLDatabase
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)

from nemo_retriever.common.modality.caption.model_profiles import DEFAULT_LOCAL_CAPTION_MODEL_ID
from nemo_retriever.common.params.utils import (
    NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL,
    validate_nemotron_parse_endpoint_list,
)
from nemo_retriever.common.remote_auth import resolve_remote_api_key

IngestorRunMode = Literal["inprocess", "batch", "service"]

# Pass as an api_key value to suppress auto-resolution from environment variables.
# Example: EmbedParams(api_key=NO_API_KEY)
NO_API_KEY = ""


_REDACTED = "***"

ENVIRONMENT_REFERENCE_PREFIX = "os.environ/"
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROTECTED_LLM_REQUEST_KEYS = frozenset(
    {
        "model",
        "messages",
        "api_key",
        "api_base",
        "timeout",
        "num_retries",
        "temperature",
        "top_p",
        "max_tokens",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "stream",
        "n",
    }
)


def environment_reference_name(value: object) -> Optional[str]:
    """Return and validate the variable name in an ``os.environ/NAME`` reference."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith(ENVIRONMENT_REFERENCE_PREFIX):
        return None
    if stripped != value:
        raise ValueError("environment references must not contain surrounding whitespace")
    name = stripped.removeprefix(ENVIRONMENT_REFERENCE_PREFIX)
    if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(name):
        raise ValueError("environment references must use os.environ/VARIABLE_NAME with a valid variable name")
    return name


def resolve_environment_reference(value: Optional[str]) -> Optional[str]:
    """Resolve an explicit environment reference, leaving literal values unchanged."""
    name = environment_reference_name(value)
    if name is None:
        return value
    resolved = (os.environ.get(name) or "").strip()
    if not resolved:
        raise ValueError(f"required credential environment variable {name!r} is not set")
    return resolved


def validate_llm_extra_params(extra_params: Optional[dict[str, Any]], *, source: str) -> None:
    """Reject extension parameters that would replace protected request state."""
    if extra_params is None:
        return
    if not isinstance(extra_params, dict):
        raise TypeError(f"{source} must be a dictionary")
    protected = sorted(PROTECTED_LLM_REQUEST_KEYS.intersection(extra_params))
    if protected:
        raise ValueError(f"{source} may not override protected request fields: {', '.join(protected)}")


def _is_api_key_field(field_name: str) -> bool:
    """Return True when ``field_name`` should be masked in ``repr`` / logs."""
    return field_name == "api_key" or field_name.endswith("_api_key")


def _is_secret_display_field(field_name: str) -> bool:
    """Return whether a nested diagnostic field may contain credentials."""
    normalized = field_name.lower().replace("-", "_")
    compact = normalized.replace("_", "")
    parts = set(normalized.split("_"))
    if _is_api_key_field(normalized):
        return True
    if parts & {
        "authorization",
        "bearer",
        "cookie",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
    }:
        return True
    return (
        "accesskey" in compact
        or "accountkey" in compact
        or compact.startswith(("authorization", "bearer", "cookie", "credential", "password", "secret"))
        or compact.endswith(("token", "privatekey", "secretkey"))
        or compact.endswith("apikey")
    )


def _redact_param_display(
    value: Any,
    *,
    field_name: Optional[str] = None,
    seen: Optional[set[int]] = None,
) -> Any:
    """Build a recursively redacted, repr-safe diagnostic value."""
    if field_name is not None and _is_secret_display_field(field_name):
        return _REDACTED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return "<recursive>"
    seen.add(value_id)
    if isinstance(value, BaseModel):
        return {
            name: _redact_param_display(
                getattr(value, name),
                field_name=name,
                seen=seen,
            )
            for name in type(value).model_fields
        }
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            display_key = (
                key
                if isinstance(key, (str, int, float, bool)) or key is None
                else f"<{type(key).__module__}.{type(key).__qualname__}>"
            )
            redacted[display_key] = _redact_param_display(
                item,
                field_name=key if isinstance(key, str) else None,
                seen=seen,
            )
        return redacted
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_param_display(item, seen=seen) for item in value]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class _ParamsModel(BaseModel):
    """Shared base for all remote-transport Pydantic params models.

    Two cross-cutting behaviours live here:

    * :meth:`_resolve_api_keys` auto-fills unset ``*api_key`` fields from
      ``NVIDIA_API_KEY`` / ``NGC_API_KEY`` (see
      :func:`nemo_retriever.utils.remote_auth.resolve_remote_api_key`).
    * :meth:`__repr__` redacts every field whose name matches
      :func:`_is_api_key_field` so that logging a transport object (or
      letting Pydantic's default error formatter echo one back) never
      prints a bearer token.  The underlying field still serialises as
      a plain ``str`` via ``.model_dump()`` / ``getattr(self, field)``
      so no downstream consumer needs changes.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    # Keep the explicit no-auth intent after NO_API_KEY is normalized to
    # None for existing runtime consumers. Graph persistence uses this
    # private provenance to distinguish no-auth from worker-side env lookup.
    _no_api_key_fields: set[str] = PrivateAttr(default_factory=set)
    _api_key_env_references: dict[str, str] = PrivateAttr(default_factory=dict)
    _api_key_env_values: dict[str, str] = PrivateAttr(default_factory=dict)
    _auto_resolve_unset_api_keys: ClassVar[bool] = True

    @model_validator(mode="after")
    def _resolve_api_keys(self) -> "_ParamsModel":
        for field_name in type(self).model_fields:
            if _is_api_key_field(field_name):
                value = getattr(self, field_name, None)
                if value is None:
                    if field_name in self._no_api_key_fields:
                        continue
                    if not type(self)._auto_resolve_unset_api_keys:
                        self._api_key_env_references.pop(field_name, None)
                        self._api_key_env_values.pop(field_name, None)
                        continue
                    resolved = resolve_remote_api_key()
                    setattr(self, field_name, resolved)
                    source_name = next(
                        (name for name in ("NVIDIA_API_KEY", "NGC_API_KEY") if (os.environ.get(name) or "").strip()),
                        None,
                    )
                    if resolved and source_name:
                        self._api_key_env_references[field_name] = f"{ENVIRONMENT_REFERENCE_PREFIX}{source_name}"
                        self._api_key_env_values[field_name] = resolved
                    else:
                        self._api_key_env_references.pop(field_name, None)
                        self._api_key_env_values.pop(field_name, None)
                elif value == NO_API_KEY:
                    self._no_api_key_fields.add(field_name)
                    self._api_key_env_references.pop(field_name, None)
                    self._api_key_env_values.pop(field_name, None)
                    setattr(self, field_name, None)
                else:
                    self._no_api_key_fields.discard(field_name)
                    explicit_reference = environment_reference_name(value)
                    if explicit_reference is not None:
                        self._api_key_env_references[field_name] = value
                        if type(self)._auto_resolve_unset_api_keys:
                            value = resolve_environment_reference(value)
                            setattr(self, field_name, value)
                        self._api_key_env_values[field_name] = value
                    else:
                        prior_value = self._api_key_env_values.get(field_name)
                        if prior_value != value:
                            self._api_key_env_references.pop(field_name, None)
                            self._api_key_env_values.pop(field_name, None)
        return self

    def _api_key_env_reference(self, field_name: str) -> Optional[str]:
        """Return the exact environment reference that supplied an API key."""
        value = getattr(self, field_name, None)
        if isinstance(value, str) and value.startswith(ENVIRONMENT_REFERENCE_PREFIX):
            return value
        reference = self._api_key_env_references.get(field_name)
        if reference is None:
            return None
        if self._api_key_env_values.get(field_name) != value:
            return None
        return reference

    def _uses_no_api_key(self, field_name: str) -> bool:
        """Return whether an API-key field was explicitly disabled.

        This is an internal persistence hook. Runtime callers continue to
        observe None for NO_API_KEY exactly as before.
        """
        return field_name in self._no_api_key_fields and getattr(self, field_name, None) is None

    def __repr__(self) -> str:
        parts: list[str] = []
        for field_name in type(self).model_fields:
            value = getattr(self, field_name, None)
            if _is_api_key_field(field_name):
                if value:
                    parts.append(f"{field_name}={_REDACTED}")
                else:
                    parts.append(f"{field_name}={value!r}")
            elif field_name == "storage_options" and value:
                parts.append(f"{field_name}={_REDACTED}")
            else:
                display_value = _redact_param_display(value, field_name=field_name)
                parts.append(f"{field_name}={display_value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    __str__ = __repr__


class RemoteRetryParams(_ParamsModel):
    remote_max_pool_workers: int = 32
    remote_max_retries: int = 5
    remote_max_429_retries: int = 3


class RemoteInvokeParams(_ParamsModel):
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0


class ModelRuntimeParams(_ParamsModel):
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192
    model_name: Optional[str] = None
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False


class IngestorCreateParams(_ParamsModel):
    documents: list[str] = Field(default_factory=list)
    ray_address: Optional[str] = None
    ray_log_to_driver: bool = True
    debug: bool = False
    base_url: str = "http://localhost:7670"
    allow_no_gpu: bool = False
    node_overrides: Optional[dict[str, dict[str, Any]]] = None
    api_key: Optional[str] = None
    error_policy: Literal["raise", "collect"] = "raise"
    # service run mode: maximum number of concurrent page uploads.  Lower
    # values (e.g. 2-4) reduce burst pressure on Kubernetes NodePort /
    # kube-proxy paths that otherwise reset connections under heavy load.
    max_concurrency: Optional[int] = None


class UrlFetchParams(_ParamsModel):
    """Controls for fetching documents configured with ``ingestor.urls()``."""

    headers: dict[str, str] = Field(default_factory=dict, description="HTTP request headers shared by all URL fetches.")
    request_timeout_s: float = Field(
        default=30.0, gt=0, description="Timeout in seconds passed to the HTTP client for URL fetch requests."
    )
    follow_redirects: bool = Field(default=True, description="Whether URL fetch requests follow HTTP redirects.")
    max_response_bytes: int = Field(
        default=10_000_000, ge=1, description="Maximum response size in bytes allowed for each URL."
    )
    max_concurrency: int = Field(default=8, ge=1, description="Maximum number of concurrent URL fetch requests.")


class IngestExecuteParams(_ParamsModel):
    show_progress: bool = False
    return_failures: bool = False
    return_traces: bool = False
    return_results: bool = True
    result_schema: Literal["legacy", "compact"] = "legacy"
    return_embeddings: bool = False
    return_images: bool = False
    parallel: bool = False
    max_workers: Optional[int] = None
    gpu_devices: list[str] = Field(default_factory=list)
    page_chunk_size: int = 32
    runtime_metrics_dir: Optional[str] = None
    runtime_metrics_prefix: Optional[str] = None


class PdfSplitParams(_ParamsModel):
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class TextChunkParams(_ParamsModel):
    max_tokens: int = 1024
    overlap_tokens: int = 0
    tokenizer_model_id: Optional[str] = None
    encoding: str = "utf-8"
    tokenizer_cache_dir: Optional[str] = None


class HtmlChunkParams(TextChunkParams):
    pass


class AudioChunkParams(_ParamsModel):
    """Params for media chunking (audio/video split). Aligned with `nemo_retriever.api` dataloader.

    Set ``enabled=False`` (when wired through ``VideoSplitActor``) to skip
    audio chunking and ASR on a video pipeline — useful for visual-only
    recall benchmarks. ``MediaChunkActor`` ignores this flag for the
    audio-only pipeline since chunking is the whole point there.

    ``audio_only=True`` on a video input extracts only the audio track,
    runs ASR over it, and skips the visual branch entirely — no frame
    extraction, no OCR, no audio/visual fusion.

    ``video_audio_separate`` is accepted for compatibility but ignored by
    ``MediaChunkActor`` on video inputs: this ASR chunking path always demuxes
    videos to ASR-safe audio chunks and does not emit video-container chunks.
    Use ``VideoSplitActor`` or the video pipeline when you need audio+visual
    video processing.
    """

    enabled: bool = True
    split_type: Literal["size", "time", "frame"] = "size"
    split_interval: int = 450
    audio_only: bool = False
    video_audio_separate: bool = False


class ASRParams(_ParamsModel):
    """Params for ASR (Parakeet/Riva gRPC or local transformers backend).

    Choice of remote-NIM vs local-model is made by the :class:`ASRActor`
    archetype (CPU variant = remote, GPU variant = local), not by a flag here.
    Pass ``audio_endpoints`` to force the remote variant on any host; leave
    them empty to let the archetype pick GPU (local) when a GPU is present
    and fall back to remote (NVCF default) when not.
    """

    audio_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    audio_infer_protocol: str = "grpc"
    # ``auto``: streaming (online) for NVCF; offline recognize for other gRPC
    # endpoints (e.g. Helm Parakeet NIM with ``mode=ofl``).
    audio_infer_mode: Literal["auto", "online", "offline"] = "auto"
    function_id: Optional[str] = None
    auth_token: Optional[str] = None
    segment_audio: bool = False


class VideoFrameParams(_ParamsModel):
    """Params for video frame extraction (ffmpeg fps + perceptual-hash dedup).

    Set ``enabled=False`` to skip frame extraction entirely; the video
    pipeline then produces only audio (ASR) rows — no frame OCR, no
    audio+visual fusion. Useful for ablating the visual modality or for
    audio-only recall benchmarks against video corpora.

    ``dedup`` activates perceptual-hash (dhash) dedup before OCR. dhash
    catches visually-identical adjacent frames that byte-level hashing
    misses (encoder noise, brightness drift, etc.). On a 60s slide-heavy
    sample we measured ~91% duplicates collapsed at distance 5 vs ~11%
    for MD5 — a near-10x cut in OCR cost on slide content. Tune
    ``dedup_max_hamming_distance`` upward for more aggressive merging or
    down to 0 to require exact perceptual-hash matches.
    """

    enabled: bool = True
    fps: float = Field(default=1.0, gt=0.0)
    max_frames: Optional[int] = None
    dedup: bool = True
    dedup_max_hamming_distance: int = 5
    dedup_max_dropped_frames: int = 2


class VideoFrameTextDedupParams(_ParamsModel):
    """Params for merging consecutive video_frame rows with identical OCR text.

    After full-frame OCR, slides that are visible for many seconds produce a
    flood of frames with the same text (image-hash dedup misses them when
    encoder noise differs frame-to-frame). This stage groups by
    ``(source_path, text)`` and merges adjacent runs into a single row whose
    ``segment_start_seconds`` / ``segment_end_seconds`` cover the union of
    the run.

    Tolerance is expressed in **dropped frames**, not seconds, so it scales
    with ``video_frame_fps``: at runtime the dedup reads each group's
    ``metadata.fps`` and converts to ``max_gap_seconds = max_dropped_frames / fps``.
    Default 2 means we bridge gaps of up to 2 missing frames in a run —
    a typical safety margin for image-hash dedup leaving small holes.
    """

    enabled: bool = True
    max_dropped_frames: int = 2


class AudioVisualFuseParams(_ParamsModel):
    """Toggle for :class:`~nemo_retriever.video.AudioVisualFuser`."""

    enabled: bool = True


class LanceDbParams(_ParamsModel):
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"
    overwrite: bool = True
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256
    embedding_column: str = "text_embeddings_1b_v2"
    embedding_key: str = "embedding"
    include_text: bool = True
    text_column: str = "text"
    hybrid: bool = False
    fts_language: str = "English"


class BatchTuningParams(_ParamsModel):
    debug_run_id: str = "unknown"
    pdf_split_batch_size: int = 1
    pdf_extract_batch_size: int = 4
    pdf_extract_num_cpus: float = 2
    pdf_extract_workers: Optional[int] = None
    page_elements_batch_size: int = 24
    detect_batch_size: int = 24
    ocr_inference_batch_size: Optional[int] = None
    page_elements_workers: Optional[int] = None
    ocr_workers: Optional[int] = None
    detect_workers: Optional[int] = None
    page_elements_cpus_per_actor: float = 1
    ocr_cpus_per_actor: float = 1
    table_structure_workers: Optional[int] = None
    table_structure_batch_size: Optional[int] = None
    table_structure_cpus_per_actor: float = 1
    embed_workers: Optional[int] = None
    embed_batch_size: int = 32
    embed_cpus_per_actor: float = 1
    gpu_page_elements: Optional[float] = None
    gpu_ocr: Optional[float] = None
    gpu_table_structure: Optional[float] = None
    gpu_embed: Optional[float] = None
    nemotron_parse_workers: Optional[int] = None
    gpu_nemotron_parse: Optional[float] = None
    nemotron_parse_batch_size: Optional[int] = None
    store_workers: Optional[int] = None
    inference_batch_size: int = 8


class GpuAllocationParams(_ParamsModel):
    gpu_devices: list[str] = Field(default_factory=list)
    startup_timeout: float = 600.0


class ExtractParams(_ParamsModel):
    # Extraction flags
    extract_text: bool = True
    extract_images: bool = True
    extract_nested_images: bool = Field(
        default=False,
        description=(
            "When extract_images is enabled for PDFium extraction, also emit decoded raster IMAGE objects nested "
            "inside Form XObjects. Each placement is emitted, including repeated uses of the same bitmap."
        ),
    )
    extract_tables: bool = True
    extract_charts: bool = True
    extract_infographics: bool = False
    extract_page_as_image: Optional[bool] = True

    # Extraction options
    method: Literal["pdfium", "pdfium_hybrid", "ocr", "nemotron_parse", "audio"] = Field(
        default="pdfium",
        description=(
            "Extraction method. PDF extraction supports 'pdfium', 'pdfium_hybrid', 'ocr', and "
            "'nemotron_parse'; 'audio' is retained for the legacy params-driven audio path."
        ),
    )
    # Run PageElementDetection (layout/yolox). Required by TableStructure and
    # OCR. Safe to disable for text-only ingests.
    use_page_elements: bool = True
    use_table_structure: bool = False
    table_output_format: Optional[Literal["pseudo_markdown", "markdown"]] = None
    dpi: int = 200
    image_format: str = "jpeg"
    jpeg_quality: int = 100
    render_mode: Literal["full_dpi", "fit_to_model"] = "fit_to_model"
    inference_batch_size: int = 8
    ocr_model_dir: Optional[str] = None
    ocr_version: Literal["v1", "v2"] = "v2"
    ocr_lang: Optional[Literal["multi", "english"]] = None

    # Service endpoints
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0
    page_elements_invoke_url: Optional[str] = None
    page_elements_api_key: Optional[str] = None
    page_elements_request_timeout_s: Optional[float] = None
    ocr_invoke_url: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_request_timeout_s: Optional[float] = None
    table_structure_invoke_url: Optional[str] = None
    nemotron_parse_invoke_url: Optional[str] = None
    nemotron_parse_model: Optional[str] = None

    # Output columns
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"

    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)

    @model_validator(mode="after")
    def _auto_enable_features(self) -> "ExtractParams":
        """Auto-configure feature flags from remote endpoints.

        * Enable ``use_table_structure`` when ``table_structure_invoke_url``
          is provided.
        * Default ``table_output_format`` to ``"markdown"`` when the stage is
          enabled and the caller did not explicitly choose a format.
        """
        if self.table_structure_invoke_url and not self.use_table_structure:
            self.use_table_structure = True
        if self.table_output_format is None:
            self.table_output_format = "markdown" if self.use_table_structure else "pseudo_markdown"
        if self.ocr_version == "v1" and self.ocr_lang is not None:
            raise ValueError("ocr_lang is only supported when ocr_version='v2'.")
        if self.extract_nested_images and not self.extract_images:
            raise ValueError(
                "extract_nested_images=True requires extract_images=True because nested images extend image output."
            )
        if self.method != "nemotron_parse" and (
            self.nemotron_parse_invoke_url is not None or self.nemotron_parse_model is not None
        ):
            raise ValueError(
                "`nemotron_parse_invoke_url` and `nemotron_parse_model` require "
                "`method='nemotron_parse'`; Parse-specific configuration is otherwise ignored."
            )
        if self.method == "nemotron_parse":
            parse_endpoints = validate_nemotron_parse_endpoint_list(self.nemotron_parse_invoke_url or self.invoke_url)
            if (
                not parse_endpoints
                and self.nemotron_parse_model is not None
                and self.nemotron_parse_model != NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL
            ):
                raise ValueError(
                    f"Local Nemotron Parse supports only `{NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL}` in this release; "
                    f"received `{self.nemotron_parse_model}`. Configure `nemotron_parse_invoke_url` or `invoke_url` "
                    "to use a compatible remote model."
                )
        if not self.use_page_elements:
            consumers = [("use_table_structure", self.use_table_structure and self.extract_tables)]
            enabled = [name for name, on in consumers if on]
            if enabled:
                raise ValueError(f"use_page_elements=False is incompatible with: {', '.join(enabled)}")
        return self


VALID_EMBED_MODALITIES: frozenset[str] = frozenset({"text", "image", "text_image"})
IMAGE_MODALITIES: frozenset[str] = frozenset({"image", "text_image"})


class EmbedParams(_ParamsModel):
    model_name: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embed_invoke_url: Optional[str] = None
    embed_model_name: Optional[str] = None
    embed_model_revision: Optional[str] = None
    embed_model_provider_prefix: Optional[str] = None
    api_key: Optional[str] = None
    input_type: str = "passage"
    embed_modality: str = "text"  # "text", "image", or "text_image" — default for all element types
    embed_granularity: Literal["element", "page"] = "element"  # "element" = per-element rows, "page" = one row per page
    text_elements_modality: Optional[str] = None  # per-type override for page-text rows
    structured_elements_modality: Optional[str] = None  # per-type override for table/chart/infographic rows
    text_column: str = "text"
    inference_batch_size: int = 32
    output_column: str = "text_embeddings_1b_v2"
    embedding_dim_column: str = "text_embeddings_1b_v2_dim"
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding"
    embed_output_column: str = "text_embeddings_1b_v2"
    embed_inference_batch_size: int = 16

    local_ingest_embed_backend: str = (
        "vllm"  # "vllm" or "hf" — selects ingest-time embedder backend for both text and VL models
    )
    query_max_length: int = 128
    dimensions: Optional[int] = None

    # Concurrent HTTP embedding requests per Ray batch (OpenAI-compatible NIM).
    nim_http_max_concurrent: int = 32
    request_timeout_s: float = 600.0

    runtime: ModelRuntimeParams = Field(default_factory=ModelRuntimeParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)

    @field_validator("local_ingest_embed_backend", mode="before")
    @classmethod
    def _validate_local_ingest_embed_backend(cls, v: str) -> str:
        from nemo_retriever.models import (
            _LOCAL_INGEST_EMBED_BACKENDS,
            normalize_backend,
        )

        return normalize_backend(
            str(v) if v is not None else None,
            _LOCAL_INGEST_EMBED_BACKENDS,
            field_name="local_ingest_embed_backend",
            default="vllm",
        )

    @field_validator(
        "embed_modality",
        "text_elements_modality",
        "structured_elements_modality",
        mode="before",
    )
    @classmethod
    def _validate_modality(cls, v: str | None) -> str | None:
        if v is None:
            return None
        modality = str(v).strip()
        if modality == "image_text":
            raise ValueError("Use 'text_image' instead of 'image_text'.")
        if modality not in VALID_EMBED_MODALITIES:
            raise ValueError(f"Modality must be one of {sorted(VALID_EMBED_MODALITIES)}")
        return modality

    @model_validator(mode="after")
    def _warn_page_granularity_overrides(self) -> "EmbedParams":
        if self.embed_granularity == "page" and (
            self.text_elements_modality is not None or self.structured_elements_modality is not None
        ):
            warnings.warn(
                "text_elements_modality and structured_elements_modality are ignored when "
                "embed_granularity='page' (only embed_modality is used).",
                UserWarning,
                stacklevel=2,
            )
        return self


MetaJoinKey = Literal["auto", "source_id", "source_name"]


class VdbUploadParams(_ParamsModel):
    """Post-graph vector DB upload configuration.

    Sidecar metadata (``meta_*``) matches ``nv_ingest_client`` / ``metadata_and_filtered_search.ipynb``:
    all three fields must be set together to merge columns into each chunk's ``content_metadata``.
    """

    vdb_op: str = "lancedb"
    vdb_kwargs: dict[str, Any] = Field(default_factory=dict)
    meta_dataframe: Optional[Any] = None
    """Path to csv/json/parquet or an in-memory :class:`pandas.DataFrame`."""
    meta_source_field: Optional[str] = None
    meta_fields: Optional[list[str]] = None
    meta_join_key: MetaJoinKey = "auto"
    """How to match rows to documents: ``source_id`` (full path), ``source_name`` (basename), or ``auto`` (try both)."""

    @model_validator(mode="after")
    def _validate_sidecar_triplet(self) -> "VdbUploadParams":
        trio = (self.meta_dataframe, self.meta_source_field, self.meta_fields)
        if all(x is None for x in trio):
            return self
        if any(x is None for x in trio):
            raise ValueError(
                "meta_dataframe, meta_source_field, and meta_fields must all be set together "
                "when attaching sidecar metadata."
            )
        if not self.meta_fields:
            raise ValueError("meta_fields must be a non-empty list when sidecar metadata is enabled.")
        return self

    def to_ingest_operator_kwargs(self) -> dict[str, Any]:
        """Flatten into kwargs for :class:`~nemo_retriever.vdb.IngestVdbOperator`."""
        out = dict(self.vdb_kwargs or {})
        if self.meta_dataframe is not None:
            out["meta_dataframe"] = self.meta_dataframe
            out["meta_source_field"] = self.meta_source_field
            out["meta_fields"] = list(self.meta_fields or [])
            out["meta_join_key"] = self.meta_join_key
        return out


class StoreParams(_ParamsModel):
    storage_uri: str = "stored_images"
    storage_options: dict[str, Any] = Field(default_factory=dict)
    image_format: str = "png"
    strip_base64: bool = True
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)

    @model_validator(mode="after")
    def _resolve_local_storage_uri(self) -> "StoreParams":
        """Resolve relative local paths to absolute so they survive Ray serialization."""
        if not urlparse(self.storage_uri).scheme:
            self.storage_uri = str(UPath(self.storage_uri).resolve())
        return self


class PageElementsParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"


class OcrParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    extract_tables: bool = False
    extract_charts: bool = False
    extract_infographics: bool = False


class TableParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    output_column: str = "table_structure_v1"
    num_detections_column: str = "table_structure_v1_num_detections"
    counts_by_label_column: str = "table_structure_v1_counts_by_label"


class ChartParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8


class LLMInferenceParams(_ParamsModel):
    """Reusable LLM sampling / generation parameters.

    Inherit from this model to add temperature, top_p, and max_tokens
    to any task that invokes an LLM (captioning, summarization, etc.).
    """

    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    max_tokens: int = 1024

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v

    def to_sampling_kwargs(self) -> dict[str, Any]:
        """Build a dict of sampling parameters suitable for LLM inference calls.

        ``top_p`` is only included when explicitly set (not ``None``), because
        many backends (vLLM, OpenAI, NIM) change behaviour when the key is
        present vs. absent.
        """
        kw: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        return kw


class LLMRemoteClientParams(_ParamsModel):
    """Transport / connection parameters for any remote LLM client.

    Pairs with :class:`LLMInferenceParams` (sampling) to fully specify a
    call. ``api_key=None`` is left unset so LiteLLM can perform provider-native
    environment lookup on the worker.
    """

    _auto_resolve_unset_api_keys: ClassVar[bool] = False

    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    num_retries: int = 3
    timeout: float = 120.0
    extra_params: dict[str, Any] = Field(default_factory=dict)
    rag_system_prompt: Optional[str] = None
    rag_system_prompt_prefix: Optional[str] = None
    reasoning_enabled: bool = True

    @field_validator("extra_params")
    @classmethod
    def _check_extra_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        validate_llm_extra_params(value, source="LLMRemoteClientParams.extra_params")
        return value

    @field_validator("num_retries")
    @classmethod
    def _check_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_retries must be >= 0")
        return v

    @field_validator("timeout")
    @classmethod
    def _check_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout must be > 0")
        return v


class LLMSamplingOverrides(_ParamsModel):
    """Partial sampling overrides resolved on top of task-specific defaults."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v

    @model_validator(mode="after")
    def _reject_explicit_null_max_tokens(self) -> "LLMSamplingOverrides":
        if "max_tokens" in self.model_fields_set and self.max_tokens is None:
            raise ValueError("max_tokens cannot be None; omit it to inherit the task default")
        return self

    @model_serializer(mode="plain")
    def _serialize_only_explicit_overrides(self) -> dict[str, Any]:
        """Preserve omitted-vs-null state across model and JSON round trips."""
        return {
            name: getattr(self, name)
            for name in ("temperature", "top_p", "max_tokens")
            if name in self.model_fields_set
        }

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LLMSamplingOverrides):
            return self.model_fields_set == other.model_fields_set and super().__eq__(other)
        return super().__eq__(other)

    def resolve(self, defaults: LLMInferenceParams) -> LLMInferenceParams:
        """Apply explicitly supplied fields to defaults."""
        values = defaults.model_dump()
        for name in self.model_fields_set:
            value = getattr(self, name)
            values[name] = value
        return LLMInferenceParams(**values)


_SAMPLING_UNSET = object()


class TextGenerationParams(_ParamsModel):
    """Transport, task controls, and partial sampling for text generation."""

    transport: LLMRemoteClientParams
    sampling: LLMSamplingOverrides = Field(default_factory=LLMSamplingOverrides)
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    reasoning_enabled: Optional[bool] = None
    max_workers: int = Field(default=8, ge=1)

    def resolve_sampling(self, defaults: LLMInferenceParams) -> LLMInferenceParams:
        """Resolve explicit sampling fields over a task's defaults."""
        return self.sampling.resolve(defaults)

    @classmethod
    def from_kwargs(
        cls,
        *,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Any = _SAMPLING_UNSET,
        top_p: Any = _SAMPLING_UNSET,
        max_tokens: Any = _SAMPLING_UNSET,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        rag_system_prompt: Optional[str] = None,
        rag_system_prompt_prefix: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_workers: int = 8,
    ) -> "TextGenerationParams":
        """Construct structured text-generation params from flat kwargs."""
        sampling_values: dict[str, Any] = {}
        for name, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("max_tokens", max_tokens),
        ):
            if value is not _SAMPLING_UNSET:
                sampling_values[name] = value

        transport_reasoning = True if reasoning_enabled is None else reasoning_enabled
        return cls(
            transport=LLMRemoteClientParams(
                model=model,
                api_base=api_base,
                api_key=api_key,
                num_retries=num_retries,
                timeout=timeout,
                extra_params=extra_params or {},
                rag_system_prompt=rag_system_prompt,
                rag_system_prompt_prefix=rag_system_prompt_prefix,
                reasoning_enabled=transport_reasoning,
            ),
            sampling=LLMSamplingOverrides(**sampling_values),
            prompt=prompt,
            system_prompt=system_prompt,
            reasoning_enabled=reasoning_enabled,
            max_workers=max_workers,
        )


class CaptionParams(LLMInferenceParams):
    endpoint_url: Optional[str] = None
    model_name: str = Field(
        default=DEFAULT_LOCAL_CAPTION_MODEL_ID,
        description=(
            "Caption model identifier. The default local BF16 checkpoint has approximately 62 GiB of weights; "
            "set this explicitly to select a smaller local model or an API model for a remote endpoint."
        ),
    )
    api_key: Optional[str] = None
    prompt: str = "Caption the content of this image:"
    system_prompt: Optional[str] = "/no_think"
    batch_size: int = 8
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    context_text_max_chars: int = 0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        gt=0,
        le=1,
        description="Fraction of GPU memory reserved for local vLLM captioning; defaults to the model profile.",
    )
    caption_infographics: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("temperature")
    @classmethod
    def _require_temperature(cls, value: Optional[float]) -> float:
        if value is None:
            raise ValueError("temperature cannot be None for captioning")
        return value


class WebhookParams(_ParamsModel):
    """Configuration for the webhook notification stage.

    When ``endpoint_url`` is set, selected columns from the processed batch
    are serialised to JSON and HTTP-POSTed to that URL.  If ``endpoint_url``
    is ``None`` the stage is a no-op.
    """

    endpoint_url: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 30.0
    max_retries: int = 3


class DedupParams(_ParamsModel):
    content_hash: bool = True
    bbox_iou: bool = True
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Structured (database) ingestion params
# ---------------------------------------------------------------------------


class TabularExtractParams(_ParamsModel):
    """Params for step 1: extract schema metadata and write to Neo4j.

    Covers SQLAlchemy reflection of a live database and/or parsing of
    pre-existing SQL DDL/query files.  Produces Database, Schema, Table,
    Column, View and Query nodes together with their relationships.
    The Neo4j connection is provided by get_neo4j_conn() (see
    tabular_data.neo4j) and is not configured here.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    connector: Optional[SQLDatabase] = None
