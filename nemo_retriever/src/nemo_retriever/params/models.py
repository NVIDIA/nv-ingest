# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import warnings


from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunMode = Literal["inprocess", "batch", "fused", "online"]


class _ParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RemoteRetryParams(_ParamsModel):
    remote_max_pool_workers: int = 16
    remote_max_retries: int = 10
    remote_max_429_retries: int = 5


class RemoteInvokeParams(_ParamsModel):
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 120.0


class ModelRuntimeParams(_ParamsModel):
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192
    model_name: Optional[str] = None


class IngestorCreateParams(_ParamsModel):
    documents: list[str] = Field(default_factory=list)
    ray_address: Optional[str] = None
    ray_log_to_driver: bool = True
    debug: bool = False
    base_url: str = "http://localhost:7670"


class IngestExecuteParams(_ParamsModel):
    show_progress: bool = False
    return_failures: bool = False
    save_to_disk: bool = False
    return_traces: bool = False
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
    """Params for media chunking (audio/video split). Aligned with nv-ingest-api dataloader."""

    split_type: Literal["size", "time", "frame"] = "size"
    split_interval: int = 450
    audio_only: bool = False
    video_audio_separate: bool = False


class ASRParams(_ParamsModel):
    """Params for ASR (Parakeet/Riva gRPC or local transformers backend)."""

    audio_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    audio_infer_protocol: str = "grpc"
    function_id: Optional[str] = None
    auth_token: Optional[str] = None
    segment_audio: bool = False


class LanceDbParams(_ParamsModel):
    lancedb_uri: str = "lancedb"
    table_name: str = "nv-ingest"
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
    embed_workers: Optional[int] = None
    embed_batch_size: int = 256
    embed_cpus_per_actor: float = 1
    gpu_page_elements: Optional[float] = None
    gpu_ocr: Optional[float] = None
    gpu_embed: Optional[float] = None
    nemotron_parse_workers: float = 0.0
    gpu_nemotron_parse: float = 0.0
    nemotron_parse_batch_size: float = 0.0
    inference_batch_size: int = 8


class FusedTuningParams(_ParamsModel):
    fused_workers: int = 1
    fused_batch_size: int = 64
    fused_cpus_per_actor: float = 1
    fused_gpus_per_actor: float = 1.0


class GpuAllocationParams(_ParamsModel):
    gpu_devices: list[str] = Field(default_factory=list)
    startup_timeout: float = 600.0


class ExtractParams(_ParamsModel):
    # Extraction flags
    extract_text: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    extract_charts: bool = True
    extract_infographics: bool = True
    extract_page_as_image: Optional[bool] = None

    # Extraction options
    method: str = "pdfium"
    use_table_structure: bool = False
    table_output_format: Optional[Literal["pseudo_markdown", "markdown"]] = None
    use_graphic_elements: bool = False
    dpi: int = 200
    image_format: str = "jpeg"
    jpeg_quality: int = 100
    inference_batch_size: int = 8
    ocr_model_dir: Optional[str] = None

    # Service endpoints
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 120.0
    page_elements_invoke_url: Optional[str] = None
    page_elements_api_key: Optional[str] = None
    page_elements_request_timeout_s: Optional[float] = None
    ocr_invoke_url: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_request_timeout_s: Optional[float] = None
    graphic_elements_invoke_url: Optional[str] = None
    table_structure_invoke_url: Optional[str] = None

    # Output columns
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"

    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)

    @model_validator(mode="after")
    def _auto_enable_features(self) -> "ExtractParams":
        """Auto-configure feature flags from remote endpoints.

        * Enable ``use_graphic_elements`` when ``graphic_elements_invoke_url``
          is provided.
        * Enable ``use_table_structure`` when ``table_structure_invoke_url``
          is provided.
        * Default ``table_output_format`` to ``"markdown"`` when the stage is
          enabled and the caller did not explicitly choose a format.
        """
        if self.graphic_elements_invoke_url and not self.use_graphic_elements:
            self.use_graphic_elements = True
        if self.table_structure_invoke_url and not self.use_table_structure:
            self.use_table_structure = True
        if self.table_output_format is None:
            self.table_output_format = "markdown" if self.use_table_structure else "pseudo_markdown"
        return self


IMAGE_MODALITIES: frozenset[str] = frozenset({"image", "text_image", "image_text"})


class EmbedParams(_ParamsModel):
    model_name: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embed_invoke_url: Optional[str] = None
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

    runtime: ModelRuntimeParams = Field(default_factory=ModelRuntimeParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)
    fused_tuning: FusedTuningParams = Field(default_factory=FusedTuningParams)

    @field_validator("embed_modality", "text_elements_modality", "structured_elements_modality", mode="before")
    @classmethod
    def _normalize_modality(cls, v: str | None) -> str | None:
        if v == "image_text":
            return "text_image"
        return v

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


class VdbUploadParams(_ParamsModel):
    purge_results_after_upload: bool = True
    lancedb: LanceDbParams = Field(default_factory=LanceDbParams)


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


class InfographicParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    allowed_page_element_labels: Sequence[str] = ("infographic", "title")
    output_column: str = "infographic_elements_v1"
    num_detections_column: str = "infographic_elements_v1_num_detections"
    counts_by_label_column: str = "infographic_elements_v1_counts_by_label"
