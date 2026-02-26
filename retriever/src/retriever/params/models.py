# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

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
    max_tokens: int = 512
    overlap_tokens: int = 0
    tokenizer_model_id: Optional[str] = None
    encoding: str = "utf-8"
    tokenizer_cache_dir: Optional[str] = None


class HtmlChunkParams(TextChunkParams):
    pass


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
    page_elements_workers: int = 1
    ocr_workers: int = 1
    detect_workers: Optional[int] = None
    page_elements_cpus_per_actor: float = 1
    ocr_cpus_per_actor: float = 1
    embed_workers: int = 1
    embed_batch_size: int = 256
    embed_cpus_per_actor: float = 1
    gpu_page_elements: Optional[float] = None
    gpu_ocr: Optional[float] = None
    gpu_embed: Optional[float] = None


class FusedTuningParams(_ParamsModel):
    fused_workers: int = 1
    fused_batch_size: int = 64
    fused_cpus_per_actor: float = 1
    fused_gpus_per_actor: float = 1.0


class GpuAllocationParams(_ParamsModel):
    gpu_devices: list[str] = Field(default_factory=list)
    startup_timeout: float = 600.0


class ExtractParams(_ParamsModel):
    method: Optional[str] = None
    extract_text: bool = False
    extract_images: bool = False
    extract_tables: bool = False
    extract_charts: bool = False
    extract_infographics: bool = False
    extract_page_as_image: Optional[bool] = None
    dpi: int = 200

    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 120.0
    page_elements_invoke_url: Optional[str] = None
    page_elements_api_key: Optional[str] = None
    page_elements_request_timeout_s: Optional[float] = None
    ocr_invoke_url: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_request_timeout_s: Optional[float] = None

    inference_batch_size: int = 8
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"
    ocr_model_dir: Optional[str] = None

    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)


class EmbedParams(_ParamsModel):
    model_name: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embed_invoke_url: Optional[str] = None
    input_type: str = "passage"
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
