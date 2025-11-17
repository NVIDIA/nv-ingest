# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Optional, Union, Annotated

from pydantic import Field, field_validator, model_validator

from nv_ingest_api.internal.schemas.meta.base_model_noext import BaseModelNoExt
from nv_ingest_api.internal.enums.common import ContentTypeEnum, TaskTypeEnum, DocumentTypeEnum

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Schemas: Common and Task-Specific
# ------------------------------------------------------------------------------


# Tracing Options Schema
class TracingOptionsSchema(BaseModelNoExt):
    trace: bool = False
    ts_send: Optional[int] = None
    trace_id: Optional[str] = None
    # V2 PDF splitting support
    parent_job_id: Optional[str] = None
    page_num: Optional[int] = None
    total_pages: Optional[int] = None


# PDF Configuration Schema
class PdfConfigSchema(BaseModelNoExt):
    """PDF-specific configuration options for job submission.

    Note: split_page_count accepts any positive integer but will be clamped
    to [1, 128] range by the server at runtime.
    """

    split_page_count: Annotated[int, Field(ge=1)] = 32


class RoutingOptionsSchema(BaseModelNoExt):
    # Queue routing hint for QoS scheduler
    queue_hint: Optional[str] = None

    @field_validator("queue_hint")
    @classmethod
    def validate_queue_hint(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("queue_hint must be a string")
        s = v.lower()
        allowed = {"default", "immediate", "micro", "small", "medium", "large"}
        if s not in allowed:
            raise ValueError("queue_hint must be one of: default, immediate, micro, small, medium, large")
        return s


# Ingest Task Schemas


class IngestTaskSplitSchema(BaseModelNoExt):
    tokenizer: Optional[str] = None
    chunk_size: Annotated[int, Field(gt=0)] = 1024
    chunk_overlap: Annotated[int, Field(ge=0)] = 150
    params: dict = Field(default_factory=dict)

    @field_validator("chunk_overlap")
    def check_chunk_overlap(cls, v, values, **kwargs):
        if v is not None and "chunk_size" in values.data and v >= values.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class IngestTaskExtractSchema(BaseModelNoExt):
    document_type: DocumentTypeEnum
    method: str
    params: dict = Field(default_factory=dict)

    @field_validator("document_type", mode="before")
    @classmethod
    def case_insensitive_document_type(cls, v):
        if isinstance(v, str):
            v = v.lower()
        try:
            return DocumentTypeEnum(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid DocumentTypeEnum value")


class IngestTaskStoreEmbedSchema(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskStoreSchema(BaseModelNoExt):
    structured: bool = True
    images: bool = False
    method: str
    params: dict = Field(default_factory=dict)


# Captioning: All fields are optional and override default parameters.
class IngestTaskCaptionSchema(BaseModelNoExt):
    api_key: Optional[str] = Field(default=None, repr=False)
    endpoint_url: Optional[str] = None
    prompt: Optional[str] = None
    model_name: Optional[str] = None


class IngestTaskFilterParamsSchema(BaseModelNoExt):
    min_size: int = 128
    max_aspect_ratio: Union[float, int] = 5.0
    min_aspect_ratio: Union[float, int] = 0.2
    filter: bool = False


class IngestTaskFilterSchema(BaseModelNoExt):
    # TODO: Ensure ContentTypeEnum is imported/defined as needed.
    content_type: ContentTypeEnum = ContentTypeEnum.IMAGE
    params: IngestTaskFilterParamsSchema = IngestTaskFilterParamsSchema()


class IngestTaskDedupParams(BaseModelNoExt):
    filter: bool = False


class IngestTaskDedupSchema(BaseModelNoExt):
    # TODO: Ensure ContentTypeEnum is imported/defined as needed.
    content_type: ContentTypeEnum = ContentTypeEnum.IMAGE
    params: IngestTaskDedupParams = IngestTaskDedupParams()


class IngestTaskEmbedSchema(BaseModelNoExt):
    endpoint_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = Field(default=None, repr=False)
    filter_errors: bool = False
    text_elements_modality: Optional[str] = None
    image_elements_modality: Optional[str] = None
    structured_elements_modality: Optional[str] = None
    audio_elements_modality: Optional[str] = None
    custom_content_field: Optional[str] = None
    result_target_field: Optional[str] = None
    dimensions: Optional[int] = None


class IngestTaskVdbUploadSchema(BaseModelNoExt):
    bulk_ingest: bool = False
    bulk_ingest_path: Optional[str] = None
    params: Optional[dict] = None
    filter_errors: bool = True


class IngestTaskAudioExtraction(BaseModelNoExt):
    auth_token: Optional[str] = Field(default=None, repr=False)
    grpc_endpoint: Optional[str] = None
    http_endpoint: Optional[str] = None
    infer_protocol: Optional[str] = None
    function_id: Optional[str] = None
    use_ssl: Optional[bool] = None
    ssl_cert: Optional[str] = Field(default=None, repr=False)
    segment_audio: Optional[bool] = None


class IngestTaskTableExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskChartExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskInfographicExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskOCRExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskUDFSchema(BaseModelNoExt):
    udf_function: str
    udf_function_name: str
    phase: Optional[int] = Field(default=None, ge=1, le=5)
    run_before: bool = Field(default=False, description="Execute UDF before the target stage")
    run_after: bool = Field(default=False, description="Execute UDF after the target stage")
    target_stage: Optional[str] = Field(
        default=None, description="Name of the stage to target (e.g., 'image_dedup', 'text_extract')"
    )

    @model_validator(mode="after")
    def validate_stage_targeting(self):
        """Validate that stage targeting configuration is consistent"""
        # Must specify either phase or target_stage, but not both
        has_phase = self.phase is not None
        has_target_stage = self.target_stage is not None

        if has_phase and has_target_stage:
            raise ValueError("Cannot specify both 'phase' and 'target_stage'. Please specify only one.")
        elif not has_phase and not has_target_stage:
            raise ValueError("Must specify either 'phase' or 'target_stage'.")

        # If using run_before or run_after, must specify target_stage
        if self.run_before or self.run_after:
            if not self.target_stage:
                raise ValueError("target_stage must be specified when using run_before or run_after")

        # If target_stage is specified, must have at least one timing
        if self.target_stage and not (self.run_before or self.run_after):
            raise ValueError("At least one of run_before or run_after must be True when target_stage is specified")

        return self


class IngestTaskSchema(BaseModelNoExt):
    type: TaskTypeEnum
    task_properties: Union[
        IngestTaskSplitSchema,
        IngestTaskExtractSchema,
        IngestTaskStoreEmbedSchema,
        IngestTaskStoreSchema,
        IngestTaskEmbedSchema,
        IngestTaskCaptionSchema,
        IngestTaskDedupSchema,
        IngestTaskFilterSchema,
        IngestTaskVdbUploadSchema,
        IngestTaskAudioExtraction,
        IngestTaskTableExtraction,
        IngestTaskChartExtraction,
        IngestTaskInfographicExtraction,
        IngestTaskOCRExtraction,
        IngestTaskUDFSchema,
    ]
    raise_on_failure: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_task_properties_type(cls, values):
        task_type = values.get("type")
        task_properties = values.get("task_properties", {})

        # Ensure task_type is lowercased and converted to enum early
        if isinstance(task_type, str):
            task_type = task_type.lower()
            try:
                task_type = TaskTypeEnum(task_type)
            except ValueError:
                raise ValueError(f"{task_type} is not a valid TaskTypeEnum value")

        task_type_to_schema = {
            TaskTypeEnum.CAPTION: IngestTaskCaptionSchema,
            TaskTypeEnum.DEDUP: IngestTaskDedupSchema,
            TaskTypeEnum.EMBED: IngestTaskEmbedSchema,
            TaskTypeEnum.EXTRACT: IngestTaskExtractSchema,
            TaskTypeEnum.FILTER: IngestTaskFilterSchema,
            TaskTypeEnum.SPLIT: IngestTaskSplitSchema,
            TaskTypeEnum.STORE_EMBEDDING: IngestTaskStoreEmbedSchema,
            TaskTypeEnum.STORE: IngestTaskStoreSchema,
            TaskTypeEnum.VDB_UPLOAD: IngestTaskVdbUploadSchema,
            TaskTypeEnum.AUDIO_DATA_EXTRACT: IngestTaskAudioExtraction,
            TaskTypeEnum.TABLE_DATA_EXTRACT: IngestTaskTableExtraction,
            TaskTypeEnum.CHART_DATA_EXTRACT: IngestTaskChartExtraction,
            TaskTypeEnum.INFOGRAPHIC_DATA_EXTRACT: IngestTaskInfographicExtraction,
            TaskTypeEnum.OCR_DATA_EXTRACT: IngestTaskOCRExtraction,
            TaskTypeEnum.UDF: IngestTaskUDFSchema,
        }

        expected_schema_cls = task_type_to_schema.get(task_type)
        if expected_schema_cls is None:
            raise ValueError(f"Unsupported or missing task_type '{task_type}'")

        validated_task_properties = expected_schema_cls(**task_properties)
        values["type"] = task_type  # ensure type is now always the enum
        values["task_properties"] = validated_task_properties

        return values

    @field_validator("type", mode="before")
    @classmethod
    def case_insensitive_task_type(cls, v):
        if isinstance(v, str):
            v = v.lower()
        try:
            return TaskTypeEnum(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid TaskTypeEnum value")


# ------------------------------------------------------------------------------
# Schemas: Job Schemas
# ------------------------------------------------------------------------------


class JobPayloadSchema(BaseModelNoExt):
    content: List[Union[str, bytes]]
    source_name: List[str]
    source_id: List[Union[str, int]]
    document_type: List[str]


class IngestJobSchema(BaseModelNoExt):
    job_payload: JobPayloadSchema
    job_id: Union[str, int]
    tasks: List[IngestTaskSchema]
    tracing_options: Optional[TracingOptionsSchema] = None
    routing_options: Optional[RoutingOptionsSchema] = None
    pdf_config: Optional[PdfConfigSchema] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_queue_hint(cls, values):
        """
        Backward-compatibility shim: if a legacy client sends
        tracing_options.queue_hint, move it into routing_options.queue_hint.
        """
        try:
            topt = values.get("tracing_options") or {}
            ropt = values.get("routing_options") or {}
            if isinstance(topt, dict) and "queue_hint" in topt and "queue_hint" not in ropt:
                ropt["queue_hint"] = topt.pop("queue_hint")
                values["routing_options"] = ropt
                values["tracing_options"] = topt
        except Exception:
            pass
        return values


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def validate_ingest_job(job_data: Dict[str, Any]) -> IngestJobSchema:
    """
    Validates a dictionary representing an ingest_job using the IngestJobSchema.

    Parameters:
    - job_data: Dictionary representing an ingest job.

    Returns:
    - IngestJobSchema: The validated ingest job.

    Raises:
    - ValidationError: If the input data does not conform to the IngestJobSchema.
    """

    return IngestJobSchema(**job_data)
