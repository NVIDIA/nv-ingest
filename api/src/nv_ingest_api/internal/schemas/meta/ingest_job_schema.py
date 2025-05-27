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
    ts_send: int
    trace_id: Optional[str] = None


# Ingest Task Schemas


class IngestTaskSplitSchema(BaseModelNoExt):
    tokenizer: Optional[str] = None
    chunk_size: Annotated[int, Field(gt=0)] = 1024
    chunk_overlap: Annotated[int, Field(ge=0)] = 150
    params: dict

    @field_validator("chunk_overlap")
    def check_chunk_overlap(cls, v, values, **kwargs):
        if v is not None and "chunk_size" in values.data and v >= values.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class IngestTaskExtractSchema(BaseModelNoExt):
    document_type: DocumentTypeEnum
    method: str
    params: dict

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
    params: dict


class IngestTaskStoreSchema(BaseModelNoExt):
    structured: bool = True
    images: bool = False
    method: str
    params: dict


# Captioning: All fields are optional and override default parameters.
class IngestTaskCaptionSchema(BaseModelNoExt):
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    prompt: Optional[str] = None
    caption_model_name: Optional[str] = None


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
    api_key: Optional[str] = None
    filter_errors: bool = False


class IngestTaskVdbUploadSchema(BaseModelNoExt):
    bulk_ingest: bool = False
    bulk_ingest_path: Optional[str] = None
    params: Optional[dict] = None
    filter_errors: bool = True


class IngestTaskAudioExtraction(BaseModelNoExt):
    auth_token: Optional[str] = None
    grpc_endpoint: Optional[str] = None
    http_endpoint: Optional[str] = None
    infer_protocol: Optional[str] = None
    function_id: Optional[str] = None
    use_ssl: Optional[bool] = None
    ssl_cert: Optional[str] = None


class IngestTaskTableExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskChartExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


class IngestTaskInfographicExtraction(BaseModelNoExt):
    params: dict = Field(default_factory=dict)


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
