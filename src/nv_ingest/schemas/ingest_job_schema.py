# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import field_validator, model_validator, Field

from nv_ingest.schemas.base_model_noext import BaseModelNoExt
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


# Enums
class DocumentTypeEnum(str, Enum):
    bmp = "bmp"
    docx = "docx"
    html = "html"
    jpeg = "jpeg"
    pdf = "pdf"
    png = "png"
    pptx = "pptx"
    svg = "svg"
    tiff = "tiff"
    txt = "text"


class TaskTypeEnum(str, Enum):
    caption = "caption"
    dedup = "dedup"
    embed = "embed"
    extract = "extract"
    filter = "filter"
    split = "split"
    store = "store"
    store_embedding = "store_embedding"
    vdb_upload = "vdb_upload"
    table_data_extract = "table_data_extract"
    chart_data_extract = "chart_data_extract"


class FilterTypeEnum(str, Enum):
    image = "image"


class TracingOptionsSchema(BaseModelNoExt):
    trace: bool = False
    ts_send: int
    trace_id: Optional[str] = None


class IngestTaskSplitSchema(BaseModelNoExt):
    split_by: Literal["word", "sentence", "passage"]
    split_length: Annotated[int, Field(gt=0)]
    split_overlap: Annotated[int, Field(ge=0)]
    max_character_length: Optional[Annotated[int, Field(gt=0)]] = None
    sentence_window_size: Optional[Annotated[int, Field(ge=0)]] = None

    @field_validator("sentence_window_size")
    def check_sentence_window_size(cls, v, values, **kwargs):
        if v is not None and v > 0 and values.data["split_by"] != "sentence":
            raise ValueError("When using sentence_window_size, split_by must be 'sentence'.")
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


# All optional, the captioning stage requires default parameters, each of these are just overrides.
class IngestTaskCaptionSchema(BaseModelNoExt):
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    prompt: Optional[str] = None
    model_name: Optional[str] = None


class IngestTaskFilterParamsSchema(BaseModelNoExt):
    min_size: int = 128
    max_aspect_ratio: Union[float, int] = 5.0
    min_aspect_ratio: Union[float, int] = 0.2
    filter: bool = False


class IngestTaskFilterSchema(BaseModelNoExt):
    content_type: ContentTypeEnum = ContentTypeEnum.IMAGE
    params: IngestTaskFilterParamsSchema = IngestTaskFilterParamsSchema()


class IngestTaskDedupParams(BaseModelNoExt):
    filter: bool = False


class IngestTaskDedupSchema(BaseModelNoExt):
    content_type: ContentTypeEnum = ContentTypeEnum.IMAGE
    params: IngestTaskDedupParams = IngestTaskDedupParams()


class IngestTaskEmbedSchema(BaseModelNoExt):
    filter_errors: bool = False


class IngestTaskVdbUploadSchema(BaseModelNoExt):
    bulk_ingest: bool = False
    bulk_ingest_path: str = None
    params: dict = None
    filter_errors: bool = True


class IngestTaskTableExtraction(BaseModelNoExt):
    params: Dict = {}


class IngestChartTableExtraction(BaseModelNoExt):
    params: Dict = {}


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
        IngestTaskTableExtraction,
        IngestChartTableExtraction,
    ]
    raise_on_failure: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_task_properties_type(cls, values):
        task_type, task_properties = values.get("type"), values.get("task_properties")
        if task_type and task_properties:
            expected_type = {
                TaskTypeEnum.caption: IngestTaskCaptionSchema,
                TaskTypeEnum.dedup: IngestTaskDedupSchema,
                TaskTypeEnum.embed: IngestTaskEmbedSchema,
                TaskTypeEnum.extract: IngestTaskExtractSchema,
                TaskTypeEnum.filter: IngestTaskFilterSchema,  # Extend this mapping as necessary
                TaskTypeEnum.split: IngestTaskSplitSchema,
                TaskTypeEnum.store_embedding: IngestTaskStoreEmbedSchema,
                TaskTypeEnum.store: IngestTaskStoreSchema,
                TaskTypeEnum.vdb_upload: IngestTaskVdbUploadSchema,
                TaskTypeEnum.table_data_extract: IngestTaskTableExtraction,
                TaskTypeEnum.chart_data_extract: IngestChartTableExtraction,
            }.get(task_type.lower())

            # logger.debug(f"Checking task_properties type for task type '{task_type}'")

            # Ensure task_properties is validated against the expected schema
            validated_task_properties = expected_type(**task_properties)
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
