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

from pydantic import conint
from pydantic import root_validator
from pydantic import validator

from nv_ingest.schemas.base_model_noext import BaseModelNoExt
from nv_ingest.schemas.metadata_schema import ContentTypeEnum

logger = logging.getLogger(__name__)


# Enums
class DocumentTypeEnum(str, Enum):
    pdf = "pdf"
    txt = "text"
    docx = "docx"
    pptx = "pptx"
    jpeg = "jpeg"
    bmp = "bmp"
    png = "png"
    svg = "svg"
    html = "html"


class TaskTypeEnum(str, Enum):
    caption = "caption"
    dedup = "dedup"
    embed = "embed"
    extract = "extract"
    filter = "filter"
    split = "split"
    store = "store"
    vdb_upload = "vdb_upload"


class FilterTypeEnum(str, Enum):
    image = "image"


class TracingOptionsSchema(BaseModelNoExt):
    trace: bool = False
    ts_send: int
    trace_id: Optional[str] = None


class IngestTaskSplitSchema(BaseModelNoExt):
    split_by: Literal["word", "sentence", "passage"]
    split_length: conint(gt=0)
    split_overlap: conint(ge=0)
    max_character_length: Optional[conint(gt=0)]
    sentence_window_size: Optional[conint(ge=0)]

    @validator("sentence_window_size")
    def check_sentence_window_size(cls, v, values, **kwargs):
        if v is not None and v > 0 and values["split_by"] != "sentence":
            raise ValueError("When using sentence_window_size, split_by must be 'sentence'.")
        return v


class IngestTaskExtractSchema(BaseModelNoExt):
    document_type: DocumentTypeEnum
    method: str
    params: dict

    @validator("document_type", pre=True)
    def case_insensitive_document_type(cls, v):
        if isinstance(v, str):
            v = v.lower()
        try:
            return DocumentTypeEnum(v)
        except ValueError:
            raise ValueError(f"{v} is not a valid DocumentTypeEnum value")


class IngestTaskStoreSchema(BaseModelNoExt):
    structured: bool = True
    images: bool = False
    method: str
    params: dict


class IngestTaskCaptionSchema(BaseModelNoExt):
    content_type: str = "image"
    n_neighbors: int = 5


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
    text: bool = True
    tables: bool = True
    filter_errors: bool = False


class IngestTaskVdbUploadSchema(BaseModelNoExt):
    filter_errors: bool = True


class IngestTaskSchema(BaseModelNoExt):
    type: TaskTypeEnum
    task_properties: Union[
        IngestTaskSplitSchema,
        IngestTaskExtractSchema,
        IngestTaskStoreSchema,
        IngestTaskEmbedSchema,
        IngestTaskCaptionSchema,
        IngestTaskDedupSchema,
        IngestTaskFilterSchema,
        IngestTaskVdbUploadSchema,
    ]
    raise_on_failure: bool = False

    @root_validator(pre=True)
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
                TaskTypeEnum.store: IngestTaskStoreSchema,
                TaskTypeEnum.vdb_upload: IngestTaskVdbUploadSchema,
            }.get(task_type.lower())

            # logger.debug(f"Checking task_properties type for task type '{task_type}'")

            # Ensure task_properties is validated against the expected schema
            validated_task_properties = expected_type(**task_properties)
            values["task_properties"] = validated_task_properties
        return values

    @validator("type", pre=True)
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
    tracing_options: Optional[TracingOptionsSchema]


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
