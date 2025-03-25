# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import field_validator, model_validator, Field

from nv_ingest_api.internal.enums.common import (
    AccessLevelEnum,
    ContentTypeEnum,
    TextTypeEnum,
    LanguageEnum,
    TableFormatEnum,
    StatusEnum,
    DocumentTypeEnum,
    TaskTypeEnum,
)
from nv_ingest_api.internal.schemas.meta.base_model_noext import BaseModelNoExt
from nv_ingest_api.util.converters import datetools

logger = logging.getLogger(__name__)


# Sub schemas
class SourceMetadataSchema(BaseModelNoExt):
    """
    Schema for the knowledge base file from which content
    and metadata is extracted.
    """

    source_name: str
    source_id: str
    source_location: str = ""
    source_type: Union[DocumentTypeEnum, str]
    collection_id: str = ""
    date_created: str = datetime.now().isoformat()
    last_modified: str = datetime.now().isoformat()
    summary: str = ""
    partition_id: int = -1
    access_level: Union[AccessLevelEnum, int] = AccessLevelEnum.UNKNOWN

    @field_validator("date_created", "last_modified")
    @classmethod
    def validate_fields(cls, field_value):
        datetools.validate_iso8601(field_value)
        return field_value


class NearbyObjectsSubSchema(BaseModelNoExt):
    """
    Schema to hold related extracted object.
    """

    content: List[str] = Field(default_factory=list)
    bbox: List[tuple] = Field(default_factory=list)
    type: List[str] = Field(default_factory=list)


class NearbyObjectsSchema(BaseModelNoExt):
    """
    Schema to hold types of related extracted objects.
    """

    text: NearbyObjectsSubSchema = NearbyObjectsSubSchema()
    images: NearbyObjectsSubSchema = NearbyObjectsSubSchema()
    structured: NearbyObjectsSubSchema = NearbyObjectsSubSchema()


class ContentHierarchySchema(BaseModelNoExt):
    """
    Schema for the extracted content hierarchy.
    """

    page_count: int = -1
    page: int = -1
    block: int = -1
    line: int = -1
    span: int = -1
    nearby_objects: NearbyObjectsSchema = NearbyObjectsSchema()


class ContentMetadataSchema(BaseModelNoExt):
    """
    Data extracted from a source; generally Text or Image.
    """

    type: ContentTypeEnum
    description: str = ""
    page_number: int = -1
    hierarchy: ContentHierarchySchema = ContentHierarchySchema()
    subtype: Union[ContentTypeEnum, str] = ""


class TextMetadataSchema(BaseModelNoExt):
    text_type: TextTypeEnum
    summary: str = ""
    keywords: Union[str, List[str], Dict] = ""
    language: LanguageEnum = "en"  # default to Unknown? Maybe do some kind of heuristic check
    text_location: tuple = (0, 0, 0, 0)
    text_location_max_dimensions: tuple = (0, 0, 0, 0)


class ImageMetadataSchema(BaseModelNoExt):
    image_type: Union[DocumentTypeEnum, str]
    structured_image_type: ContentTypeEnum = ContentTypeEnum.NONE
    caption: str = ""
    text: str = ""
    image_location: tuple = (0, 0, 0, 0)
    image_location_max_dimensions: tuple = (0, 0)
    uploaded_image_url: str = ""
    width: int = 0
    height: int = 0

    @field_validator("image_type")
    def validate_image_type(cls, v):
        if not isinstance(v, (DocumentTypeEnum, str)):
            raise ValueError("image_type must be a string or DocumentTypeEnum")
        return v

    @field_validator("width", "height")
    def clamp_non_negative(cls, v, field):
        if v < 0:
            logger.warning(f"{field.field_name} is negative; clamping to 0. Original value: {v}")
            return 0
        return v


class TableMetadataSchema(BaseModelNoExt):
    caption: str = ""
    table_format: TableFormatEnum
    table_content: str = ""
    table_content_format: Union[TableFormatEnum, str] = ""
    table_location: tuple = (0, 0, 0, 0)
    table_location_max_dimensions: tuple = (0, 0)
    uploaded_image_uri: str = ""


class ChartMetadataSchema(BaseModelNoExt):
    caption: str = ""
    table_format: TableFormatEnum
    table_content: str = ""
    table_content_format: Union[TableFormatEnum, str] = ""
    table_location: tuple = (0, 0, 0, 0)
    table_location_max_dimensions: tuple = (0, 0)
    uploaded_image_uri: str = ""


class AudioMetadataSchema(BaseModelNoExt):
    audio_transcript: str = ""
    audio_type: str = ""


# TODO consider deprecating this in favor of info msg...
class ErrorMetadataSchema(BaseModelNoExt):
    task: TaskTypeEnum
    status: StatusEnum
    source_id: str = ""
    error_msg: str


class InfoMessageMetadataSchema(BaseModelNoExt):
    task: TaskTypeEnum
    status: StatusEnum
    message: str
    filter: bool


# Main metadata schema
class MetadataSchema(BaseModelNoExt):
    content: str = ""
    content_url: str = ""
    embedding: Optional[List[float]] = None
    source_metadata: Optional[SourceMetadataSchema] = None
    content_metadata: Optional[ContentMetadataSchema] = None
    audio_metadata: Optional[AudioMetadataSchema] = None
    text_metadata: Optional[TextMetadataSchema] = None
    image_metadata: Optional[ImageMetadataSchema] = None
    table_metadata: Optional[TableMetadataSchema] = None
    chart_metadata: Optional[ChartMetadataSchema] = None
    error_metadata: Optional[ErrorMetadataSchema] = None
    info_message_metadata: Optional[InfoMessageMetadataSchema] = None
    debug_metadata: Optional[Dict[str, Any]] = None
    raise_on_failure: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_metadata_type(cls, values):
        content_type = values.get("content_metadata", {}).get("type", None)
        if content_type != ContentTypeEnum.AUDIO:
            values["audio_metadata"] = None
        if content_type != ContentTypeEnum.IMAGE:
            values["image_metadata"] = None
        if content_type != ContentTypeEnum.TEXT:
            values["text_metadata"] = None
        if content_type != ContentTypeEnum.STRUCTURED:
            values["table_metadata"] = None
        return values


def validate_metadata(metadata: Dict[str, Any]) -> MetadataSchema:
    """
    Validates the given metadata dictionary against the MetadataSchema.

    Parameters:
    - metadata: A dictionary representing metadata to be validated.

    Returns:
    - An instance of MetadataSchema if validation is successful.

    Raises:
    - ValidationError: If the metadata does not conform to the schema.
    """
    return MetadataSchema(**metadata)
