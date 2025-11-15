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
    """The name of the source file."""

    source_id: str
    """The ID of the source file."""

    source_location: str = ""
    """The URL, URI, or pointer to the storage location of the source file."""

    source_type: Union[DocumentTypeEnum, str]
    """The type of the source file, such as pdf, docx, pptx, or txt."""

    collection_id: str = ""
    """The ID of the collection in which the source is contained."""

    date_created: str = datetime.now().isoformat()
    """The date the source was created."""

    last_modified: str = datetime.now().isoformat()
    """The date the source was last modified."""

    summary: str = ""
    """A summary of the source."""

    partition_id: int = -1
    """The offset of this data fragment within a larger set of fragments."""

    access_level: Union[AccessLevelEnum, int] = AccessLevelEnum.UNKNOWN
    """The role-based access control for the source."""

    custom_content: Optional[Dict[str, Any]] = None

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
    """The type of the content. Text, Image, Structured, Table, or Chart."""

    description: str = ""
    """A text description of the content object."""

    page_number: int = -1
    """The page number of the content in the source."""

    hierarchy: ContentHierarchySchema = ContentHierarchySchema()
    """The location or order of the content within the source."""

    subtype: Union[ContentTypeEnum, str] = ""
    """The type of the content for structured data types, such as table or chart."""

    start_time: int = -1
    """The timestamp of the start of a piece of audio content."""

    end_time: int = -1
    """The timestamp of the end of a piece of audio content."""

    custom_content: Optional[Dict[str, Any]] = None


class TextMetadataSchema(BaseModelNoExt):
    """
    The schema for the extracted text content.
    """

    text_type: TextTypeEnum
    """The type of the text, such as header or body."""

    summary: str = ""
    """An abbreviated summary of the content."""

    keywords: Union[str, List[str], Dict] = ""
    """Keywords, named entities, or other phrases."""

    language: LanguageEnum = LanguageEnum.EN  # default to Unknown? Maybe do some kind of heuristic check
    """The language of the content."""

    text_location: tuple = (0, 0, 0, 0)
    """The bounding box of the text, in the format (x1,y1,x2,y2)."""

    text_location_max_dimensions: tuple = (0, 0)
    """The maximum dimensions of the bounding box of the text, in the format (x_max,y_max)."""

    custom_content: Optional[Dict[str, Any]] = None


class ImageMetadataSchema(BaseModelNoExt):
    """
    The schema for the extracted image content.
    """

    image_type: Union[DocumentTypeEnum, str]
    """The type of the image, such as structured, natural, hybrid, and others."""

    structured_image_type: ContentTypeEnum = ContentTypeEnum.NONE
    """The type of the content for structured data types, such as bar chart, pie chart, and others."""

    caption: str = ""
    """A caption or subheading associated with the image."""

    text: str = ""
    """Extracted text from a structured chart."""

    image_location: tuple = (0, 0, 0, 0)
    """The bounding box of the image, in the format (x1,y1,x2,y2)."""

    image_location_max_dimensions: tuple = (0, 0)
    """The maximum dimensions of the bounding box of the image, in the format (x_max,y_max)."""

    uploaded_image_url: str = ""
    """A mirror of source_metadata.source_location."""

    width: int = 0
    """The width of the image."""

    height: int = 0
    """The height of the image."""

    custom_content: Optional[Dict[str, Any]] = None

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
    """
    The schema for the extracted table content.
    """

    caption: str = ""
    """The caption for the table."""

    table_format: TableFormatEnum
    """
    The format of the table.  One of Structured (dataframe / lists of rows and columns), or serialized as markdown,
    html, latex, simple (cells separated as spaces).
    """

    table_content: str = ""
    """Extracted text content, formatted according to table_metadata.table_format."""

    table_content_format: Union[TableFormatEnum, str] = ""

    table_location: tuple = (0, 0, 0, 0)
    """The bounding box of the table, in the format (x1,y1,x2,y2)."""

    table_location_max_dimensions: tuple = (0, 0)
    """The maximum dimensions of the bounding box of the table, in the format (x_max,y_max)."""

    uploaded_image_uri: str = ""
    """A mirror of source_metadata.source_location."""

    custom_content: Optional[Dict[str, Any]] = None


class ChartMetadataSchema(BaseModelNoExt):
    """
    The schema for table content extracted from charts.
    """

    caption: str = ""
    """The caption for the chart."""

    table_format: TableFormatEnum
    """
    The format of the table.  One of Structured (dataframe / lists of rows and columns), or serialized as markdown,
    html, latex, simple (cells separated as spaces).
    """

    table_content: str = ""
    """Extracted text content, formatted according to chart_metadata.table_format."""

    table_content_format: Union[TableFormatEnum, str] = ""

    table_location: tuple = (0, 0, 0, 0)
    """The bounding box of the chart, in the format (x1,y1,x2,y2)."""

    table_location_max_dimensions: tuple = (0, 0)
    """The maximum dimensions of the bounding box of the chart, in the format (x_max,y_max)."""

    uploaded_image_uri: str = ""
    """A mirror of source_metadata.source_location."""

    custom_content: Optional[Dict[str, Any]] = None


class AudioMetadataSchema(BaseModelNoExt):
    """
    The schema for extracted audio content.
    """

    audio_transcript: str = ""
    """A transcript of the audio content."""

    audio_type: str = ""
    """The type or format of the audio, such as mp3, wav."""

    custom_content: Optional[Dict[str, Any]] = None


# TODO consider deprecating this in favor of info msg...
class ErrorMetadataSchema(BaseModelNoExt):
    task: TaskTypeEnum
    status: StatusEnum
    source_id: str = ""
    error_msg: str
    custom_content: Optional[Dict[str, Any]] = None


class InfoMessageMetadataSchema(BaseModelNoExt):
    task: TaskTypeEnum
    status: StatusEnum
    message: str
    filter: bool
    custom_content: Optional[Dict[str, Any]] = None


# Main metadata schema
class MetadataSchema(BaseModelNoExt):
    """
    The primary container schema for extraction results.
    """

    content: str = ""
    """The actual textual content extracted from the source."""

    content_url: str = ""
    """A URL that points to the location of the content, if applicable."""

    embedding: Optional[List[float]] = None
    """An optional numerical vector representation (embedding) of the content."""

    source_metadata: Optional[SourceMetadataSchema] = None
    """Metadata about the original source of the content."""

    content_metadata: Optional[ContentMetadataSchema] = None
    """General metadata about the extracted content itself."""

    audio_metadata: Optional[AudioMetadataSchema] = None
    """Specific metadata for audio content. Automatically set to None if content_metadata.type is not AUDIO."""

    text_metadata: Optional[TextMetadataSchema] = None
    """Specific metadata for text content. Automatically set to None if content_metadata.type is not TEXT."""

    image_metadata: Optional[ImageMetadataSchema] = None
    """Specific metadata for image content. Automatically set to None if content_metadata.type is not IMAGE."""

    table_metadata: Optional[TableMetadataSchema] = None
    """Specific metadata for tabular content. Automatically set to None if content_metadata.type is not STRUCTURED."""

    chart_metadata: Optional[ChartMetadataSchema] = None
    """Specific metadata for chart content. Automatically set to None if content_metadata.type is not STRUCTURED."""

    error_metadata: Optional[ErrorMetadataSchema] = None
    """Metadata that describes any errors encountered during processing."""

    info_message_metadata: Optional[InfoMessageMetadataSchema] = None
    """Informational messages related to the processing."""

    debug_metadata: Optional[Dict[str, Any]] = None
    """A dictionary for storing any arbitrary debug information."""

    raise_on_failure: bool = False
    """If True, indicates that processing should halt on failure."""

    total_pages: Optional[int] = None
    """Total number of pages in the source document (V2 API)."""

    original_source_id: Optional[str] = None
    """The original source identifier before any splitting or chunking (V2 API)."""

    original_source_name: Optional[str] = None
    """The original source name before any splitting or chunking (V2 API)."""

    custom_content: Optional[Dict[str, Any]] = None

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
