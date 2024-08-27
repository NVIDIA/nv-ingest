# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import root_validator
from pydantic import validator

from nv_ingest.schemas.base_model_noext import BaseModelNoExt
from nv_ingest.util.converters import datetools


# Do we want types and similar items to be enums or just strings?
class SourceTypeEnum(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    source_type_1 = "source_type_1"
    source_type_2 = "source_type_2"


class AccessLevelEnum(int, Enum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


class ContentTypeEnum(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    STRUCTURED = "structured"
    INFO_MSG = "info_message"


class StdContentDescEnum(str, Enum):
    PDF_TEXT = "Unstructured text from PDF document."
    PDF_IMAGE = "Image extracted from PDF document."
    PDF_TABLE = "Structured table extracted from PDF document."
    PDF_CHART = "Structured chart extracted from PDF document."
    DOCX_TEXT = "Unstructured text from DOCX document."
    DOCX_IMAGE = "Image extracted from DOCX document."
    DOCX_TABLE = "Structured table extracted from DOCX document."
    PPTX_TEXT = "Unstructured text from PPTX presentation."
    PPTX_IMAGE = "Image extracted from PPTX presentation."
    PPTX_TABLE = "Structured table extracted from PPTX presentation."


class TextTypeEnum(str, Enum):
    HEADER = "header"
    BODY = "body"
    SPAN = "span"
    LINE = "line"
    BLOCK = "block"
    PAGE = "page"
    DOCUMENT = "document"
    NEARBY_BLOCK = "nearby_block"
    OTHER = "other"


class LanguageEnum(str, Enum):
    AF = "af"
    AR = "ar"
    BG = "bg"
    BN = "bn"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    FA = "fa"
    FI = "fi"
    FR = "fr"
    GU = "gu"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HU = "hu"
    ID = "id"
    IT = "it"
    JA = "ja"
    KN = "kn"
    KO = "ko"
    LT = "lt"
    LV = "lv"
    MK = "mk"
    ML = "ml"
    MR = "mr"
    NE = "ne"
    NL = "nl"
    NO = "no"
    PA = "pa"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SK = "sk"
    SL = "sl"
    SO = "so"
    SQ = "sq"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TH = "th"
    TL = "tl"
    TR = "tr"
    UK = "uk"
    UR = "ur"
    VI = "vi"
    ZH_CN = "zh-cn"
    ZH_TW = "zh-tw"
    UNKNOWN = "unknown"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ImageTypeEnum(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    GIF = "gif"
    TIFF = "tiff"

    image_type_1 = "image_type_1"  # until classifier developed
    image_type_2 = "image_type_2"  # until classifier developed

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class TableFormatEnum(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    LATEX = "latex"
    IMAGE = "image"


class TaskTypeEnum(str, Enum):
    CAPTION = "caption"
    EMBED = "embed"
    EXTRACT = "extract"
    FILTER = "filter"
    SPLIT = "split"
    TRANSFORM = "transform"


class StatusEnum(str, Enum):
    ERROR: str = "error"
    SUCCESS: str = "success"


class ContentSubtypeEnum(str, Enum):
    TABLE = "table"
    CHART = "chart"


# Sub schemas
class SourceMetadataSchema(BaseModelNoExt):
    """
    Schema for the knowledge base file from which content
    and metadata is extracted.
    """

    source_name: str
    source_id: str
    source_location: str = ""
    source_type: Union[SourceTypeEnum, str]
    collection_id: str = ""
    date_created: str = datetime.now().isoformat()
    last_modified: str = datetime.now().isoformat()
    summary: str = ""
    partition_id: int = -1
    access_level: Union[AccessLevelEnum, int] = -1

    @validator("date_created", "last_modified")
    @classmethod
    def validate_fields(cls, field_value):
        datetools.validate_iso8601(field_value)
        return field_value


class NearbyObjectsSubSchema(BaseModelNoExt):
    """
    Schema to hold related extracted object
    """

    content: List[str] = []
    bbox: List[tuple] = []


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
    subtype: Union[ContentSubtypeEnum, str] = ""


class TextMetadataSchema(BaseModelNoExt):
    text_type: TextTypeEnum
    summary: str = ""
    keywords: Union[str, List[str], Dict] = ""
    language: LanguageEnum = "en"  # default to Unknown? Maybe do some kind of heuristic check
    text_location: tuple = (0, 0, 0, 0)


class ImageMetadataSchema(BaseModelNoExt):
    image_type: Union[ImageTypeEnum, str]
    structured_image_type: ImageTypeEnum = ImageTypeEnum.image_type_1
    caption: str = ""
    text: str = ""
    image_location: tuple = (0, 0, 0, 0)
    uploaded_image_url: str = ""
    width: int = 0
    height: int = 0


class TableMetadataSchema(BaseModelNoExt):
    caption: str = ""
    table_format: TableFormatEnum
    table_content: str = ""
    table_location: tuple = (0, 0, 0, 0)
    uploaded_image_uri: str = ""


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
    embedding: Optional[List[float]] = None
    source_metadata: Optional[SourceMetadataSchema] = None
    content_metadata: Optional[ContentMetadataSchema] = None
    text_metadata: Optional[TextMetadataSchema] = None
    image_metadata: Optional[ImageMetadataSchema] = None
    table_metadata: Optional[TableMetadataSchema] = None
    error_metadata: Optional[ErrorMetadataSchema] = None
    info_message_metadata: Optional[InfoMessageMetadataSchema] = None
    debug_metadata: Optional[Dict[str, Any]] = None
    raise_on_failure: bool = False

    @root_validator(pre=True)
    def check_metadata_type(cls, values):
        content_type = values.get("content_metadata", {}).get("type", None)
        if content_type != ContentTypeEnum.TEXT:
            values["text_metadata"] = None
        if content_type != ContentTypeEnum.IMAGE:
            values["image_metadata"] = None
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
