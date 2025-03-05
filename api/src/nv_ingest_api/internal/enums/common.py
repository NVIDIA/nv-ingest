# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from enum import Enum
from typing import Type, Any

logger = logging.getLogger(__name__)


class SourceTypeEnum(str, Enum):
    """
    Enum for representing different source file types.

    Attributes
    ----------
    PDF : str
        Represents a PDF file type.
    DOCX : str
        Represents a DOCX file type.
    PPTX : str
        Represents a PPTX file type.
    source_type_1 : str
        Represents a custom source type 1.
    source_type_2 : str
        Represents a custom source type 2.
    """

    PDF: str = "pdf"
    DOCX: str = "docx"
    PPTX: str = "pptx"
    source_type_1: str = "source_type_1"
    source_type_2: str = "source_type_2"


class AccessLevelEnum(int, Enum):
    """
    Note: This is for future use, and currently has no functional use case.

    Enum for representing different access levels.

    Attributes
    ----------
    LEVEL_1 : int
        Represents access level 1.
    LEVEL_2 : int
        Represents access level 2.
    LEVEL_3 : int
        Represents access level 3.
    """

    LEVEL_1: int = 1
    LEVEL_2: int = 2
    LEVEL_3: int = 3


class ContentTypeEnum(str, Enum):
    """
    Enum for representing various content types.

    Attributes
    ----------
    AUDIO : str
        Represents audio content.
    EMBEDDING : str
        Represents embedding content.
    IMAGE : str
        Represents image content.
    INFO_MSG : str
        Represents an informational message.
    STRUCTURED : str
        Represents structured content.
    TEXT : str
        Represents text content.
    UNSTRUCTURED : str
        Represents unstructured content.
    VIDEO : str
        Represents video content.
    """

    AUDIO: str = "audio"
    EMBEDDING: str = "embedding"
    IMAGE: str = "image"
    INFO_MSG: str = "info_message"
    STRUCTURED: str = "structured"
    TEXT: str = "text"
    VIDEO: str = "video"


class StdContentDescEnum(str, Enum):
    """
    Enum for standard content descriptions extracted from different source types.

    Attributes
    ----------
    DOCX_IMAGE : str
        Description for image extracted from DOCX document.
    DOCX_TABLE : str
        Description for structured table extracted from DOCX document.
    DOCX_TEXT : str
        Description for unstructured text from DOCX document.
    PDF_CHART : str
        Description for structured chart extracted from PDF document.
    PDF_IMAGE : str
        Description for image extracted from PDF document.
    PDF_INFOGRAPHIC : str
        Description for structured infographic extracted from PDF document.
    PDF_TABLE : str
        Description for structured table extracted from PDF document.
    PDF_TEXT : str
        Description for unstructured text from PDF document.
    PPTX_IMAGE : str
        Description for image extracted from PPTX presentation.
    PPTX_TABLE : str
        Description for structured table extracted from PPTX presentation.
    PPTX_TEXT : str
        Description for unstructured text from PPTX presentation.
    """

    DOCX_IMAGE: str = "Image extracted from DOCX document."
    DOCX_TABLE: str = "Structured table extracted from DOCX document."
    DOCX_TEXT: str = "Unstructured text from DOCX document."
    PDF_CHART: str = "Structured chart extracted from PDF document."
    PDF_IMAGE: str = "Image extracted from PDF document."
    PDF_INFOGRAPHIC: str = "Structured infographic extracted from PDF document."
    PDF_TABLE: str = "Structured table extracted from PDF document."
    PDF_TEXT: str = "Unstructured text from PDF document."
    PPTX_IMAGE: str = "Image extracted from PPTX presentation."
    PPTX_TABLE: str = "Structured table extracted from PPTX presentation."
    PPTX_TEXT: str = "Unstructured text from PPTX presentation."


class TextTypeEnum(str, Enum):
    """
    Enum for representing different types of text segments.

    Attributes
    ----------
    BLOCK : str
        Represents a text block.
    BODY : str
        Represents body text.
    DOCUMENT : str
        Represents an entire document.
    HEADER : str
        Represents a header text.
    LINE : str
        Represents a single line of text.
    NEARBY_BLOCK : str
        Represents a block of text in close proximity to another.
    OTHER : str
        Represents other unspecified text type.
    PAGE : str
        Represents a page of text.
    SPAN : str
        Represents an inline text span.
    """

    BLOCK: str = "block"
    BODY: str = "body"
    DOCUMENT: str = "document"
    HEADER: str = "header"
    LINE: str = "line"
    NEARBY_BLOCK: str = "nearby_block"
    OTHER: str = "other"
    PAGE: str = "page"
    SPAN: str = "span"


class LanguageEnum(str, Enum):
    """
    Enum for representing various language codes.

    Attributes
    ----------
    AF : str
        Afrikaans language code.
    AR : str
        Arabic language code.
    BG : str
        Bulgarian language code.
    BN : str
        Bengali language code.
    CA : str
        Catalan language code.
    CS : str
        Czech language code.
    CY : str
        Welsh language code.
    DA : str
        Danish language code.
    DE : str
        German language code.
    EL : str
        Greek language code.
    EN : str
        English language code.
    ES : str
        Spanish language code.
    ET : str
        Estonian language code.
    FA : str
        Persian language code.
    FI : str
        Finnish language code.
    FR : str
        French language code.
    GU : str
        Gujarati language code.
    HE : str
        Hebrew language code.
    HI : str
        Hindi language code.
    HR : str
        Croatian language code.
    HU : str
        Hungarian language code.
    ID : str
        Indonesian language code.
    IT : str
        Italian language code.
    JA : str
        Japanese language code.
    KN : str
        Kannada language code.
    KO : str
        Korean language code.
    LT : str
        Lithuanian language code.
    LV : str
        Latvian language code.
    MK : str
        Macedonian language code.
    ML : str
        Malayalam language code.
    MR : str
        Marathi language code.
    NE : str
        Nepali language code.
    NL : str
        Dutch language code.
    NO : str
        Norwegian language code.
    PA : str
        Punjabi language code.
    PL : str
        Polish language code.
    PT : str
        Portuguese language code.
    RO : str
        Romanian language code.
    RU : str
        Russian language code.
    SK : str
        Slovak language code.
    SL : str
        Slovenian language code.
    SO : str
        Somali language code.
    SQ : str
        Albanian language code.
    SV : str
        Swedish language code.
    SW : str
        Swahili language code.
    TA : str
        Tamil language code.
    TE : str
        Telugu language code.
    TH : str
        Thai language code.
    TL : str
        Tagalog language code.
    TR : str
        Turkish language code.
    UK : str
        Ukrainian language code.
    UR : str
        Urdu language code.
    VI : str
        Vietnamese language code.
    ZH_CN : str
        Chinese (Simplified) language code.
    ZH_TW : str
        Chinese (Traditional) language code.
    UNKNOWN : str
        Represents an unknown language.
    """

    AF: str = "af"
    AR: str = "ar"
    BG: str = "bg"
    BN: str = "bn"
    CA: str = "ca"
    CS: str = "cs"
    CY: str = "cy"
    DA: str = "da"
    DE: str = "de"
    EL: str = "el"
    EN: str = "en"
    ES: str = "es"
    ET: str = "et"
    FA: str = "fa"
    FI: str = "fi"
    FR: str = "fr"
    GU: str = "gu"
    HE: str = "he"
    HI: str = "hi"
    HR: str = "hr"
    HU: str = "hu"
    ID: str = "id"
    IT: str = "it"
    JA: str = "ja"
    KN: str = "kn"
    KO: str = "ko"
    LT: str = "lt"
    LV: str = "lv"
    MK: str = "mk"
    ML: str = "ml"
    MR: str = "mr"
    NE: str = "ne"
    NL: str = "nl"
    NO: str = "no"
    PA: str = "pa"
    PL: str = "pl"
    PT: str = "pt"
    RO: str = "ro"
    RU: str = "ru"
    SK: str = "sk"
    SL: str = "sl"
    SO: str = "so"
    SQ: str = "sq"
    SV: str = "sv"
    SW: str = "sw"
    TA: str = "ta"
    TE: str = "te"
    TH: str = "th"
    TL: str = "tl"
    TR: str = "tr"
    UK: str = "uk"
    UR: str = "ur"
    VI: str = "vi"
    ZH_CN: str = "zh-cn"
    ZH_TW: str = "zh-tw"
    UNKNOWN: str = "unknown"

    @classmethod
    def has_value(cls: Type["LanguageEnum"], value: Any) -> bool:
        """
        Check if the enum contains the given value.

        Parameters
        ----------
        value : Any
            The value to check against the enum members.

        Returns
        -------
        bool
            True if value exists in the enum, False otherwise.
        """
        return value in cls._value2member_map_


class ImageTypeEnum(str, Enum):
    """
    Enum for representing different image file types.

    Attributes
    ----------
    BMP : str
        Represents BMP image type.
    GIF : str
        Represents GIF image type.
    JPEG : str
        Represents JPEG image type.
    PNG : str
        Represents PNG image type.
    TIFF : str
        Represents TIFF image type.
    image_type_1 : str
        Custom image type 1 (until classifier developed).
    image_type_2 : str
        Custom image type 2 (until classifier developed).
    """

    BMP: str = "bmp"
    GIF: str = "gif"
    JPEG: str = "jpeg"
    PNG: str = "png"
    TIFF: str = "tiff"
    image_type_1: str = "image_type_1"  # until classifier developed
    image_type_2: str = "image_type_2"  # until classifier developed

    @classmethod
    def has_value(cls: Type["ImageTypeEnum"], value: Any) -> bool:
        """
        Check if the enum contains the given image type value.

        Parameters
        ----------
        value : Any
            The value to check against the enum members.

        Returns
        -------
        bool
            True if value exists in the enum, False otherwise.
        """
        return value in cls._value2member_map_


class TableFormatEnum(str, Enum):
    """
    Enum for representing table formats.

    Attributes
    ----------
    HTML : str
        Represents HTML table format.
    IMAGE : str
        Represents image table format.
    LATEX : str
        Represents LaTeX table format.
    MARKDOWN : str
        Represents Markdown table format.
    PSEUDO_MARKDOWN : str
        Represents pseudo Markdown table format.
    SIMPLE : str
        Represents simple table format.
    """

    HTML: str = "html"
    IMAGE: str = "image"
    LATEX: str = "latex"
    MARKDOWN: str = "markdown"
    PSEUDO_MARKDOWN: str = "pseudo_markdown"
    SIMPLE: str = "simple"


class PrimaryTaskTypeEnum(str, Enum):
    """
    Enum for representing different types of tasks.

    Attributes
    ----------
    CAPTION : str
        Represents a captioning task.
    EMBED : str
        Represents an embedding task.
    EXTRACT : str
        Represents an extraction task.
    FILTER : str
        Represents a filtering task.
    SPLIT : str
        Represents a splitting task.
    TRANSFORM : str
        Represents a transforming task.
    """

    CAPTION: str = "caption"
    EMBED: str = "embed"
    EXTRACT: str = "extract"
    FILTER: str = "filter"
    SPLIT: str = "split"
    TRANSFORM: str = "transform"


class StatusEnum(str, Enum):
    """
    Enum for representing status messages.

    Attributes
    ----------
    ERROR : str
        Represents an error status.
    SUCCESS : str
        Represents a success status.
    """

    ERROR: str = "error"
    SUCCESS: str = "success"


class ContentSubtypeEnum(str, Enum):
    """
    Enum for representing different content subtypes.

    Attributes
    ----------
    TABLE : str
        Represents a table subtype.
    CHART : str
        Represents a chart subtype.
    INFOGRAPHIC : str
        Represents an infographic subtype.
    """

    TABLE: str = "table"
    CHART: str = "chart"
    INFOGRAPHIC: str = "infographic"


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
    mp3 = "mp3"
    wav = "wav"


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
    audio_data_extract = "audio_data_extract"
    table_data_extract = "table_data_extract"
    chart_data_extract = "chart_data_extract"
    infographic_data_extract = "infographic_data_extract"
