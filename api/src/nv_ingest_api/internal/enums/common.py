# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from enum import Enum
from typing import Type, Any

logger = logging.getLogger(__name__)


class AccessLevelEnum(int, Enum):
    """
    Note
    ----
    This is for future use, and currently has no functional use case.

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

    UNKNOWN: int = -1
    LEVEL_1: int = 1
    LEVEL_2: int = 2
    LEVEL_3: int = 3


class ContentDescriptionEnum(str, Enum):
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
    PDF_PAGE_IMAGE : str
        Description for a full-page image rendered from a PDF document.
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
    PDF_PAGE_IMAGE: str = "Full-page image rendered from a PDF document."
    PDF_TABLE: str = "Structured table extracted from PDF document."
    PDF_TEXT: str = "Unstructured text from PDF document."
    PPTX_IMAGE: str = "Image extracted from PPTX presentation."
    PPTX_TABLE: str = "Structured table extracted from PPTX presentation."
    PPTX_TEXT: str = "Unstructured text from PPTX presentation."


class ContentTypeEnum(str, Enum):
    """
    Enum for representing various content types.

    Note: Content type declares the broad category of the content, such as text, image, audio, etc.
    This is not equivalent to the Document type, which is a specific file format.

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
    PAGE_IMAGE : str
        Represents a full-page image rendered from a document.
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
    CHART: str = "chart"
    EMBEDDING: str = "embedding"
    IMAGE: str = "image"
    INFOGRAPHIC: str = "infographic"
    INFO_MSG: str = "info_message"
    NONE: str = "none"
    PAGE_IMAGE: str = "page_image"
    STRUCTURED: str = "structured"
    TABLE: str = "table"
    TEXT: str = "text"
    UNKNOWN: str = "unknown"
    VIDEO: str = "video"


class DocumentTypeEnum(str, Enum):
    """
    Enum for representing various document file types.

    Note: Document type refers to the specific file format of the content, such as PDF, DOCX, etc.
    This is not equivalent to the Content type, which is a broad category of the content.

    Attributes
    ----------
    BMP: str
        BMP image format.
    DOCX: str
        Microsoft Word document format.
    HTML: str
        HTML document.
    JPEG: str
        JPEG image format.
    PDF: str
        PDF document format.
    PNG: str
        PNG image format.
    PPTX: str
        PowerPoint presentation format.
    SVG: str
        SVG image format.
    TIFF: str
        TIFF image format.
    TXT: str
        Plain text file.
    MP3: str
        MP3 audio format.
    WAV: str
        WAV audio format.
    MP4: str
        MP4 video format.
    MOV: str
        MOV video format.
    AVI: str
        AVI video format.
    MKV: str
        MKV video format.
    """

    BMP: str = "bmp"
    DOCX: str = "docx"
    HTML: str = "html"
    JPEG: str = "jpeg"
    PDF: str = "pdf"
    PNG: str = "png"
    PPTX: str = "pptx"
    SVG: str = "svg"
    TIFF: str = "tiff"
    TXT: str = "text"
    MD: str = "text"
    MP3: str = "mp3"
    WAV: str = "wav"
    MP4: str = "mp4"
    MOV: str = "mov"
    AVI: str = "avi"
    MKV: str = "mkv"
    UNKNOWN: str = "unknown"


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
            True if the value exists in the enum, False otherwise.
        """
        return value in cls._value2member_map_


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


class PipelinePhase(int, Enum):
    """
    The logical phase of a pipeline stage.

    Attributes
    ----------
    PRE_PROCESSING : int
        Pre-processing phase.
    EXTRACTION : int
        Extraction phase.
    POST_PROCESSING : int
        Post-processing phase.
    MUTATION : int
        Mutation phase.
    TRANSFORM : int
        Transform phase.
    RESPONSE : int
        Response phase.
    TELEMETRY : int
        Telemetry phase.
    DRAIN : int
        Drain phase.
    """

    PRE_PROCESSING = 0
    EXTRACTION = 1
    POST_PROCESSING = 2
    MUTATION = 3
    TRANSFORM = 4
    RESPONSE = 5
    TELEMETRY = 6
    DRAIN = 7


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


class TaskTypeEnum(str, Enum):
    """
    Enum for representing various task types.

    Attributes
    ----------
    CAPTION : str
        Represents a caption task.
    DEDUP : str
        Represents a deduplication task.
    EMBED : str
        Represents an embedding task.
    EXTRACT : str
        Represents an extraction task.
    FILTER : str
        Represents a filtering task.
    SPLIT : str
        Represents a splitting task.
    STORE : str
        Represents a storing task.
    STORE_EMBEDDING : str
        Represents a task for storing embeddings.
    VDB_UPLOAD : str
        Represents a task for uploading to a vector database.
    AUDIO_DATA_EXTRACT : str
        Represents a task for extracting audio data.
    TABLE_DATA_EXTRACT : str
        Represents a task for extracting table data.
    CHART_DATA_EXTRACT : str
        Represents a task for extracting chart data.
    INFOGRAPHIC_DATA_EXTRACT : str
        Represents a task for extracting infographic data.
    UDF : str
        Represents a user-defined function task.
    """

    AUDIO_DATA_EXTRACT: str = "audio_data_extract"
    CAPTION: str = "caption"
    CHART_DATA_EXTRACT: str = "chart_data_extract"
    DEDUP: str = "dedup"
    EMBED: str = "embed"
    EXTRACT: str = "extract"
    FILTER: str = "filter"
    INFOGRAPHIC_DATA_EXTRACT: str = "infographic_data_extract"
    OCR_DATA_EXTRACT: str = "ocr_data_extract"
    SPLIT: str = "split"
    STORE_EMBEDDING: str = "store_embedding"
    STORE: str = "store"
    TABLE_DATA_EXTRACT: str = "table_data_extract"
    UDF: str = "udf"
    VDB_UPLOAD: str = "vdb_upload"


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
