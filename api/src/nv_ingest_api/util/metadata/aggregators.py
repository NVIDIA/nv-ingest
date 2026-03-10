# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import io
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import pypdfium2 as pdfium
from PIL import Image
from pypdfium2 import PdfImage

from nv_ingest_api.internal.enums.common import ContentDescriptionEnum, DocumentTypeEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import NearbyObjectsSchema
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
from nv_ingest_api.util.converters import datetools
from nv_ingest_api.util.detectors.language import detect_language
from nv_ingest_api.util.exception_handlers.pdf import pdfium_exception_handler


@dataclass
class CroppedImageWithContent:
    content: str
    image: str
    bbox: Tuple[int, int, int, int]
    max_width: int
    max_height: int
    type_string: str
    content_format: str = ""


@dataclass
class LatexTable:
    latex: pd.DataFrame
    bbox: Tuple[int, int, int, int]
    max_width: int
    max_height: int


@dataclass
class Base64Image:
    image: str
    bbox: Tuple[int, int, int, int]
    width: int
    height: int
    max_width: int
    max_height: int


@dataclass
class PDFMetadata:
    """
    A data object to store metadata information extracted from a PDF document.
    """

    page_count: int
    filename: str
    last_modified: str
    date_created: str
    keywords: List[str]
    source_type: str = "PDF"


def extract_pdf_metadata(doc: pdfium.PdfDocument, source_id: str) -> PDFMetadata:
    """
    Extracts metadata and relevant information from a PDF document.

    Parameters
    ----------
    pdf_stream : bytes
        The PDF document data as a byte stream.
    source_id : str
        The identifier for the source document, typically the filename.

    Returns
    -------
    PDFMetadata
        An object containing extracted metadata and information including:
        - `page_count`: The total number of pages in the PDF.
        - `filename`: The source filename or identifier.
        - `last_modified`: The last modified date of the PDF document.
        - `date_created`: The creation date of the PDF document.
        - `keywords`: Keywords associated with the PDF document.
        - `source_type`: The type/format of the source, e.g., "PDF".

    Raises
    ------
    PdfiumError
        If there is an issue processing the PDF document.
    """
    page_count: int = len(doc)
    filename: str = source_id

    # Extract document metadata
    doc_meta = doc.get_metadata_dict()

    # Extract and process the last modified date
    last_modified: str = doc_meta.get("ModDate")
    if last_modified in (None, ""):
        last_modified = datetools.remove_tz(datetime.now()).isoformat()
    else:
        last_modified = datetools.datetimefrompdfmeta(last_modified)

    # Extract and process the creation date
    date_created: str = doc_meta.get("CreationDate")
    if date_created in (None, ""):
        date_created = datetools.remove_tz(datetime.now()).isoformat()
    else:
        date_created = datetools.datetimefrompdfmeta(date_created)

    # Extract keywords, defaulting to an empty list if not found
    keywords: List[str] = doc_meta.get("Keywords", [])

    # Create the PDFMetadata object
    metadata = PDFMetadata(
        page_count=page_count,
        filename=filename,
        last_modified=last_modified,
        date_created=date_created,
        keywords=keywords,
    )

    return metadata


def construct_text_metadata(
    accumulated_text,
    keywords,
    page_idx,
    block_idx,
    line_idx,
    span_idx,
    page_count,
    text_depth,
    source_metadata,
    base_unified_metadata,
    delimiter=" ",
    bbox_max_dimensions: Tuple[int, int] = (-1, -1),
    nearby_objects: Optional[Dict[str, Any]] = None,
):
    extracted_text = delimiter.join(accumulated_text)

    content_metadata = {
        "type": ContentTypeEnum.TEXT,
        "description": ContentDescriptionEnum.PDF_TEXT,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": -1,
            "line": -1,
            "span": -1,
            "nearby_objects": nearby_objects or NearbyObjectsSchema(),
        },
    }

    language = detect_language(extracted_text)

    # TODO(Devin) - Implement bounding box logic for text
    bbox = (-1, -1, -1, -1)

    text_metadata = {
        "text_type": text_depth,
        "summary": "",
        "keywords": keywords,
        "language": language,
        "text_location": bbox,
        "text_location_max_dimensions": bbox_max_dimensions,
    }

    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": extracted_text,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "text_metadata": text_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.TEXT, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


def construct_image_metadata_from_base64(
    base64_image: str,
    page_idx: int,
    page_count: int,
    source_metadata: Dict[str, Any],
    base_unified_metadata: Dict[str, Any],
    subtype: None | ContentTypeEnum | str = "",
    text: str = "",
) -> List[Any]:
    """
    Extracts image data from a base64-encoded image string, decodes the image to get
    its dimensions and bounding box, and constructs metadata for the image.

    Parameters
    ----------
    base64_image : str
        A base64-encoded string representing the image.
    page_idx : int
        The index of the current page being processed.
    page_count : int
        The total number of pages in the PDF document.
    source_metadata : Dict[str, Any]
        Metadata related to the source of the PDF document.
    base_unified_metadata : Dict[str, Any]
        The base unified metadata structure to be updated with the extracted image information.

    Returns
    -------
    List[Any]
        A list containing the content type, validated metadata dictionary, and a UUID string.

    Raises
    ------
    ValueError
        If the image cannot be decoded from the base64 string.
    """
    # Decode the base64 image
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Failed to decode image from base64: {e}")

    # Extract image dimensions and bounding box
    width, height = image.size
    bbox = (0, 0, width, height)  # Assuming the full image as the bounding box

    # Construct content metadata
    content_metadata: Dict[str, Any] = {
        "type": ContentTypeEnum.IMAGE,
        "description": ContentDescriptionEnum.PDF_IMAGE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": -1,
            "line": -1,
            "span": -1,
        },
        "subtype": subtype or "",
    }

    # Construct image metadata
    image_metadata: Dict[str, Any] = {
        "image_type": DocumentTypeEnum.PNG,
        "structured_image_type": ContentTypeEnum.UNKNOWN,
        "caption": "",
        "text": text,
        "image_location": bbox,
        "image_location_max_dimensions": (width, height),
        "height": height,
    }

    # Update the unified metadata with the extracted image information
    unified_metadata: Dict[str, Any] = base_unified_metadata.copy()
    unified_metadata.update(
        {
            "content": base64_image,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    # Validate and return the unified metadata
    validated_unified_metadata = validate_metadata(unified_metadata)
    return [ContentTypeEnum.IMAGE, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


def construct_image_metadata_from_pdf_image(
    pdf_image: PdfImage,
    page_idx: int,
    page_count: int,
    source_metadata: Dict[str, Any],
    base_unified_metadata: Dict[str, Any],
) -> List[Any]:
    """
    Extracts image data from a PdfImage object, converts it to a base64-encoded string,
    and constructs metadata for the image.

    Parameters
    ----------
    image_obj : PdfImage
        The PdfImage object from which the image will be extracted.
    page_idx : int
        The index of the current page being processed.
    page_count : int
        The total number of pages in the PDF document.
    source_metadata : dict
        Metadata related to the source of the PDF document.
    base_unified_metadata : dict
        The base unified metadata structure to be updated with the extracted image information.

    Returns
    -------
    List[Any]
        A list containing the content type, validated metadata dictionary, and a UUID string.

    Raises
    ------
    PdfiumError
        If the image cannot be extracted due to an issue with the PdfImage object.
        :param pdf_image:
    """

    # Construct content metadata
    content_metadata: Dict[str, Any] = {
        "type": ContentTypeEnum.IMAGE,
        "description": ContentDescriptionEnum.PDF_IMAGE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": -1,
            "line": -1,
            "span": -1,
        },
    }

    # Construct image metadata
    image_metadata: Dict[str, Any] = {
        "image_type": DocumentTypeEnum.PNG,
        "structured_image_type": ContentTypeEnum.UNKNOWN,
        "caption": "",
        "text": "",
        "image_location": pdf_image.bbox,
        "image_location_max_dimensions": (max(pdf_image.max_width, 0), max(pdf_image.max_height, 0)),
        "height": pdf_image.height,
        "width": pdf_image.width,
    }

    # Update the unified metadata with the extracted image information
    unified_metadata: Dict[str, Any] = base_unified_metadata.copy()
    unified_metadata.update(
        {
            "content": pdf_image.image,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    # Validate and return the unified metadata
    validated_unified_metadata = validate_metadata(unified_metadata)
    return [ContentTypeEnum.IMAGE, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


def _construct_text_image_primitive(
    cropped_image: CroppedImageWithContent,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
) -> List[Any]:
    """Constructs an 'image' primitive for a detected text block, intended for downstream OCR."""
    content_metadata = {
        "type": ContentTypeEnum.TEXT,
        "description": ContentDescriptionEnum.PDF_TEXT,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
        },
        "subtype": cropped_image.type_string,
    }

    text_metadata = {
        "text_type": "page",
        "text_location": cropped_image.bbox,
        "text_location_max_dimensions": (cropped_image.max_width, cropped_image.max_height),
    }

    unified_metadata = base_unified_metadata.copy()
    unified_metadata.update(
        {
            "content": cropped_image.image,  # The base64 image of the text block
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "text_metadata": text_metadata,
        }
    )

    validated_metadata = validate_metadata(unified_metadata)
    return [ContentTypeEnum.TEXT, validated_metadata.model_dump(), str(uuid.uuid4())]


# TODO(Devin): Disambiguate tables and charts, create two distinct processing methods
@pdfium_exception_handler(descriptor="pdfium")
def construct_page_element_metadata(
    structured_image: CroppedImageWithContent,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    """
    +--------------------------------+--------------------------+------------+---+
    | Table/Chart Metadata           |                          | Extracted  | Y |
    | (tables within documents)      |                          |            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table format                   | Structured (dataframe /  | Extracted  |   |
    |                                | lists of rows and        |            |   |
    |                                | columns), or serialized  |            |   |
    |                                | as markdown, html,       |            |   |
    |                                | latex, simple (cells     |            |   |
    |                                | separated just as spaces)|            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table content                  | Extracted text content   |            |   |
    |                                |                          |            |   |
    |                                | Important: Tables should |            |   |
    |                                | not be chunked           |            |   |
    +--------------------------------+--------------------------+------------+---+
    | Table location                 | Bounding box of the table|            |   |
    +--------------------------------+--------------------------+------------+---+
    | Caption                        | Detected captions for    |            |   |
    |                                | the table/chart          |            |   |
    +--------------------------------+--------------------------+------------+---+
    | uploaded_image_uri             | Mirrors                  |            |   |
    |                                | source_metadata.         |            |   |
    |                                | source_location          |            |   |
    +--------------------------------+--------------------------+------------+---+
    """
    text_types = {"paragraph", "title", "header_footer"}
    if structured_image.type_string in text_types:
        return _construct_text_image_primitive(
            structured_image, page_idx, page_count, source_metadata, base_unified_metadata
        )

    if structured_image.type_string in ("table",):
        content = structured_image.image
        structured_content_text = structured_image.content
        structured_content_format = structured_image.content_format
        table_format = TableFormatEnum.IMAGE
        subtype = ContentTypeEnum.TABLE
        description = ContentDescriptionEnum.PDF_TABLE
        meta_name = "table_metadata"

    elif structured_image.type_string in ("chart",):
        content = structured_image.image
        structured_content_text = structured_image.content
        structured_content_format = structured_image.content_format
        table_format = TableFormatEnum.IMAGE
        subtype = ContentTypeEnum.CHART
        description = ContentDescriptionEnum.PDF_CHART
        # TODO(Devin) swap this to chart_metadata after we confirm metadata schema changes.
        meta_name = "table_metadata"

    elif structured_image.type_string in ("infographic",):
        content = structured_image.image
        structured_content_text = structured_image.content
        structured_content_format = structured_image.content_format
        table_format = TableFormatEnum.IMAGE
        subtype = ContentTypeEnum.INFOGRAPHIC
        description = ContentDescriptionEnum.PDF_INFOGRAPHIC
        meta_name = "table_metadata"

    else:
        raise ValueError(f"Unknown table/chart/infographic type: {structured_image.type_string}")

    content_metadata = {
        "type": ContentTypeEnum.STRUCTURED,
        "description": description,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "line": -1,
            "span": -1,
        },
        "subtype": subtype,
    }

    structured_metadata = {
        "caption": "",
        "table_format": table_format,
        "table_content": structured_content_text,
        "table_content_format": structured_content_format,
        "table_location": structured_image.bbox,
        "table_location_max_dimensions": (structured_image.max_width, structured_image.max_height),
    }

    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": content,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            meta_name: structured_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.STRUCTURED, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


# TODO: remove this alias
construct_table_and_chart_metadata = construct_page_element_metadata
