# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import operator
import re
import uuid
from datetime import datetime
from typing import Dict
from typing import Optional

import pandas as pd
from pptx import Presentation
from pptx.enum.dml import MSO_COLOR_TYPE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.slide import Slide

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import bytetools
from nv_ingest.util.detectors.language import detect_language

logger = logging.getLogger(__name__)


# Define a helper function to use python-pptx to extract text from a base64
# encoded bytestram PPTX
def python_pptx(pptx_stream, extract_text: bool, extract_images: bool, extract_tables: bool, **kwargs):
    """
    Helper function to use python-pptx to extract text from a bytestream PPTX.

    A document has five levels - presentation, slides, shapes, paragraphs, and runs.
    To align with the pdf extraction, we map the levels as follows:
    - Document -> Presention
    - Pages -> Slides
    - Blocks -> Shapes
    - Lines -> Paragraphs
    - Spans -> Runs

    Parameters
    ----------
    pptx_stream : io.BytesIO
        A bytestream PPTX.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.
    """

    logger.debug("Extracting PPTX with python-pptx backend.")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    # Not configurable anywhere at the moment
    paragraph_format = kwargs.get("paragraph_format", "markdown")
    identify_nearby_objects = kwargs.get("identify_nearby_objects", True)

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")
    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}
    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    presentation = Presentation(pptx_stream)

    # Collect source metadata from the core properties of the document.
    last_modified = (
        presentation.core_properties.modified.isoformat()
        if presentation.core_properties.modified
        else datetime.now().isoformat()
    )
    date_created = (
        presentation.core_properties.created.isoformat()
        if presentation.core_properties.created
        else datetime.now().isoformat()
    )
    keywords = presentation.core_properties.keywords
    source_type = SourceTypeEnum.PPTX
    source_metadata = {
        "source_name": source_id,  # python-pptx doesn't maintain filename; re-use source_id
        "source_id": source_id,
        "source_location": source_location,
        "source_type": source_type,
        "collection_id": collection_id,
        "date_created": date_created,
        "last_modified": last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    slide_count = len(presentation.slides)

    accumulated_text = []
    extracted_data = []

    for slide_idx, slide in enumerate(presentation.slides):
        shapes = sorted(ungroup_shapes(slide.shapes), key=operator.attrgetter("top", "left"))

        page_nearby_blocks = {
            "text": {"content": [], "bbox": []},
            "images": {"content": [], "bbox": []},
            "structured": {"content": [], "bbox": []},
        }

        for shape_idx, shape in enumerate(shapes):
            block_text = []
            added_title = added_subtitle = False

            if extract_text and shape.has_text_frame:
                for paragraph_idx, paragraph in enumerate(shape.text_frame.paragraphs):
                    if not paragraph.text.strip():
                        continue

                    for run_idx, run in enumerate(paragraph.runs):
                        text = run.text
                        if not text:
                            continue
                        text = escape_text(text)

                        if paragraph_format == "markdown":
                            # For titles/subtitles, process them on the block/shape level, and
                            # skip formatting.
                            if is_title(shape):
                                if added_title:
                                    continue
                                text = process_title(shape)
                                added_title = True
                            elif is_subtitle(shape):
                                if added_subtitle:
                                    continue
                                text = process_subtitle(shape)
                                added_subtitle = True
                            else:
                                if run.hyperlink.address:
                                    text = get_hyperlink(text, run.hyperlink.address)

                                if is_accent(paragraph.font) or is_accent(run.font):
                                    text = format_text(text, italic=True)
                                elif is_strong(paragraph.font) or is_strong(run.font):
                                    text = format_text(text, bold=True)
                                elif is_underlined(paragraph.font) or is_underlined(run.font):
                                    text = format_text(text, underline=True)

                                if is_list_block(shape):
                                    text = "  " * paragraph.level + "* " + text

                        accumulated_text.append(text)

                        if extract_images and identify_nearby_objects:
                            block_text.append(text)

                        if text_depth == TextTypeEnum.SPAN:
                            text_extraction = _construct_text_metadata(
                                presentation,
                                shape,
                                accumulated_text,
                                keywords,
                                slide_idx,
                                shape_idx,
                                paragraph_idx,
                                run_idx,
                                slide_count,
                                text_depth,
                                source_metadata,
                                base_unified_metadata,
                            )

                            if len(text_extraction) > 0:
                                extracted_data.append(text_extraction)

                            accumulated_text = []

                    # Avoid excessive newline characters and add them only at
                    # the line/paragraph level or higher.
                    if accumulated_text and not accumulated_text[-1].endswith("\n\n"):
                        accumulated_text.append("\n\n")

                    if text_depth == TextTypeEnum.LINE:
                        text_extraction = _construct_text_metadata(
                            presentation,
                            shape,
                            accumulated_text,
                            keywords,
                            slide_idx,
                            shape_idx,
                            paragraph_idx,
                            -1,
                            slide_count,
                            text_depth,
                            source_metadata,
                            base_unified_metadata,
                        )

                        if len(text_extraction) > 0:
                            extracted_data.append(text_extraction)

                        accumulated_text = []

                if text_depth == TextTypeEnum.BLOCK:
                    text_extraction = _construct_text_metadata(
                        presentation,
                        shape,
                        accumulated_text,
                        keywords,
                        slide_idx,
                        shape_idx,
                        -1,
                        -1,
                        slide_count,
                        text_depth,
                        source_metadata,
                        base_unified_metadata,
                    )

                    if len(text_extraction) > 0:
                        extracted_data.append(text_extraction)

                    accumulated_text = []

            if extract_images and identify_nearby_objects and (len(block_text) > 0):
                page_nearby_blocks["text"]["content"].append("".join(block_text))
                page_nearby_blocks["text"]["bbox"].append(get_bbox(shape_object=shape))

            if extract_images and (
                shape.shape_type == MSO_SHAPE_TYPE.PICTURE
                or (
                    shape.is_placeholder
                    and shape.placeholder_format.type == PP_PLACEHOLDER.OBJECT
                    and hasattr(shape, "image")
                    and getattr(shape, "image")
                )
            ):
                try:
                    image_extraction = _construct_image_metadata(
                        shape,
                        shape_idx,
                        slide_idx,
                        slide_count,
                        source_metadata,
                        base_unified_metadata,
                        page_nearby_blocks,
                    )
                    extracted_data.append(image_extraction)
                except ValueError as e:
                    # Handle the specific case where no embedded image is found
                    logger.warning(f"No embedded image found for shape {shape_idx} on slide {slide_idx}: {e}")
                except Exception as e:
                    # Handle any other exceptions that might occur
                    logger.warning(f"An error occurred while processing shape {shape_idx} on slide {slide_idx}: {e}")

            if extract_tables and shape.has_table:
                table_extraction = _construct_table_metadata(
                    shape, slide_idx, slide_count, source_metadata, base_unified_metadata
                )
                extracted_data.append(table_extraction)

        # Extract text - slide (b)
        if (extract_text) and (text_depth == TextTypeEnum.PAGE):
            text_extraction = _construct_text_metadata(
                presentation,
                shape,
                accumulated_text,
                keywords,
                slide_idx,
                -1,
                -1,
                -1,
                slide_count,
                text_depth,
                source_metadata,
                base_unified_metadata,
            )

            if len(text_extraction) > 0:
                extracted_data.append(text_extraction)

            accumulated_text = []

    # Extract text - presentation (c)
    if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
        text_extraction = _construct_text_metadata(
            presentation,
            shape,
            accumulated_text,
            keywords,
            -1,
            -1,
            -1,
            -1,
            slide_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

        accumulated_text = []

    return extracted_data


def _construct_text_metadata(
    presentation_object,
    shape_object,
    accumulated_text,
    keywords,
    slide_idx,
    shape_idx,
    paragraph_idx,
    run_idx,
    slide_count,
    text_depth,
    source_metadata,
    base_unified_metadata,
):
    extracted_text = "".join(accumulated_text)

    content_metadata = {
        "type": ContentTypeEnum.TEXT,
        "description": StdContentDescEnum.PPTX_TEXT,
        "page_number": slide_idx,
        "hierarchy": {
            "page_count": slide_count,
            "page": slide_idx,
            "block": shape_idx,
            "line": paragraph_idx,
            "span": run_idx,
        },
    }

    language = detect_language(extracted_text)
    bbox = get_bbox(
        presentation_object=presentation_object,
        shape_object=shape_object,
        text_depth=text_depth,
    )

    text_metadata = {
        "text_type": text_depth,
        "summary": "",
        "keywords": keywords,
        "language": language,
        "text_location": bbox,
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

    return [ContentTypeEnum.TEXT, validated_unified_metadata.dict(), str(uuid.uuid4())]


# need to add block text to hierarchy/nearby_objects, including bbox
def _construct_image_metadata(
    shape, shape_idx, slide_idx, slide_count, source_metadata, base_unified_metadata, page_nearby_blocks
):
    image_type = shape.image.ext
    if ImageTypeEnum.has_value(image_type):
        image_type = ImageTypeEnum[image_type.upper()]

    base64_img = bytetools.base64frombytes(shape.image.blob)

    bbox = get_bbox(shape_object=shape)
    width = shape.width
    height = shape.height

    content_metadata = {
        "type": ContentTypeEnum.IMAGE,
        "description": StdContentDescEnum.PPTX_IMAGE,
        "page_number": slide_idx,
        "hierarchy": {
            "page_count": slide_count,
            "page": slide_idx,
            "block": shape_idx,
            "line": -1,
            "span": -1,
            "nearby_objects": page_nearby_blocks,
        },
    }

    image_metadata = {
        "image_type": image_type,
        "structured_image_type": ImageTypeEnum.image_type_1,
        "caption": "",
        "text": "",
        "image_location": bbox,
        "width": width,
        "height": height,
    }

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update(
        {
            "content": base64_img,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [ContentTypeEnum.IMAGE, validated_unified_metadata.dict(), str(uuid.uuid4())]


def _construct_table_metadata(
    shape,
    slide_idx: int,
    slide_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    table = [[cell.text for cell in row.cells] for row in shape.table.rows]
    df = pd.DataFrame(table[1:], columns=table[0])
    # As df is eventually converted to markdown,
    # remove any newlines, tabs, or extra spaces from the column names
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)

    bbox = get_bbox(shape_object=shape)

    content_metadata = {
        "type": ContentTypeEnum.STRUCTURED,
        "description": StdContentDescEnum.PPTX_TABLE,
        "page_number": slide_idx,
        "hierarchy": {
            "page_count": slide_count,
            "page": slide_idx,
            "line": -1,
            "span": -1,
        },
    }
    table_metadata = {
        "caption": "",
        "table_format": TableFormatEnum.MARKDOWN,
        "table_location": bbox,
    }
    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": df.to_markdown(index=False),
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "table_metadata": table_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.STRUCTURED, validated_unified_metadata.dict(), str(uuid.uuid4())]


def get_bbox(
    presentation_object: Optional[Presentation] = None,
    shape_object: Optional[Slide] = None,
    text_depth: Optional[TextTypeEnum] = None,
):
    bbox = (-1, -1, -1, -1)
    if text_depth == TextTypeEnum.DOCUMENT:
        bbox = (-1, -1, -1, -1)
    elif text_depth == TextTypeEnum.PAGE:
        top = left = 0
        width = presentation_object.slide_width
        height = presentation_object.slide_height
        bbox = (top, left, top + height, left + width)
    elif shape_object:
        top = shape_object.top
        left = shape_object.left
        width = shape_object.width
        height = shape_object.height
        bbox = (top, left, top + height, left + width)
    return bbox


def ungroup_shapes(shapes):
    result = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            result.extend(ungroup_shapes(shape.shapes))
        else:
            result.append(shape)
    return result


def is_title(shape):
    if shape.is_placeholder and (
        shape.placeholder_format.type == PP_PLACEHOLDER.TITLE
        or shape.placeholder_format.type == PP_PLACEHOLDER.VERTICAL_TITLE
        or shape.placeholder_format.type == PP_PLACEHOLDER.CENTER_TITLE
    ):
        return True
    else:
        return False


def process_title(shape):
    title = shape.text_frame.text.strip()
    extracted_text = f"{title}\n{'=' * len(title)}"
    return extracted_text


def is_subtitle(shape):
    if shape.is_placeholder and (shape.placeholder_format.type == PP_PLACEHOLDER.SUBTITLE):
        return True
    else:
        return False


def process_subtitle(shape):
    subtitle = shape.text_frame.text.strip()
    extracted_text = f"{subtitle}\n{'-' * len(subtitle)}"
    return extracted_text


def is_list_block(shape):
    levels = set()
    for paragraph in shape.text_frame.paragraphs:
        if paragraph.level not in levels:
            levels.add(paragraph.level)
        if paragraph.level != 0 or len(levels) > 1:
            return True
    return False


def escape_text(text):
    def escape_repl(match_obj):
        return "\\" + match_obj.group(0)

    escape_regex_1 = re.compile(r"([\\\*`!_\{\}\[\]\(\)#\+-\.])")
    escape_regex_2 = re.compile(r"(<[^>]+>)")
    text = re.sub(escape_regex_1, escape_repl, text)
    text = re.sub(escape_regex_2, escape_repl, text)

    return text


def get_hyperlink(text, url):
    result = f"[{text}]({url})"
    return result


def is_accent(font):
    if font.italic or (
        font.color.type == MSO_COLOR_TYPE.SCHEME
        and (
            font.color.theme_color == MSO_THEME_COLOR.ACCENT_1
            or font.color.theme_color == MSO_THEME_COLOR.ACCENT_2
            or font.color.theme_color == MSO_THEME_COLOR.ACCENT_3
            or font.color.theme_color == MSO_THEME_COLOR.ACCENT_4
            or font.color.theme_color == MSO_THEME_COLOR.ACCENT_5
            or font.color.theme_color == MSO_THEME_COLOR.ACCENT_6
        )
    ):
        return True
    else:
        return False


def is_underlined(font):
    if font.underline:
        return True
    else:
        return False


def format_text(text: str, bold: bool = False, italic: bool = False, underline: bool = False) -> str:
    if not text.strip():
        return text

    prefix, suffix = "", ""
    # Exclude leading and trailing spaces from style
    trailing_space_pattern = re.compile(r"(^\s*)(.*?)(\s*$)", re.DOTALL)
    match = trailing_space_pattern.match(text)
    if match:
        prefix, text, suffix = match.groups()

    # Apply style
    if bold:
        text = f"**{text}**"
    if italic:
        text = f"*{text}*"
    if underline:
        text = f"<u>{text}</u>"

    # Add back leading and trailing spaces
    text = prefix + text + suffix

    return text


def is_strong(font):
    if font.bold or (
        font.color.type == MSO_COLOR_TYPE.SCHEME
        and (font.color.theme_color == MSO_THEME_COLOR.DARK_1 or font.color.theme_color == MSO_THEME_COLOR.DARK_2)
    ):
        return True
    else:
        return False
