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
import io
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, IO
from typing import Optional

import pandas as pd
from pptx import Presentation
from pptx.enum.dml import MSO_COLOR_TYPE
from pptx.enum.dml import MSO_THEME_COLOR  # noqa
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.shapes import PP_PLACEHOLDER  # noqa
from pptx.shapes.autoshape import Shape
from pptx.slide import Slide

from nv_ingest_api.internal.enums.common import AccessLevelEnum, DocumentTypeEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.enums.common import ContentDescriptionEnum
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
from nv_ingest_api.internal.extract.image.image_helpers.common import (
    load_and_preprocess_image,
    extract_page_elements_from_images,
)
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXConfigSchema
from nv_ingest_api.util.converters import bytetools
from nv_ingest_api.util.detectors.language import detect_language
from nv_ingest_api.util.metadata.aggregators import construct_page_element_metadata

logger = logging.getLogger(__name__)


def _finalize_images(
    pending_images: List[Tuple[Shape, int, int, int, dict, dict, dict]],
    extracted_data: List,
    pptx_extraction_config: PPTXConfigSchema,
    extract_tables: bool = False,
    extract_charts: bool = False,
    trace_info: Optional[Dict] = None,
):
    """
    Post-process all pending images.
      - Convert shape image -> NumPy or base64
      - If `extract_tables` or `extract_charts`, do detection (table/chart)
      - Build the appropriate metadata, either table/chart or image.

    This mimics the docx approach, but adapted for python-pptx shapes.
    """
    if not pending_images:
        return

    # Convert each shape to image data (base64 or ndarray).
    # We'll store them for a single call to your model if you'd like (batching).
    image_arrays = []
    image_contexts = []
    for (
        shape,
        shape_idx,
        slide_idx,
        slide_count,
        page_nearby_blocks,
        source_metadata,
        base_unified_metadata,
    ) in pending_images:
        try:
            image_bytes = shape.image.blob
            image_array = load_and_preprocess_image(io.BytesIO(image_bytes))
            base64_img = bytetools.base64frombytes(image_bytes)

            image_arrays.append(image_array)
            image_contexts.append(
                (
                    shape_idx,
                    slide_idx,
                    slide_count,
                    page_nearby_blocks,
                    source_metadata,
                    base_unified_metadata,
                    base64_img,
                )
            )
        except Exception as e:
            logger.warning(f"Unable to process shape image: {e}")

    # If you want table/chart detection for these images, do it now
    # (similar to docx approach). This might use your YOLO or another method:
    detection_map = defaultdict(list)  # image_idx -> list of CroppedImageWithContent
    if extract_tables or extract_charts:
        try:
            # For example, a call to your function that checks for tables/charts
            detection_results = extract_page_elements_from_images(
                images=image_arrays,
                config=ImageConfigSchema(**(pptx_extraction_config.model_dump())),
                trace_info=trace_info,
            )
            # detection_results is something like [(image_idx, CroppedImageWithContent), ...]
            for img_idx, cropped_obj in detection_results:
                detection_map[img_idx].append(cropped_obj)
        except Exception as e:
            logger.error(f"Error while running table/chart detection on PPTX images: {e}")
            detection_map = {}

    # Now build the final metadata objects
    for i, context in enumerate(image_contexts):
        (shape_idx, slide_idx, slide_count, page_nearby_blocks, source_metadata, base_unified_metadata, base64_img) = (
            context
        )

        # If there's a detection result for this image, handle it
        if i in detection_map and detection_map[i]:
            # We found table(s)/chart(s) in the image
            for cropped_item in detection_map[i]:
                structured_entry = construct_page_element_metadata(
                    structured_image=cropped_item,
                    page_idx=slide_idx,
                    page_count=slide_count,
                    source_metadata=source_metadata,
                    base_unified_metadata=base_unified_metadata,
                )
                extracted_data.append(structured_entry)
        else:
            # No table detected => build normal image metadata
            image_entry = _construct_image_metadata(
                shape_idx=shape_idx,
                slide_idx=slide_idx,
                slide_count=slide_count,
                page_nearby_blocks=page_nearby_blocks,
                base64_img=base64_img,
                source_metadata=source_metadata,
                base_unified_metadata=base_unified_metadata,
            )
            extracted_data.append(image_entry)


def _safe_position(shape):
    top = shape.top if shape.top is not None else float("inf")
    left = shape.left if shape.left is not None else float("inf")
    return (top, left)


# -----------------------------------------------------------------------------
# Helper Function: Recursive Image Extraction
# -----------------------------------------------------------------------------
def process_shape(
    shape, shape_idx, slide_idx, slide_count, pending_images, page_nearby_blocks, source_metadata, base_unified_metadata
):
    """
    Recursively process a shape:
      - If the shape is a group, iterate over its child shapes.
      - If it is a picture or a placeholder with an embedded image, append it to pending_images.
    """
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sub_idx, sub_shape in enumerate(shape.shapes):
            # Create a composite index (e.g., "2.1" for the first child of shape 2)
            composite_idx = f"{shape_idx}.{sub_idx}"
            process_shape(
                sub_shape,
                composite_idx,
                slide_idx,
                slide_count,
                pending_images,
                page_nearby_blocks,
                source_metadata,
                base_unified_metadata,
            )
    else:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or (
            shape.is_placeholder and shape.placeholder_format.type == PP_PLACEHOLDER.OBJECT and hasattr(shape, "image")
        ):
            try:
                pending_images.append(
                    (
                        shape,  # so we can later pull shape.image.blob
                        shape_idx,
                        slide_idx,
                        slide_count,
                        page_nearby_blocks,
                        source_metadata,
                        base_unified_metadata,
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing shape {shape_idx} on slide {slide_idx}: {e}")
                raise


# -----------------------------------------------------------------------------
# Main Extraction Function
# -----------------------------------------------------------------------------
def python_pptx(
    *,
    pptx_stream: IO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extraction_config: dict,
    execution_trace_log: Optional[List] = None,
):
    _ = extract_infographics
    _ = execution_trace_log

    row_data = extraction_config.get("row_data")
    source_id = row_data["source_id"]

    text_depth = TextTypeEnum[extraction_config.get("text_depth", "page").upper()]
    paragraph_format = extraction_config.get("paragraph_format", "markdown")
    identify_nearby_objects = extraction_config.get("identify_nearby_objects", True)

    metadata_col = extraction_config.get("metadata_column", "metadata")
    pptx_extractor_config = extraction_config.get("pptx_extraction_config", {})
    trace_info = extraction_config.get("trace_info", {})

    base_unified_metadata = row_data.get(metadata_col, {})
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    source_location = base_source_metadata.get("source_location", "")
    collection_id = base_source_metadata.get("collection_id", "")
    partition_id = base_source_metadata.get("partition_id", -1)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.UNKNOWN)

    try:
        presentation = Presentation(pptx_stream)
    except Exception as e:
        logger.error("Failed to open PPTX presentation: %s", e)
        return []

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
    source_metadata = {
        "source_name": source_id,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": DocumentTypeEnum.PPTX,
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
    pending_images = []

    for slide_idx, slide in enumerate(presentation.slides):
        try:
            shapes = sorted(ungroup_shapes(slide.shapes), key=_safe_position)
        except Exception as e:
            logger.error("Slide %d: Failed to ungroup or sort shapes: %s", slide_idx, e)
            continue

        page_nearby_blocks = {
            "text": {"content": [], "bbox": []},
            "images": {"content": [], "bbox": []},
            "structured": {"content": [], "bbox": []},
        }

        for shape_idx, shape in enumerate(shapes):
            try:
                block_text = []
                added_title = added_subtitle = False

                # Text extraction
                if extract_text and shape.has_text_frame:
                    for paragraph_idx, paragraph in enumerate(shape.text_frame.paragraphs):
                        if not paragraph.text.strip():
                            continue

                        for run_idx, run in enumerate(paragraph.runs):
                            try:
                                text = run.text
                                if not text:
                                    continue

                                text = escape_text(text)

                                if paragraph_format == "markdown":
                                    if is_title(shape) and not added_title:
                                        text = process_title(shape)
                                        added_title = True
                                    elif is_subtitle(shape) and not added_subtitle:
                                        text = process_subtitle(shape)
                                        added_subtitle = True
                                    elif is_title(shape) or is_subtitle(shape):
                                        continue  # already added

                                    if run.hyperlink and run.hyperlink.address:
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
                                    extracted_data.append(
                                        _construct_text_metadata(
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
                                    )
                                    accumulated_text = []

                            except Exception as e:
                                logger.warning(
                                    "Slide %d Shape %d Run %d: Failed to process run: %s",
                                    slide_idx,
                                    shape_idx,
                                    run_idx,
                                    e,
                                )

                        if accumulated_text and not accumulated_text[-1].endswith("\n\n"):
                            accumulated_text.append("\n\n")

                        if text_depth == TextTypeEnum.LINE:
                            extracted_data.append(
                                _construct_text_metadata(
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
                            )
                            accumulated_text = []

                    if text_depth == TextTypeEnum.BLOCK:
                        extracted_data.append(
                            _construct_text_metadata(
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
                        )
                        accumulated_text = []

                if extract_images and identify_nearby_objects and block_text:
                    page_nearby_blocks["text"]["content"].append("".join(block_text))
                    page_nearby_blocks["text"]["bbox"].append(get_bbox(shape_object=shape))

                # Image processing (deferred)
                if extract_images or extract_tables or extract_charts:
                    try:
                        process_shape(
                            shape,
                            shape_idx,
                            slide_idx,
                            slide_count,
                            pending_images,
                            page_nearby_blocks,
                            source_metadata,
                            base_unified_metadata,
                        )
                    except Exception as e:
                        logger.warning("Slide %d Shape %d: Failed to process image shape: %s", slide_idx, shape_idx, e)

                # Table extraction
                if extract_tables and shape.has_table:
                    try:
                        extracted_data.append(
                            _construct_table_metadata(
                                shape, slide_idx, slide_count, source_metadata, base_unified_metadata
                            )
                        )
                    except Exception as e:
                        logger.warning("Slide %d Shape %d: Failed to extract table: %s", slide_idx, shape_idx, e)

            except Exception as e:
                logger.warning("Slide %d Shape %d: Top-level failure: %s", slide_idx, shape_idx, e)

        if extract_text and text_depth == TextTypeEnum.PAGE and accumulated_text:
            extracted_data.append(
                _construct_text_metadata(
                    presentation,
                    None,
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
            )
            accumulated_text = []

    if extract_text and text_depth == TextTypeEnum.DOCUMENT and accumulated_text:
        extracted_data.append(
            _construct_text_metadata(
                presentation,
                None,
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
        )

    if extract_images or extract_tables or extract_charts:
        try:
            _finalize_images(
                pending_images,
                extracted_data,
                pptx_extractor_config,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                trace_info=trace_info,
            )
        except Exception as e:
            logger.error("Finalization of images failed: %s", e)

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
        "description": ContentDescriptionEnum.PPTX_TEXT,
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

    return [ContentTypeEnum.TEXT, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


# need to add block text to hierarchy/nearby_objects, including bbox
def _construct_image_metadata(
    shape_idx: int,
    slide_idx: int,
    slide_count: int,
    page_nearby_blocks: Dict,
    base64_img: str,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    """
    Build standard PPTX image metadata.
    """
    # Example bounding box
    bbox = (0, 0, 0, 0)  # or extract from shape.left, shape.top, shape.width, shape.height if desired

    content_metadata = {
        "type": ContentTypeEnum.IMAGE,
        "description": ContentDescriptionEnum.PPTX_IMAGE,
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
        "image_type": DocumentTypeEnum.PNG,
        "structured_image_type": ContentTypeEnum.UNKNOWN,
        "caption": "",  # could attempt to guess a caption from nearby text
        "text": "",
        "image_location": bbox,
    }

    unified_metadata = base_unified_metadata.copy() if base_unified_metadata else {}
    unified_metadata.update(
        {
            "content": base64_img,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [
        ContentTypeEnum.IMAGE.value,
        validated_unified_metadata.model_dump(),
        str(uuid.uuid4()),
    ]


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
        "description": ContentDescriptionEnum.PPTX_TABLE,
        "page_number": slide_idx,
        "hierarchy": {
            "page_count": slide_count,
            "page": slide_idx,
            "line": -1,
            "span": -1,
        },
        "subtype": ContentTypeEnum.TABLE,
    }
    table_metadata = {
        "caption": "",
        "table_format": TableFormatEnum.MARKDOWN,
        "table_location": bbox,
        "table_content": df.to_markdown(index=False),
    }
    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": "",
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "table_metadata": table_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.STRUCTURED, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


def get_bbox(
    presentation_object: Optional[Presentation] = None,
    shape_object: Optional[Slide] = None,
    text_depth: Optional[TextTypeEnum] = None,
):
    """
    Safely computes bounding box for a slide, shape, or document.
    Ensures that missing or None values are gracefully handled.

    Returns
    -------
    Tuple[int, int, int, int]
        Bounding box as (top, left, bottom, right).
        Defaults to (-1, -1, -1, -1) if invalid or unsupported.
    """
    try:
        if text_depth == TextTypeEnum.DOCUMENT:
            return (-1, -1, -1, -1)

        elif text_depth == TextTypeEnum.PAGE and presentation_object:
            top = left = 0
            width = presentation_object.slide_width
            height = presentation_object.slide_height
            return (top, left, top + height, left + width)

        elif shape_object:
            top = shape_object.top if shape_object.top is not None else -1
            left = shape_object.left if shape_object.left is not None else -1
            width = shape_object.width if shape_object.width is not None else -1
            height = shape_object.height if shape_object.height is not None else -1

            # If all are valid, return normally, else return placeholder
            if -1 in [top, left, width, height]:
                return (-1, -1, -1, -1)

            return (top, left, top + height, left + width)

    except Exception as e:
        logger.warning(f"get_bbox: Failed to compute bbox due to {e}")
        return (-1, -1, -1, -1)

    return (-1, -1, -1, -1)


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
