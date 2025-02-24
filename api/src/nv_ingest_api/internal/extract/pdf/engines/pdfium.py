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

import concurrent.futures
import logging

import pypdfium2 as libpdfium

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest_api.internal.primitives.nim.default_values import YOLOX_MAX_WIDTH, YOLOX_MAX_BATCH_SIZE, YOLOX_MAX_HEIGHT
from nv_ingest_api.util.metadata.aggregators import (
    Base64Image,
    construct_image_metadata_from_pdf_image,
    construct_table_and_chart_metadata,
    extract_pdf_metadata,
    construct_text_metadata,
)
from nv_ingest_api.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest_api.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest_api.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy
from nv_ingest_api.util.image_processing.processing import extract_tables_and_charts_yolox
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64

logger = logging.getLogger(__name__)


def _extract_page_text(page) -> str:
    """
    Always extract text from the given page and return it as a raw string.
    The caller decides whether to use per-page or doc-level logic.
    """
    textpage = page.get_textpage()
    return textpage.get_text_bounded()


def _extract_page_images(
    page,
    page_idx: int,
    page_width: float,
    page_height: float,
    page_count: int,
    source_metadata: dict,
    base_unified_metadata: dict,
) -> list:
    """
    Always extract images from the given page and return a list of image metadata items.
    The caller decides whether to call this based on a flag.
    """
    extracted_images = []
    for obj in page.get_objects():
        obj_type = PDFIUM_PAGEOBJ_MAPPING.get(obj.type, "UNKNOWN")
        if obj_type == "IMAGE":
            try:
                image_numpy = pdfium_try_get_bitmap_as_numpy(obj)
                image_base64 = numpy_to_base64(image_numpy)
                image_bbox = obj.get_pos()
                image_size = obj.get_size()

                image_data = Base64Image(
                    image=image_base64,
                    bbox=image_bbox,
                    width=image_size[0],
                    height=image_size[1],
                    max_width=page_width,
                    max_height=page_height,
                )

                image_meta = construct_image_metadata_from_pdf_image(
                    image_data,
                    page_idx,
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                )
                extracted_images.append(image_meta)
            except Exception as e:
                logger.error(f"Unhandled error extracting image on page {page_idx}: {e}")
                # continue extracting other images

    return extracted_images


def _extract_tables_and_charts(
    pages: list,
    pdfium_config: PDFiumConfigSchema,
    page_count: int,
    source_metadata: dict,
    base_unified_metadata: dict,
    paddle_output_format,
    trace_info=None,
) -> list:
    """
    Always extract tables and charts from the given pages using YOLOX-based logic.
    The caller decides whether to call it.
    """
    extracted_table_chart = []

    table_chart_results = extract_tables_and_charts_yolox(pages, pdfium_config.model_dump(), trace_info=trace_info)

    # Build metadata for each
    for page_idx, table_or_chart in table_chart_results:
        # If we want all tables and charts, we assume the caller wouldn't call
        # this function unless we truly want them.
        if table_or_chart.type_string == "table":
            table_or_chart.content_format = paddle_output_format

        table_chart_meta = construct_table_and_chart_metadata(
            table_or_chart,
            page_idx,
            page_count,
            source_metadata,
            base_unified_metadata,
        )
        extracted_table_chart.append(table_chart_meta)

    return extracted_table_chart


def pdfium_extractor(
    pdf_stream,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    extractor_config: dict,
    trace_info=None,
):
    # --- Extract and validate extractor_config ---
    if extractor_config is None or not isinstance(extractor_config, dict):
        raise ValueError("`extractor_config` must be provided as a dictionary.")

    # Validate and extract row_data
    row_data = extractor_config.get("row_data")
    if row_data is None:
        raise ValueError("`extractor_config` must include a valid 'row_data' dictionary.")
    if "source_id" not in row_data:
        raise ValueError("The 'row_data' dictionary must contain the 'source_id' key.")

    # Validate and extract text_depth
    text_depth_str = extractor_config.get("text_depth", "page")
    try:
        text_depth = TextTypeEnum[text_depth_str.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid text_depth: {text_depth_str}. Valid options: {list(TextTypeEnum.__members__.keys())}"
        )

    # Validate and extract paddle_output_format
    paddle_output_format_str = extractor_config.get("paddle_output_format", "pseudo_markdown")
    try:
        paddle_output_format = TableFormatEnum[paddle_output_format_str.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid paddle_output_format: {paddle_output_format_str}. "
            f"Valid options: {list(TableFormatEnum.__members__.keys())}"
        )

    # Extract metadata_column
    metadata_column = extractor_config.get("metadata_column", "metadata")

    # Process pdfium_config
    pdfium_config_raw = extractor_config.get("pdfium_config", {})
    if isinstance(pdfium_config_raw, dict):
        pdfium_config = PDFiumConfigSchema(**pdfium_config_raw)
    elif isinstance(pdfium_config_raw, PDFiumConfigSchema):
        pdfium_config = pdfium_config_raw
    else:
        raise ValueError("`pdfium_config` must be a dictionary or a PDFiumConfigSchema instance.")
    # --- End extractor_config extraction ---

    logger.debug("Extracting PDF with pdfium backend.")
    source_id = row_data["source_id"]

    # Retrieve unified metadata robustly (supporting pandas Series or dict)
    if hasattr(row_data, "index"):
        base_unified_metadata = row_data[metadata_column] if metadata_column in row_data.index else {}
    else:
        base_unified_metadata = row_data.get(metadata_column, {})

    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    source_location = base_source_metadata.get("source_location", "")
    collection_id = base_source_metadata.get("collection_id", "")
    partition_id = base_source_metadata.get("partition_id", -1)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    doc = libpdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)
    page_count = pdf_metadata.page_count

    source_metadata = {
        "source_name": pdf_metadata.filename,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": pdf_metadata.source_type,
        "collection_id": collection_id,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    logger.debug(f"PDF has {page_count} pages.")
    logger.debug(
        f"extract_text={extract_text}, extract_images={extract_images}, "
        f"extract_tables={extract_tables}, extract_charts={extract_charts}"
    )

    # Decide if text extraction should be done at the PAGE or DOCUMENT level
    if text_depth != TextTypeEnum.PAGE:
        text_depth = TextTypeEnum.DOCUMENT

    extracted_data = []
    accumulated_text = []

    # Prepare for table/chart extraction
    pages_for_tables = []  # Accumulate tuples of (page_idx, np_image)
    futures = []  # To track asynchronous table/chart extraction tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=pdfium_config.workers_per_progress_engine) as executor:
        # PAGE LOOP
        for page_idx in range(page_count):
            page = doc.get_page(page_idx)
            page_width, page_height = page.get_size()

            # Text extraction
            if extract_text:
                page_text = _extract_page_text(page)
                if text_depth == TextTypeEnum.PAGE:
                    text_meta = construct_text_metadata(
                        [page_text],
                        pdf_metadata.keywords,
                        page_idx,
                        -1,
                        -1,
                        -1,
                        page_count,
                        text_depth,
                        source_metadata,
                        base_unified_metadata,
                    )
                    extracted_data.append(text_meta)
                else:
                    accumulated_text.append(page_text)

            # Image extraction
            if extract_images:
                image_data = _extract_page_images(
                    page,
                    page_idx,
                    page_width,
                    page_height,
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                )
                extracted_data.extend(image_data)

            # Table/Chart extraction: rasterize the page and queue for processing.
            if extract_tables or extract_charts:
                image, _ = pdfium_pages_to_numpy(
                    [page], scale_tuple=(YOLOX_MAX_WIDTH, YOLOX_MAX_HEIGHT), trace_info=trace_info
                )
                pages_for_tables.append((page_idx, image[0]))
                if len(pages_for_tables) >= YOLOX_MAX_BATCH_SIZE:
                    future = executor.submit(
                        _extract_tables_and_charts,
                        pages_for_tables[:],
                        pdfium_config,
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                        paddle_output_format,
                        trace_info=trace_info,
                    )
                    futures.append(future)
                    pages_for_tables.clear()

            page.close()

        # Process any remaining pages queued for table/chart extraction.
        if (extract_tables or extract_charts) and pages_for_tables:
            future = executor.submit(
                _extract_tables_and_charts,
                pages_for_tables[:],
                pdfium_config,
                page_count,
                source_metadata,
                base_unified_metadata,
                paddle_output_format,
                trace_info=trace_info,
            )
            futures.append(future)
            pages_for_tables.clear()

        # Wait for all asynchronous jobs to complete.
        for fut in concurrent.futures.as_completed(futures):
            table_chart_items = fut.result()  # Blocks until the job is finished
            extracted_data.extend(table_chart_items)

    # For document-level text extraction, combine the accumulated text.
    if extract_text and text_depth == TextTypeEnum.DOCUMENT and accumulated_text:
        doc_text_meta = construct_text_metadata(
            accumulated_text,
            pdf_metadata.keywords,
            -1,
            -1,
            -1,
            -1,
            page_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )
        extracted_data.append(doc_text_meta)

    return extracted_data
