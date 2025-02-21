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

from nv_ingest_api.primitives.nim.default_values import *
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_pdf_image
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy
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


# TODO: Replace with common function
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
    trace_info=None,
    **kwargs,
):
    logger.debug("Extracting PDF with pdfium backend.")

    row_data = kwargs.get("row_data")
    source_id = row_data["source_id"]

    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    paddle_output_format = kwargs.get("paddle_output_format", "pseudo_markdown")
    paddle_output_format = TableFormatEnum[paddle_output_format.upper()]

    # Basic config
    metadata_col = kwargs.get("metadata_column", "metadata")
    pdfium_config = kwargs.get("pdfium_config", {})
    if isinstance(pdfium_config, dict):
        pdfium_config = PDFiumConfigSchema(**pdfium_config)

    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}
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

    # Decide if text_depth is PAGE or DOCUMENT
    if text_depth != TextTypeEnum.PAGE:
        text_depth = TextTypeEnum.DOCUMENT

    extracted_data = []
    accumulated_text = []

    # Prepare for table/chart extraction
    pages_for_tables = []  # We'll accumulate (page_idx, np_image) here
    futures = []  # We'll keep track of all the Future objects for table/charts

    with concurrent.futures.ThreadPoolExecutor(max_workers=pdfium_config.workers_per_progress_engine) as executor:
        # PAGE LOOP
        for page_idx in range(page_count):
            page = doc.get_page(page_idx)
            page_width, page_height = page.get_size()

            # If we want text, extract text now.
            if extract_text:
                page_text = _extract_page_text(page)
                if text_depth == TextTypeEnum.PAGE:
                    # Build a page-level text metadata item
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
                    # doc-level => accumulate
                    accumulated_text.append(page_text)

            # If we want images, extract images now.
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

            # If we want tables or charts, rasterize the page and store it
            if extract_tables or extract_charts:
                image, _ = pdfium_pages_to_numpy(
                    [page], scale_tuple=(YOLOX_MAX_WIDTH, YOLOX_MAX_HEIGHT), trace_info=trace_info
                )
                pages_for_tables.append((page_idx, image[0]))

                # Whenever pages_for_tables hits YOLOX_MAX_BATCH_SIZE, submit a job
                if len(pages_for_tables) >= YOLOX_MAX_BATCH_SIZE:
                    future = executor.submit(
                        _extract_tables_and_charts,
                        pages_for_tables[:],  # pass a copy
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

        # After page loop, if we still have leftover pages_for_tables, submit one last job
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

        # Now wait for all futures to complete
        for fut in concurrent.futures.as_completed(futures):
            table_chart_items = fut.result()  # blocks until finished
            extracted_data.extend(table_chart_items)

    # DOC-LEVEL TEXT added last
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
