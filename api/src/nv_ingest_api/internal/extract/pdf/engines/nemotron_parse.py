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

import io
import logging
import math
import uuid
import concurrent.futures
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import List

import numpy as np
import pypdfium2 as pdfium

from nv_ingest_api.internal.extract.pdf.engines.pdfium import _extract_page_elements
from nv_ingest_api.internal.primitives.nim.model_interface import nemotron_parse as nemotron_parse_utils
from nv_ingest_api.internal.enums.common import AccessLevelEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.enums.common import ContentDescriptionEnum
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YOLOX_PAGE_IMAGE_PREPROC_WIDTH,
    YOLOX_PAGE_IMAGE_PREPROC_HEIGHT,
    YOLOX_PAGE_IMAGE_FORMAT,
)
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import NemotronParseConfigSchema
from nv_ingest_api.util.metadata.aggregators import (
    extract_pdf_metadata,
    LatexTable,
    Base64Image,
    construct_image_metadata_from_pdf_image,
    construct_text_metadata,
)
from nv_ingest_api.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest_api.internal.primitives.nim.default_values import YOLOX_MAX_BATCH_SIZE
from nv_ingest_api.util.exception_handlers.pdf import pdfium_exception_handler
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64, crop_image
from nv_ingest_api.util.nim import create_inference_client


logger = logging.getLogger(__name__)

NEMOTRON_PARSE_RENDER_DPI = 300
NEMOTRON_PARSE_MAX_WIDTH = 1024
NEMOTRON_PARSE_MAX_HEIGHT = 1280
NEMOTRON_PARSE_MAX_BATCH_SIZE = 8


# Define a helper function to use nemotron_parse to extract text from a base64 encoded bytestram PDF
def nemotron_parse_extractor(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extractor_config: dict,
    execution_trace_log: Optional[List[Any]] = None,
) -> str:
    """
    Helper function to use nemotron_parse to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_infographics : bool
        Specifies whether to extract infographics.
    extract_charts : bool
        Specifies whether to extract charts.
    execution_trace_log : Optional[List], optional
        Trace information for debugging purposes (default is None).
    extractor_config : dict
        A dictionary containing additional extraction parameters. Expected keys include:
            - row_data : dict
            - text_depth : str, optional (default is "page")
            - extract_tables_method : str, optional (default is "yolox")
            - identify_nearby_objects : bool, optional (default is True)
            - table_output_format : str, optional (default is "pseudo_markdown")
            - pdfium_config : dict, optional (configuration for PDFium)
            - nemotron_parse_config : dict, optional (configuration for Nemotron Parse)
            - metadata_column : str, optional (default is "metadata")

    Returns
    -------
    str
        A string of extracted text.

    Raises
    ------
    ValueError
        If required keys are missing in extractor_config or invalid values are provided.
    KeyError
        If required keys are missing in row_data.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Extracting PDF with nemotron_parse backend.")

    # Retrieve row_data from extractor_config.
    row_data = extractor_config.get("row_data")
    if row_data is None:
        raise ValueError("Missing 'row_data' in extractor_config.")

    # Get source_id from row_data.
    try:
        source_id = row_data["source_id"]
    except KeyError:
        raise KeyError("row_data must contain 'source_id'.")

    # Get and validate text_depth.
    text_depth_str = extractor_config.get("text_depth", "page")
    try:
        text_depth = TextTypeEnum[text_depth_str.upper()]
    except KeyError:
        valid_options = [e.name.lower() for e in TextTypeEnum]
        raise ValueError(f"Invalid text_depth value: {text_depth_str}. Expected one of: {valid_options}")

    # Get extraction method for tables.
    extract_tables_method = extractor_config.get("extract_tables_method", "yolox")

    # Flag for identifying nearby objects.
    identify_nearby_objects = extractor_config.get("identify_nearby_objects", True)

    # Get and validate table_output_format.
    table_output_format_str = extractor_config.get("table_output_format", "pseudo_markdown")
    try:
        table_output_format = TableFormatEnum[table_output_format_str.upper()]
    except KeyError:
        valid_options = [e.name.lower() for e in TableFormatEnum]
        raise ValueError(
            f"Invalid table_output_format value: {table_output_format_str}. Expected one of: {valid_options}"
        )

    # Process nemotron_parse configuration.
    nemotron_parse_config_raw = extractor_config.get("nemotron_parse_config", {})
    if isinstance(nemotron_parse_config_raw, dict):
        nemotron_parse_config = NemotronParseConfigSchema(**nemotron_parse_config_raw)
    elif isinstance(nemotron_parse_config_raw, NemotronParseConfigSchema):
        nemotron_parse_config = nemotron_parse_config_raw
    else:
        raise ValueError("`nemotron_parse_config` must be a dictionary or a NemotronParseConfigSchema instance.")

    # Get base metadata.
    metadata_col = extractor_config.get("metadata_column", "metadata")
    if hasattr(row_data, "index") and metadata_col in row_data.index:
        base_unified_metadata = row_data[metadata_col]
    else:
        base_unified_metadata = row_data.get(metadata_col, {})

    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.UNKNOWN)

    extracted_data = []
    doc = pdfium.PdfDocument(pdf_stream)
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

    accumulated_text = []
    accumulated_tables = []
    accumulated_images = []

    pages_for_ocr = []  # We'll accumulate (page_idx, np_image) here
    pages_for_tables = []  # We'll accumulate (page_idx, np_image) here
    futures = []  # We'll keep track of all the Future objects for table/charts

    nemotron_parse_client = None
    if extract_text:
        nemotron_parse_client = _create_clients(nemotron_parse_config)

    max_workers = nemotron_parse_config.workers_per_progress_engine
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        for page_idx in range(page_count):
            page = doc.get_page(page_idx)

            page_image, padding_offset = _convert_pdfium_page_to_numpy_for_parser(page)
            pages_for_ocr.append((page_idx, page_image))
            page_image_for_tables, padding_offset_for_tables = _convert_pdfium_page_to_numpy_for_yolox(page)
            pages_for_tables.append((page_idx, page_image_for_tables, padding_offset_for_tables))

            page.close()

            # Whenever pages_as_images hits NEMOTRON_PARSE_MAX_BATCH_SIZE, submit a job
            if (extract_text) and (len(pages_for_ocr) >= NEMOTRON_PARSE_MAX_BATCH_SIZE):
                future_parser = executor.submit(
                    lambda *args, **kwargs: ("parser", _extract_text_and_bounding_boxes(*args, **kwargs)),
                    pages_for_ocr[:],  # pass a copy
                    nemotron_parse_client,
                    execution_trace_log=execution_trace_log,
                )
                futures.append(future_parser)
                pages_for_ocr.clear()

            # Whenever pages_as_images hits YOLOX_MAX_BATCH_SIZE, submit a job
            if (
                (extract_tables_method == "yolox")
                and (extract_tables or extract_charts or extract_infographics)
                and (len(pages_for_tables) >= YOLOX_MAX_BATCH_SIZE)
            ):
                future_yolox = executor.submit(
                    lambda *args, **kwargs: ("yolox", _extract_page_elements(*args, **kwargs)),
                    pages_for_tables[:],  # pass a copy
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                    extract_tables,
                    extract_charts,
                    extract_infographics,
                    {},  # page_to_text_flag_map
                    table_output_format,
                    nemotron_parse_config.yolox_endpoints,
                    nemotron_parse_config.yolox_infer_protocol,
                    nemotron_parse_config.auth_token,
                    execution_trace_log=execution_trace_log,
                )
                futures.append(future_yolox)
                pages_for_tables.clear()

        # After page loop, if we still have leftover pages_as_images, submit one last job
        if extract_text and pages_for_ocr:
            future_parser = executor.submit(
                lambda *args, **kwargs: ("parser", _extract_text_and_bounding_boxes(*args, **kwargs)),
                pages_for_ocr[:],  # pass a copy
                nemotron_parse_client,
                execution_trace_log=execution_trace_log,
            )
            futures.append(future_parser)
            pages_for_ocr.clear()

        if (
            (extract_tables_method == "yolox")
            and (extract_tables or extract_charts or extract_infographics)
            and pages_for_tables
        ):
            future_yolox = executor.submit(
                lambda *args, **kwargs: ("yolox", _extract_page_elements(*args, **kwargs)),
                pages_for_tables[:],
                page_count,
                source_metadata,
                base_unified_metadata,
                extract_tables,
                extract_charts,
                extract_infographics,
                {},  # page_to_text_flag_map
                table_output_format,
                nemotron_parse_config.yolox_endpoints,
                nemotron_parse_config.yolox_infer_protocol,
                nemotron_parse_config.auth_token,
                execution_trace_log=execution_trace_log,
            )
            futures.append(future_yolox)
            pages_for_tables.clear()

        parser_results = []
        # Now wait for all futures to complete
        for fut in concurrent.futures.as_completed(futures):
            model_name, extracted_items = fut.result()  # blocks until finished
            if (model_name == "yolox") and (extract_tables or extract_charts or extract_infographics):
                extracted_data.extend(extracted_items)
            elif model_name == "parser":
                parser_results.extend(extracted_items)

    for page_idx, parser_output in parser_results:
        page = None
        page_image = None
        page_text = []

        page_nearby_blocks = {
            "text": {"content": [], "bbox": [], "type": []},
            "images": {"content": [], "bbox": [], "type": []},
            "structured": {"content": [], "bbox": [], "type": []},
        }

        for bbox_dict in parser_output:
            cls = bbox_dict["type"]
            bbox = bbox_dict["bbox"]
            txt = bbox_dict["text"]

            transformed_bbox = [
                math.floor(bbox["xmin"] * NEMOTRON_PARSE_MAX_WIDTH),
                math.floor(bbox["ymin"] * NEMOTRON_PARSE_MAX_HEIGHT),
                math.ceil(bbox["xmax"] * NEMOTRON_PARSE_MAX_WIDTH),
                math.ceil(bbox["ymax"] * NEMOTRON_PARSE_MAX_HEIGHT),
            ]

            if cls not in nemotron_parse_utils.ACCEPTED_CLASSES:
                continue

            if identify_nearby_objects:
                _insert_page_nearby_blocks(page_nearby_blocks, cls, txt, transformed_bbox)

            if extract_text:
                page_text.append(txt)

            if (extract_tables_method == "nemotron_parse") and (extract_tables) and (cls == "Table"):
                table = LatexTable(
                    latex=txt,
                    bbox=transformed_bbox,
                    max_width=NEMOTRON_PARSE_MAX_WIDTH,
                    max_height=NEMOTRON_PARSE_MAX_HEIGHT,
                )
                accumulated_tables.append(table)

            if extract_images and (cls == "Picture"):
                if page is None:
                    page = doc.get_page(page_idx)
                if page_image is None:
                    page_image, _ = _convert_pdfium_page_to_numpy_for_parser(page)

                img_numpy = crop_image(page_image, transformed_bbox)

                if img_numpy is not None:
                    base64_img = numpy_to_base64(img_numpy, format=YOLOX_PAGE_IMAGE_FORMAT)
                    image = Base64Image(
                        image=base64_img,
                        bbox=transformed_bbox,
                        width=img_numpy.shape[1],
                        height=img_numpy.shape[0],
                        max_width=NEMOTRON_PARSE_MAX_WIDTH,
                        max_height=NEMOTRON_PARSE_MAX_HEIGHT,
                    )
                    accumulated_images.append(image)

        # If Nemotron Parse fails to extract anything, fall back to using pdfium.
        if not "".join(page_text).strip():
            if page is None:
                page = doc.get_page(page_idx)
            page_text = [page.get_textpage().get_text_bounded()]

        accumulated_text.extend(page_text)

        # Construct tables
        if extract_tables:
            for table in accumulated_tables:
                extracted_data.append(
                    _construct_table_metadata(
                        table,
                        page_idx,
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                    )
                )
            accumulated_tables = []

        # Construct images
        if extract_images:
            for image in accumulated_images:
                extracted_data.append(
                    construct_image_metadata_from_pdf_image(
                        image,
                        page_idx,
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                    )
                )
            accumulated_images = []

        # Construct text - page
        if (extract_text) and (text_depth == TextTypeEnum.PAGE):
            extracted_data.append(
                construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.keywords,
                    page_idx,
                    -1,
                    -1,
                    -1,
                    page_count,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                    delimiter="\n\n",
                    bbox_max_dimensions=(NEMOTRON_PARSE_MAX_WIDTH, NEMOTRON_PARSE_MAX_HEIGHT),
                    nearby_objects=page_nearby_blocks,
                )
            )
            accumulated_text = []

    # Construct text - document
    if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
        text_extraction = construct_text_metadata(
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
            delimiter="\n\n",
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

    if nemotron_parse_client:
        nemotron_parse_client.close()
    doc.close()

    return extracted_data


def _extract_text_and_bounding_boxes(
    pages: list,
    nemotron_parse_client,
    execution_trace_log=None,
) -> list:

    # Collect all page indices and images in order.
    image_page_indices = [page[0] for page in pages]
    original_images = [page[1] for page in pages]

    # Prepare the data payload with all images.
    data = {"images": original_images}

    # Perform inference using the NimClient.
    inference_results = nemotron_parse_client.infer(
        data=data,
        model_name="nemotron_parse",
        stage_name="pdf_extraction",
        max_batch_size=NEMOTRON_PARSE_MAX_BATCH_SIZE,
        execution_trace_log=execution_trace_log,
    )

    return list(zip(image_page_indices, inference_results))


def _create_clients(nemotron_parse_config):
    model_interface = nemotron_parse_utils.NemotronParseModelInterface(
        model_name=nemotron_parse_config.nemotron_parse_model_name,
    )
    nemotron_parse_client = create_inference_client(
        nemotron_parse_config.nemotron_parse_endpoints,
        model_interface,
        nemotron_parse_config.auth_token,
        nemotron_parse_config.nemotron_parse_infer_protocol,
        nemotron_parse_config.timeout,
    )

    return nemotron_parse_client


def _send_inference_request(
    nemotron_parse_client,
    image_array: np.ndarray,
) -> Dict[str, Any]:

    try:
        # NIM only supports processing one page at a time (batch size = 1).
        data = {"image": image_array}
        response = nemotron_parse_client.infer(
            data=data,
            model_name="nemotron_parse",
        )
    except Exception as e:
        logger.exception(f"Unhandled error during Nemotron Parse inference: {e}")
        raise e

    return response


def _convert_pdfium_page_to_numpy_for_parser(
    page: pdfium.PdfPage,
    render_dpi: int = NEMOTRON_PARSE_RENDER_DPI,
    scale_tuple: Tuple[int, int] = (NEMOTRON_PARSE_MAX_WIDTH, NEMOTRON_PARSE_MAX_HEIGHT),
    padding_tuple: Tuple[int, int] = (NEMOTRON_PARSE_MAX_WIDTH, NEMOTRON_PARSE_MAX_HEIGHT),
) -> np.ndarray:
    page_images, padding_offsets = pdfium_pages_to_numpy(
        [page], render_dpi=render_dpi, scale_tuple=scale_tuple, padding_tuple=padding_tuple
    )

    return page_images[0], padding_offsets[0]


def _convert_pdfium_page_to_numpy_for_yolox(
    page: pdfium.PdfPage,
    scale_tuple: Tuple[int, int] = (YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
    padding_tuple: Tuple[int, int] = (YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
) -> np.ndarray:
    page_images, padding_offsets = pdfium_pages_to_numpy([page], scale_tuple=scale_tuple, padding_tuple=padding_tuple)

    return page_images[0], padding_offsets[0]


def _insert_page_nearby_blocks(
    page_nearby_blocks: Dict[str, Any],
    cls: str,
    txt: str,
    bbox: str,
):
    if cls in nemotron_parse_utils.ACCEPTED_TEXT_CLASSES:
        nearby_blocks_key = "text"
    elif cls in nemotron_parse_utils.ACCEPTED_TABLE_CLASSES:
        nearby_blocks_key = "structured"
    elif cls in nemotron_parse_utils.ACCEPTED_IMAGE_CLASSES:
        nearby_blocks_key = "images"

    page_nearby_blocks[nearby_blocks_key]["content"].append(txt)
    page_nearby_blocks[nearby_blocks_key]["bbox"].append(bbox)
    page_nearby_blocks[nearby_blocks_key]["type"].append(cls)


@pdfium_exception_handler(descriptor="nemotron_parse")
def _construct_table_metadata(
    table: LatexTable,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    content = table.latex
    table_format = TableFormatEnum.LATEX
    subtype = ContentTypeEnum.TABLE
    description = ContentDescriptionEnum.PDF_TABLE

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
    table_metadata = {
        "caption": "",
        "table_content": content,
        "table_format": table_format,
        "table_location": table.bbox,
        "table_location_max_dimensions": (table.max_width, table.max_height),
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
