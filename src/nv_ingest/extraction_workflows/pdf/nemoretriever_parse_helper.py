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
import math
import traceback
import uuid
import concurrent.futures
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import List

import numpy as np
import pypdfium2 as pdfium

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentSubtypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.schemas.pdf_extractor_schema import NemoRetrieverParseConfigSchema
from nv_ingest.util.exception_handlers.pdf import pdfium_exception_handler
from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim import nemoretriever_parse as nemoretriever_parse_utils
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import LatexTable
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_pdf_image
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.extraction_workflows.pdf.pdfium_helper import _extract_page_elements
from nv_ingest.extraction_workflows.pdf.pdfium_helper import YOLOX_MAX_BATCH_SIZE
from nv_ingest.extraction_workflows.pdf.pdfium_helper import YOLOX_PAGE_IMAGE_PREPROC_HEIGHT
from nv_ingest.extraction_workflows.pdf.pdfium_helper import YOLOX_PAGE_IMAGE_PREPROC_WIDTH


logger = logging.getLogger(__name__)

NEMORETRIEVER_PARSE_RENDER_DPI = 300
NEMORETRIEVER_PARSE_MAX_WIDTH = 1024
NEMORETRIEVER_PARSE_MAX_HEIGHT = 1280
NEMORETRIEVER_PARSE_MAX_BATCH_SIZE = 8


# Define a helper function to use nemoretriever_parse to extract text from a base64 encoded bytestram PDF
def nemoretriever_parse(
    pdf_stream,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    trace_info: Optional[List] = None,
    **kwargs,
):
    """
    Helper function to use nemoretriever_parse to extract text from a bytestream PDF.

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
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.
    """
    logger.debug("Extracting PDF with nemoretriever_parse backend.")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    extract_infographics = kwargs.get("extract_infographics", False)
    extract_tables_method = kwargs.get("extract_tables_method", "yolox")
    identify_nearby_objects = kwargs.get("identify_nearby_objects", True)
    paddle_output_format = kwargs.get("paddle_output_format", "pseudo_markdown")
    paddle_output_format = TableFormatEnum[paddle_output_format.upper()]

    if (extract_tables_method == "yolox") and (extract_tables or extract_charts or extract_infographics):
        pdfium_config = kwargs.get("pdfium_config", {})
        if isinstance(pdfium_config, dict):
            pdfium_config = PDFiumConfigSchema(**pdfium_config)
    nemoretriever_parse_config = kwargs.get("nemoretriever_parse_config", {})
    if isinstance(nemoretriever_parse_config, dict):
        nemoretriever_parse_config = NemoRetrieverParseConfigSchema(**nemoretriever_parse_config)

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

    nemoretriever_parse_client = None
    if extract_text:
        nemoretriever_parse_client = _create_clients(nemoretriever_parse_config)

    max_workers = nemoretriever_parse_config.workers_per_progress_engine
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        for page_idx in range(page_count):
            page = doc.get_page(page_idx)

            page_image, padding_offset = _convert_pdfium_page_to_numpy_for_parser(page)
            pages_for_ocr.append((page_idx, page_image))
            page_image_for_tables, padding_offset_for_tables = _convert_pdfium_page_to_numpy_for_yolox(page)
            pages_for_tables.append((page_idx, page_image_for_tables, padding_offset_for_tables))

            page.close()

            # Whenever pages_as_images hits NEMORETRIEVER_PARSE_MAX_BATCH_SIZE, submit a job
            if (extract_text) and (len(pages_for_ocr) >= NEMORETRIEVER_PARSE_MAX_BATCH_SIZE):
                future_parser = executor.submit(
                    lambda *args, **kwargs: ("parser", _extract_text_and_bounding_boxes(*args, **kwargs)),
                    pages_for_ocr[:],  # pass a copy
                    nemoretriever_parse_client,
                    trace_info=trace_info,
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
                    pdfium_config,
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                    extract_tables,
                    extract_charts,
                    extract_infographics,
                    paddle_output_format,
                    trace_info=trace_info,
                )
                futures.append(future_yolox)
                pages_for_tables.clear()

        # After page loop, if we still have leftover pages_as_images, submit one last job
        if extract_text and pages_for_ocr:
            future_parser = executor.submit(
                lambda *args, **kwargs: ("parser", _extract_text_and_bounding_boxes(*args, **kwargs)),
                pages_for_ocr[:],  # pass a copy
                nemoretriever_parse_client,
                trace_info=trace_info,
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
                pdfium_config,
                page_count,
                source_metadata,
                base_unified_metadata,
                extract_tables,
                extract_charts,
                extract_infographics,
                paddle_output_format,
                trace_info=trace_info,
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
                math.floor(bbox["xmin"] * NEMORETRIEVER_PARSE_MAX_WIDTH),
                math.floor(bbox["ymin"] * NEMORETRIEVER_PARSE_MAX_HEIGHT),
                math.ceil(bbox["xmax"] * NEMORETRIEVER_PARSE_MAX_WIDTH),
                math.ceil(bbox["ymax"] * NEMORETRIEVER_PARSE_MAX_HEIGHT),
            ]

            if cls not in nemoretriever_parse_utils.ACCEPTED_CLASSES:
                continue

            if identify_nearby_objects:
                _insert_page_nearby_blocks(page_nearby_blocks, cls, txt, transformed_bbox)

            if extract_text:
                page_text.append(txt)

            if (extract_tables_method == "nemoretriever_parse") and (extract_tables) and (cls == "Table"):
                table = LatexTable(
                    latex=txt,
                    bbox=transformed_bbox,
                    max_width=NEMORETRIEVER_PARSE_MAX_WIDTH,
                    max_height=NEMORETRIEVER_PARSE_MAX_HEIGHT,
                )
                accumulated_tables.append(table)

            if extract_images and (cls == "Picture"):
                if page is None:
                    page = doc.get_page(page_idx)
                if page_image is None:
                    page_image, _ = _convert_pdfium_page_to_numpy_for_parser(page)

                img_numpy = crop_image(page_image, transformed_bbox)

                if img_numpy is not None:
                    base64_img = numpy_to_base64(img_numpy)
                    image = Base64Image(
                        image=base64_img,
                        bbox=transformed_bbox,
                        width=img_numpy.shape[1],
                        height=img_numpy.shape[0],
                        max_width=NEMORETRIEVER_PARSE_MAX_WIDTH,
                        max_height=NEMORETRIEVER_PARSE_MAX_HEIGHT,
                    )
                    accumulated_images.append(image)

        # If NemoRetrieverParse fails to extract anything, fall back to using pdfium.
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
                    bbox_max_dimensions=(NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT),
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

    if nemoretriever_parse_client:
        nemoretriever_parse_client.close()
    doc.close()

    return extracted_data


def _extract_text_and_bounding_boxes(
    pages: list,
    nemoretriever_parse_client,
    trace_info=None,
) -> list:

    # Collect all page indices and images in order.
    image_page_indices = [page[0] for page in pages]
    original_images = [page[1] for page in pages]

    # Prepare the data payload with all images.
    data = {"images": original_images}

    # Perform inference using the NimClient.
    inference_results = nemoretriever_parse_client.infer(
        data=data,
        model_name="nemoretriever_parse",
        stage_name="pdf_content_extractor",
        max_batch_size=NEMORETRIEVER_PARSE_MAX_BATCH_SIZE,
        trace_info=trace_info,
    )

    return list(zip(image_page_indices, inference_results))


def _create_clients(nemoretriever_parse_config):
    model_interface = nemoretriever_parse_utils.NemoRetrieverParseModelInterface(
        model_name=nemoretriever_parse_config.model_name,
    )
    nemoretriever_parse_client = create_inference_client(
        nemoretriever_parse_config.nemoretriever_parse_endpoints,
        model_interface,
        nemoretriever_parse_config.auth_token,
        nemoretriever_parse_config.nemoretriever_parse_infer_protocol,
        nemoretriever_parse_config.timeout,
    )

    return nemoretriever_parse_client


def _send_inference_request(
    nemoretriever_parse_client,
    image_array: np.ndarray,
) -> Dict[str, Any]:

    try:
        # NIM only supports processing one page at a time (batch size = 1).
        data = {"image": image_array}
        response = nemoretriever_parse_client.infer(
            data=data,
            model_name="nemoretriever_parse",
        )
    except Exception as e:
        logger.error(f"Unhandled error during NemoRetrieverParse inference: {e}")
        traceback.print_exc()
        raise e

    return response


def _convert_pdfium_page_to_numpy_for_parser(
    page: pdfium.PdfPage,
    render_dpi: int = NEMORETRIEVER_PARSE_RENDER_DPI,
    scale_tuple: Tuple[int, int] = (NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT),
    padding_tuple: Tuple[int, int] = (NEMORETRIEVER_PARSE_MAX_WIDTH, NEMORETRIEVER_PARSE_MAX_HEIGHT),
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
    if cls in nemoretriever_parse_utils.ACCEPTED_TEXT_CLASSES:
        nearby_blocks_key = "text"
    elif cls in nemoretriever_parse_utils.ACCEPTED_TABLE_CLASSES:
        nearby_blocks_key = "structured"
    elif cls in nemoretriever_parse_utils.ACCEPTED_IMAGE_CLASSES:
        nearby_blocks_key = "images"

    page_nearby_blocks[nearby_blocks_key]["content"].append(txt)
    page_nearby_blocks[nearby_blocks_key]["bbox"].append(bbox)
    page_nearby_blocks[nearby_blocks_key]["type"].append(cls)


@pdfium_exception_handler(descriptor="nemoretriever_parse")
def _construct_table_metadata(
    table: LatexTable,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    content = table.latex
    table_format = TableFormatEnum.LATEX
    subtype = ContentSubtypeEnum.TABLE
    description = StdContentDescEnum.PDF_TABLE

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
