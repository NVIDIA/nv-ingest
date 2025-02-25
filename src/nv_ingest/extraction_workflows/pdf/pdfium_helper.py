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
import traceback
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as libpdfium

import nv_ingest.util.nim.yolox as yolox_utils
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.yolox import YOLOX_PAGE_IMAGE_PREPROC_HEIGHT
from nv_ingest.util.nim.yolox import YOLOX_PAGE_IMAGE_PREPROC_WIDTH
from nv_ingest.util.nim.yolox import get_yolox_model_name
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_pdf_image
from nv_ingest.util.pdf.metadata_aggregators import construct_page_element_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy

YOLOX_MAX_BATCH_SIZE = 8

logger = logging.getLogger(__name__)


def extract_page_elements_using_image_ensemble(
    pages: List[Tuple[int, np.ndarray, Tuple[int, int]]],
    config: PDFiumConfigSchema,
    trace_info: Optional[List] = None,
) -> List[Tuple[int, object]]:
    """
    Given a list of (page_index, image) tuples, this function calls the YOLOX-based
    inference service to extract page element annotations from all pages.

    Returns
    -------
    List[Tuple[int, object]]
        For each page, returns (page_index, joined_content) where joined_content
        is the result of combining annotations from the inference.
    """
    page_elements = []
    yolox_client = None

    # Obtain yolox_version
    # Assuming that the http endpoint is at index 1
    yolox_http_endpoint = config.yolox_endpoints[1]
    yolox_model_name = get_yolox_model_name(yolox_http_endpoint)

    try:
        model_interface = yolox_utils.YoloxPageElementsModelInterface(yolox_model_name=yolox_model_name)
        yolox_client = create_inference_client(
            config.yolox_endpoints,
            model_interface,
            config.auth_token,
            config.yolox_infer_protocol,
        )

        # Collect all page indices and images in order.
        # Optionally, collect padding offsets if present.
        image_page_indices = []
        original_images = []
        padding_offsets = []
        for page in pages:
            image_page_indices.append(page[0])
            original_images.append(page[1])
            if len(pages[0]) > 2:
                padding_offset = page[2]
            else:
                padding_offset = 0
            padding_offsets.append(padding_offset)

        # Prepare the data payload with all images.
        data = {"images": original_images}

        # Perform inference using the NimClient.
        inference_results = yolox_client.infer(
            data,
            model_name="yolox",
            max_batch_size=YOLOX_MAX_BATCH_SIZE,
            trace_info=trace_info,
            stage_name="pdf_content_extractor",
        )

        # Process results: iterate over each image's inference output.
        for annotation_dict, page_index, original_image, padding_offset in zip(
            inference_results, image_page_indices, original_images, padding_offsets
        ):
            extract_page_element_images(
                annotation_dict,
                original_image,
                page_index,
                page_elements,
                padding_offset,
            )

    except TimeoutError:
        logger.error("Timeout error during page element extraction.")
        raise

    except Exception as e:
        logger.error(f"Unhandled error during page element extraction: {str(e)}")
        traceback.print_exc()
        raise

    finally:
        if yolox_client:
            yolox_client.close()

    logger.debug(f"Extracted {len(page_elements)} page elements.")
    return page_elements


# Handle individual page element extraction and model inference
def extract_page_element_images(
    annotation_dict,
    original_image,
    page_idx,
    page_elements,
    padding_offset=(0, 0),
):
    """
    Handle the extraction of page elements from the inference results and run additional model inference.

    Parameters
    ----------
    annotation_dict : dict/
        A dictionary containing detected objects and their bounding boxes.
    original_image : np.ndarray
        The original image from which objects were detected.
    page_idx : int
        The index of the current page being processed.
    page_elements : List[Tuple[int, ImageTable]]
        A list to which extracted page elements will be appended.

    Notes
    -----
    This function iterates over detected objects, crops the original image to the bounding boxes,
    and runs additional inference on the cropped images to extract detailed information about page
    elements.

    Examples
    --------
    >>> annotation_dict = {"table": [], "chart": []}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> page_elements = []
    >>> extract_page_element_images(annotation_dict, original_image, 0, page_elements)
    """
    orig_width, orig_height, *_ = original_image.shape
    pad_width, pad_height = padding_offset

    for label in ["table", "chart", "infographic"]:
        if not annotation_dict:
            continue

        if label not in annotation_dict:
            continue

        objects = annotation_dict[label]

        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            w1, h1, w2, h2 = bbox

            cropped = crop_image(original_image, (int(w1), int(h1), int(w2), int(h2)))
            base64_img = numpy_to_base64(cropped)

            bbox_in_orig_coord = (
                int(w1) - pad_width,
                int(h1) - pad_height,
                int(w2) - pad_width,
                int(h2) - pad_height,
            )
            max_width = orig_width - 2 * pad_width
            max_height = orig_height - 2 * pad_height

            page_element_data = CroppedImageWithContent(
                content="",
                image=base64_img,
                bbox=bbox_in_orig_coord,
                max_width=max_width,
                max_height=max_height,
                type_string=label,
            )
            page_elements.append((page_idx, page_element_data))


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

    return extracted_images


def _extract_page_elements(
    pages: list,
    pdfium_config: PDFiumConfigSchema,
    page_count: int,
    source_metadata: dict,
    base_unified_metadata: dict,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    paddle_output_format: str,
    trace_info=None,
) -> list:
    """
    Always extract page elements from the given pages using YOLOX-based logic.
    The caller decides whether to call it.
    """
    extracted_page_elements = []

    page_element_results = extract_page_elements_using_image_ensemble(pages, pdfium_config, trace_info=trace_info)

    # Build metadata for each
    for page_idx, page_element in page_element_results:
        if (not extract_tables) and (page_element.type_string == "table"):
            continue
        if (not extract_charts) and (page_element.type_string == "chart"):
            continue
        if (not extract_infographics) and (page_element.type_string == "infographic"):
            continue

        if page_element.type_string == "table":
            page_element.content_format = paddle_output_format

        page_element_meta = construct_page_element_metadata(
            page_element,
            page_idx,
            page_count,
            source_metadata,
            base_unified_metadata,
        )
        extracted_page_elements.append(page_element_meta)

    return extracted_page_elements


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

    extract_infographics = kwargs.get("extract_infographics", False)
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
        f"extract_tables={extract_tables}, extract_charts={extract_charts}, "
        f"extract_infographics={extract_infographics}"
    )

    # Decide if text_depth is PAGE or DOCUMENT
    if text_depth != TextTypeEnum.PAGE:
        text_depth = TextTypeEnum.DOCUMENT

    extracted_data = []
    accumulated_text = []

    # Prepare for table/chart extraction
    pages_for_tables = []  # We'll accumulate (page_idx, np_image, padding_offset) here
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
            if extract_tables or extract_charts or extract_infographics:
                image, padding_offsets = pdfium_pages_to_numpy(
                    [page],
                    scale_tuple=(YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
                    padding_tuple=(YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
                    trace_info=trace_info,
                )
                pages_for_tables.append((page_idx, image[0], padding_offsets[0]))

                # Whenever pages_for_tables hits YOLOX_MAX_BATCH_SIZE, submit a job
                if len(pages_for_tables) >= YOLOX_MAX_BATCH_SIZE:
                    future = executor.submit(
                        _extract_page_elements,
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
                    futures.append(future)
                    pages_for_tables.clear()

            page.close()

        # After page loop, if we still have leftover pages_for_tables, submit one last job
        if (extract_tables or extract_charts or extract_infographics) and pages_for_tables:
            future = executor.submit(
                _extract_page_elements,
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
