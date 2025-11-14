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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypdfium2 as libpdfium

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.nim.default_values import YOLOX_MAX_BATCH_SIZE
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YOLOX_PAGE_IMAGE_PREPROC_WIDTH,
    YOLOX_PAGE_IMAGE_PREPROC_HEIGHT,
    YoloxPageElementsModelInterface,
    YOLOX_PAGE_IMAGE_FORMAT,
)
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFiumConfigSchema
from nv_ingest_api.internal.enums.common import TableFormatEnum, TextTypeEnum, AccessLevelEnum
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YOLOX_PAGE_DEFAULT_VERSION,
    YOLOX_PAGE_CLASS_LABELS,
    get_yolox_page_version,
)
from nv_ingest_api.util.metadata.aggregators import (
    construct_image_metadata_from_base64,
    construct_image_metadata_from_pdf_image,
    extract_pdf_metadata,
    construct_text_metadata,
    construct_page_element_metadata,
    CroppedImageWithContent,
)
from nv_ingest_api.util.nim import create_inference_client
from nv_ingest_api.util.pdf.pdfium import (
    extract_nested_simple_images_from_pdfium_page,
    extract_image_like_objects_from_pdfium_page,
    is_scanned_page,
    pdfium_pages_to_numpy,
)
from nv_ingest_api.util.image_processing import scale_image_to_encoding_size
from nv_ingest_api.util.image_processing.transforms import numpy_to_base64, crop_image

logger = logging.getLogger(__name__)


def _extract_page_elements_using_image_ensemble(
    pages: List[Tuple[int, np.ndarray, Tuple[int, int]]],
    yolox_client,
    execution_trace_log: Optional[List] = None,
) -> List[Tuple[int, object]]:
    """
    Given a list of (page_index, image) tuples and a YOLOX client, this function performs
    inference to extract page element annotations from all pages.

    Parameters
    ----------
    pages : List[Tuple[int, np.ndarray, Tuple[int, int]]]
        List of tuples containing page index, image data as numpy array,
        and optional padding offset information.
    yolox_client : object
        A pre-configured client instance for the YOLOX inference service.
    execution_trace_log : Optional[List], default=None
        List for accumulating execution trace information.

    Returns
    -------
    List[Tuple[int, object]]
        For each page, returns (page_index, joined_content) where joined_content
        is the result of combining annotations from the inference.
    """
    page_elements = []

    try:
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
            model_name="yolox_ensemble",
            max_batch_size=YOLOX_MAX_BATCH_SIZE,
            input_names=["INPUT_IMAGES", "THRESHOLDS"],
            dtypes=["BYTES", "FP32"],
            output_names=["OUTPUT"],
            trace_info=execution_trace_log,
            stage_name="pdf_extraction",
        )

        # Process results: iterate over each image's inference output.
        for annotation_dict, page_index, original_image, padding_offset in zip(
            inference_results, image_page_indices, original_images, padding_offsets
        ):
            _extract_page_element_images(
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
        logger.exception(f"Unhandled error during page element extraction: {str(e)}")
        raise

    logger.debug(f"Extracted {len(page_elements)} page elements.")
    return page_elements


# Handle individual page element extraction and model inference
def _extract_page_element_images(
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
    >>> _extract_page_element_images(annotation_dict, original_image, 0, page_elements)
    """
    orig_width, orig_height, *_ = original_image.shape
    pad_width, pad_height = padding_offset

    if annotation_dict and (set(YOLOX_PAGE_CLASS_LABELS) <= annotation_dict.keys()):
        labels = YOLOX_PAGE_CLASS_LABELS
    else:
        labels = ["table", "chart", "infographics"]

    for label in labels:
        if not annotation_dict:
            continue

        if label not in annotation_dict:
            continue

        objects = annotation_dict[label]

        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            w1, h1, w2, h2 = bbox

            cropped = crop_image(original_image, (int(w1), int(h1), int(w2), int(h2)))
            if cropped is None:
                continue

            base64_img = numpy_to_base64(cropped, format=YOLOX_PAGE_IMAGE_FORMAT)

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
    extract_images_method: str,
    page,
    page_idx: int,
    page_width: float,
    page_height: float,
    page_count: int,
    source_metadata: dict,
    base_unified_metadata: dict,
    **extract_images_params,
) -> list:
    """
    Always extract images from the given page and return a list of image metadata items.
    The caller decides whether to call this based on a flag.
    """
    if extract_images_method == "simple":
        extracted_image_data = extract_nested_simple_images_from_pdfium_page(page)
    else:  # if extract_images_method == "group"
        extracted_image_data = extract_image_like_objects_from_pdfium_page(page, merge=True, **extract_images_params)

    extracted_images = []
    for image_data in extracted_image_data:
        try:
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


def _extract_page_elements(
    pages: list,
    page_count: int,
    source_metadata: dict,
    base_unified_metadata: dict,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    page_to_text_flag_map: Dict[int, bool],
    table_output_format: str,
    yolox_endpoints: Tuple[Optional[str], Optional[str]],
    yolox_infer_protocol: str = "http",
    auth_token: Optional[str] = None,
    execution_trace_log=None,
) -> list:
    """
    Extract page elements from the given pages using YOLOX-based inference.

    This function creates a YOLOX client using the provided parameters, extracts elements
    from pages, and builds metadata for each extracted element based on the specified
    extraction flags.

    Parameters
    ----------
    pages : list
        List of page images to process.
    page_count : int
        Total number of pages in the document.
    source_metadata : dict
        Metadata about the source document.
    base_unified_metadata : dict
        Base metadata to include in all extracted elements.
    extract_tables : bool
        Flag indicating whether to extract tables.
    extract_charts : bool
        Flag indicating whether to extract charts.
    extract_infographics : bool
        Flag indicating whether to extract infographics.
    table_output_format : str
        Format to use for table content.
    yolox_endpoints : Tuple[Optional[str], Optional[str]]
        A tuple containing the gRPC and HTTP endpoints for the YOLOX service.
    yolox_infer_protocol : str, default="http"
        Protocol to use for inference (either "http" or "grpc").
    auth_token : Optional[str], default=None
        Authentication token for the inference service.
    execution_trace_log : optional
        List for accumulating execution trace information.

    Returns
    -------
    list
        List of extracted page elements with their metadata.
    """
    extracted_page_elements = []
    yolox_client = None

    try:
        # Default model name
        yolox_version = YOLOX_PAGE_DEFAULT_VERSION

        # Get the HTTP endpoint to determine the model name if needed
        yolox_http_endpoint = yolox_endpoints[1]
        if yolox_http_endpoint:
            try:
                yolox_version = get_yolox_page_version(yolox_http_endpoint)
            except Exception as e:
                logger.warning(f"Failed to get YOLOX model name from endpoint: {e}. Using default.")

        # Create the model interface
        model_interface = YoloxPageElementsModelInterface(version=yolox_version)
        # Create the inference client
        yolox_client = create_inference_client(
            yolox_endpoints,
            model_interface,
            auth_token,
            yolox_infer_protocol,
        )

        # Extract page elements using the client
        page_element_results = _extract_page_elements_using_image_ensemble(
            pages, yolox_client, execution_trace_log=execution_trace_log
        )

        # Process each extracted element based on extraction flags
        for page_idx, page_element in page_element_results:
            process_text_for_this_page = page_to_text_flag_map.get(page_idx, False)
            element_type = page_element.type_string

            page_reading_index = page_idx + 1

            # Skip elements that shouldn't be extracted based on flags
            if (not extract_tables) and (element_type == "table"):
                continue
            if (not extract_charts) and (element_type == "chart"):
                continue
            if (not extract_infographics) and (element_type == "infographic"):
                continue
            if (not process_text_for_this_page) and (element_type in {"title", "paragraph", "header_footer"}):
                continue

            # Set content format for tables
            if page_element.type_string == "table":
                page_element.content_format = table_output_format

            # Construct metadata for the page element
            page_element_meta = construct_page_element_metadata(
                page_element,
                page_reading_index,
                page_count,
                source_metadata,
                base_unified_metadata,
            )
            extracted_page_elements.append(page_element_meta)

    except Exception as e:
        logger.exception(f"Error in page element extraction: {str(e)}")
        raise

    return extracted_page_elements


def pdfium_extractor(
    pdf_stream,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_page_as_image: bool,
    extractor_config: dict,
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
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

    # Validate and extract table_output_format
    table_output_format_str = extractor_config.get("table_output_format", "pseudo_markdown")
    try:
        table_output_format = TableFormatEnum[table_output_format_str.upper()]
    except KeyError:
        raise ValueError(
            f"Invalid table_output_format: {table_output_format_str}. "
            f"Valid options: {list(TableFormatEnum.__members__.keys())}"
        )

    text_extraction_method = extractor_config.get("extract_method", "pdfium")
    extract_images_method = extractor_config.get("extract_images_method", "group")
    extract_images_params = extractor_config.get("extract_images_params", {})

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
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.UNKNOWN)

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

    # Decide if text extraction should be done at the PAGE or DOCUMENT level
    if text_depth != TextTypeEnum.PAGE:
        text_depth = TextTypeEnum.DOCUMENT

    extracted_data = []
    accumulated_text = []

    # Prepare for table/chart/infographic/text OCR extraction
    pages_for_extractions = []  # Accumulate tuples of (page_idx, np_image)
    page_to_text_flag_map = {}  # Maps page_idx -> bool (True if OCR text extraction is needed)
    futures = []  # To track asynchronous table/chart extraction tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=pdfium_config.workers_per_progress_engine) as executor:
        # PAGE LOOP
        for page_idx in range(page_count):
            page = doc.get_page(page_idx)
            page_width, page_height = page.get_size()
            page_reading_index = page_idx + 1

            is_scanned = is_scanned_page(page)
            extraction_needed_for_text = extract_text and (
                (text_extraction_method == "pdfium_hybrid" and is_scanned) or text_extraction_method == "ocr"
            )
            extraction_needed_for_structured = extract_tables or extract_charts or extract_infographics
            page_to_text_flag_map[page_idx] = extraction_needed_for_text

            # Text extraction
            if extract_text and not extraction_needed_for_text:
                page_text = _extract_page_text(page)
                if text_depth == TextTypeEnum.PAGE:
                    text_meta = construct_text_metadata(
                        [page_text],
                        pdf_metadata.keywords,
                        page_reading_index,
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
                    extract_images_method,
                    page,
                    page_reading_index,
                    page_width,
                    page_height,
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                    **extract_images_params,
                )
                extracted_data.extend(image_data)

            # Full page image extraction
            if extract_page_as_image:
                if text_extraction_method == "ocr":
                    page_text = ""
                else:
                    page_text = _extract_page_text(page)
                image, _ = pdfium_pages_to_numpy([page], scale_tuple=(16384, 16384), trace_info=execution_trace_log)
                base64_image = numpy_to_base64(image[0])
                if len(base64_image) > 2**24 - 1:
                    base64_image, _ = scale_image_to_encoding_size(base64_image, max_base64_size=2**24 - 1)
                image_meta = construct_image_metadata_from_base64(
                    base64_image,
                    page_reading_index,
                    page_count,
                    source_metadata,
                    base_unified_metadata,
                    subtype=ContentTypeEnum.PAGE_IMAGE,
                    text=page_text,
                )
                extracted_data.append(image_meta)

            # If we want OCR extraction, rasterize the page and store it
            if extraction_needed_for_text or extraction_needed_for_structured:
                image, padding_offsets = pdfium_pages_to_numpy(
                    [page],
                    scale_tuple=(YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
                    padding_tuple=(YOLOX_PAGE_IMAGE_PREPROC_WIDTH, YOLOX_PAGE_IMAGE_PREPROC_HEIGHT),
                    trace_info=execution_trace_log,
                )
                pages_for_extractions.append((page_idx, image[0], padding_offsets[0]))

                # Whenever pages_for_extractions hits YOLOX_MAX_BATCH_SIZE, submit a job
                if len(pages_for_extractions) >= YOLOX_MAX_BATCH_SIZE:
                    future = executor.submit(
                        _extract_page_elements,
                        pages_for_extractions[:],  # pass a copy
                        page_count,
                        source_metadata,
                        base_unified_metadata,
                        extract_tables=extract_tables,
                        extract_charts=extract_charts,
                        extract_infographics=extract_infographics,
                        page_to_text_flag_map=page_to_text_flag_map,
                        table_output_format=table_output_format,
                        yolox_endpoints=pdfium_config.yolox_endpoints,
                        yolox_infer_protocol=pdfium_config.yolox_infer_protocol,
                        auth_token=pdfium_config.auth_token,
                        execution_trace_log=execution_trace_log,
                    )
                    futures.append(future)
                    pages_for_extractions.clear()

            page.close()

        # After page loop, if we still have leftover pages_for_extractions, submit one last job
        if (extraction_needed_for_text or extraction_needed_for_structured) and pages_for_extractions:
            future = executor.submit(
                _extract_page_elements,
                pages_for_extractions[:],
                page_count,
                source_metadata,
                base_unified_metadata,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_infographics=extract_infographics,
                page_to_text_flag_map=page_to_text_flag_map,
                table_output_format=table_output_format,
                yolox_endpoints=pdfium_config.yolox_endpoints,
                yolox_infer_protocol=pdfium_config.yolox_infer_protocol,
                auth_token=pdfium_config.auth_token,
                execution_trace_log=execution_trace_log,
            )
            futures.append(future)

            pages_for_extractions.clear()

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

    doc.close()

    logger.debug(f"Extracted {len(extracted_data)} items from PDF.")
    return extracted_data
