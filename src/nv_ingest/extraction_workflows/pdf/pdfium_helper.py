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
import traceback

from math import log
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as libpdfium
import tritonclient.grpc as grpcclient
import nv_ingest.util.nim.yolox as yolox_utils

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.helpers import perform_model_inference
from nv_ingest.util.nim.yolox import prepare_images_for_inference
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_pdf_image
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import PDFIUM_PAGEOBJ_MAPPING
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy
from nv_ingest.util.pdf.pdfium import pdfium_try_get_bitmap_as_numpy

YOLOX_MAX_BATCH_SIZE = 8
YOLOX_MAX_WIDTH = 1536
YOLOX_MAX_HEIGHT = 1536
YOLOX_NUM_CLASSES = 3
YOLOX_CONF_THRESHOLD = 0.01
YOLOX_IOU_THRESHOLD = 0.5
YOLOX_MIN_SCORE = 0.1
YOLOX_FINAL_SCORE = 0.48

logger = logging.getLogger(__name__)


def extract_tables_and_charts_using_image_ensemble(
        pages: List[libpdfium.PdfPage],
        config: PDFiumConfigSchema,
        max_batch_size: int = YOLOX_MAX_BATCH_SIZE,
        num_classes: int = YOLOX_NUM_CLASSES,
        conf_thresh: float = YOLOX_CONF_THRESHOLD,
        iou_thresh: float = YOLOX_IOU_THRESHOLD,
        min_score: float = YOLOX_MIN_SCORE,
        final_thresh: float = YOLOX_FINAL_SCORE,
        trace_info: Optional[List] = None,
) -> List[Tuple[int, CroppedImageWithContent]]:
    """
    Extract tables and charts from a series of document pages using an ensemble of image-based models.

    This function processes a list of document pages to detect and extract tables and charts.
    It uses a sequence of models hosted on different inference servers to achieve this.

    Parameters
    ----------
    pages : List[libpdfium.PdfPage]
        A list of document pages to process.
    yolox_nim_endpoint_url : str
        The URL of the Triton inference server endpoint for the primary model.
    model_name : str
        The name of the model to use on the Triton server.
    max_batch_size : int, optional
        The maximum number of pages to process in a single batch (default is 16).
    num_classes : int, optional
        The number of classes the model is trained to detect (default is 3).
    conf_thresh : float, optional
        The confidence threshold for detection (default is 0.01).
    iou_thresh : float, optional
        The Intersection Over Union (IoU) threshold for non-maximum suppression (default is 0.5).
    min_score : float, optional
        The minimum score threshold for considering a detection valid (default is 0.1).
    final_thresh: float, optional
        Threshold for keeping a bounding box applied after postprocessing. (default is 0.48)


    Returns
    -------
    List[Tuple[int, ImageTable]]
        A list of tuples, each containing the page index and an `ImageTable` or `ImageChart` object
        representing the detected table or chart along with its associated metadata.

    Notes
    -----
    This function centralizes the management of inference clients, handles batch processing
    of pages, and manages the inference and post-processing of results from multiple models.
    It ensures that the results are properly associated with their corresponding pages and
    regions within those pages.

    Examples
    --------
    >>> pages = [libpdfium.PdfPage(), libpdfium.PdfPage()]  # List of pages from a document
    >>> tables_and_charts = extract_tables_and_charts_using_image_ensemble(
    ...     pages,
    ...     yolox_nim_endpoint_url="http://localhost:8000",
    ...     model_name="model",
    ...     max_batch_size=8,
    ...     num_classes=3,
    ...     conf_thresh=0.5,
    ...     iou_thresh=0.5,
    ...     min_score=0.2
    ... )
    >>> for page_idx, image_obj in tables_and_charts:
    ...     print(f"Page: {page_idx}, Object: {image_obj}")
    """
    tables_and_charts = []

    yolox_client = None
    try:
        yolox_client = create_inference_client(config.yolox_endpoints, config.auth_token)

        batches = []
        i = 0
        while i < len(pages):
            batch_size = min(2 ** int(log(len(pages) - i, 2)), max_batch_size)
            batches.append(pages[i: i + batch_size])  # noqa: E203
            i += batch_size

        page_index = 0
        for batch in batches:
            original_images, _ = pdfium_pages_to_numpy(
                batch, scale_tuple=(YOLOX_MAX_WIDTH, YOLOX_MAX_HEIGHT), trace_info=trace_info
            )

            # original images is an implicitly indexed list of pages
            original_image_shapes = [image.shape for image in original_images]
            input_array = prepare_images_for_inference(original_images)

            output_array = perform_model_inference(yolox_client, "yolox", input_array, trace_info=trace_info)

            # Get back inference results
            yolox_annotated_detections = process_inference_results(
                output_array, original_image_shapes, num_classes, conf_thresh, iou_thresh, min_score, final_thresh
            )

            for annotation_dict, original_image in zip(yolox_annotated_detections, original_images):
                extract_table_and_chart_images(
                    annotation_dict,
                    original_image,
                    page_index,
                    tables_and_charts,
                )

                page_index += 1

    except Exception as e:
        logger.error(f"Unhandled error during table/chart extraction: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if isinstance(yolox_client, grpcclient.InferenceServerClient):
            yolox_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts.")

    return tables_and_charts


def process_inference_results(
        output_array: np.ndarray,
        original_image_shapes: List[Tuple[int, int]],
        num_classes: int,
        conf_thresh: float,
        iou_thresh: float,
        min_score: float,
        final_thresh: float,
):
    """
    Process the model output to generate detection results and expand bounding boxes.

    Parameters
    ----------
    output_array : np.ndarray
        The raw output from the model inference.
    original_image_shapes : List[Tuple[int, int]]
        The shapes of the original images before resizing, used for scaling bounding boxes.
    num_classes : int
        The number of classes the model can detect.
    conf_thresh : float
        The confidence threshold for detecting objects.
    iou_thresh : float
        The Intersection Over Union (IoU) threshold for non-maximum suppression.
    min_score : float
        The minimum score for keeping a detection.
    final_thresh: float
        Threshold for keeping a bounding box applied after postprocessing.


    Returns
    -------
    List[dict]
        A list of dictionaries, each containing processed detection results including expanded bounding boxes.

    Notes
    -----
    This function applies non-maximum suppression to the model's output and scales the bounding boxes back to the
    original image size.

    Examples
    --------
    >>> output_array = np.random.rand(2, 100, 85)
    >>> original_image_shapes = [(1536, 1536), (1536, 1536)]
    >>> results = process_inference_results(output_array, original_image_shapes, 80, 0.5, 0.5, 0.1)
    >>> len(results)
    2
    """
    pred = yolox_utils.postprocess_model_prediction(
        output_array, num_classes, conf_thresh, iou_thresh, class_agnostic=True
    )
    results = yolox_utils.postprocess_results(pred, original_image_shapes, min_score=min_score)

    annotation_dicts = [yolox_utils.expand_chart_bboxes(annotation_dict) for annotation_dict in results]
    inference_results = []

    # Filter out bounding boxes below the final threshold
    for annotation_dict in annotation_dicts:
        new_dict = {}
        if "table" in annotation_dict:
            new_dict["table"] = [bb for bb in annotation_dict["table"] if bb[4] >= final_thresh]
        if "chart" in annotation_dict:
            new_dict["chart"] = [bb for bb in annotation_dict["chart"] if bb[4] >= final_thresh]
        if "title" in annotation_dict:
            new_dict["title"] = annotation_dict["title"]
        inference_results.append(new_dict)

    return inference_results


# Handle individual table/chart extraction and model inference
def extract_table_and_chart_images(
        annotation_dict,
        original_image,
        page_idx,
        tables_and_charts,
):
    """
    Handle the extraction of tables and charts from the inference results and run additional model inference.

    Parameters
    ----------
    annotation_dict : dict/
        A dictionary containing detected objects and their bounding boxes.
    original_image : np.ndarray
        The original image from which objects were detected.
    page_idx : int
        The index of the current page being processed.
    tables_and_charts : List[Tuple[int, ImageTable]]
        A list to which extracted tables and charts will be appended.

    Notes
    -----
    This function iterates over detected objects, crops the original image to the bounding boxes,
    and runs additional inference on the cropped images to extract detailed information about tables
    and charts.

    Examples
    --------
    >>> annotation_dict = {"table": [], "chart": []}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> tables_and_charts = []
    >>> extract_table_and_chart_images(annotation_dict, original_image, 0, tables_and_charts)
    """

    width, height, *_ = original_image.shape
    for label in ["table", "chart"]:
        if not annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = bbox * np.array([height, width, height, width])

            cropped = crop_image(original_image, (h1, w1, h2, w2))
            base64_img = numpy_to_base64(cropped)

            table_data = CroppedImageWithContent(
                content="", image=base64_img, bbox=(w1, h1, w2, h2), max_width=width,
                max_height=height, type_string=label
            )
            tables_and_charts.append((page_idx, table_data))


# Define a helper function to use unstructured-io to extract text from a base64
# encoded bytestream PDF
def pdfium_extractor(
        pdf_stream,
        extract_text: bool,
        extract_images: bool,
        extract_tables: bool,
        extract_charts: bool,
        trace_info=None,
        **kwargs,
):
    """
    Helper function to use pdfium to extract text from a bytestream PDF.

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
    extract_charts : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

        kwargs.pdfium_config : dict, optional[PDFiumConfigSchema]

    Returns
    -------
    str
        A string of extracted text.
    """
    logger.debug("Extracting PDF with pdfium backend.")

    row_data = kwargs.get("row_data")
    source_id = row_data["source_id"]
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")

    pdfium_config = kwargs.get("pdfium_config", {})
    pdfium_config = pdfium_config if pdfium_config is not None else {}

    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}

    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    source_location = base_source_metadata.get("source_location", "")
    collection_id = base_source_metadata.get("collection_id", "")
    partition_id = base_source_metadata.get("partition_id", -1)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    pages = []
    extracted_data = []
    doc = libpdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)

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

    logger.debug(f"Extracting text from PDF with {pdf_metadata.page_count} pages.")
    logger.debug(f"Extract text: {extract_text}")
    logger.debug(f"extract images: {extract_images}")
    logger.debug(f"extract tables: {extract_tables}")
    logger.debug(f"extract tables: {extract_charts}")

    # Pdfium does not support text extraction at the document level
    accumulated_text = []
    text_depth = text_depth if text_depth == TextTypeEnum.PAGE else TextTypeEnum.DOCUMENT
    for page_idx in range(pdf_metadata.page_count):
        page = doc.get_page(page_idx)
        page_width, page_height = doc.get_page_size(page_idx)

        # https://pypdfium2.readthedocs.io/en/stable/python_api.html#module-pypdfium2._helpers.textpage
        if extract_text:
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            accumulated_text.append(page_text)

            if text_depth == TextTypeEnum.PAGE:
                text_extraction = construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.keywords,
                    page_idx,
                    -1,
                    -1,
                    -1,
                    pdf_metadata.page_count,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                )

                extracted_data.append(text_extraction)
                accumulated_text = []

        # Image extraction
        if extract_images:
            for obj in page.get_objects():
                obj_type = PDFIUM_PAGEOBJ_MAPPING.get(obj.type, "UNKNOWN")
                if obj_type == "IMAGE":
                    try:
                        # Attempt to retrieve the image bitmap
                        image_numpy: np.ndarray = pdfium_try_get_bitmap_as_numpy(obj)  # noqa
                        image_base64: str = numpy_to_base64(image_numpy)
                        image_bbox = obj.get_pos()
                        image_size = obj.get_size()
                        image_data = Base64Image(
                            image=image_base64, bbox=image_bbox, width=image_size[0], height=image_size[1],
                            max_width=page_width, max_height=page_height
                        )

                        extracted_image_data = construct_image_metadata_from_pdf_image(
                            image_data,
                            page_idx,
                            pdf_metadata.page_count,
                            source_metadata,
                            base_unified_metadata,
                        )

                        extracted_data.append(extracted_image_data)
                    except Exception as e:
                        logger.error(f"Unhandled error extracting image: {e}")
                        pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

        # Table and chart collection
        if extract_tables or extract_charts:
            pages.append(page)

    if extract_text and text_depth == TextTypeEnum.DOCUMENT:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.keywords,
            -1,
            -1,
            -1,
            -1,
            pdf_metadata.page_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        extracted_data.append(text_extraction)

    if extract_tables or extract_charts:
        for page_idx, table_and_charts in extract_tables_and_charts_using_image_ensemble(
                pages,
                pdfium_config,
                trace_info=trace_info,
        ):
            if (extract_tables and (table_and_charts.type_string == "table")) or (
                extract_charts and (table_and_charts.type_string == "chart")
            ):
                extracted_data.append(
                    construct_table_and_chart_metadata(
                        table_and_charts,
                        page_idx,
                        pdf_metadata.page_count,
                        source_metadata,
                        base_unified_metadata,
                    )
                )

    logger.debug(f"Extracted {len(extracted_data)} items from PDF.")

    return extracted_data
