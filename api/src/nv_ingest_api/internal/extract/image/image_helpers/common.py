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

import os
import io
import logging
from datetime import datetime
from typing import Dict, IO, Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image

# from wand.image import Image as WandImage

from nv_ingest_api.internal.enums.common import AccessLevelEnum
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YOLOX_PAGE_CLASS_LABELS,
    YOLOX_PAGE_DEFAULT_VERSION,
    YoloxPageElementsModelInterface,
    get_yolox_page_version,
)
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.util.image_processing.transforms import crop_image, numpy_to_base64
from nv_ingest_api.util.metadata.aggregators import (
    CroppedImageWithContent,
    construct_page_element_metadata,
    construct_image_metadata_from_base64,
)
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)

YOLOX_MAX_BATCH_SIZE = 8

RAW_FILE_FORMATS = ["jpeg", "jpg", "png", "tiff", "bmp"]
PREPROC_FILE_FORMATS = ["svg"]

SUPPORTED_FILE_TYPES = RAW_FILE_FORMATS + ["svg"]


def load_and_preprocess_image(image_stream: io.BytesIO) -> np.ndarray:
    """
    Loads and preprocesses a JPEG, JPG, or PNG image from a bytestream.

    Parameters
    ----------
    image_stream : io.BytesIO
        A bytestream of the image file.

    Returns
    -------
    np.ndarray
        Preprocessed image as a numpy array.
    """
    # Load image from the byte stream
    processed_image = Image.open(image_stream).convert("RGB")

    # Convert image to numpy uint8 array
    image_array = np.asarray(processed_image)

    return image_array


def convert_svg_to_bitmap(image_stream: io.BytesIO) -> np.ndarray:
    """
    Converts an SVG image from a bytestream to a bitmap format.

    Parameters
    ----------
    image_stream : io.BytesIO
        A bytestream of the SVG file.

    Returns
    -------
    np.ndarray
        Preprocessed image as a numpy array in bitmap format.
    """

    pass
    # Convert SVG to PNG using Wand (ImageMagick)
    # with WandImage(blob=image_stream.read(), format="svg") as img:
    #    img.format = "png"
    #    png_data = img.make_blob()

    ## Reload the PNG as a PIL Image
    # processed_image = Image.open(io.BytesIO(png_data)).convert("RGB")

    ## Convert image to numpy array and normalize pixel values
    # image_array = np.asarray(processed_image, dtype=np.float32)

    # return image_array


def extract_page_element_images(
    annotation_dict: Dict[str, List[List[float]]],
    original_image: np.ndarray,
    page_idx: int,
    page_elements: List[Tuple[int, "CroppedImageWithContent"]],
) -> None:
    """
    Handle the extraction of tables and charts from the inference results and run additional model inference.

    Parameters
    ----------
    annotation_dict : dict of {str : list of list of float}
        A dictionary containing detected objects and their bounding boxes. Keys should include "table" and "chart",
        and each key's value should be a list of bounding boxes, with each bounding box represented as a list of floats.
    original_image : np.ndarray
        The original image from which objects were detected, expected to be in RGB format with shape (H, W, 3).
    page_idx : int
        The index of the current page being processed.
    page_elements : list of tuple of (int, CroppedImageWithContent)
        A list to which extracted tables and charts will be appended. Each item in the list is a tuple where the first
        element is the page index, and the second is an instance of CroppedImageWithContent representing a cropped image
        and associated metadata.

    Returns
    -------
    None

    Notes
    -----
    This function iterates over detected objects labeled as "table" or "chart". For each object, it crops the original
    image according to the bounding box coordinates, then creates an instance of `CroppedImageWithContent` containing
    the cropped image and metadata, and appends it to `page_elements`.

    Examples
    --------
    >>> annotation_dict = {"table": [[0.1, 0.1, 0.5, 0.5, 0.8]], "chart": [[0.6, 0.6, 0.9, 0.9, 0.9]]}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> page_elements = []
    >>> extract_page_element_images(annotation_dict, original_image, 0, page_elements)
    >>> len(page_elements)
    2
    """

    width, height, *_ = original_image.shape

    if annotation_dict and (set(YOLOX_PAGE_CLASS_LABELS) <= annotation_dict.keys()):
        labels = YOLOX_PAGE_CLASS_LABELS
    else:
        labels = ["table", "chart", "infographics"]

    for label in labels:
        if not annotation_dict or label not in annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = bbox

            cropped_img = crop_image(original_image, (int(h1), int(w1), int(h2), int(w2)))
            base64_img = numpy_to_base64(cropped_img) if cropped_img is not None else None

            table_data = CroppedImageWithContent(
                content="",
                image=base64_img,
                bbox=(int(w1), int(h1), int(w2), int(h2)),
                max_width=width,
                max_height=height,
                type_string=label,
            )
            page_elements.append((page_idx, table_data))


def extract_page_elements_from_images(
    images: List[np.ndarray],
    config: ImageConfigSchema,
    trace_info: Optional[List] = None,
) -> List[Tuple[int, object]]:
    """
    Detect and extract tables/charts from a list of NumPy images using YOLOX.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images in NumPy array format.
    config : ImageConfigSchema
        Configuration object containing YOLOX endpoints, auth token, etc.
    trace_info : Optional[List], optional
        Optional tracing data for debugging/performance profiling.

    Returns
    -------
    List[Tuple[int, object]]
        A list of (image_index, CroppedImageWithContent) representing extracted
        table/chart data from each image.
    """
    page_elements = []
    yolox_client = None

    # Obtain yolox_version
    # Assuming that the http endpoint is at index 1
    yolox_version = YOLOX_PAGE_DEFAULT_VERSION

    # Get the HTTP endpoint to determine the model name if needed
    yolox_http_endpoint = config.yolox_endpoints[1]
    if yolox_http_endpoint:
        try:
            yolox_version = get_yolox_page_version(yolox_http_endpoint)
        except Exception as e:
            logger.warning(f"Failed to get YOLOX model name from endpoint: {e}. Using default.")

    try:
        model_interface = YoloxPageElementsModelInterface(version=yolox_version)
        yolox_client = create_inference_client(
            config.yolox_endpoints,
            model_interface,
            config.auth_token,
            config.yolox_infer_protocol,
        )

        # Prepare the payload with all images.
        data = {"images": images}

        # Perform inference in a single call. The NimClient handles batching internally.
        inference_results = yolox_client.infer(
            data,
            model_name="yolox_ensemble",
            max_batch_size=YOLOX_MAX_BATCH_SIZE,
            input_names=["INPUT_IMAGES", "THRESHOLDS"],
            dtypes=["BYTES", "FP32"],
            output_names=["OUTPUT"],
            trace_info=trace_info,
            stage_name="pdf_extraction",
        )

        # Process each result along with its corresponding image.
        for i, (annotation_dict, original_image) in enumerate(zip(inference_results, images)):
            extract_page_element_images(
                annotation_dict,
                original_image,
                i,
                page_elements,
            )

    except TimeoutError:
        logger.error("Timeout error during table/chart extraction.")
        raise

    except Exception as e:
        logger.exception(f"Unhandled error during table/chart extraction: {str(e)}")
        raise

    logger.debug(f"Extracted {len(page_elements)} tables and charts from image.")
    return page_elements


def unstructured_image_extractor(
    *,
    image_stream: IO[bytes],
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extraction_config: Dict[str, Any],
    extraction_trace_log: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Extract primitives from an unstructured image bytestream.

    This helper function processes an image bytestream according to the provided extraction
    configuration. It supports extraction of tables, charts, and infographics from the image.
    (Note: text and additional image extraction are not supported yet for raw images.)

    Parameters
    ----------
    image_stream : IO[bytes]
        A bytestream (e.g. io.BytesIO) containing the image file data.
    image_stream : io.BytesIO
        A bytestream for the image file.
    document_type : str
        Specifies the type of the image document ('png', 'jpeg', 'jpg', 'svg', 'tiff', 'bmp').
    extract_text : bool
        Flag specifying whether to extract text (currently not supported for raw images).
    extract_images : bool
        Flag specifying whether to extract images (currently not supported for raw images).
    extract_infographics : bool
        Flag specifying whether to extract infographics.
    extract_tables : bool
        Flag specifying whether to extract tables.
    extract_charts : bool
        Flag specifying whether to extract charts.
    extraction_config : Dict[str, Any]
        A dictionary containing additional extraction parameters and configurations.
        Expected keys include "document_type", "row_data", "metadata_column", and
        "image_extraction_config".
    extraction_trace_log : Optional[Dict[str, Any]], optional
        An optional dictionary containing trace information for logging or debugging,
        by default None.

    Returns
    -------
    List[Any]
        A list of extracted data items (e.g., metadata dictionaries) from the image.

    Raises
    ------
    ValueError
        If the document type is unsupported.
    Exception
        If an error occurs during extraction.
    """
    # Note: extract_infographics flag is not currently used in extraction
    _ = extract_infographics

    # Determine the type of the document from the extraction config.
    document_type: str = extraction_config.get("document_type", "unknown")
    logger.debug(f"Extracting {document_type.upper()} image with image extractor.")

    # Ensure the document type is supported.
    if document_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported document type: {document_type}")

    # Retrieve additional row-specific data and source identifier.
    row_data: Dict[str, Any] = extraction_config.get("row_data", {})
    source_id: str = row_data.get("source_id", "unknown_source")

    # Build source metadata based on row data.
    base_unified_metadata: Dict[str, Any] = row_data.get(extraction_config.get("metadata_column", "metadata"), {})
    current_iso_datetime: str = datetime.now().isoformat()
    source_metadata: Dict[str, Any] = {
        "source_name": source_id if os.path.splitext(source_id)[1] else f"{source_id}.{document_type}",
        "source_id": source_id,
        "source_location": row_data.get("source_location", ""),
        "source_type": document_type,
        "collection_id": row_data.get("collection_id", ""),
        "date_created": row_data.get("date_created", current_iso_datetime),
        "last_modified": row_data.get("last_modified", current_iso_datetime),
        "summary": f"Raw {document_type} image extracted from source {source_id}",
        "partition_id": row_data.get("partition_id", -1),
        "access_level": row_data.get("access_level", AccessLevelEnum.UNKNOWN),
    }

    # Optionally update the extract_infographics flag based on extraction_config.
    extract_infographics = extraction_config.get("extract_infographics", False)
    text_extraction_method = extraction_config.get("extract_method", "ocr")

    # Log which primitives are requested for extraction.
    logger.debug(f"Extract text: {extract_text} (not supported yet for raw images)")
    logger.debug(f"Extract images: {extract_images} (not supported yet for raw images)")
    logger.debug(f"Extract tables: {extract_tables}")
    logger.debug(f"Extract charts: {extract_charts}")
    logger.debug(f"Extract infographics: {extract_infographics}")

    # Preprocess the image based on the document type.
    if document_type in RAW_FILE_FORMATS:
        logger.debug(f"Loading and preprocessing {document_type} image.")
        image_array = load_and_preprocess_image(image_stream)
    elif document_type in PREPROC_FILE_FORMATS:
        logger.debug(f"Converting {document_type} to bitmap.")
        image_array = convert_svg_to_bitmap(image_stream)
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

    extracted_data: List[Any] = []

    if extract_text:
        if text_extraction_method != "ocr":
            logger.warning(
                f"Text extraction method '{text_extraction_method} is not supported for raw images. "
                "Defaulting to 'ocr'."
            )

    # Extract tables, charts, or infographics if requested.
    if extract_text or extract_tables or extract_charts or extract_infographics:
        try:
            page_elements = extract_page_elements_from_images(
                [image_array],
                config=extraction_config.get("image_extraction_config"),
                trace_info=extraction_trace_log,
            )
            for item in page_elements:
                table_chart_data = item[1]

                # Skip elements that shouldn't be extracted based on flags
                element_type = table_chart_data.type_string
                if (not extract_tables) and (element_type == "table"):
                    continue
                if (not extract_charts) and (element_type == "chart"):
                    continue
                if (not extract_infographics) and (element_type == "infographic"):
                    continue
                if (not extract_text) and (element_type in {"title", "paragraph", "header_footer"}):
                    continue

                extracted_data.append(
                    construct_page_element_metadata(
                        table_chart_data,
                        page_idx=0,  # Treat single image as one page.
                        page_count=1,
                        source_metadata=source_metadata,
                        base_unified_metadata=base_unified_metadata,
                    )
                )
        except Exception as e:
            logger.error(f"Error extracting tables/charts from image: {e}")
            raise

    # Image extraction stub: if no structured elements were extracted and image extraction is requested.
    if extract_images and not extracted_data:
        extracted_data.append(
            construct_image_metadata_from_base64(
                numpy_to_base64(image_array),
                page_idx=0,  # Treat single image as one page.
                page_count=1,
                source_metadata=source_metadata,
                base_unified_metadata=base_unified_metadata,
            )
        )

    logger.debug(f"Extracted {len(extracted_data)} items from the image.")
    return extracted_data
