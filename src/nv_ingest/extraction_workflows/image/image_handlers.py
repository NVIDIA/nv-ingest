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
import traceback
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from math import log
from wand.image import Image as WandImage

import nv_ingest.util.nim.yolox as yolox_utils
from nv_ingest.extraction_workflows.pdf.doughnut_utils import crop_image
from nv_ingest.schemas.image_extractor_schema import ImageConfigSchema
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata_from_base64
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata

logger = logging.getLogger(__name__)

YOLOX_MAX_BATCH_SIZE = 8
YOLOX_MAX_WIDTH = 1536
YOLOX_MAX_HEIGHT = 1536
YOLOX_NUM_CLASSES = 3
YOLOX_CONF_THRESHOLD = 0.01
YOLOX_IOU_THRESHOLD = 0.5
YOLOX_MIN_SCORE = 0.1
YOLOX_FINAL_SCORE = 0.48

RAW_FILE_FORMATS = ["jpeg", "jpg", "png", "tiff"]
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

    # Convert image to numpy array and normalize pixel values
    image_array = np.asarray(processed_image, dtype=np.float32)

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
    # Convert SVG to PNG using Wand (ImageMagick)
    with WandImage(blob=image_stream.read(), format="svg") as img:
        img.format = "png"
        png_data = img.make_blob()

    # Reload the PNG as a PIL Image
    processed_image = Image.open(io.BytesIO(png_data)).convert("RGB")

    # Convert image to numpy array and normalize pixel values
    image_array = np.asarray(processed_image, dtype=np.float32)

    return image_array


def extract_table_and_chart_images(
    annotation_dict: Dict[str, List[List[float]]],
    original_image: np.ndarray,
    page_idx: int,
    tables_and_charts: List[Tuple[int, "CroppedImageWithContent"]],
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
    tables_and_charts : list of tuple of (int, CroppedImageWithContent)
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
    the cropped image and metadata, and appends it to `tables_and_charts`.

    Examples
    --------
    >>> annotation_dict = {"table": [[0.1, 0.1, 0.5, 0.5, 0.8]], "chart": [[0.6, 0.6, 0.9, 0.9, 0.9]]}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> tables_and_charts = []
    >>> extract_table_and_chart_images(annotation_dict, original_image, 0, tables_and_charts)
    >>> len(tables_and_charts)
    2
    """

    width, height, *_ = original_image.shape
    for label in ["table", "chart"]:
        if not annotation_dict or label not in annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = np.array(bbox) * np.array([height, width, height, width])

            base64_img = crop_image(original_image, (int(h1), int(w1), int(h2), int(w2)))

            table_data = CroppedImageWithContent(
                content="",
                image=base64_img,
                bbox=(int(w1), int(h1), int(w2), int(h2)),
                max_width=width,
                max_height=height,
                type_string=label,
            )
            tables_and_charts.append((page_idx, table_data))


def extract_tables_and_charts_from_images(
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
    config : PDFiumConfigSchema
        Configuration object containing YOLOX endpoints, auth token, etc.
    trace_info : Optional[List], optional
        Optional tracing data for debugging/performance profiling.

    Returns
    -------
    List[Tuple[int, object]]
        A list of (image_index, CroppedImageWithContent)
        representing extracted table/chart data from each image.
    """
    tables_and_charts = []
    yolox_client = None

    try:
        model_interface = yolox_utils.YoloxPageElementsModelInterface()
        yolox_client = create_inference_client(
            config.yolox_endpoints,
            model_interface,
            config.auth_token,
            config.yolox_infer_protocol,
        )

        max_batch_size = YOLOX_MAX_BATCH_SIZE
        batches = []
        i = 0
        while i < len(images):
            batch_size = min(2 ** int(log(len(images) - i, 2)), max_batch_size)
            batches.append(images[i : i + batch_size])  # noqa: E203
            i += batch_size

        img_index = 0
        for batch in batches:
            data = {"images": batch}

            # NimClient inference
            inference_results = yolox_client.infer(
                data,
                model_name="yolox",
                max_batch_size=YOLOX_MAX_BATCH_SIZE,
                num_classes=YOLOX_NUM_CLASSES,
                conf_thresh=YOLOX_CONF_THRESHOLD,
                iou_thresh=YOLOX_IOU_THRESHOLD,
                min_score=YOLOX_MIN_SCORE,
                final_thresh=YOLOX_FINAL_SCORE,
                trace_info=trace_info,  # traceable_func arg
                stage_name="pdf_content_extractor",  # traceable_func arg
            )

            # 5) Extract table/chart info from each image's annotations
            for annotation_dict, original_image in zip(inference_results, batch):
                extract_table_and_chart_images(
                    annotation_dict,
                    original_image,
                    img_index,
                    tables_and_charts,
                )
                img_index += 1

    except TimeoutError:
        logger.error("Timeout error during table/chart extraction.")
        raise

    except Exception as e:
        logger.error(f"Unhandled error during table/chart extraction: {str(e)}")
        traceback.print_exc()
        raise e

    finally:
        if yolox_client:
            yolox_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts from image.")

    return tables_and_charts


def image_data_extractor(
    image_stream,
    document_type: str,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    trace_info: dict = None,
    **kwargs,
):
    """
    Helper function to extract text, images, tables, and charts from an image bytestream.

    Parameters
    ----------
    image_stream : io.BytesIO
        A bytestream for the image file.
    document_type : str
        Specifies the type of the image document ('png', 'jpeg', 'jpg', 'svg', 'tiff').
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_charts : bool
        Specifies whether to extract charts.
    trace_info : dict, optional
        Tracing information for logging or debugging purposes.
    **kwargs
        Additional extraction parameters.

    Returns
    -------
    list
        A list of extracted data items.
    """
    logger.debug(f"Extracting {document_type.upper()} image with image extractor.")

    if document_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported document type: {document_type}")

    row_data = kwargs.get("row_data")
    source_id = row_data.get("source_id", "unknown_source")

    # Metadata extraction setup
    base_unified_metadata = row_data.get(kwargs.get("metadata_column", "metadata"), {})
    current_iso_datetime = datetime.now().isoformat()
    source_metadata = {
        "source_name": f"{source_id}_{document_type}",
        "source_id": source_id,
        "source_location": row_data.get("source_location", ""),
        "source_type": document_type,
        "collection_id": row_data.get("collection_id", ""),
        "date_created": row_data.get("date_created", current_iso_datetime),
        "last_modified": row_data.get("last_modified", current_iso_datetime),
        "summary": f"Raw {document_type} image extracted from source {source_id}",
        "partition_id": row_data.get("partition_id", -1),
        "access_level": row_data.get("access_level", AccessLevelEnum.LEVEL_1),
    }

    # Prepare for extraction
    extracted_data = []
    logger.debug(f"Extract text: {extract_text} (not supported yet for raw images)")
    logger.debug(f"Extract images: {extract_images} (not supported yet for raw images)")
    logger.debug(f"Extract tables: {extract_tables}")
    logger.debug(f"Extract charts: {extract_charts}")

    # Preprocess based on image type
    if document_type in RAW_FILE_FORMATS:
        logger.debug(f"Loading and preprocessing {document_type} image.")
        image_array = load_and_preprocess_image(image_stream)
    elif document_type in PREPROC_FILE_FORMATS:
        logger.debug(f"Converting {document_type} to bitmap.")
        image_array = convert_svg_to_bitmap(image_stream)
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

    # Text extraction stub
    if extract_text:
        # Future function for text extraction based on document_type
        logger.warning("Text extraction is not supported for raw images.")

    # Table and chart extraction
    if extract_tables or extract_charts:
        try:
            tables_and_charts = extract_tables_and_charts_from_images(
                [image_array],
                config=kwargs.get("image_extraction_config"),
                trace_info=trace_info,
            )
            for item in tables_and_charts:
                table_chart_data = item[1]
                extracted_data.append(
                    construct_table_and_chart_metadata(
                        table_chart_data,
                        page_idx=0,  # Single image treated as one page
                        page_count=1,
                        source_metadata=source_metadata,
                        base_unified_metadata=base_unified_metadata,
                    )
                )
        except Exception as e:
            logger.error(f"Error extracting tables/charts from image: {e}")
            raise

        # Image extraction stub
    if extract_images and not extracted_data:  # It's not an unstructured image if we extracted a sturctured image
        # Placeholder for image-specific extraction process
        extracted_data.append(
            construct_image_metadata_from_base64(
                numpy_to_base64(image_array),
                page_idx=0,  # Single image treated as one page
                page_count=1,
                source_metadata=source_metadata,
                base_unified_metadata=base_unified_metadata,
            )
        )

    logger.debug(f"Extracted {len(extracted_data)} items from the image.")

    return extracted_data
