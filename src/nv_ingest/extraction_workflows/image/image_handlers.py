# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
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

import cairosvg

from typing import List
from typing import Optional
from typing import Tuple

import cv2
from PIL import Image
import io

import numpy as np
import tritonclient.grpc as grpcclient

from nv_ingest.extraction_workflows.pdf.doughnut_utils import crop_image
import nv_ingest.util.nim.yolox as yolox_utils
from nv_ingest.schemas.image_extractor_schema import ImageExtractorSchema
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.helpers import perform_model_inference
from nv_ingest.util.pdf.metadata_aggregators import CroppedImageWithContent
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

    # Specify the file path for saving
    output_path_png = '/workspace/processed_image.png'
    output_path_jpeg = '/workspace/processed_image.jpeg'

    # Save the image as PNG
    processed_image.save(output_path_png, format='PNG')
    print(f"Image saved as PNG at {output_path_png}")

    # Save the image as JPEG with 95% quality
    processed_image.save(output_path_jpeg, format='JPEG', quality=95)
    print(f"Image saved as JPEG at {output_path_jpeg}")

    # Resize or normalize image as required by the model (example size 640x640)
    # processed_image = processed_image.resize((YOLOX_MAX_WIDTH, YOLOX_MAX_HEIGHT))

    # Convert image to numpy array and normalize pixel values
    image_array = np.asarray(processed_image, dtype=np.float32) / 255.0

    # Expand dimensions if required by the model (e.g., [1, H, W, C])
    # image_array = np.expand_dims(image_array, axis=0)

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
    # Convert SVG to PNG byte stream using cairosvg
    png_data = cairosvg.svg2png(bytestring=image_stream.read())

    # Reload the PNG as a PIL Image
    processed_image = Image.open(io.BytesIO(png_data)).convert("RGB")

    # Convert image to numpy array and normalize pixel values
    image_array = np.asarray(processed_image, dtype=np.float32) / 255.0

    # Expand dimensions if required by the model
    # image_array = np.expand_dims(image_array, axis=0)

    return image_array


# TODO(Devin): Move to common file
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
    logger.debug(f"Number of results: {len(results)}")
    logger.debug(f"Results: {results}")

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


# TODO(Devin): Move to common file
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
    annotation_dict : dict
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


def extract_tables_and_charts_from_image(
        image: np.ndarray,
        config: ImageExtractorSchema,
        num_classes: int = YOLOX_NUM_CLASSES,
        conf_thresh: float = YOLOX_CONF_THRESHOLD,
        iou_thresh: float = YOLOX_IOU_THRESHOLD,
        min_score: float = YOLOX_MIN_SCORE,
        final_thresh: float = YOLOX_FINAL_SCORE,
        trace_info: Optional[List] = None,
) -> List[CroppedImageWithContent]:
    """
    Extract tables and charts from a single image using an ensemble of image-based models.

    This function processes a single image to detect and extract tables and charts.
    It uses a sequence of models hosted on different inference servers to achieve this.

    Parameters
    ----------
    image : np.ndarray
        A preprocessed image array for table and chart detection.
    config : ImageExtractorSchema
        Configuration for the inference client, including endpoint URLs and authentication.
    num_classes : int, optional
        The number of classes the model is trained to detect (default is 3).
    conf_thresh : float, optional
        The confidence threshold for detection (default is 0.01).
    iou_thresh : float, optional
        The Intersection Over Union (IoU) threshold for non-maximum suppression (default is 0.5).
    min_score : float, optional
        The minimum score threshold for considering a detection valid (default is 0.1).
    final_thresh: float, optional
        Threshold for keeping a bounding box applied after postprocessing (default is 0.48).
    trace_info : Optional[List], optional
        Tracing information for logging or debugging purposes.

    Returns
    -------
    List[CroppedImageWithContent]
        A list of `CroppedImageWithContent` objects representing detected tables or charts,
        each containing metadata about the detected region.
    """
    tables_and_charts = []

    yolox_client = None
    try:
        logger.debug("Initializing YOLOX inference client.")
        yolox_client = create_inference_client(config.yolox_endpoints, config.auth_token)

        logger.debug("Preparing image for inference.")
        input_image = yolox_utils.prepare_images_for_inference([image])
        image_shape = image.shape
        logger.debug(f"Image shape for inference: {image_shape}")

        logger.debug("Performing model inference.")
        output_array = perform_model_inference(yolox_client, "yolox", input_image, trace_info=trace_info)
        logger.debug("Model inference completed.")

        logger.debug("Processing inference results.")
        logger.debug(f"Output array: {output_array}")
        yolox_annotated_detections = process_inference_results(
            output_array, [image_shape], num_classes, conf_thresh, iou_thresh, min_score, final_thresh
        )
        logger.debug(f"Number of detections: {len(yolox_annotated_detections)}")
        logger.debug(f"Actual detections: {yolox_annotated_detections}")

        logger.debug("Extracting tables and charts from detected regions.")
        for annotation_dict in yolox_annotated_detections:
            extract_table_and_chart_images(
                annotation_dict,
                image,
                page_idx=0,  # Single image treated as one page
                tables_and_charts=tables_and_charts,
            )
        logger.debug("Table and chart extraction completed.")

    except Exception as e:
        logger.error(f"Error during table/chart extraction from image: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if isinstance(yolox_client, grpcclient.InferenceServerClient):
            logger.debug("Closing YOLOX inference client.")
            yolox_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts from image.")

    return tables_and_charts


def image_data_extractor(image_stream,
                         document_type: str,
                         extract_text: bool,
                         extract_images: bool,
                         extract_tables: bool,
                         extract_charts: bool,
                         trace_info: dict = None,
                         **kwargs):
    """
    Helper function to extract text, images, tables, and charts from an image bytestream.

    Parameters
    ----------
    image_stream : io.BytesIO
        A bytestream for the image file.
    document_type : str
        Specifies the type of the image document ('png', 'jpeg', 'jpg', 'svg').
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_charts : bool
        Specifies whether to extract charts.
    **kwargs
        Additional extraction parameters.

    Returns
    -------
    list
        A list of extracted data items.
    """
    logger.debug(f"Extracting {document_type.upper()} image with image extractor.")

    # TODO(Devin): Remove once schema is updated to enforce supported types
    if (document_type not in ["jpeg", "jpg", "png", "svg"]):
        raise ValueError(f"Unsupported document type: {document_type}")

    row_data = kwargs.get("row_data")
    source_id = row_data.get("source_id", "unknown_source")

    # Metadata extraction setup
    base_unified_metadata = row_data.get(kwargs.get("metadata_column", "metadata"), {})
    source_metadata = {
        "source_name": f"{source_id}_{document_type}",
        "source_id": source_id,
        "source_location": row_data.get("source_location", ""),
        "source_type": document_type,
        "collection_id": row_data.get("collection_id", ""),
        "date_created": row_data.get("date_created", ""),
        "last_modified": row_data.get("last_modified", ""),
        "summary": "",
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
    if document_type in {"jpeg", "jpg", "png"}:
        logger.debug("Loading and preprocessing image.")
        image_array = load_and_preprocess_image(image_stream)
    elif document_type == "svg":
        logger.debug("Converting SVG to bitmap.")
        image_array = convert_svg_to_bitmap(image_stream)
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

    # Text extraction stub
    if extract_text:
        # Future function for text extraction based on document_type
        logger.warning("Text extraction is not supported for raw images.")

    # Image extraction stub
    if extract_images:
        # Placeholder for image-specific extraction process
        logger.warning("Image extraction is not supported for raw images.")

    logger.debug(json.dumps(kwargs, indent=2, default=str))
    # Table and chart extraction
    if extract_tables or extract_charts:
        try:
            tables_and_charts = extract_tables_and_charts_from_image(
                image_array,
                config=kwargs.get("image_extraction_config"),
                trace_info=trace_info,
            )
            logger.debug(f"Extracted table/chart data from image")
            for _, table_chart_data in tables_and_charts:
                extracted_data.append(
                    construct_table_and_chart_metadata(
                        table_chart_data,
                        page_idx=0,  # Single image treated as one page
                        total_pages=1,
                        source_metadata=source_metadata,
                        base_unified_metadata=base_unified_metadata,
                    )
                )
        except Exception as e:
            logger.error(f"Error extracting tables/charts from image: {e}")

    logger.debug(f"Extracted {len(extracted_data)} items from the image.")

    return extracted_data
