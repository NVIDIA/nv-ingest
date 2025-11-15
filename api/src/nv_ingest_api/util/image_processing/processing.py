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
import numpy as np
from typing import List, Tuple, Optional

from nv_ingest_api.internal.primitives.nim.default_values import (
    YOLOX_MAX_BATCH_SIZE,
    YOLOX_CONF_THRESHOLD,
    YOLOX_IOU_THRESHOLD,
    YOLOX_MIN_SCORE,
    YOLOX_FINAL_SCORE,
)
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxPageElementsModelInterface
from nv_ingest_api.util.image_processing.transforms import crop_image, numpy_to_base64
from nv_ingest_api.util.metadata.aggregators import CroppedImageWithContent
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)


def extract_tables_and_charts_from_image(annotation_dict, original_image, page_idx, tables_and_charts):
    """
    Extract and process table and chart regions from the provided image based on detection annotations.

    Parameters
    ----------
    annotation_dict : dict
        A dictionary containing detected objects and their bounding boxes, e.g. keys "table" and "chart".
    original_image : np.ndarray
        The original image from which objects were detected.
    page_idx : int
        The index of the current page being processed.
    tables_and_charts : list of tuple
        A list to which extracted table/chart data will be appended. Each item is a tuple
        (page_idx, CroppedImageWithContent).

    Notes
    -----
    This function iterates over the detected table and chart objects. For each detected object, it:
      - Crops the original image based on the bounding box.
      - Converts the cropped image to a base64 encoded string.
      - Wraps the encoded image along with its bounding box and the image dimensions in a standardized data structure.

    Additional model inference or post-processing can be added where needed.

    Examples
    --------
    >>> annotation_dict = {"table": [ [...], [...] ], "chart": [ [...], [...] ]}
    >>> original_image = np.random.rand(1536, 1536, 3)
    >>> tables_and_charts = []
    >>> extract_tables_and_charts(annotation_dict, original_image, 0, tables_and_charts)
    """

    width, height, *_ = original_image.shape
    for label in ["table", "chart"]:
        if not annotation_dict:
            continue

        objects = annotation_dict[label]
        for idx, bboxes in enumerate(objects):
            *bbox, _ = bboxes
            h1, w1, h2, w2 = bbox

            cropped = crop_image(original_image, (int(h1), int(w1), int(h2), int(w2)))
            base64_img = numpy_to_base64(cropped)

            element_data = CroppedImageWithContent(
                content="",
                image=base64_img,
                bbox=(int(w1), int(h1), int(w2), int(h2)),
                max_width=width,
                max_height=height,
                type_string=label,
            )
            tables_and_charts.append((page_idx, element_data))


def extract_tables_and_charts_yolox(
    pages: List[Tuple[int, np.ndarray]],
    config: dict,
    trace_info: Optional[List] = None,
) -> List[Tuple[int, object]]:
    """
    Given a list of (page_index, image) tuples and a configuration dictionary,
    this function calls the YOLOX-based inference service to extract table and chart
    annotations from all pages.

    Parameters
    ----------
    pages : List[Tuple[int, np.ndarray]]
        A list of tuples containing the page index and the corresponding image.
    config : dict
        A dictionary containing configuration parameters such as:
            - 'yolox_endpoints'
            - 'auth_token'
            - 'yolox_infer_protocol'
    trace_info : Optional[List], optional
        Optional tracing information for logging/debugging purposes.

    Returns
    -------
    List[Tuple[int, object]]
        For each page, returns a tuple (page_index, joined_content) where
        joined_content is the result of combining annotations from the inference.
    """
    tables_and_charts = []
    yolox_client = None

    try:
        model_interface = YoloxPageElementsModelInterface()
        yolox_client = create_inference_client(
            config["yolox_endpoints"],
            model_interface,
            config["auth_token"],
            config["yolox_infer_protocol"],
        )

        # Collect all page indices and images in order.
        image_page_indices = [page[0] for page in pages]
        original_images = [page[1] for page in pages]

        # Prepare the data payload with all images.
        data = {"images": original_images}

        # Perform inference using the YOLOX client.
        inference_results = yolox_client.infer(
            data,
            model_name="yolox",
            max_batch_size=YOLOX_MAX_BATCH_SIZE,
            conf_thresh=YOLOX_CONF_THRESHOLD,
            iou_thresh=YOLOX_IOU_THRESHOLD,
            min_score=YOLOX_MIN_SCORE,
            final_thresh=YOLOX_FINAL_SCORE,
            trace_info=trace_info,
            stage_name="pdf_extraction",
        )

        # Process results: iterate over each image's inference output.
        for annotation_dict, page_index, original_image in zip(inference_results, image_page_indices, original_images):
            extract_tables_and_charts_from_image(
                annotation_dict,
                original_image,
                page_index,
                tables_and_charts,
            )

    except TimeoutError:
        logger.error("Timeout error during table/chart extraction.")
        raise

    except Exception as e:
        err_msg = f"Error during table/chart extraction: {str(e)}"
        logger.exception(err_msg)
        raise

    finally:
        if yolox_client:
            yolox_client.close()

    logger.debug(f"Extracted {len(tables_and_charts)} tables and charts.")
    return tables_and_charts
