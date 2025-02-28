# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from nv_ingest_api.util.image_processing.table_and_chart import join_yolox_graphic_elements_and_paddle_output
from nv_ingest_api.util.image_processing.table_and_chart import process_yolox_graphic_elements
from nv_ingest_api.internal.primitives.nim.model_interface.paddle import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxGraphicElementsModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(
    base64_images: List[str],
    yolox_client: NimClient,
    paddle_client: NimClient,
    trace_info: Dict,
    worker_pool_size: int = 8,  # Not currently used.
) -> List[Tuple[str, Dict]]:
    """
    Given a list of base64-encoded chart images, this function calls both the Yolox and Paddle
    inference services concurrently to extract chart data for all images.

    For each base64-encoded image, returns:
      (original_image_str, joined_chart_content_dict)
    """
    logger.debug("Running chart extraction using updated concurrency handling.")

    # Initialize the results list in the same order as base64_images.
    results: List[Tuple[str, Any]] = [("", None)] * len(base64_images)

    valid_images: List[str] = []
    valid_arrays: List[np.ndarray] = []
    valid_indices: List[int] = []

    # Pre-decode image dimensions and filter valid images.
    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
            valid_images.append(img)
            valid_arrays.append(array)
            valid_indices.append(i)
        else:
            # Image is too small; mark as skipped.
            results[i] = (img, None)

    # Prepare data payloads for both clients.
    data_yolox = {"images": valid_arrays}
    data_paddle = {"base64_images": valid_images}

    _ = worker_pool_size
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_yolox = executor.submit(
            yolox_client.infer,
            data=data_yolox,
            model_name="yolox",
            stage_name="chart_data_extraction",
            max_batch_size=8,
            trace_info=trace_info,
        )
        future_paddle = executor.submit(
            paddle_client.infer,
            data=data_paddle,
            model_name="paddle",
            stage_name="chart_data_extraction",
            max_batch_size=1 if paddle_client.protocol == "grpc" else 2,
            trace_info=trace_info,
        )

        try:
            yolox_results = future_yolox.result()
        except Exception as e:
            logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
            raise

        try:
            paddle_results = future_paddle.result()
        except Exception as e:
            logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
            raise

    # Ensure both clients returned lists of results matching the number of input images.
    if not (isinstance(yolox_results, list) and isinstance(paddle_results, list)):
        raise ValueError("Expected list results from both yolox_client and paddle_client infer calls.")

    if len(yolox_results) != len(valid_arrays):
        raise ValueError(f"Expected {len(valid_arrays)} yolox results, got {len(yolox_results)}")
    if len(paddle_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} paddle results, got {len(paddle_results)}")

    # Join the corresponding results from both services for each image.
    for idx, (yolox_res, paddle_res) in enumerate(zip(yolox_results, paddle_results)):
        bounding_boxes, text_predictions = paddle_res
        yolox_elements = join_yolox_graphic_elements_and_paddle_output(yolox_res, bounding_boxes, text_predictions)
        chart_content = process_yolox_graphic_elements(yolox_elements)
        original_index = valid_indices[idx]
        results[original_index] = (base64_images[original_index], chart_content)

    return results


def _create_clients(
    yolox_endpoints: Tuple[str, str],
    yolox_protocol: str,
    paddle_endpoints: Tuple[str, str],
    paddle_protocol: str,
    auth_token: str,
) -> Tuple[NimClient, NimClient]:
    yolox_model_interface = YoloxGraphicElementsModelInterface()
    paddle_model_interface = PaddleOCRModelInterface()

    logger.debug(f"Inference protocols: yolox={yolox_protocol}, paddle={paddle_protocol}")

    yolox_client = create_inference_client(
        endpoints=yolox_endpoints,
        model_interface=yolox_model_interface,
        auth_token=auth_token,
        infer_protocol=yolox_protocol,
    )

    paddle_client = create_inference_client(
        endpoints=paddle_endpoints,
        model_interface=paddle_model_interface,
        auth_token=auth_token,
        infer_protocol=paddle_protocol,
    )

    return yolox_client, paddle_client


def extract_data_from_chart_image_internal(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts chart data from a DataFrame in a bulk fashion rather than row-by-row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which chart data is to be extracted.
    task_props : Dict[str, Any]
        Dictionary containing task properties and configurations.
    validated_config : Any
        The validated configuration object for chart extraction.
    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.

    Raises
    ------
    Exception
        If any error occurs during the chart data extraction process.
    """

    _ = task_props  # unused

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df.empty:
        return df, trace_info

    stage_config = validated_config.stage_config
    yolox_client, paddle_client = _create_clients(
        stage_config.yolox_endpoints,
        stage_config.yolox_infer_protocol,
        stage_config.paddle_endpoints,
        stage_config.paddle_infer_protocol,
        stage_config.auth_token,
    )

    try:
        # 1) Identify rows that meet criteria in a single pass
        #    - metadata exists
        #    - content_metadata.type == "structured"
        #    - content_metadata.subtype == "chart"
        #    - table_metadata not None
        #    - base64_image not None or ""
        def meets_criteria(row):
            m = row.get("metadata", {})
            if not m:
                return False

            content_md = m.get("content_metadata", {})
            if (
                content_md.get("type") == "structured"
                and content_md.get("subtype") == "chart"
                and m.get("table_metadata") is not None
                and m.get("content") not in [None, ""]
            ):
                return True

            return False

        mask = df.apply(meets_criteria, axis=1)
        valid_indices = df[mask].index.tolist()

        # If no rows meet the criteria, just return.
        if not valid_indices:
            return df, {"trace_info": trace_info}

        # 2) Extract base64 images + keep track of row -> image mapping.
        base64_images = []
        for idx in valid_indices:
            meta = df.at[idx, "metadata"]
            base64_images.append(meta["content"])  # guaranteed by meets_criteria

        # 3) Call our bulk _update_metadata to get all results.
        bulk_results = _update_metadata(
            base64_images=base64_images,
            yolox_client=yolox_client,
            paddle_client=paddle_client,
            worker_pool_size=stage_config.workers_per_progress_engine,
            trace_info=trace_info,
        )

        # 4) Write the results back to each row’s table_metadata
        #    The order of base64_images in bulk_results should match their original
        #    indices if we process them in the same order.
        for row_id, idx in enumerate(valid_indices):
            _, chart_content = bulk_results[row_id]
            df.at[idx, "metadata"]["table_metadata"]["table_content"] = chart_content

        return df, {"trace_info": trace_info}

    except Exception:
        logger.error("Error occurred while extracting chart data.", exc_info=True)

        raise

    finally:
        try:
            if paddle_client is not None:
                paddle_client.close()
            if yolox_client is not None:
                yolox_client.close()

        except Exception as close_err:
            logger.error(f"Error closing clients: {close_err}", exc_info=True)
