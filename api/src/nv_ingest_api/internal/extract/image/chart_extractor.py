# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskChartExtraction
from nv_ingest_api.util.image_processing.table_and_chart import join_yolox_graphic_elements_and_ocr_output
from nv_ingest_api.util.image_processing.table_and_chart import process_yolox_graphic_elements
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import PaddleOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import NemoRetrieverOCRModelInterface
from nv_ingest_api.internal.primitives.nim.model_interface.ocr import get_ocr_model_name
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxGraphicElementsModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32

logger = logging.getLogger(f"ray.{__name__}")


def _filter_valid_chart_images(
    base64_images: List[str],
) -> Tuple[List[str], List[np.ndarray], List[int], List[Tuple[str, Optional[Dict]]]]:
    """
    Filter base64-encoded images based on minimum dimensions for chart extraction.

    Returns:
      - valid_images: Base64 strings meeting size requirements.
      - valid_arrays: Corresponding numpy arrays.
      - valid_indices: Original indices of valid images.
      - results: Initial results list where invalid images are set to (img, None).
    """
    results: List[Tuple[str, Optional[Dict]]] = [("", None)] * len(base64_images)
    valid_images: List[str] = []
    valid_arrays: List[np.ndarray] = []
    valid_indices: List[int] = []

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
    return valid_images, valid_arrays, valid_indices, results


def _run_chart_inference(
    yolox_client: Any,
    ocr_client: Any,
    ocr_model_name: str,
    valid_arrays: List[np.ndarray],
    valid_images: List[str],
    trace_info: Dict,
) -> Tuple[List[Any], List[Any]]:
    """
    Run concurrent inference for chart extraction using YOLOX and Paddle.

    Returns a tuple of (yolox_results, ocr_results).
    """
    data_yolox = {"images": valid_arrays}
    data_ocr = {"base64_images": valid_images}

    future_yolox_kwargs = dict(
        data=data_yolox,
        model_name="yolox_ensemble",
        stage_name="chart_extraction",
        input_names=["INPUT_IMAGES", "THRESHOLDS"],
        dtypes=["BYTES", "FP32"],
        output_names=["OUTPUT"],
        trace_info=trace_info,
        max_batch_size=8,
    )
    future_ocr_kwargs = dict(
        data=data_ocr,
        stage_name="chart_extraction",
        trace_info=trace_info,
    )
    if ocr_model_name == "paddle":
        future_ocr_kwargs.update(
            model_name="paddle",
            max_batch_size=1 if ocr_client.protocol == "grpc" else 2,
        )
    elif ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"}:
        future_ocr_kwargs.update(
            model_name=ocr_model_name,
            input_names=["INPUT_IMAGE_URLS", "MERGE_LEVELS"],
            output_names=["OUTPUT"],
            dtypes=["BYTES", "BYTES"],
            merge_level="paragraph",
        )
    else:
        raise ValueError(f"Unknown OCR model name: {ocr_model_name}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_yolox = executor.submit(yolox_client.infer, **future_yolox_kwargs)
        future_ocr = executor.submit(ocr_client.infer, **future_ocr_kwargs)

        try:
            yolox_results = future_yolox.result()
        except Exception as e:
            logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
            raise

        try:
            ocr_results = future_ocr.result()
        except Exception as e:
            logger.error(f"Error calling ocr_client.infer: {e}", exc_info=True)
            raise

    return yolox_results, ocr_results


def _validate_chart_inference_results(
    yolox_results: Any,
    ocr_results: Any,
    valid_arrays: List[Any],
    valid_images: List[str],
) -> Tuple[List[Any], List[Any]]:
    """
    Ensure inference results are lists and have expected lengths.

    Raises:
      ValueError if results do not match expected types or lengths.
    """
    if not (isinstance(yolox_results, list) and isinstance(ocr_results, list)):
        raise ValueError("Expected list results from both yolox_client and ocr_client infer calls.")

    if len(yolox_results) != len(valid_arrays):
        raise ValueError(f"Expected {len(valid_arrays)} yolox results, got {len(yolox_results)}")
    if len(ocr_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} ocr results, got {len(ocr_results)}")
    return yolox_results, ocr_results


def _merge_chart_results(
    base64_images: List[str],
    valid_indices: List[int],
    yolox_results: List[Any],
    ocr_results: List[Any],
    initial_results: List[Tuple[str, Optional[Dict]]],
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Merge inference results into the initial results list using the original indices.

    For each valid image, processes the results from both inference calls and updates the
    corresponding entry in the results list.
    """
    for idx, (yolox_res, ocr_res) in enumerate(zip(yolox_results, ocr_results)):
        # Unpack ocr result into bounding boxes and text predictions.
        bounding_boxes, text_predictions, _ = ocr_res
        yolox_elements = join_yolox_graphic_elements_and_ocr_output(yolox_res, bounding_boxes, text_predictions)
        chart_content = process_yolox_graphic_elements(yolox_elements)
        original_index = valid_indices[idx]
        initial_results[original_index] = (base64_images[original_index], chart_content)
    return initial_results


def _update_chart_metadata(
    base64_images: List[str],
    yolox_client: Any,
    ocr_client: Any,
    ocr_model_name: str,
    trace_info: Dict,
    worker_pool_size: int = 8,  # Not currently used.
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Given a list of base64-encoded chart images, concurrently call both YOLOX and Paddle
    inference services to extract chart data.

    For each base64-encoded image, returns:
      (original_image_str, joined_chart_content_dict)

    Images that do not meet minimum size requirements are marked as skipped.
    """
    logger.debug("Running chart extraction using updated concurrency handling.")

    # Initialize results with placeholders and filter valid images.
    valid_images, valid_arrays, valid_indices, results = _filter_valid_chart_images(base64_images)

    # Run concurrent inference only for valid images.
    yolox_results, ocr_results = _run_chart_inference(
        yolox_client=yolox_client,
        ocr_client=ocr_client,
        ocr_model_name=ocr_model_name,
        valid_arrays=valid_arrays,
        valid_images=valid_images,
        trace_info=trace_info,
    )

    # Validate that the returned inference results are lists of the expected length.
    yolox_results, ocr_results = _validate_chart_inference_results(
        yolox_results, ocr_results, valid_arrays, valid_images
    )

    # Merge the inference results into the results list.
    return _merge_chart_results(base64_images, valid_indices, yolox_results, ocr_results, results)


def _create_yolox_client(
    yolox_endpoints: Tuple[str, str],
    yolox_protocol: str,
    auth_token: str,
) -> NimClient:
    yolox_model_interface = YoloxGraphicElementsModelInterface()

    yolox_client = create_inference_client(
        endpoints=yolox_endpoints,
        model_interface=yolox_model_interface,
        auth_token=auth_token,
        infer_protocol=yolox_protocol,
    )

    return yolox_client


def _create_ocr_client(
    ocr_endpoints: Tuple[str, str],
    ocr_protocol: str,
    ocr_model_name: str,
    auth_token: str,
) -> NimClient:
    ocr_model_interface = (
        NemoRetrieverOCRModelInterface()
        if ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"}
        else PaddleOCRModelInterface()
    )

    ocr_client = create_inference_client(
        endpoints=ocr_endpoints,
        model_interface=ocr_model_interface,
        auth_token=auth_token,
        infer_protocol=ocr_protocol,
        enable_dynamic_batching=(
            True if ocr_model_name in {"scene_text_ensemble", "scene_text_wrapper", "scene_text_python"} else False
        ),
        dynamic_batch_memory_budget_mb=32,
    )

    return ocr_client


def extract_chart_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[IngestTaskChartExtraction, Dict[str, Any]],
    extraction_config: ChartExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts chart data from a DataFrame in a bulk fashion rather than row-by-row.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which chart data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for chart extraction.
    execution_trace_log : Optional[Dict], optional
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
    _ = task_config  # Unused variable

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_extraction_ledger.empty:
        return df_extraction_ledger, execution_trace_log

    endpoint_config = extraction_config.endpoint_config

    # Get the grpc endpoint to determine the model if needed
    ocr_grpc_endpoint = endpoint_config.ocr_endpoints[0]
    ocr_model_name = get_ocr_model_name(ocr_grpc_endpoint)

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

        mask = df_extraction_ledger.apply(meets_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, just return.
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # 2) Extract base64 images + keep track of row -> image mapping.
        base64_images = []
        for idx in valid_indices:
            meta = df_extraction_ledger.at[idx, "metadata"]
            base64_images.append(meta["content"])  # guaranteed by meets_criteria

        # 3) Call our bulk _update_metadata to get all results.
        yolox_client = _create_yolox_client(
            endpoint_config.yolox_endpoints,
            endpoint_config.yolox_infer_protocol,
            endpoint_config.auth_token,
        )

        ocr_client = _create_ocr_client(
            endpoint_config.ocr_endpoints,
            endpoint_config.ocr_infer_protocol,
            ocr_model_name,
            endpoint_config.auth_token,
        )

        bulk_results = _update_chart_metadata(
            base64_images=base64_images,
            yolox_client=yolox_client,
            ocr_client=ocr_client,
            ocr_model_name=ocr_model_name,
            worker_pool_size=endpoint_config.workers_per_progress_engine,
            trace_info=execution_trace_log,
        )

        # 4) Write the results back to each row’s table_metadata
        #    The order of base64_images in bulk_results should match their original
        #    indices if we process them in the same order.
        for row_id, idx in enumerate(valid_indices):
            _, chart_content = bulk_results[row_id]
            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content"] = chart_content

        return df_extraction_ledger, {"trace_info": execution_trace_log}

    except Exception:
        logger.error("Error occurred while extracting chart data.", exc_info=True)

        raise
