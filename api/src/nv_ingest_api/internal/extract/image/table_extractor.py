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

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskTableExtraction
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.primitives.nim.model_interface.paddle import PaddleOCRModelInterface
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.util.image_processing.table_and_chart import join_yolox_table_structure_and_paddle_output
from nv_ingest_api.util.image_processing.table_and_chart import convert_paddle_response_to_psuedo_markdown
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YoloxTableStructureModelInterface
from nv_ingest_api.util.image_processing.transforms import base64_to_numpy
from nv_ingest_api.util.nim import create_inference_client

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _filter_valid_images(base64_images: List[str]) -> Tuple[List[str], List[np.ndarray], List[int]]:
    """
    Filter base64-encoded images by their dimensions.

    Returns three lists:
      - valid_images: The base64 strings that meet minimum size requirements.
      - valid_arrays: The corresponding numpy arrays.
      - valid_indices: The original indices in the input list.
    """
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
            # Image is too small; skip it.
            continue

    return valid_images, valid_arrays, valid_indices


def _run_inference(
    enable_yolox: bool,
    yolox_client: Any,
    paddle_client: Any,
    valid_arrays: List[np.ndarray],
    valid_images: List[str],
    trace_info: Optional[Dict] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    Run inference concurrently for YOLOX (if enabled) and Paddle.

    Returns a tuple of (yolox_results, paddle_results).
    """
    data_paddle = {"base64_images": valid_images}
    if enable_yolox:
        data_yolox = {"images": valid_arrays}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_yolox = None
        if enable_yolox:
            future_yolox = executor.submit(
                yolox_client.infer,
                data=data_yolox,
                model_name="yolox",
                stage_name="table_data_extraction",
                max_batch_size=8,
                trace_info=trace_info,
            )
        future_paddle = executor.submit(
            paddle_client.infer,
            data=data_paddle,
            model_name="paddle",
            stage_name="table_data_extraction",
            max_batch_size=1 if paddle_client.protocol == "grpc" else 2,
            trace_info=trace_info,
        )

        if enable_yolox:
            try:
                yolox_results = future_yolox.result()
            except Exception as e:
                logger.error(f"Error calling yolox_client.infer: {e}", exc_info=True)
                raise
        else:
            yolox_results = [None] * len(valid_images)

        try:
            paddle_results = future_paddle.result()
        except Exception as e:
            logger.error(f"Error calling paddle_client.infer: {e}", exc_info=True)
            raise

    return yolox_results, paddle_results


def _validate_inference_results(
    yolox_results: Any,
    paddle_results: Any,
    valid_arrays: List[Any],
    valid_images: List[str],
) -> Tuple[List[Any], List[Any]]:
    """
    Validate that both inference results are lists and have the expected lengths.

    If not, default values are assigned. Raises a ValueError if the lengths do not match.
    """
    if not isinstance(yolox_results, list) or not isinstance(paddle_results, list):
        logger.warning(
            "Unexpected result types from inference clients: yolox_results=%s, paddle_results=%s. "
            "Proceeding with available results.",
            type(yolox_results).__name__,
            type(paddle_results).__name__,
        )
        if not isinstance(yolox_results, list):
            yolox_results = [None] * len(valid_arrays)
        if not isinstance(paddle_results, list):
            paddle_results = [(None, None)] * len(valid_images)

    if len(yolox_results) != len(valid_arrays):
        raise ValueError(f"Expected {len(valid_arrays)} yolox results, got {len(yolox_results)}")
    if len(paddle_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} paddle results, got {len(paddle_results)}")

    return yolox_results, paddle_results


def _update_table_metadata(
    base64_images: List[str],
    yolox_client: Any,
    paddle_client: Any,
    worker_pool_size: int = 8,  # Not currently used
    enable_yolox: bool = False,
    trace_info: Optional[Dict] = None,
) -> List[Tuple[str, Any, Any, Any]]:
    """
    Given a list of base64-encoded images, this function filters out images that do not meet
    the minimum size requirements and then calls the PaddleOCR model via paddle_client.infer
    to extract table data.

    For each base64-encoded image, the result is a tuple:
        (base64_image, yolox_result, paddle_text_predictions, paddle_bounding_boxes)

    Images that do not meet the minimum size are skipped (resulting in placeholders).
    The paddle_client is expected to handle any necessary batching and concurrency.
    """
    logger.debug(f"Running table extraction using protocol {paddle_client.protocol}")

    # Initialize the results list with default placeholders.
    results: List[Tuple[str, Any, Any, Any]] = [("", None, None, None)] * len(base64_images)

    # Filter valid images based on size requirements.
    valid_images, valid_arrays, valid_indices = _filter_valid_images(base64_images)

    if not valid_images:
        return results

    # Run inference concurrently.
    yolox_results, paddle_results = _run_inference(
        enable_yolox=enable_yolox,
        yolox_client=yolox_client,
        paddle_client=paddle_client,
        valid_arrays=valid_arrays,
        valid_images=valid_images,
        trace_info=trace_info,
    )

    # Validate that the inference results have the expected structure.
    yolox_results, paddle_results = _validate_inference_results(
        yolox_results, paddle_results, valid_arrays, valid_images
    )

    # Combine results with the original order.
    for idx, (yolox_res, paddle_res) in enumerate(zip(yolox_results, paddle_results)):
        original_index = valid_indices[idx]
        results[original_index] = (base64_images[original_index], yolox_res, paddle_res[0], paddle_res[1])

    return results


def _create_clients(
    yolox_endpoints: Tuple[str, str],
    yolox_protocol: str,
    paddle_endpoints: Tuple[str, str],
    paddle_protocol: str,
    auth_token: str,
) -> Tuple[NimClient, NimClient]:
    yolox_model_interface = YoloxTableStructureModelInterface()
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


def extract_table_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[IngestTaskTableExtraction, Dict[str, Any]],
    extraction_config: TableExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts table data from a DataFrame in a bulk fashion rather than row-by-row,
    following the chart extraction pattern.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which table data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for table extraction.
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.
    """

    _ = task_config  # unused

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df_extraction_ledger.empty:
        return df_extraction_ledger, execution_trace_log

    endpoint_config = extraction_config.endpoint_config
    yolox_client, paddle_client = _create_clients(
        endpoint_config.yolox_endpoints,
        endpoint_config.yolox_infer_protocol,
        endpoint_config.paddle_endpoints,
        endpoint_config.paddle_infer_protocol,
        endpoint_config.auth_token,
    )

    try:
        # 1) Identify rows that meet criteria (structured, subtype=table, table_metadata != None, content not empty)
        def meets_criteria(row):
            m = row.get("metadata", {})
            if not m:
                return False
            content_md = m.get("content_metadata", {})
            if (
                content_md.get("type") == "structured"
                and content_md.get("subtype") == "table"
                and m.get("table_metadata") is not None
                and m.get("content") not in [None, ""]
            ):
                return True
            return False

        mask = df_extraction_ledger.apply(meets_criteria, axis=1)
        valid_indices = df_extraction_ledger[mask].index.tolist()

        # If no rows meet the criteria, just return
        if not valid_indices:
            return df_extraction_ledger, {"trace_info": execution_trace_log}

        # 2) Extract base64 images in the same order
        base64_images = []
        for idx in valid_indices:
            meta = df_extraction_ledger.at[idx, "metadata"]
            base64_images.append(meta["content"])

        # 3) Call our bulk _update_metadata to get all results
        table_content_format = (
            df_extraction_ledger.at[valid_indices[0], "metadata"]["table_metadata"].get("table_content_format")
            or TableFormatEnum.PSEUDO_MARKDOWN
        )
        enable_yolox = True if table_content_format in (TableFormatEnum.MARKDOWN,) else False

        bulk_results = _update_table_metadata(
            base64_images=base64_images,
            yolox_client=yolox_client,
            paddle_client=paddle_client,
            worker_pool_size=endpoint_config.workers_per_progress_engine,
            enable_yolox=enable_yolox,
            trace_info=execution_trace_log,
        )

        # 4) Write the results (bounding_boxes, text_predictions) back
        for row_id, idx in enumerate(valid_indices):
            # unpack (base64_image, (yolox_predictions, paddle_bounding boxes, paddle_text_predictions))
            _, cell_predictions, bounding_boxes, text_predictions = bulk_results[row_id]

            if table_content_format == TableFormatEnum.SIMPLE:
                table_content = " ".join(text_predictions)
            elif table_content_format == TableFormatEnum.PSEUDO_MARKDOWN:
                table_content = convert_paddle_response_to_psuedo_markdown(bounding_boxes, text_predictions)
            elif table_content_format == TableFormatEnum.MARKDOWN:
                table_content = join_yolox_table_structure_and_paddle_output(
                    cell_predictions, bounding_boxes, text_predictions
                )
            else:
                raise ValueError(f"Unexpected table format: {table_content_format}")

            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content"] = table_content
            df_extraction_ledger.at[idx, "metadata"]["table_metadata"]["table_content_format"] = table_content_format

        return df_extraction_ledger, {"trace_info": execution_trace_log}

    except Exception:
        logger.exception("Error occurred while extracting table data.", exc_info=True)
        raise
    finally:
        yolox_client.close()
        paddle_client.close()
