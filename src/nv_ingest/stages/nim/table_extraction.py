# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from morpheus.config import Config

from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import create_inference_client, NimClient, get_version
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _update_metadata(
    base64_images: List[str],
    paddle_client: NimClient,
    worker_pool_size: int = 8,  # Not currently used
    trace_info: Dict = None,
) -> List[Tuple[str, Tuple[Any, Any]]]:
    """
    Given a list of base64-encoded images, this function filters out images that do not meet the minimum
    size requirements and then calls the PaddleOCR model via paddle_client.infer to extract table data.

    For each base64-encoded image, the result is:
        (base64_image, (table_content, table_content_format))

    Images that do not meet the minimum size are skipped (resulting in ("", "") for that image).
    The paddle_client is expected to handle any necessary batching and concurrency.
    """
    logger.debug(f"Running table extraction using protocol {paddle_client.protocol}")

    # Initialize the results list in the same order as base64_images.
    results: List[Optional[Tuple[str, Tuple[Any, Any]]]] = [None] * len(base64_images)

    valid_images: List[str] = []
    valid_indices: List[int] = []

    _ = worker_pool_size
    # Pre-decode image dimensions and filter valid images.
    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
            valid_images.append(img)
            valid_indices.append(i)
        else:
            # Image is too small; mark as skipped.
            results[i] = (img, ("", ""))

    if valid_images:
        data = {"base64_images": valid_images}
        try:
            # Call infer once for all valid images. The NimClient will handle batching internally.
            paddle_result = paddle_client.infer(
                data=data,
                model_name="paddle",
                stage_name="table_data_extraction",
                max_batch_size=1,
                trace_info=trace_info,
            )

            if not isinstance(paddle_result, list):
                raise ValueError(f"Expected a list of tuples, got {type(paddle_result)}")
            if len(paddle_result) != len(valid_images):
                raise ValueError(f"Expected {len(valid_images)} results, got {len(paddle_result)}")

            # Assign each result back to its original position.
            for idx, result in enumerate(paddle_result):
                original_index = valid_indices[idx]
                results[original_index] = (base64_images[original_index], result)

        except Exception as e:
            logger.error(f"Error processing images. Error: {e}", exc_info=True)
            for i in valid_indices:
                results[i] = (base64_images[i], ("", ""))
            raise

    return results


def _create_paddle_client(stage_config) -> NimClient:
    """
    Helper to create a NimClient for PaddleOCR, retrieving the paddle version from the endpoint.
    """
    # Attempt to obtain PaddleOCR version from the second endpoint
    paddle_endpoint = stage_config.paddle_endpoints[1]
    try:
        paddle_version = get_version(paddle_endpoint)
        if not paddle_version:
            logger.warning("Failed to obtain PaddleOCR version from the endpoint. Falling back to the latest version.")
            paddle_version = None
    except Exception:
        logger.warning("Failed to get PaddleOCR version after 30 seconds. Falling back to the latest version.")
        paddle_version = None

    paddle_model_interface = PaddleOCRModelInterface(paddle_version=paddle_version)

    paddle_client = create_inference_client(
        endpoints=stage_config.paddle_endpoints,
        model_interface=paddle_model_interface,
        auth_token=stage_config.auth_token,
        infer_protocol=stage_config.paddle_infer_protocol,
    )

    return paddle_client


def _extract_table_data(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts table data from a DataFrame in a bulk fashion rather than row-by-row,
    following the chart extraction pattern.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which table data is to be extracted.
    task_props : Dict[str, Any]
        Dictionary containing task properties and configurations.
    validated_config : Any
        The validated configuration object for table extraction.
    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.
    """

    _ = task_props  # unused

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df.empty:
        return df, trace_info

    stage_config = validated_config.stage_config
    paddle_client = _create_paddle_client(stage_config)

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

        mask = df.apply(meets_criteria, axis=1)
        valid_indices = df[mask].index.tolist()

        # If no rows meet the criteria, just return
        if not valid_indices:
            return df, {"trace_info": trace_info}

        # 2) Extract base64 images in the same order
        base64_images = []
        for idx in valid_indices:
            meta = df.at[idx, "metadata"]
            base64_images.append(meta["content"])

        # 3) Call our bulk _update_metadata to get all results
        bulk_results = _update_metadata(
            base64_images=base64_images,
            paddle_client=paddle_client,
            worker_pool_size=stage_config.workers_per_progress_engine,
            trace_info=trace_info,
        )

        # 4) Write the results (table_content, table_content_format) back
        for row_id, idx in enumerate(valid_indices):
            # unpack (base64_image, (content, format))
            _, (table_content, table_content_format) = bulk_results[row_id]

            df.at[idx, "metadata"]["table_metadata"]["table_content"] = table_content
            df.at[idx, "metadata"]["table_metadata"]["table_content_format"] = table_content_format

        return df, {"trace_info": trace_info}

    except Exception:
        logger.error("Error occurred while extracting table data.", exc_info=True)
        raise
    finally:
        paddle_client.close()


def generate_table_extractor_stage(
    c: Config,
    stage_config: Dict[str, Any],
    task: str = "table_data_extract",
    task_desc: str = "table_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform table data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the table content extractor, passed as a dictionary
        validated against the `TableExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific table extraction process.
        Default is "table_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during table extraction.
        Default is "table_data_extraction".

    pe_count : int, optional
        The number of process engines to use for table data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles table data extraction
        from PDF content.
    """

    print(f"TableExtractorSchema stage_config: {stage_config}")
    validated_config = TableExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_extract_table_data, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
