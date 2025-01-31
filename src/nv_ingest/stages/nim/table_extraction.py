# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from morpheus.config import Config

from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.table_and_chart import convert_paddle_response_to_psuedo_markdown
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _update_metadata(
    base64_images: List[str],
    paddle_client: NimClient,
    batch_size: int = 1,
    worker_pool_size: int = 1,
    trace_info: Dict = None,
) -> List[Tuple[str, Tuple[Any, Any]]]:
    """
    Given a list of base64-encoded images, this function processes them either individually
    (if paddle_client.protocol == 'grpc') or in batches (if paddle_client.protocol == 'http'),
    then calls the PaddleOCR model to extract data.

    For each base64-encoded image, the result is:
        (base64_image, (text_predictions, bounding_boxes))

    Images that do not meet the minimum size are skipped (("", "")).
    """
    logger.debug(
        f"Running table extraction: batch_size={batch_size}, "
        f"worker_pool_size={worker_pool_size}, protocol={paddle_client.protocol}"
    )

    # We'll build the final results in the same order as base64_images.
    results: List[Optional[Tuple[str, Tuple[Any, Any]]]] = [None] * len(base64_images)

    # Pre-decode dimensions once (optional, but efficient if we want to skip small images).
    decoded_shapes = []
    for img in base64_images:
        array = base64_to_numpy(img)
        decoded_shapes.append(array.shape)  # e.g. (height, width, channels)

    # ------------------------------------------------
    # GRPC path: submit one request per valid image.
    # ------------------------------------------------
    if paddle_client.protocol == "grpc":
        with ThreadPoolExecutor(max_workers=worker_pool_size) as executor:
            future_to_index = {}

            # Submit individual requests
            for i, b64_image in enumerate(base64_images):
                height, width = decoded_shapes[i][0], decoded_shapes[i][1]
                if width < PADDLE_MIN_WIDTH or height < PADDLE_MIN_HEIGHT:
                    # Too small, skip inference
                    results[i] = (b64_image, (None, None))
                    continue

                # Enqueue a single-image inference
                data = {"base64_images": [b64_image]}  # single item
                future = executor.submit(
                    paddle_client.infer,
                    data=data,
                    model_name="paddle",
                    trace_info=trace_info,
                    stage_name="table_data_extraction",
                )
                future_to_index[future] = i

            # Gather results
            for future, i in future_to_index.items():
                b64_image = base64_images[i]
                try:
                    paddle_result = future.result()
                    # We expect exactly one result for one image
                    if not isinstance(paddle_result, list) or len(paddle_result) != 1:
                        raise ValueError(f"Expected 1 result list, got: {paddle_result}")
                    bounding_boxes, text_predictions = paddle_result[0]
                    results[i] = (b64_image, (bounding_boxes, text_predictions))
                except Exception as e:
                    logger.error(f"Error processing image {i}. Error: {e}", exc_info=True)
                    results[i] = (b64_image, (None, None))

    # ------------------------------------------------
    # HTTP path: submit requests in batches.
    # ------------------------------------------------
    else:
        with ThreadPoolExecutor(max_workers=worker_pool_size) as executor:
            # Process images in chunks
            for start_idx in range(0, len(base64_images), batch_size):
                chunk_indices = range(start_idx, min(start_idx + batch_size, len(base64_images)))
                valid_indices = []
                valid_images = []

                # Check dimensions & collect valid images
                for i in chunk_indices:
                    height, width = decoded_shapes[i][0], decoded_shapes[i][1]
                    if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
                        valid_indices.append(i)
                        valid_images.append(base64_images[i])
                    else:
                        # Too small, skip inference
                        results[i] = (base64_images[i], (None, None))

                if not valid_images:
                    # All images in this chunk were too small
                    continue

                # Submit a single batch inference
                data = {"base64_images": valid_images}
                future = executor.submit(
                    paddle_client.infer,
                    data=data,
                    model_name="paddle",
                    trace_info=trace_info,
                    stage_name="table_data_extraction",
                )

                try:
                    # This should be a list of (text_predictions, bounding_boxes)
                    # in the same order as valid_images
                    paddle_result = future.result()

                    if not isinstance(paddle_result, list):
                        raise ValueError(f"Expected a list of tuples, got {type(paddle_result)}")

                    if len(paddle_result) != len(valid_images):
                        raise ValueError(f"Expected {len(valid_images)} results, got {len(paddle_result)}")

                    # Match each result back to its original index
                    for idx_in_batch, (tc, tf) in enumerate(paddle_result):
                        i = valid_indices[idx_in_batch]
                        results[i] = (base64_images[i], (tc, tf))

                except Exception as e:
                    logger.error(f"Error processing batch {valid_images}. Error: {e}", exc_info=True)
                    # If inference fails, we can fill them with empty or re-raise
                    for vi in valid_indices:
                        results[vi] = (base64_images[vi], (None, None))
                    raise

    # 'results' now has an entry for every image in base64_images
    return results


def _create_paddle_client(stage_config) -> NimClient:
    """
    Helper to create a NimClient for PaddleOCR, retrieving the paddle version from the endpoint.
    """
    # Attempt to obtain PaddleOCR version from the second endpoint
    paddle_model_interface = PaddleOCRModelInterface()

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
            batch_size=stage_config.nim_batch_size,
            worker_pool_size=stage_config.workers_per_progress_engine,
            trace_info=trace_info,
        )

        # 4) Write the results (bounding_boxes, text_predictions) back
        table_content_format = df.at[valid_indices[0], "metadata"]["table_metadata"].get(
            "table_content_format", TableFormatEnum.PSEUDO_MARKDOWN
        )

        for row_id, idx in enumerate(valid_indices):
            # unpack (base64_image, (bounding boxes, text_predictions))
            _, (bounding_boxes, text_predictions) = bulk_results[row_id]

            if table_content_format == TableFormatEnum.SIMPLE:
                table_content = " ".join(text_predictions)
            elif table_content_format == TableFormatEnum.PSEUDO_MARKDOWN:
                table_content = convert_paddle_response_to_psuedo_markdown(text_predictions, bounding_boxes)
            else:
                raise ValueError(f"Unexpected table format: {table_content_format}")

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
