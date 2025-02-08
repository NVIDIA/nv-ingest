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

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.table_and_chart import join_yolox_and_paddle_output
from nv_ingest.util.image_processing.table_and_chart import process_yolox_graphic_elements
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface
from nv_ingest.util.nim.yolox import YoloxGraphicElementsModelInterface

logger = logging.getLogger(f"morpheus.{__name__}")

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _update_metadata(
    base64_images: List[str],
    yolox_client: NimClient,
    paddle_client: NimClient,
    trace_info: Dict,
    batch_size: int = 1,
    worker_pool_size: int = 1,
) -> List[Tuple[str, Dict]]:
    """
    Given a list of base64-encoded chart images, this function:
      - Splits them into batches of size `batch_size`.
      - Calls Yolox and Paddle with *all images* in each batch in a single request if protocol != 'grpc'.
        If protocol == 'grpc', calls Yolox and Paddle individually for each image in the batch.
      - Joins the results for each image into a final combined inference result.

    Returns
    -------
    List[Tuple[str, Dict]]
      For each base64-encoded image, returns (original_image_str, joined_chart_content_dict).
    """
    logger.debug(f"Running chart extraction: batch_size={batch_size}, worker_pool_size={worker_pool_size}")

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    results = []
    image_arrays = [base64_to_numpy(img) for img in base64_images]

    with ThreadPoolExecutor(max_workers=worker_pool_size) as executor:
        for batch, arrays in zip(chunk_list(base64_images, batch_size), chunk_list(image_arrays, batch_size)):
            # 1) Yolox calls
            # Single request for the entire batch
            data = {"images": arrays}
            yolox_futures = executor.submit(
                yolox_client.infer,
                data=data,
                model_name="yolox",
                stage_name="chart_data_extraction",
                trace_info=trace_info,
            )

            # 2) Paddle calls
            paddle_futures = []
            if paddle_client.protocol == "grpc":
                # Submit each image in the batch separately
                paddle_futures = []
                for image_str, image_arr in zip(batch, arrays):
                    width, height = image_arr.shape[:2]
                    if width < PADDLE_MIN_WIDTH or height < PADDLE_MIN_HEIGHT:
                        # Too small, skip inference
                        continue

                    data = {"base64_images": [image_str]}
                    fut = executor.submit(
                        paddle_client.infer,
                        data=data,
                        model_name="paddle",
                        stage_name="chart_data_extraction",
                        max_batch_size=1,
                        trace_info=trace_info,
                    )
                    paddle_futures.append(fut)
            else:
                # Single request for the entire batch
                data = {"base64_images": batch}
                paddle_futures = executor.submit(
                    paddle_client.infer,
                    data=data,
                    model_name="paddle",
                    stage_name="chart_data_extraction",
                    max_batch_size=batch_size,
                    trace_info=trace_info,
                )

            try:
                # Retrieve results from Yolox
                yolox_results = yolox_futures.result()

                # 3) Retrieve results from Yolox
                if paddle_client.protocol == "grpc":
                    # Each future should return a single-element list
                    # We take the 0th item to align with single-image results
                    paddle_results = []
                    for fut in paddle_futures:
                        res = fut.result()
                        if isinstance(res, list) and len(res) == 1:
                            paddle_results.append(res[0])
                        else:
                            # Fallback in case the service returns something unexpected
                            logger.warning(f"Unexpected PaddleOCR result format: {res}")
                            paddle_results.append(res)
                else:
                    # Single call returning a list of the same length as 'batch'
                    paddle_results = paddle_futures.result()

                # 4) Zip them together, one by one
                for img_str, yolox_res, paddle_res in zip(batch, yolox_results, paddle_results):
                    bounding_boxes, text_predictions = paddle_res
                    yolox_elements = join_yolox_and_paddle_output(yolox_res, bounding_boxes, text_predictions)
                    chart_content = process_yolox_graphic_elements(yolox_elements)
                    results.append((img_str, chart_content))

            except Exception as e:
                logger.error(f"Error processing batch: {batch}, error: {e}", exc_info=True)
                raise

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


def _extract_chart_data(
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

        # If no rows meet the criteria, just return
        if not valid_indices:
            return df, {"trace_info": trace_info}

        # 2) Extract base64 images + keep track of row -> image mapping
        base64_images = []
        for idx in valid_indices:
            meta = df.at[idx, "metadata"]
            base64_images.append(meta["content"])  # guaranteed by meets_criteria

        # 3) Call our bulk update_metadata to get all results
        bulk_results = _update_metadata(
            base64_images=base64_images,
            yolox_client=yolox_client,
            paddle_client=paddle_client,
            batch_size=stage_config.nim_batch_size,
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
        yolox_client.close()
        paddle_client.close()


def generate_chart_extractor_stage(
    c: Config,
    stage_config: Dict[str, Any],
    task: str = "chart_data_extract",
    task_desc: str = "chart_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform chart data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the chart content extractor, passed as a dictionary
        validated against the `ChartExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific chart extraction process.
        Default is "chart_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during chart extraction.
        Default is "chart_data_extraction".

    pe_count : int, optional
        The number of process engines to use for chart data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles chart data extraction
        from PDF content.
    """

    validated_config = ChartExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_extract_chart_data, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
