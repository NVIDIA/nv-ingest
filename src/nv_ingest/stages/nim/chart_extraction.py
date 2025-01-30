# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any, List
from typing import Dict
from typing import Optional
from typing import Tuple

import pandas as pd
from morpheus.config import Config
from concurrent.futures import ThreadPoolExecutor

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.table_and_chart import join_cached_and_deplot_output
from nv_ingest.util.nim.cached import CachedModelInterface
from nv_ingest.util.nim.deplot import DeplotModelInterface
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.helpers import NimClient

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(
    base64_images: List[str],
    cached_client: NimClient,
    deplot_client: NimClient,
    trace_info: Dict,
    batch_size: int = 1,
    worker_pool_size: int = 1,
) -> List[Tuple[str, Dict]]:
    """
    Given a list of base64-encoded chart images, this function:
      - Splits them into batches of size `batch_size`.
      - Calls Cached with *all images* in each batch in a single request if protocol != 'grpc'.
        If protocol == 'grpc', calls Cached individually for each image in the batch.
      - Calls Deplot individually (one request per image) in parallel.
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

    with ThreadPoolExecutor(max_workers=worker_pool_size) as executor:
        for batch in chunk_list(base64_images, batch_size):
            # 1) Cached calls
            if cached_client.protocol == "grpc":
                # Submit each image in the batch separately
                cached_futures = []
                for image_str in batch:
                    data = {"base64_images": [image_str]}
                    fut = executor.submit(
                        cached_client.infer,
                        data=data,
                        model_name="cached",
                        stage_name="chart_data_extraction",
                        trace_info=trace_info,
                    )
                    cached_futures.append(fut)
            else:
                # Single request for the entire batch
                data = {"base64_images": batch}
                future_cached = executor.submit(
                    cached_client.infer,
                    data=data,
                    model_name="cached",
                    stage_name="chart_data_extraction",
                    trace_info=trace_info,
                )

            # 2) Multiple calls to Deplot, one per image in the batch
            deplot_futures = []
            for image_str in batch:
                # Deplot only supports single-image calls
                deplot_data = {"base64_image": image_str}
                fut = executor.submit(
                    deplot_client.infer,
                    data=deplot_data,
                    model_name="deplot",
                    stage_name="chart_data_extraction",
                    trace_info=trace_info,
                )
                deplot_futures.append(fut)

            try:
                # 3) Retrieve results from Cached
                if cached_client.protocol == "grpc":
                    # Each future should return a single-element list
                    # We take the 0th item to align with single-image results
                    cached_results = []
                    for fut in cached_futures:
                        res = fut.result()
                        if isinstance(res, list) and len(res) == 1:
                            cached_results.append(res[0])
                        else:
                            # Fallback in case the service returns something unexpected
                            logger.warning(f"Unexpected CACHED result format: {res}")
                            cached_results.append(res)
                else:
                    # Single call returning a list of the same length as 'batch'
                    cached_results = future_cached.result()

                # Retrieve results from Deplot (each call returns a single inference result)
                deplot_results = [f.result() for f in deplot_futures]

                # 4) Zip them together, one by one
                for img_str, cached_res, deplot_res in zip(batch, cached_results, deplot_results):
                    chart_content = join_cached_and_deplot_output(cached_res, deplot_res)
                    results.append((img_str, chart_content))

            except Exception as e:
                logger.error(f"Error processing batch: {batch}, error: {e}", exc_info=True)
                raise

    return results


def _create_clients(
    cached_endpoints: Tuple[str, str],
    cached_protocol: str,
    deplot_endpoints: Tuple[str, str],
    deplot_protocol: str,
    auth_token: str,
) -> Tuple[NimClient, NimClient]:
    cached_model_interface = CachedModelInterface()
    deplot_model_interface = DeplotModelInterface()

    logger.debug(f"Inference protocols: cached={cached_protocol}, deplot={deplot_protocol}")

    cached_client = create_inference_client(
        endpoints=cached_endpoints,
        model_interface=cached_model_interface,
        auth_token=auth_token,
        infer_protocol=cached_protocol,
    )

    deplot_client = create_inference_client(
        endpoints=deplot_endpoints,
        model_interface=deplot_model_interface,
        auth_token=auth_token,
        infer_protocol=deplot_protocol,
    )

    return cached_client, deplot_client


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
    cached_client, deplot_client = _create_clients(
        stage_config.cached_endpoints,
        stage_config.cached_infer_protocol,
        stage_config.deplot_endpoints,
        stage_config.deplot_infer_protocol,
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
            cached_client=cached_client,
            deplot_client=deplot_client,
            batch_size=stage_config.nim_batch_size,
            worker_pool_size=stage_config.workers_per_progress_engine,
            trace_info=trace_info,
        )

        # 4) Write the results back to each row’s table_metadata
        #    The order of base64_images in bulk_results should match their original
        #    indices if we process them in the same order.
        for row_id, idx in enumerate(valid_indices):
            (_, chart_content) = bulk_results[row_id]
            df.at[idx, "metadata"]["table_metadata"]["table_content"] = chart_content

        return df, {"trace_info": trace_info}

    except Exception:
        logger.error("Error occurred while extracting chart data.", exc_info=True)
        raise
    finally:
        cached_client.close()
        deplot_client.close()


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
