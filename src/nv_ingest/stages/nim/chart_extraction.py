# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import tritonclient.grpc as grpcclient
from morpheus.config import Config

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.table_and_chart import join_cached_and_deplot_output
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import call_image_inference_model, create_inference_client

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(row: pd.Series, cached_client: Any, deplot_client: Any, trace_info: Dict) -> Dict:
    """
    Modifies the metadata of a row if the conditions for chart extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the chart extraction.

    cached_client : Any
        The client used to call the cached inference model.

    deplot_client : Any
        The client used to call the deplot inference model.

    trace_info : Dict
        Trace information used for logging or debugging.

    Returns
    -------
    Dict
        The modified metadata if conditions are met, otherwise the original metadata.

    Raises
    ------
    ValueError
        If critical information (such as metadata) is missing from the row.
    """
    metadata = row.get("metadata")
    if metadata is None:
        logger.error("Row does not contain 'metadata'.")
        raise ValueError("Row does not contain 'metadata'.")

    base64_image = metadata.get("content")
    content_metadata = metadata.get("content_metadata", {})
    chart_metadata = metadata.get("table_metadata")

    # Only modify if content type is structured and subtype is 'chart' and chart_metadata exists
    if ((content_metadata.get("type") != "structured") or
            (content_metadata.get("subtype") != "chart") or
            (chart_metadata is None)):
        return metadata

    # Modify chart metadata with the result from the inference model
    try:
        image_array = base64_to_numpy(base64_image)

        deplot_result = call_image_inference_model(deplot_client, "deplot", image_array, trace_info=trace_info)
        cached_result = call_image_inference_model(cached_client, "cached", image_array, trace_info=trace_info)
        chart_content = join_cached_and_deplot_output(cached_result, deplot_result)

        chart_metadata["table_content"] = chart_content
    except Exception as e:
        logger.error(f"Unhandled error calling image inference model: {e}", exc_info=True)
        raise

    return metadata


def _extract_chart_data(df: pd.DataFrame, task_props: Dict[str, Any],
                        validated_config: Any, trace_info: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts chart data from a DataFrame.

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

    deplot_client = create_inference_client(
        validated_config.stage_config.deplot_endpoints,
        validated_config.stage_config.auth_token,
        validated_config.stage_config.deplot_infer_protocol
    )

    cached_client = create_inference_client(
        validated_config.stage_config.cached_endpoints,
        validated_config.stage_config.auth_token,
        validated_config.stage_config.cached_infer_protocol
    )

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(cached_client, deplot_client, trace_info))

        return df, trace_info

    except Exception as e:
        logger.error("Error occurred while extracting chart data.", exc_info=True)
        raise
    finally:
        if (isinstance(cached_client, grpcclient.InferenceServerClient)):
            cached_client.close()
        if (isinstance(deplot_client, grpcclient.InferenceServerClient)):
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
