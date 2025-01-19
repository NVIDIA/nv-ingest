# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any
from typing import Dict
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


# Modify the _update_metadata function
def _update_metadata(row: pd.Series, yolox_client: NimClient, paddle_client: NimClient, trace_info: Dict) -> Dict:
    """
    Modifies the metadata of a row if the conditions for chart extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the chart extraction.

    yolox_client : NimClient
        The client used to call the yolox inference model.

    paddle_client : NimClient
        The client used to call the paddle inference model.

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
    if (
        (content_metadata.get("type") != "structured")
        or (content_metadata.get("subtype") != "chart")
        or (chart_metadata is None)
        or (base64_image in [None, ""])
    ):
        return metadata

    # Modify chart metadata with the result from the inference models
    try:
        base64_data = {"base64_image": base64_image}
        array_data = {"images": [base64_to_numpy(base64_image)]}

        # Perform inference using the NimClients
        yolox_result = yolox_client.infer(
            array_data,
            model_name="yolox",
            stage_name="chart_data_extraction",  # traceable_func arg
            trace_info=trace_info,  # traceable_func arg
        )
        paddle_result = paddle_client.infer(
            base64_data,
            model_name="paddle",
            stage_name="chart_data_extraction",  # traceable_func arg
            trace_info=trace_info,  # traceable_func arg
        )

        ###
        source_name = metadata.get("source_metadata", {}).get("source_name").split("/")[-1]
        page_number = metadata.get("content_metadata", {}).get("page_number")

        out_path = f"/workspace/data/tmp/{source_name}.{page_number}.yolox.pickle"

        import os
        import pickle

        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = {"data": []}

        data["data"].append({"base64_image": base64_image, "yolox": yolox_result[0], "paddle": paddle_result})

        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        ###

        text_predictions, bounding_boxes = paddle_result
        yolox_elements = join_yolox_and_paddle_output(yolox_result[0], text_predictions, bounding_boxes)
        chart_content = process_yolox_graphic_elements(yolox_elements)

        chart_metadata["table_content"] = chart_content
    except Exception as e:
        logger.error(f"Unhandled error calling image inference model: {e}", exc_info=True)
        raise

    return metadata


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

    stage_config = validated_config.stage_config
    yolox_client, paddle_client = _create_clients(
        stage_config.yolox_endpoints,
        stage_config.yolox_infer_protocol,
        stage_config.paddle_endpoints,
        stage_config.paddle_infer_protocol,
        stage_config.auth_token,
    )

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(yolox_client, paddle_client, trace_info))

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
