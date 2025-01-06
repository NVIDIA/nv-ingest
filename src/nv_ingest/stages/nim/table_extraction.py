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

from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.image_processing.transforms import check_numpy_image_size
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.helpers import get_version
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

logger = logging.getLogger(f"morpheus.{__name__}")

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _update_metadata(row: pd.Series, paddle_client: NimClient, trace_info: Dict) -> Dict:
    """
    Modifies the metadata of a row if the conditions for table extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the table extraction.

    paddle_client : NimClient
        The client used to call the PaddleOCR inference model.

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
    table_metadata = metadata.get("table_metadata")

    # Only modify if content type is structured and subtype is 'table' and table_metadata exists
    if (
        (content_metadata.get("type") != "structured")
        or (content_metadata.get("subtype") != "table")
        or (table_metadata is None)
    ):
        return metadata

    # Modify table metadata with the result from the inference model
    try:
        data = {"base64_image": base64_image}

        image_array = base64_to_numpy(base64_image)

        paddle_result = "", ""
        if check_numpy_image_size(image_array, PADDLE_MIN_WIDTH, PADDLE_MIN_HEIGHT):
            # Perform inference using the NimClient
            paddle_result = paddle_client.infer(
                data,
                model_name="paddle",
                table_content_format=table_metadata.get("table_content_format"),
                trace_info=trace_info,  # traceable_func arg
                stage_name="table_data_extraction",  # traceable_func arg
            )

        table_content, table_content_format = paddle_result
        table_metadata["table_content"] = table_content
        table_metadata["table_content_format"] = table_content_format
    except Exception as e:
        logger.error(f"Unhandled error calling PaddleOCR inference model: {e}", exc_info=True)
        raise

    return metadata


def _extract_table_data(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts table data from a DataFrame.

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

    Raises
    ------
    Exception
        If any error occurs during the table data extraction process.
    """

    _ = task_props  # unused

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df.empty:
        return df, trace_info

    stage_config = validated_config.stage_config

    # Obtain paddle_version
    # Assuming that the grpc endpoint is at index 0
    paddle_endpoint = stage_config.paddle_endpoints[1]
    try:
        paddle_version = get_version(paddle_endpoint)
        if not paddle_version:
            logger.warning("Failed to obtain PaddleOCR version from the endpoint. Falling back to the latest version.")
            paddle_version = None  # Default to the latest version
    except Exception:
        logger.warning("Failed to get PaddleOCR version after 30 seconds. Falling back to the latest verrsion.")
        paddle_version = None  # Default to the latest version

    # Create the PaddleOCRModelInterface with paddle_version
    paddle_model_interface = PaddleOCRModelInterface(paddle_version=paddle_version)

    # Create the NimClient for PaddleOCR
    paddle_client = create_inference_client(
        endpoints=stage_config.paddle_endpoints,
        model_interface=paddle_model_interface,
        auth_token=stage_config.auth_token,
        infer_protocol=stage_config.paddle_infer_protocol,
    )

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(paddle_client, trace_info))

        return df, {"trace_info": trace_info}

    except Exception:
        logger.error("Error occurred while extracting table data.", exc_info=True)
        raise
    finally:
        if isinstance(paddle_client, NimClient):
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

    validated_config = TableExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_extract_table_data, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
