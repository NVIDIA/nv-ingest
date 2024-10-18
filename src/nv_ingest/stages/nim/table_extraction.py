# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import functools
import traceback
from typing import Any
from typing import Dict

from morpheus.config import Config

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.nim.helpers import call_image_inference_model, create_inference_client

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(row, paddle_client, trace_info):
    """
    Modifies the metadata of a row if the conditions for table extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the table extraction.

    paddle_client : Any
        The client used to call the image inference model.

    trace_info : dict
        Trace information used for logging or debugging.

    Returns
    -------
    dict
        The modified metadata.
    """
    logger.info("Starting _update_metadata function.")

    metadata = row["metadata"]
    logger.info(f"Extracted metadata from row: {metadata}")

    base64_image = metadata["content"]
    content_metadata = metadata["content_metadata"]
    table_metadata = metadata.get("table_metadata")
    logger.info(f"Base64 image extracted: {base64_image[:30]}...")  # Log only the beginning of the image data for brevity
    logger.info(f"Content metadata: {content_metadata}")
    logger.info(f"Table metadata: {table_metadata}")

    # Only modify if content type is structured and subtype is 'table' and table_metadata exists
    if ((content_metadata.type != ContentTypeEnum.STRUCTURED)
            or (content_metadata.subtype not in ("table",))
            or (table_metadata is None)):
        logger.info("Conditions for table extraction not met. Returning original metadata.")
        return metadata

    logger.info("Conditions met for table extraction. Calling image inference model.")
    # Modify table metadata with the result from the inference model
    try:
        result = call_image_inference_model(paddle_client, "paddle", base64_image, trace_info)
        table_metadata.table_content = result
        logger.info(f"Table content extracted and updated in metadata: {result}")
    except Exception as e:
        logger.error(f"Error calling image inference model: {e}", exc_info=True)
        return metadata

    logger.info("Returning modified metadata.")
    # Return the modified metadata to be updated in the DataFrame
    return metadata


def _extract_table_data(df, task_props, validated_config, trace_info=None):
    """
    Function to extract table data from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which table data is to be extracted.

    task_props : dict
        Dictionary containing task properties and configurations.

    validated_config : TableExtractorSchema
        The validated configuration object for table extraction.

    trace_info : Optional[dict], default=None
        Optional trace information for debugging or logging.
    """
    logger.info(f"Starting _extract_table_data function with task properties: {task_props}")

    # TODO (Devin): Should be part of the stage_config
    logger.info("Creating inference client with validated configuration.")
    paddle_client = create_inference_client(
        validated_config.paddle_endpoints,
        validated_config.auth_token,
        validated_config.paddle_infer_protocol
    )

    logger.info(f"Inference client created: {paddle_client}")

    if trace_info is None:
        trace_info = {}
        logger.info("No trace_info provided. Initialized empty trace_info dictionary.")
    else:
        logger.info(f"Using provided trace_info: {trace_info}")

    try:
        logger.info("Applying _update_metadata function to each row in the DataFrame.")
        # Apply the modify_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(paddle_client, trace_info))
        logger.info("Successfully updated metadata for all rows in the DataFrame.")

        return df, trace_info

    except Exception as e:
        logger.error("Error occurred while extracting table data.", exc_info=True)
        raise


def generate_table_extractor_stage(
        c: Config,
        stage_config: Dict[str, Any],
        task: str = "table_data_extract",
        task_desc: str = "table_data_extraction",
        pe_count: int = 1,
):
    """
    Helper function to generate a multiprocessing stage to perform table data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : dict
        Configuration parameters for the table content extractor, passed as a dictionary
        that will be validated against the `TableExtractorSchema`.

    task : str, default="table_extraction"
        The task name for the stage worker function, defining the specific table extraction process.

    task_desc : str, default="table_data_extractor"
        A descriptor used for latency tracing and logging during table extraction.

    pe_count : int, default=1
        The number of process engines to use for table data extraction. This value controls
        how many worker processes will run concurrently.

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
