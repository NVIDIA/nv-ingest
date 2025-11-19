# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import functools
import io
import logging
from typing import Any, Optional, Dict, Union, Tuple

import pandas as pd
from pydantic import BaseModel

from nv_ingest_api.internal.extract.pptx.engines.pptx_helper import python_pptx
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


def _prepare_task_properties(
    base64_row: pd.Series, task_props: Union[Dict[str, Any], BaseModel]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Prepare and return the task properties dictionary and source identifier from a DataFrame row.

    This function converts task properties to a dictionary (if provided as a Pydantic model),
    extracts row data (excluding the "content" field), and stores it under the "row_data" key within
    the task properties. It also retrieves the "source_id" from the row if present.

    Parameters
    ----------
    base64_row : pd.Series
        A pandas Series representing a row containing base64-encoded content under the key "content"
        and optionally a "source_id".
    task_props : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model containing extraction instructions and parameters.

    Returns
    -------
    Tuple[Dict[str, Any], Optional[str]]
        A tuple where the first element is the prepared task properties dictionary (with "row_data" added)
        and the second element is the source_id if present; otherwise, None.
    """
    # If task_props is a Pydantic model, convert it to a dictionary.
    if isinstance(task_props, BaseModel):
        task_props = task_props.model_dump()
    else:
        task_props = dict(task_props)

    # Exclude the "content" field from the row data.
    row_data = base64_row.drop(labels=["content"], errors="ignore")
    if "params" not in task_props:
        task_props["params"] = {}
    # Store the row data in the parameters.
    task_props["params"]["row_data"] = row_data

    # Retrieve the source identifier if available.
    source_id = base64_row.get("source_id", None)
    return task_props, source_id


@unified_exception_handler
def _decode_and_extract_from_pptx(
    base64_row: pd.Series,
    task_props: Union[Dict[str, Any], BaseModel],
    extraction_config: Any,
    trace_info: Dict[str, Any],
) -> Any:
    """
    Decode base64-encoded PPTX content from a DataFrame row and extract data using the specified method.

    The function prepares task properties (using `_prepare_task_properties`), decodes the base64 content
    into a byte stream, determines extraction parameters, and calls the extraction function (e.g. `python_pptx`)
    with the proper flags. If extraction fails, an exception tag is returned.

    Parameters
    ----------
    base64_row : pd.Series
        A Series containing base64-encoded PPTX content under the key "content" and optionally a "source_id".
    task_props : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model containing extraction instructions (may include a "method" key and "params").
    extraction_config : Any
        A configuration object containing PPTX extraction settings (e.g. `pptx_extraction_config`).
    trace_info : Dict[str, Any]
        A dictionary with trace information for logging or debugging.

    Returns
    -------
    Any
        The extracted data from the PPTX file, or an exception tag indicating failure.
    """
    # Prepare task properties and extract source_id.
    prepared_task_props, source_id = _prepare_task_properties(base64_row, task_props)

    # Decode base64 content into bytes and create a BytesIO stream.
    base64_content: str = base64_row["content"]
    pptx_bytes: bytes = base64.b64decode(base64_content)
    pptx_stream: io.BytesIO = io.BytesIO(pptx_bytes)

    # Retrieve extraction parameters (and remove boolean flags as they are consumed).
    extract_params: Dict[str, Any] = prepared_task_props.get("params", {})
    extract_text: bool = extract_params.pop("extract_text", False)
    extract_images: bool = extract_params.pop("extract_images", False)
    extract_tables: bool = extract_params.pop("extract_tables", False)
    extract_charts: bool = extract_params.pop("extract_charts", False)
    extract_infographics: bool = extract_params.pop("extract_infographics", False)

    # Inject additional configuration and trace information.
    if getattr(extraction_config, "pptx_extraction_config", None) is not None:
        extract_params["pptx_extraction_config"] = extraction_config.pptx_extraction_config
    if trace_info is not None:
        extract_params["trace_info"] = trace_info

    # Call the PPTX extraction function.
    extracted_data = python_pptx(
        pptx_stream=pptx_stream,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_infographics=extract_infographics,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extraction_config=extract_params,
        execution_trace_log=None,
    )

    return extracted_data


@unified_exception_handler
def extract_primitives_from_pptx_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[Dict[str, Any], BaseModel],
    extraction_config: Any,  # Assuming PPTXExtractorSchema or similar type
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Process a DataFrame containing base64-encoded PPTX files and extract primitive data.

    This function applies a decoding and extraction routine to each row of the DataFrame
    (via `_decode_and_extract_from_pptx`), then explodes any list results into separate rows, drops missing values,
    and compiles the extracted data into a new DataFrame. The resulting DataFrame includes columns for document type,
    extracted metadata, and a unique identifier (UUID).

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        Input DataFrame with PPTX files in base64 encoding. Expected to include columns 'source_id' and 'content'.
    task_config : Union[Dict[str, Any], BaseModel]
        Configuration for the PPTX extraction task, as a dict or Pydantic model.
    extraction_config : Any
        Configuration object for PPTX extraction (e.g. PPTXExtractorSchema).
    execution_trace_log : Optional[Dict[str, Any]], optional
        Optional dictionary containing trace information for debugging.

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted PPTX content containing columns:
        "document_type", "metadata", and "uuid".

    Raises
    ------
    Exception
        Reraises any exception encountered during extraction with additional context.
    """
    # Create a partial function to decode and extract content from each DataFrame row.
    decode_and_extract_partial = functools.partial(
        _decode_and_extract_from_pptx,
        task_props=task_config,
        extraction_config=extraction_config,
        trace_info=execution_trace_log,
    )
    # Apply the decoding and extraction to each row.
    extraction_series = df_extraction_ledger.apply(decode_and_extract_partial, axis=1)
    # Explode list results into separate rows and remove missing values.
    extraction_series = extraction_series.explode().dropna()

    # Convert the series into a DataFrame with defined columns.
    if not extraction_series.empty:
        extracted_df = pd.DataFrame(extraction_series.to_list(), columns=["document_type", "metadata", "uuid"])
    else:
        extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

    return extracted_df, {}
