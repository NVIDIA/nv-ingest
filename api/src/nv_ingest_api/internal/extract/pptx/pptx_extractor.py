# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import functools
import io
import logging
from typing import Any, Optional, Dict, Union, Callable, Tuple

import pandas as pd
from pydantic import BaseModel

from nv_ingest.extraction_workflows import pptx
from nv_ingest.schemas.pptx_extractor_schema import PPTXExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from nv_ingest_api.util.exception_handlers.pdf import create_exception_tag

logger = logging.getLogger(__name__)


def _prepare_task_properties(
    base64_row: pd.Series, task_props: Union[Dict[str, Any], BaseModel]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Prepares and returns the task properties dictionary and the source identifier from a DataFrame row.

    This function converts the task properties to a dictionary if provided as a Pydantic model,
    extracts all row data except the "content" key, and adds this data under the "row_data" key in the
    task properties. It also retrieves the "source_id" from the row if available.

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
        A tuple where the first element is the prepared task properties dictionary with "row_data" added,
        and the second element is the source_id if present; otherwise, None.
    """
    if isinstance(task_props, BaseModel):
        task_props = task_props.model_dump()
    else:
        task_props = dict(task_props)

    # Extract row data excluding the "content" field.
    row_data = base64_row.drop(labels=["content"], errors="ignore")
    if "params" not in task_props:
        task_props["params"] = {}
    task_props["params"]["row_data"] = row_data

    source_id = base64_row.get("source_id", None)
    return task_props, source_id


def _get_extraction_function(module: Any, method_name: str, default: str) -> Callable:
    """
    Retrieves the extraction function from the specified module using the given method name.

    If the specified method does not exist in the module, the default method is used.

    Parameters
    ----------
    module : Any
        The module from which to retrieve the extraction function.
    method_name : str
        The desired extraction method name.
    default : str
        The default extraction method name to use if the specified method is not available.

    Returns
    -------
    Callable
        The extraction function from the module.
    """
    if not hasattr(module, method_name):
        method_name = default
    return getattr(module, method_name)


@unified_exception_handler
def _decode_and_extract_from_pptx(
    base64_row: pd.Series,
    task_props: Union[Dict[str, Any], BaseModel],
    validated_config: Any,
    trace_info: Dict[str, Any],
    default: str = "python_pptx",
) -> Any:
    """
    Decodes base64-encoded PPTX content from a DataFrame row and extracts data using the specified extraction method.

    This function prepares the task properties for extraction, decodes the base64 content into a byte stream,
    determines the extraction method to use, and calls the corresponding extraction function from the pptx module.
    If extraction fails, an exception tag is returned.

    Parameters
    ----------
    base64_row : pd.Series
        A Series containing the base64-encoded PPTX content under the key "content" and optionally a "source_id".
    task_props : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model containing extraction instructions and parameters. It may include a "method"
        key specifying the extraction method and a "params" key for additional parameters.
    validated_config : Any
        A configuration object that contains PPTX extraction settings, such as `pptx_extraction_config`.
    trace_info : Dict[str, Any]
        A dictionary containing trace information for logging or debugging.
    default : str, optional
        The default extraction method to use if the specified method is not available (default is "python_pptx").

    Returns
    -------
    Any
        The extracted data from the PPTX file, or an exception tag indicating failure if an error occurs.

    Raises
    ------
    Exception
        Any unhandled exception encountered during extraction is logged and re-raised wrapped with additional context.
    """
    source_id: Optional[str] = None
    try:
        # Prepare task properties and extract source_id.
        prepared_task_props, source_id = _prepare_task_properties(base64_row, task_props)

        # Retrieve base64 content and decode it.
        base64_content: str = base64_row["content"]
        pptx_bytes: bytes = base64.b64decode(base64_content)
        pptx_stream: io.BytesIO = io.BytesIO(pptx_bytes)

        # Determine the extraction method and parameters.
        extract_method: str = prepared_task_props.get("method", default)
        extract_params: Dict[str, Any] = prepared_task_props.get("params", {})

        # Inject configuration settings and trace information.
        if getattr(validated_config, "pptx_extraction_config", None) is not None:
            extract_params["pptx_extraction_config"] = validated_config.pptx_extraction_config
        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        # Retrieve the extraction function from the pptx module.
        extraction_func: Callable = _get_extraction_function(pptx, extract_method, default)
        logger.debug("decode_and_extract: Running extraction method: %s", extract_method)

        extracted_data = extraction_func(pptx_stream, **extract_params)
        return extracted_data

    except Exception as e:
        err_msg = f"decode_and_extract: Error processing PPTX for source '{source_id}'. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        exception_tag = create_exception_tag(error_message=err_msg, source_id=source_id)
        return exception_tag


@unified_exception_handler
def extract_primitives_from_pptx_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[Dict[str, Any], BaseModel],
    extraction_config: PPTXExtractorSchema,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Processes a pandas DataFrame containing PPTX files in base64 encoding and extracts text content from them.

    This function applies a decoding and extraction routine to each row of the input DataFrame using the
    `decode_and_extract` function. The resulting extraction is exploded into individual elements and any
    missing values are removed. The extracted data is then compiled into a new DataFrame with columns
    representing the document type, extracted metadata, and a unique identifier (UUID).

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        The input DataFrame containing PPTX files in base64 encoding. Expected columns include
        'source_id' and 'content', where 'content' holds the base64 encoded PPTX document.
    task_config : Union[Dict[str, Any], BaseModel]
        Configuration instructions for the PPTX extraction task. This can be provided as a dictionary
        or as a Pydantic model.
    extraction_config : Any
        A configuration object for PPTX extraction that provides necessary settings and parameters.
    execution_trace_log : Optional[Dict[str, Any]], default=None
        An optional dictionary containing trace information for logging or debugging purposes.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the extracted PPTX content. The DataFrame contains the following columns:
          - "document_type": The type of the document.
          - "metadata": A dictionary containing the extracted metadata.
          - "uuid": A unique identifier for the document.

    Raises
    ------
    Exception
        If an error occurs during the PPTX extraction process, the error is logged, a traceback is printed,
        and the exception is re-raised with additional context.
    """
    # Create a partial function to decode and extract content from each row.
    decode_and_extract_partial = functools.partial(
        _decode_and_extract_from_pptx,
        task_props=task_config,
        validated_config=extraction_config,
        trace_info=execution_trace_log,
    )
    # Apply the decode_and_extract function to each row of the DataFrame.
    extraction_series = df_extraction_ledger.apply(decode_and_extract_partial, axis=1)
    # Explode any list results into separate rows and drop missing values.
    extraction_series = extraction_series.explode().dropna()

    # Convert the resulting series into a DataFrame with specified columns.
    if not extraction_series.empty:
        extracted_df = pd.DataFrame(extraction_series.to_list(), columns=["document_type", "metadata", "uuid"])
    else:
        extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

    logger.debug("_process_pptx_bytes: Extraction complete.")
    return extracted_df
