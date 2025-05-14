# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
from typing import Optional, Dict, Any, Union, Tuple

import pandas as pd
from pydantic import BaseModel

from nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docx_helper import python_docx
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


def _prepare_task_props(
    task_config: Union[Dict[str, Any], BaseModel], base64_row: pd.Series
) -> (Dict[str, Any], Optional[str]):
    """
    Prepares the task properties by converting a Pydantic model to a dictionary (if needed)
    and injecting row-specific data.

    Parameters
    ----------
    task_config : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model containing instructions and parameters for extraction.
    base64_row : pd.Series
        A Series representing a row from the DataFrame that contains at least the "content"
        key and optionally "source_id".

    Returns
    -------
    Tuple[Dict[str, Any], Optional[str]]
        A tuple where the first element is the prepared task properties dictionary with the key
        "row_data" added under its "params" key, and the second element is the source_id (if present),
        otherwise None.
    """

    if isinstance(task_config, BaseModel):
        task_config = task_config.model_dump()
    else:
        task_config = dict(task_config)

    # Extract all row data except the "content" field.
    row_data = base64_row.drop(labels=["content"], errors="ignore")
    if "params" not in task_config:
        task_config["params"] = {}

    task_config["params"]["row_data"] = row_data

    source_id = base64_row.get("source_id", None)

    return task_config, source_id


@unified_exception_handler
def _decode_and_extract_from_docx(
    base64_row: pd.Series,
    task_config: Union[Dict[str, Any], BaseModel],
    extraction_config: Any,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Decodes base64 content from a DataFrame row and extracts data using the specified extraction method.

    The function decodes the base64-encoded content from the "content" key in the row, prepares
    extraction parameters (including additional row data and configuration), and invokes the extraction
    function from the docx module. If an error occurs, an exception tag is returned.

    Parameters
    ----------
    base64_row : pd.Series
        A Series containing the base64-encoded content under the key "content" and optionally a "source_id".
    task_config : Union[Dict[str, Any], BaseModel]
        A dictionary or Pydantic model containing extraction instructions and parameters.
        Expected to have a "params" key for additional parameters and optionally a "method" key specifying
        the extraction method.
    extraction_config : Any
        A configuration object that contains extraction-specific settings, such as `docx_extraction_config`.
    execution_trace_log : Optional[Dict[str, Any]], default=None
        A dictionary containing trace information for debugging or logging.
    default : str, optional
        The default extraction method to use if the specified method is not available (default is "python_docx").

    Returns
    -------
    Any
        The extracted data, or an exception tag if extraction fails.

    Raises
    ------
    Exception
        If an unhandled exception occurs during extraction, it is logged and a tagged error is returned.
    """
    # Prepare task properties and extract source_id
    task_config, source_id = _prepare_task_props(task_config, base64_row)

    # Retrieve base64 content and decode it into a byte stream.
    base64_content: str = base64_row["content"]
    doc_bytes: bytes = base64.b64decode(base64_content)
    doc_stream: io.BytesIO = io.BytesIO(doc_bytes)

    extract_params: Dict[str, Any] = task_config.get("params", {})

    # Extract required boolean flags from params.
    try:
        extract_text = extract_params.pop("extract_text", False)
        extract_images = extract_params.pop("extract_images", False)
        extract_tables = extract_params.pop("extract_tables", False)
        extract_charts = extract_params.pop("extract_charts", False)
        extract_infographics = extract_params.pop("extract_infographics", False)
    except KeyError as e:
        raise ValueError(f"Missing required extraction flag: {e}")

    # Inject configuration and trace info into extraction parameters.
    if getattr(extraction_config, "docx_extraction_config", None) is not None:
        extract_params["docx_extraction_config"] = extraction_config.docx_extraction_config

    if execution_trace_log is not None:
        extract_params["trace_info"] = execution_trace_log

    # extraction_func: Callable = _get_extraction_function(extract_method, default)
    extracted_data: Any = python_docx(
        docx_stream=doc_stream,
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
def extract_primitives_from_docx_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[Dict[str, Any], BaseModel],
    extraction_config: DocxExtractorSchema,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Union[Dict, None]]:
    """
    Processes a pandas DataFrame containing DOCX files encoded in base64, extracting text from
    each document and replacing the original content with the extracted text.

    This function applies a decoding and extraction routine to each row of the input DataFrame.
    The routine is provided via the `decode_and_extract` function, which is partially applied with
    task configuration, extraction configuration, and optional trace information. The results are
    exploded and any missing values are dropped, then compiled into a new DataFrame with columns
    for document type, metadata, and a UUID identifier.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        The input DataFrame containing DOCX files in base64 encoding. Expected columns include
        'source_id' and 'content'.
    task_config : Union[Dict[str, Any], BaseModel]
        Configuration instructions for the document processing task. This can be provided as a
        dictionary or a Pydantic model.
    extraction_config : Any
        A configuration object for document extraction that guides the extraction process.
    execution_trace_log : Optional[Dict[str, Any]], default=None
        An optional dictionary containing trace information for debugging or logging.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the original DOCX content replaced by the extracted text. The resulting
        DataFrame contains the columns "document_type", "metadata", and "uuid".

    Raises
    ------
    Exception
        If an error occurs during the document extraction process, the exception is logged and
        re-raised.
    """
    # Create a partial function to decode and extract using the provided configurations.
    _decode_and_extract = functools.partial(
        _decode_and_extract_from_docx,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=execution_trace_log,
    )

    # Apply the decode_and_extract function to each row in the DataFrame.
    sr_extraction = df_extraction_ledger.apply(_decode_and_extract, axis=1)

    # Explode any list results and drop missing values.
    sr_extraction = sr_extraction.explode().dropna()

    # Convert the extraction results to a DataFrame if available.
    if not sr_extraction.empty:
        extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
    else:
        extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

    return extracted_df, {}
