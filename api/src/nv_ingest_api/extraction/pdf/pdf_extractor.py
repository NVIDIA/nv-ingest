# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, NVIDIA CORPORATION.

import base64
import io
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import logging

# Import extraction functions for different engines.
from nv_ingest_api.extraction.pdf.engines import *

logger = logging.getLogger(__name__)

# Lookup table mapping extraction method names to extractor functions.
EXTRACTOR_LOOKUP = {
    "adobe": adobe_extractor,
    "llama": llama_parse_extractor,
    "nemoretriever": nemoretriever_parse_extractor,
    "pdfium": pdfium_extractor,
    "tika": tika_extractor,
    "unstructured_io": unstructured_io_extractor,
}


def work_extract_pdf(pdf_stream: io.BytesIO, extract_params: Dict[str, Any]) -> Any:
    """
    Perform PDF extraction on a decoded PDF stream.

    This work function is solely responsible for performing the extraction
    from the provided PDF stream using the given extraction parameters. It
    does not concern itself with orchestration details such as decoding or
    parameter preparation.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A BytesIO stream containing the PDF data.
    extract_params : dict
        A dictionary of extraction parameters, which may include configuration
        settings, row metadata, trace information, and other parameters required
        by the extraction method. It should include the key 'extract_method'
        indicating which extraction engine to use.

    Returns
    -------
    Any
        The result of the extraction process. The exact type depends on the
        extraction function being used.

    Raises
    ------
    Exception
        Propagates any exception raised by the underlying extraction function.
    """

    extract_method = extract_params.pop("extract_method", "pdfium")
    extractor_fn = EXTRACTOR_LOOKUP.get(extract_method, pdfium_extractor)

    return extractor_fn(pdf_stream, **extract_params)


def orchestrate_row_extraction(
    row: pd.Series, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[List[Any]] = None
) -> Any:
    """
    Orchestrate the extraction process for a single DataFrame row.

    This function handles the orchestration steps required for processing a single
    row: it decodes the base64-encoded PDF content, extracts row metadata, augments
    the extraction parameters with configuration and trace information, and then
    delegates the extraction work to the work function.

    Parameters
    ----------
    row : pd.Series
        A pandas Series representing a single row from the DataFrame. Must contain a
        'content' key with base64-encoded PDF data.
    task_props : dict
        A dictionary containing task properties including extraction parameters and
        the desired extraction method.
    validated_config : Any
        A configuration object holding settings (e.g., pdfium_config and
        nemoretriever_parse_config) for the extractor.
    trace_info : list, optional
        A list of trace information to be merged into the extraction parameters.
        Defaults to None.

    Returns
    -------
    Any
        The result returned by the work extraction function.

    Raises
    ------
    KeyError
        If the 'content' key is missing from the row.
    Exception
        If an error occurs during the extraction process.
    """
    if "content" not in row:
        err_msg = f"orchestrate_row_extraction: Missing 'content' key in row: {row}"
        logger.error(err_msg)
        raise KeyError(err_msg)

    try:
        # Decode the base64-encoded PDF content into a BytesIO stream.
        pdf_stream = io.BytesIO(base64.b64decode(row["content"]))
    except Exception as e:
        err_msg = f"orchestrate_row_extraction: Error decoding base64 content: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e

    # Prepare extraction parameters from task properties.
    extract_params = task_props.get("params", {}).copy()

    # Add row metadata (all columns except 'content').
    row_metadata = row.drop("content")
    extract_params["row_data"] = row_metadata

    # Inject configuration settings.
    if getattr(validated_config, "pdfium_config", None) is not None:
        extract_params["pdfium_config"] = validated_config.pdfium_config
    if getattr(validated_config, "nemoretriever_parse_config", None) is not None:
        extract_params["nemoretriever_parse_config"] = validated_config.nemoretriever_parse_config

    # Include trace information if provided.
    if trace_info is not None:
        extract_params["trace_info"] = trace_info

    # Delegate the actual extraction work.
    try:
        result = work_extract_pdf(pdf_stream, extract_params)
        return result
    except Exception as e:
        err_msg = f"orchestrate_row_extraction: Extraction failed for row with metadata " f"{row_metadata}: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def run_pdf_extraction(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[List[Any]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process a DataFrame of PDF documents by orchestrating extraction for each row.

    This function applies the row-level orchestration function to every row in the
    DataFrame, aggregates the results, and returns a new DataFrame with the extracted
    data along with any trace information.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing PDF documents. Must include a 'content' column
        with base64-encoded PDF data.
    task_props : dict
        A dictionary containing task properties and extraction parameters.
    validated_config : Any
        A validated configuration object for the PDF extractor.
    trace_info : list, optional
        A list for accumulating trace information during extraction. Defaults to None.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        A tuple where the first element is a DataFrame with the extracted data (with
        columns: document_type, metadata, uuid) and the second element is a dictionary
        containing trace information.

    Raises
    ------
    Exception
        If an error occurs during the extraction process on any row.
    """
    try:
        # Apply the orchestration function to each row.
        extraction_series = df.apply(
            lambda row: orchestrate_row_extraction(row, task_props, validated_config, trace_info), axis=1
        )
        # Explode the results if the extraction returns lists.
        extraction_series = extraction_series.explode().dropna()

        # Convert the extracted results into a DataFrame.
        if not extraction_series.empty:
            extracted_df = pd.DataFrame(extraction_series.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": trace_info}
    except Exception as e:
        err_msg = f"process_pdf_bytes: Error processing PDF bytes: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
