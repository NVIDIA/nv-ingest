# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
import traceback
from typing import Any, Optional, Dict

import pandas as pd
from pydantic import BaseModel
from morpheus.config import Config

from nv_ingest.extraction_workflows import pptx
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.schemas.pptx_extractor_schema import PPTXExtractorSchema
from nv_ingest.util.exception_handlers.pdf import create_exception_tag

logger = logging.getLogger(f"morpheus.{__name__}")


def decode_and_extract(base64_row, task_props, validated_config: Any, trace_info: Dict, default="python_pptx"):
    """
    Decodes base64 content from a row and extracts data from it using the specified extraction method.

    Parameters
    ----------
    base64_row : pd.Series
        A Series containing the base64-encoded content and other relevant data.
        The key "content" should contain the base64 string, and the key "source_id" is optional.
    task_props : dict or BaseModel
        A dictionary (or BaseModel instance) containing instructions and parameters for extraction.
    validated_config : Any
        Configuration object that contains `pptx_extraction_config`.
    trace_info : dict
        Dictionary containing trace information.
    default : str, optional
        The default extraction method to use if the specified method is not available
        (default is "python_pptx").

    Returns
    -------
    Any
        The extracted data, or an exception tag if extraction fails.
    """
    source_id = None
    try:
        if isinstance(task_props, BaseModel):
            task_props = task_props.model_dump()

        # Retrieve base64 content.
        base64_content = base64_row["content"]

        # Extract row data (all columns except "content") and add to parameters.
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data

        # Retrieve source_id if present.
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None

        # Decode the base64 content and create a stream.
        pptx_bytes = base64.b64decode(base64_content)
        pptx_stream = io.BytesIO(pptx_bytes)

        # Determine extraction method and parameters.
        extract_method = task_props.get("method", "python_pptx")
        extract_params = task_props.get("params", {})

        if not hasattr(pptx, extract_method):
            extract_method = default

        if validated_config.pptx_extraction_config is not None:
            extract_params["pptx_extraction_config"] = validated_config.pptx_extraction_config

        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        func = getattr(pptx, extract_method, default)
        logger.debug("decode_and_extract: Running extraction method: %s", extract_method)
        extracted_data = func(pptx_stream, **extract_params)
        return extracted_data

    except Exception as e:
        err_msg = f"decode_and_extract: Error processing PPTX for source '{source_id}'. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        # Return an exception tag to indicate extraction failure.
        exception_tag = create_exception_tag(error_message=err_msg, source_id=source_id)

        return exception_tag


def _process_pptx_bytes(df, task_props: dict, validated_config: Any, trace_info: Optional[Dict[str, Any]] = None):
    """
    Processes a pandas DataFrame containing PPTX files in base64 encoding.
    Each PPTX's content is replaced with its extracted text.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with columns 'source_id' and 'content' (base64 encoded PPTXs).
    task_props : dict or BaseModel
        Dictionary containing instructions for the PPTX processing task.
    validated_config : Any
        Configuration object for PPTX extraction.
    trace_info : dict, optional
        Dictionary containing trace information.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the PPTX content replaced by the extracted text.
    """
    try:
        _decode_and_extract = functools.partial(
            decode_and_extract, task_props=task_props, validated_config=validated_config, trace_info=trace_info
        )
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})
        logger.debug("_process_pptx_bytes: Extraction complete.")
        return extracted_df

    except Exception as e:
        err_msg = f"_process_pptx_bytes: Failed to extract text from PPTX. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        traceback.print_exc()

        raise type(e)(err_msg) from e


def generate_pptx_extractor_stage(
    c: Config,
    extractor_config: dict,
    task: str = "pptx-extract",
    task_desc: str = "pptx_content_extractor",
    pe_count: int = 1,
):
    """
    Helper function to generate a multiprocessing stage to perform PPTX content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extractor_config : dict
        Configuration parameters for PPTX content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for PPTX content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with the applied worker function.

    Raises
    ------
    Exception
        If an error occurs during stage generation.
    """
    try:
        validated_config = PPTXExtractorSchema(**extractor_config)
        _wrapped_process_fn = functools.partial(_process_pptx_bytes, validated_config=validated_config)
        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pptx"
        )
    except Exception as e:
        err_msg = f"generate_pptx_extractor_stage: Error generating PPTX extractor stage. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
