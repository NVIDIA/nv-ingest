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
    if isinstance(task_props, BaseModel):
        task_props = task_props.model_dump()

    # Base64 content to extract
    base64_content = base64_row["content"]
    # Row data to include in extraction
    bool_index = base64_row.index.isin(("content",))
    row_data = base64_row[~bool_index]
    task_props["params"]["row_data"] = row_data
    # Get source_id
    source_id = base64_row["source_id"] if "source_id" in base64_row.index else None
    # Decode the base64 content
    pptx_bytes = base64.b64decode(base64_content)

    # Load the PPTX
    pptx_stream = io.BytesIO(pptx_bytes)

    # Type of extraction method to use
    extract_method = task_props.get("method", "python_pptx")
    extract_params = task_props.get("params", {})
    if not hasattr(pptx, extract_method):
        extract_method = default
    try:
        if validated_config.pptx_extraction_config is not None:
            extract_params["pptx_extraction_config"] = validated_config.pptx_extraction_config

        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        func = getattr(pptx, extract_method, default)
        logger.debug("Running extraction method: %s", extract_method)
        extracted_data = func(pptx_stream, **extract_params)

        return extracted_data

    except Exception as e:
        traceback.print_exc()
        log_error_message = f"Error loading extractor:{e}"
        logger.error(log_error_message)
        logger.error(f"Failed on file:{source_id}")

    # Propagate error back and tag message as failed.
    exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)

    return exception_tag


def _process_pptx_bytes(df, task_props: dict, validated_config: Any, trace_info: Optional[Dict[str, Any]] = None):
    """
    Processes a cuDF DataFrame containing PPTX files in base64 encoding.
    Each PPTX's content is replaced with its extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded PPTXs).
    - task_props: dictionary containing instructions for the pptx processing task.

    Returns:
    - A pandas DataFrame with the PPTX content replaced by the extracted text.
    """
    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(
            decode_and_extract, task_props=task_props, validated_config=validated_config, trace_info=trace_info
        )

        # logger.debug(f"processing ({task_props.get('method', None)})")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PPTX: {e}")
        raise

    return df


def generate_pptx_extractor_stage(
    c: Config,
    extractor_config: dict,
    task: str = "pptx-extract",
    task_desc: str = "pptx_content_extractor",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform pptx content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    extractor_config : dict
        Configuration parameters for document content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pptx content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """

    validated_config = PPTXExtractorSchema(**extractor_config)
    _wrapped_process_fn = functools.partial(_process_pptx_bytes, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pptx"
    )
