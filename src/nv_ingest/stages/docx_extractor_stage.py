# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
import traceback
from typing import Optional, Dict, Any

import pandas as pd
from pydantic import BaseModel
from morpheus.config import Config

from nv_ingest.extraction_workflows import docx
from nv_ingest.schemas.docx_extractor_schema import DocxExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.exception_handlers.pdf import create_exception_tag

logger = logging.getLogger(f"morpheus.{__name__}")


def decode_and_extract(base64_row, task_props, validated_config: Any, trace_info: Dict, default="python_docx"):
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
    doc_bytes = base64.b64decode(base64_content)

    # Load the document
    doc_stream = io.BytesIO(doc_bytes)

    # Type of extraction method to use
    extract_method = task_props.get("method", "python_docx")
    extract_params = task_props.get("params", {})
    try:
        if validated_config.docx_extraction_config is not None:
            extract_params["docx_extraction_config"] = validated_config.docx_extraction_config

        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        if not hasattr(docx, extract_method):
            extract_method = default

        func = getattr(docx, extract_method, default)
        logger.debug("Running extraction method: %s", extract_method)
        extracted_data = func(doc_stream, **extract_params)

        return extracted_data

    except Exception as error:
        traceback.print_exc()
        log_error_message = f"Error loading extractor:{error}"
        logger.error(log_error_message)
        logger.error(f"Failed on file:{source_id}")

    # Propagate error back and tag message as failed.
    exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)

    return exception_tag


def _process_docx_bytes(df, task_props, validated_config: Any, trace_info: Optional[Dict[str, Any]] = None):
    """
    Processes a cuDF DataFrame containing docx files in base64 encoding.
    Each document's content is replaced with its extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded documents).
    - task_props: dictionary containing instructions for the document processing task.

    Returns:
    - A pandas DataFrame with the docx content replaced by the extracted text.
    """

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(
            decode_and_extract, task_props=task_props, validated_config=validated_config, trace_info=trace_info
        )
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        logger.debug("extracted_df %s", extracted_df)
        return extracted_df

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from document: {e}")
        raise

    return df


def generate_docx_extractor_stage(
    c: Config,
    extractor_config: dict,
    task: str = "docx-extract",
    task_desc: str = "docx_content_extractor",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform document content extraction.

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
        Integer for how many process engines to use for document content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """
    validated_config = DocxExtractorSchema(**extractor_config)
    _wrapped_process_fn = functools.partial(_process_docx_bytes, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="docx"
    )
