# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
import traceback

import pandas as pd
from morpheus.config import Config

from nv_ingest.extraction_workflows import docx
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.exception_handlers.pdf import create_exception_tag

logger = logging.getLogger(f"morpheus.{__name__}")


def _process_docx_bytes(df, task_props):
    """
    Processes a cuDF DataFrame containing docx files in base64 encoding.
    Each document's content is replaced with its extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded documents).
    - task_props: dictionary containing instructions for the document processing task.

    Returns:
    - A pandas DataFrame with the docx content replaced by the extracted text.
    """

    def decode_and_extract(base64_row, task_props, default="python_docx"):
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
        if not hasattr(docx, extract_method):
            extract_method = default
        try:
            func = getattr(docx, extract_method, default)
            logger.debug("Running extraction method: %s", extract_method)
            extracted_data = func(doc_stream, **extract_params)

            return extracted_data

        except Exception as e:
            traceback.print_exc()
            log_error_message = f"Error loading extractor:{e}"
            logger.error(log_error_message)
            logger.error(f"Failed on file:{source_id}")

        # Propagate error back and tag message as failed.
        exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)

        return exception_tag

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(decode_and_extract, task_props=task_props)
        logger.debug(f"processing ({task_props.get('method', None)})")
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

    return df


def generate_docx_extractor_stage(
    c: Config,
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

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_process_docx_bytes, document_type="docx"
    )
