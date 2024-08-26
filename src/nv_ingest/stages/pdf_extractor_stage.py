# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
import traceback
from typing import Any
from typing import Dict

import pandas as pd
from morpheus.config import Config

from nv_ingest.extraction_workflows import pdf
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.exception_handlers.pdf import create_exception_tag

logger = logging.getLogger(f"morpheus.{__name__}")


def _process_pdf_bytes(df, task_props, validated_config):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded PDFs).
    - task_props: dictionary containing instructions for the pdf processing task.

    Returns:
    - A pandas DataFrame with the PDF content replaced by the extracted text.
    """

    def decode_and_extract(base64_row, task_props, default="pdfium"):
        # Base64 content to extract
        try:
            base64_content = base64_row["content"]
        except KeyError:
            log_error_message = f"NO CONTENT FOUND IN ROW:\n{base64_row}"
            logger.error(log_error_message)
            raise

        # Row data to include in extraction
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data
        # Get source_id
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None
        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content)

        # Load the PDF
        pdf_stream = io.BytesIO(pdf_bytes)

        # Type of extraction method to use
        extract_method = task_props.get("method", "pdfium")
        extract_params = task_props.get("params", {})

        if validated_config.pdfium_config is not None:
            extract_params["pdfium_config"] = validated_config.pdfium_config

        if not hasattr(pdf, extract_method):
            extract_method = default
        try:
            func = getattr(pdf, extract_method, default)
            logger.debug("Running extraction method: %s", extract_method)
            extracted_data = func(pdf_stream, **extract_params)

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

        return extracted_df

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PDF: {e}")

    return df


def generate_pdf_extractor_stage(
    c: Config,
    extractor_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "pdf_content_extractor",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform pdf content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    extractor_config : dict
        Configuration parameters for pdf content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pdf content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """
    validated_config = PDFExtractorSchema(**extractor_config)
    _wrapped_process_fn = functools.partial(_process_pdf_bytes, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pdf"
    )
