# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import functools
import io
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from morpheus.config import Config
from nv_ingest.extraction_workflows import pdf
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage

logger = logging.getLogger(f"morpheus.{__name__}")


def decode_and_extract(
    base64_row: Dict[str, Any],
    task_props: Dict[str, Any],
    validated_config: Any,
    default: str = "pdfium",
    trace_info: Optional[List] = None,
) -> Any:
    """
    Decodes base64 content from a row and extracts data from it using the specified extraction method.

    Parameters
    ----------
    base64_row : dict
        A dictionary containing the base64-encoded content and other relevant data.
        The key "content" should contain the base64 string, and the key "source_id" is optional.
    task_props : dict
        A dictionary containing task properties. It should have the keys:
        - "method" (str): The extraction method to use (e.g., "pdfium").
        - "params" (dict): Parameters to pass to the extraction function.
    validated_config : Any
        Configuration object that contains `pdfium_config`. Used if the `pdfium` method is selected.
    default : str, optional
        The default extraction method to use if the specified method in `task_props` is not available (default is "pdfium").

    Returns
    -------
    Any
        The extracted data from the decoded content. The exact return type depends on the extraction method used.

    Raises
    ------
    KeyError
        If the "content" key is missing from `base64_row`.
    Exception
        For any other unhandled exceptions during extraction, an error is logged, and the exception is re-raised.
    """

    try:
        base64_content = base64_row["content"]
    except KeyError:
        log_error_message = f"NO CONTENT FOUND IN ROW:\n{base64_row}"
        logger.error(log_error_message)
        raise

    try:
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
        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        if not hasattr(pdf, extract_method):
            extract_method = default

        func = getattr(pdf, extract_method, default)
        logger.debug("Running extraction method: %s", extract_method)
        extracted_data = func(pdf_stream, **extract_params)

        return extracted_data

    except Exception as e:
        err_msg = f"Unhandled exception in decode_and_extract for '{source_id}':\n{e}"
        logger.error(err_msg)

        raise

    # Propagate error back and tag message as failed.
    # exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)


def process_pdf_bytes(df, task_props, validated_config, trace_info=None):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded PDFs).
    - task_props: dictionary containing instructions for the pdf processing task.

    Returns:
    - A pandas DataFrame with the PDF content replaced by the extracted text.
    """
    if trace_info is None:
        trace_info = {}

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(
            decode_and_extract, task_props=task_props, validated_config=validated_config, trace_info=trace_info
        )
        logger.debug(f"processing ({task_props.get('method', None)})")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": trace_info}

    except Exception as e:
        err_msg = f"Unhandled exception in process_pdf_bytes: {e}"
        logger.error(err_msg)

        raise


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
    _wrapped_process_fn = functools.partial(process_pdf_bytes, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pdf"
    )
