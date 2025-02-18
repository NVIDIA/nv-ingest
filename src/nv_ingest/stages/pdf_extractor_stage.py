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
from typing import List
from typing import Optional

import pandas as pd
from pydantic import BaseModel
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
        The default extraction method to use if the specified method in `task_props` is not available.
    trace_info : Optional[List], optional
        An optional list for trace information to pass to the extraction function.

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
    except KeyError as e:
        err_msg = f"decode_and_extract: Missing 'content' key in row: {base64_row}"
        logger.error(err_msg, exc_info=True)
        raise KeyError(err_msg) from e

    try:
        # Extract row data excluding the "content" column.
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data

        # Get source_id if available.
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None

        # Decode the base64 content.
        pdf_bytes = base64.b64decode(base64_content)
        pdf_stream = io.BytesIO(pdf_bytes)

        # Determine the extraction method and parameters.
        extract_method = task_props.get("method", "pdfium")
        extract_params = task_props.get("params", {})

        if validated_config.pdfium_config is not None:
            extract_params["pdfium_config"] = validated_config.pdfium_config
        if validated_config.nemoretriever_parse_config is not None:
            extract_params["nemoretriever_parse_config"] = validated_config.nemoretriever_parse_config
        if trace_info is not None:
            extract_params["trace_info"] = trace_info

        if not hasattr(pdf, extract_method):
            extract_method = default

        func = getattr(pdf, extract_method, default)
        logger.debug("decode_and_extract: Running extraction method: %s", extract_method)
        extracted_data = func(pdf_stream, **extract_params)

        return extracted_data

    except Exception as e:
        err_msg = f"decode_and_extract: Error processing PDF for source '{source_id}'. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        traceback.print_exc()

        raise type(e)(err_msg) from e


def process_pdf_bytes(df, task_props, validated_config, trace_info=None):
    """
    Processes a pandas DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced by the extracted text.

    Parameters:
    - df: pandas DataFrame with columns 'source_id' and 'content' (base64 encoded PDFs).
    - task_props: dictionary containing instructions for the PDF processing task.
    - validated_config: configuration object for the extractor.
    - trace_info: optional trace information to include in extraction.

    Returns:
    - A tuple containing:
      - A pandas DataFrame with the PDF content replaced by the extracted text.
      - A dictionary with trace information.
    """
    if trace_info is None:
        trace_info = {}

    if isinstance(task_props, BaseModel):
        task_props = task_props.model_dump()

    try:
        _decode_and_extract = functools.partial(
            decode_and_extract,
            task_props=task_props,
            validated_config=validated_config,
            trace_info=trace_info,
        )
        logger.debug(f"process_pdf_bytes: Processing PDFs with extraction method: {task_props.get('method', None)}")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"trace_info": trace_info}

    except Exception as e:
        err_msg = f"process_pdf_bytes: Error processing PDF bytes. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def generate_pdf_extractor_stage(
    c: Config,
    extractor_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "pdf_content_extractor",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform PDF content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extractor_config : dict
        Configuration parameters for the PDF content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for PDF content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with the applied worker function.
    """
    try:
        validated_config = PDFExtractorSchema(**extractor_config)
        _wrapped_process_fn = functools.partial(process_pdf_bytes, validated_config=validated_config)
        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pdf"
        )
    except Exception as e:
        err_msg = f"generate_pdf_extractor_stage: Error generating PDF extractor stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
