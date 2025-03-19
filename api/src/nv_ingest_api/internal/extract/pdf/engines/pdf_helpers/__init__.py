# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA CORPORATION.

import base64
import io

import pandas as pd
from typing import Any, Dict, List, Optional
import logging

from nv_ingest_api.internal.extract.pdf.engines import (
    adobe_extractor,
    llama_parse_extractor,
    nemoretriever_parse_extractor,
    pdfium_extractor,
    tika_extractor,
    unstructured_io_extractor,
)
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

# Import extraction functions for different engines.

logger = logging.getLogger(__name__)

# Lookup table mapping extraction method names to extractor functions.
EXTRACTOR_LOOKUP = {
    "adobe": adobe_extractor,
    "llama": llama_parse_extractor,
    "nemoretriever_parse": nemoretriever_parse_extractor,
    "pdfium": pdfium_extractor,
    "tika": tika_extractor,
    "unstructured_io": unstructured_io_extractor,
}


def _work_extract_pdf(
    *,
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extractor_config: dict,
    execution_trace_log=None,
) -> Any:
    """
    Perform PDF extraction on a decoded PDF stream using the given extraction parameters.
    """

    extract_method = extractor_config["extract_method"]
    extractor_fn = EXTRACTOR_LOOKUP.get(extract_method, pdfium_extractor)
    return extractor_fn(
        pdf_stream,
        extract_text,
        extract_images,
        extract_infographics,
        extract_tables,
        extract_charts,
        extractor_config,
        execution_trace_log,
    )


@unified_exception_handler
def _orchestrate_row_extraction(
    row: pd.Series,
    task_config: Dict[str, Any],
    extractor_config: Any,
    execution_trace_log: Optional[List[Any]] = None,
) -> Any:
    """
    Orchestrate extraction for a single DataFrame row by decoding the PDF stream,
    building an extractor_config, and then delegating to the work function.
    """
    if "content" not in row:
        err_msg = f"Missing 'content' key in row: {row}"
        logger.error(err_msg)
        raise KeyError(err_msg)

    try:
        pdf_stream = io.BytesIO(base64.b64decode(row["content"]))
    except Exception as e:
        err_msg = f"Error decoding base64 content: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e

    # Begin with a copy of the task parameters.
    params = task_config.get("params", {}).copy()

    # Extract required boolean flags from params.
    try:
        extract_text = params.pop("extract_text", False)
        extract_images = params.pop("extract_images", False)
        extract_tables = params.pop("extract_tables", False)
        extract_charts = params.pop("extract_charts", False)
        extract_infographics = params.pop("extract_infographics", False)
        extract_method = params.get("extract_method", "pdfium")
    except KeyError as e:
        raise ValueError(f"Missing required extraction flag: {e}")

    # Add row metadata (all columns except 'content') into the config.
    row_metadata = row.drop("content")
    params["row_data"] = row_metadata

    extract_method = task_config.get("method", extract_method)
    params["extract_method"] = extract_method

    # Construct the config key based on the extraction method
    config_key = f"{extract_method}_config"

    # Handle both object and dictionary cases for extractor_config
    if hasattr(extractor_config, config_key):
        # Object case: extractor_config is a Pydantic model with attribute access
        method_config = getattr(extractor_config, config_key)
    elif isinstance(extractor_config, dict) and config_key in extractor_config:
        # Dictionary case: extractor_config is a dict with key access
        method_config = extractor_config[config_key]
    else:
        # If no matching config is found, log a warning but don't fail
        logger.warning(f"No {config_key} found in extractor_config: {extractor_config}")
        method_config = None

    # Add the method-specific config to the parameters if available
    if method_config is not None:
        params[config_key] = method_config
        logger.debug(f"Added {config_key} to extraction parameters")

    # The resulting parameters constitute the complete extractor_config
    extractor_config = params
    logger.debug(f"Final extractor_config: {extractor_config}")

    result = _work_extract_pdf(
        pdf_stream=pdf_stream,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_infographics=extract_infographics,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extractor_config=extractor_config,
        execution_trace_log=execution_trace_log,
    )

    return result
