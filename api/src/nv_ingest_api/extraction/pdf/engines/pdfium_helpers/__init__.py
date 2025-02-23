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


def _work_extract_pdf(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    extractor_config: dict,
    trace_info=None,
) -> Any:
    """
    Perform PDF extraction on a decoded PDF stream using the given extraction parameters.
    """
    # Pop 'extract_method' from the config if provided, defaulting to 'pdfium'.
    extract_method = extractor_config.pop("extract_method", "pdfium")
    extractor_fn = EXTRACTOR_LOOKUP.get(extract_method, pdfium_extractor)
    return extractor_fn(
        pdf_stream,
        extract_text,
        extract_images,
        extract_tables,
        extract_charts,
        extractor_config,
        trace_info,
    )


def _orchestrate_row_extraction(
    row: pd.Series,
    task_config: Dict[str, Any],
    extractor_config: Any,
    trace_info: Optional[List[Any]] = None,
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
        extract_text = params.pop("extract_text")
        extract_images = params.pop("extract_images")
        extract_tables = params.pop("extract_tables")
        extract_charts = params.pop("extract_charts")
    except KeyError as e:
        raise ValueError(f"Missing required extraction flag: {e}")

    # Add row metadata (all columns except 'content') into the config.
    row_metadata = row.drop("content")
    params["row_data"] = row_metadata

    # Inject configuration settings, if provided.
    if getattr(extractor_config, "pdfium_config", None) is not None:
        params["pdfium_config"] = extractor_config.pdfium_config
    if getattr(extractor_config, "nemoretriever_parse_config", None) is not None:
        params["nemoretriever_parse_config"] = extractor_config.nemoretriever_parse_config

    # The remaining parameters constitute the extractor_config.
    extractor_config = params

    try:
        result = _work_extract_pdf(
            pdf_stream,
            extract_text,
            extract_images,
            extract_tables,
            extract_charts,
            extractor_config,
            trace_info,
        )
        return result
    except Exception as e:
        err_msg = f"Extraction failed for row with metadata {row_metadata}: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
