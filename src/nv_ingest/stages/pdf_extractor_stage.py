# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import logging
from typing import Any
from typing import Dict

from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.extraction.pdf import process_pdf_bytes

logger = logging.getLogger(f"morpheus.{__name__}")


def generate_pdf_extractor_stage(
    c: Any,
    extractor_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "pdf_content_extractor",
    pe_count: int = 24,
) -> Any:
    """
    Generate a multiprocessing stage for PDF extraction.

    This function acts as an adapter between the pipeline architecture and the
    underlying work function. It validates the extractor configuration, wraps the
    DataFrame-level processing function in a partial function, and creates a
    MultiProcessingBaseStage for parallel processing of PDF extraction.

    Parameters
    ----------
    c : Any
        The global configuration object for the pipeline.
    extractor_config : dict
        A dictionary containing configuration parameters for the PDF extractor.
    task : str, optional
        The name of the extraction task. Defaults to "extract".
    task_desc : str, optional
        A descriptor for the task used in latency tracing. Defaults to
        "pdf_content_extractor".
    pe_count : int, optional
        The number of processing engines to use for extraction. Defaults to 24.

    Returns
    -------
    Any
        A MultiProcessingBaseStage object configured for PDF extraction.

    Raises
    ------
    Exception
        If an error occurs during the creation of the PDF extractor stage.
    """
    try:
        validated_config = PDFExtractorSchema(**extractor_config)
        wrapped_process_fn = functools.partial(process_pdf_bytes, validated_config=validated_config)

        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=wrapped_process_fn, document_type="pdf"
        )
    except Exception as e:
        err_msg = f"generate_pdf_extractor_stage: Error generating PDF extractor stage: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
