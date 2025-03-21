# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import logging
from typing import Any, Optional, List, Tuple
from typing import Dict

import pandas as pd

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema

logger = logging.getLogger(f"morpheus.{__name__}")


def _inject_validated_config(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict,
    execution_trace_log: Optional[List[Any]] = None,
    validated_config: Any = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Helper function that injects the validated_config into the config dictionary and
    calls extract_primitives_from_pdf.

    Parameters
    ----------
    df_payload : pd.DataFrame
        A DataFrame containing PDF documents.
    task_config : dict
        A dictionary of configuration parameters. Expected to include 'task_props'.
    execution_trace_log : list, optional
        Optional list for trace information.
    validated_config : Any, optional
        The validated configuration to be injected.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The result from extract_primitives_from_pdf.
    """

    return extract_primitives_from_pdf_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extractor_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


def generate_pdf_extractor_stage(
    c: Any,
    extractor_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "pdf_content_extractor",
    pe_count: int = 24,
) -> Any:
    """
    Generate a multiprocessing stage for PDF extraction.

    This function validates the extractor configuration, creates a partial function
    wrapper to inject the validated configuration into the config dict, and returns
    a MultiProcessingBaseStage for parallel PDF extraction.

    Parameters
    ----------
    c : Any
        The global configuration object for the pipeline.
    extractor_config : dict
        A dictionary containing configuration parameters for the PDF extractor.
    task : str, optional
        The name of the extraction task. Defaults to "extract".
    task_desc : str, optional
        A descriptor for the task used in latency tracing. Defaults to "pdf_content_extractor".
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
        validated_extractor_config = PDFExtractorSchema(**extractor_config)
        wrapped_process_fn = functools.partial(_inject_validated_config, validated_config=validated_extractor_config)

        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=wrapped_process_fn, document_type="pdf"
        )
    except Exception as e:
        err_msg = f"generate_pdf_extractor_stage: Error generating PDF extractor stage: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
