# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import logging

from morpheus.config import Config

from nv_ingest.schemas.docx_extractor_schema import DocxExtractorSchema
from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.docx.docx_extractor import extract_primitives_from_docx_internal

logger = logging.getLogger(__name__)


def generate_docx_extractor_stage(
    c: Config,
    extraction_config: dict,
    task: str = "docx-extract",
    task_desc: str = "docx_content_extractor",
    pe_count: int = 8,
):
    """
    Helper function to generate a multiprocessing stage to perform document content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extraction_config : dict
        Configuration parameters for document content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for document content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with the applied worker function.

    Raises
    ------
    Exception
        If an error occurs during stage generation.
    """
    try:
        validated_config = DocxExtractorSchema(**extraction_config)
        _wrapped_process_fn = functools.partial(
            extract_primitives_from_docx_internal, extraction_config=validated_config
        )
        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="docx"
        )
    except Exception as e:
        err_msg = f"generate_docx_extractor_stage: Error generating document extractor stage. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
