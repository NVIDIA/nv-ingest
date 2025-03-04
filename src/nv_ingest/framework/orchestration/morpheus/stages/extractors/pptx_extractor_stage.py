# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import logging

from morpheus.config import Config

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.pptx.pptx_extractor import extract_primitives_from_pptx_internal
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema

logger = logging.getLogger(__name__)


def generate_pptx_extractor_stage(
    c: Config,
    extraction_config: dict,
    task: str = "pptx-extract",
    task_desc: str = "pptx_content_extractor",
    pe_count: int = 8,
):
    """
    Helper function to generate a multiprocessing stage to perform PPTX content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extraction_config : dict
        Configuration parameters for PPTX content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for PPTX content extraction.

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
        validated_config = PPTXExtractorSchema(**extraction_config)
        _wrapped_process_fn = functools.partial(
            extract_primitives_from_pptx_internal, extraction_config=validated_config
        )
        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn, document_type="pptx"
        )
    except Exception as e:
        err_msg = f"generate_pptx_extractor_stage: Error generating PPTX extractor stage. " f"Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
