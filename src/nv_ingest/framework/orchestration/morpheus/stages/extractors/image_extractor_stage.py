# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import logging
from typing import Any
from typing import Dict

from morpheus.config import Config

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.image.image_extractor import extract_primitives_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema

logger = logging.getLogger(__name__)


def generate_image_extractor_stage(
    c: Config,
    extraction_config: Dict[str, Any],
    task: str = "extract",
    task_desc: str = "image_content_extractor",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform image content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    extraction_config : dict
        Configuration parameters for image content extractor.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use for image content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with the applied worker function.
    """
    try:
        validated_config = ImageExtractorSchema(**extraction_config)
        _wrapped_process_fn = functools.partial(
            extract_primitives_from_image_internal, extraction_config=validated_config
        )
        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=_wrapped_process_fn,
            document_type="regex:^(png|svg|jpeg|jpg|tiff)$",
        )
    except Exception as e:
        err_msg = f"generate_image_extractor_stage: Error generating image extractor stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
