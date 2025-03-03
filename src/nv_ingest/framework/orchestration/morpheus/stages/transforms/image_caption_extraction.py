# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import partial
from typing import Any
from typing import Dict

from morpheus.config import Config

from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal

logger = logging.getLogger(__name__)


def generate_caption_extraction_stage(
    c: Config,
    transform_config: Dict[str, Any],
    task: str = "caption",
    task_desc: str = "caption_extraction",
    pe_count: int = 8,
):
    """
    Generates a caption extraction stage with the specified configuration.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    transform_config : dict
        Configuration parameters for caption extraction.
    task : str, optional
        The task name to match for the stage worker function, by default "caption".
    task_desc : str, optional
        A descriptor to be used in latency tracing, by default "caption_extraction".
    pe_count : int, optional
        Number of processing elements to use, by default 8.

    Returns
    -------
    MultiProcessingBaseStage
        The generated caption extraction stage.

    Raises
    ------
    ValueError
        If an error occurs during stage generation.
    """
    try:
        validated_config = ImageCaptionExtractionSchema(**transform_config)
        _wrapped_caption_extract = partial(
            transform_image_create_vlm_caption_internal, transform_config=validated_config
        )

        logger.debug(f"Generating caption extraction stage with {pe_count} processing elements. Task: {task}")

        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_caption_extract
        )

    except Exception as e:
        err_msg = f"generate_caption_extraction_stage: Error generating caption extraction stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
