# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from functools import partial
from typing import Any
from typing import Dict

import pandas as pd
from morpheus.config import Config
from pydantic import BaseModel

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.framework.orchestration.morpheus.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.mutate.filter import filter_images_internal

logger = logging.getLogger(__name__)


def image_filter_stage(df, task_props, validated_config) -> pd.DataFrame:
    try:
        if isinstance(task_props, BaseModel):
            task_props = task_props.model_dump()

        task_props.get("content_type")
        task_params = task_props.get("params", {})
        filter_flag = task_params.get("filter", True)

        logger.debug(f"Filtering images by scale with filter_flag={filter_flag}")

        df_result = filter_images_internal(df, task_params)

        return df_result

    except Exception as e:
        err_msg = f"image_filter_stage: Error filtering images. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e


def generate_image_filter_stage(
    c: Config,
    caption_config: Dict[str, Any],
    task: str = "filter",
    task_desc: str = "image_filter",
    pe_count: int = 8,
):
    """
    Generates a caption extraction stage with the specified configuration.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    caption_config : dict
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
        validated_config = ImageFilterSchema(**caption_config)
        _wrapped_caption_extract = partial(image_filter_stage, validated_config=validated_config)

        logger.debug(
            f"Generating image filtering stage with {pe_count} processing elements. task: {task}, document_type: *"
        )

        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=_wrapped_caption_extract,
            filter_properties={"content_type": ContentTypeEnum.IMAGE.value},
        )

    except Exception as e:
        err_msg = f"generate_image_filter_stage: Error generating image filter stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e
