# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from functools import partial
from typing import Any, Optional, List
from typing import Dict

import pandas as pd
from morpheus.config import Config

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.framework.orchestration.morpheus.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.mutate.filter import filter_images_internal

logger = logging.getLogger(__name__)


def image_filter_stage(
    df_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    mutate_config: Any,
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Apply the image filtering stage to the ledger DataFrame.

    This function extracts image filtering parameters from the provided task
    configuration and delegates processing to the internal filter_images_internal
    function.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        The ledger DataFrame containing image metadata. This DataFrame must include
        the required columns for filtering.
    task_config : Dict[str, Any]
        A dictionary containing the task configuration. Expected to have a key "params"
        holding the filtering parameters.
    mutate_config : Any
        Additional mutation configuration (passed directly to the internal function).
    execution_trace_log : Optional[List[Any]], optional
        An optional list for execution trace logging, by default None.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame after the image filtering stage has been applied.

    Raises
    ------
    Exception
        Any exception raised during the filtering process is logged and re-raised with
        additional context.
    """
    try:
        task_params: Dict[str, Any] = task_config.get("params", {})
        df_result = filter_images_internal(
            df_ledger=df_ledger,
            task_config=task_params,
            mutate_config=mutate_config,
            execution_trace_log=execution_trace_log,
        )
        return df_result
    except Exception as e:
        err_msg = f"image_filter_stage: Error filtering images. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def generate_image_filter_stage(
    c: Config,
    image_filter_config: Dict[str, Any],
    task: str = "filter",
    task_desc: str = "image_filter",
    pe_count: int = 8,
) -> MultiProcessingBaseStage:
    """
    Generate an image filter stage with the specified configuration.

    This function validates the image filter configuration and wraps the image_filter_stage
    function to produce a multi-processing stage for filtering images.

    Parameters
    ----------
    c : Config
        The global configuration object.
    image_filter_config : Dict[str, Any]
        A dictionary containing configuration parameters for image filtering.
    task : str, optional
        The task name to be assigned to the stage. Default is "filter".
    task_desc : str, optional
        A descriptor for latency tracing. Default is "image_filter".
    pe_count : int, optional
        The number of processing elements to use. Default is 8.

    Returns
    -------
    MultiProcessingBaseStage
        The generated multi-processing stage configured for image filtering.

    Raises
    ------
    Exception
        Any exception raised during stage generation is logged and re-raised with additional context.
    """
    try:
        validated_config = ImageFilterSchema(**image_filter_config)
        wrapped_filter_fn = partial(image_filter_stage, mutate_config=validated_config)

        logger.debug(
            f"Generating image filtering stage with {pe_count} processing elements. "
            f"Task: {task}, Document Type: {ContentTypeEnum.IMAGE.value}"
        )

        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=wrapped_filter_fn,
            filter_properties={"content_type": ContentTypeEnum.IMAGE.value},
        )
    except Exception as e:
        err_msg = f"generate_image_filter_stage: Error generating image filter stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
