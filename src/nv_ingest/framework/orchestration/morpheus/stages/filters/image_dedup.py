# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any
from typing import Dict

import pandas as pd
from morpheus.config import Config
from morpheus.utils.module_utils import ModuleLoaderFactory

from nv_ingest.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.mutate.deduplicate import deduplicate_images_internal

logger = logging.getLogger(__name__)

MODULE_NAME = "dedup_images"
MODULE_NAMESPACE = "nv-ingest"
ImageDedupLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageDedupSchema)


def dedup_image_stage(df_ledger: pd.DataFrame, task_config: Dict[str, Any], validated_config: Any) -> pd.DataFrame:
    """
    Deduplicates images in the provided DataFrame based on the task properties.

    This function processes a DataFrame containing images and applies a deduplication filter
    based on the `filter` parameter within the task properties. The deduplication is performed
    by identifying and removing duplicate images, or by marking them with informational messages,
    depending on the value of the `filter_flag`.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        The DataFrame containing the data to be deduplicated. It must have columns that include
        image metadata and document types.
    task_config : dict of {str: Any}
        A dictionary containing task properties, which may include the content type and parameters for filtering.
    mutate_config : Any
        The validated configuration object containing settings related to the deduplication task.

    Returns
    -------
    pd.DataFrame
        The DataFrame with duplicates either filtered out or marked as informational messages, depending on the
        `filter_flag`.

    Notes
    -----
    - The deduplication process operates on the rows where `document_type` is `ContentTypeEnum.IMAGE`.
    - The `filter_flag` parameter, extracted from `task_props`, determines whether duplicates are removed or marked.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "document_type": ["IMAGE", "IMAGE", "TEXT"],
    ...     "metadata": [{"content": "image1"}, {"content": "image1"}, {"content": "text"}]
    ... })
    >>> task_props = {"params": {"filter": True}}
    >>> result_df = dedup_image_stage(df_ledger, task_config, mutate_config)
    >>> print(result_df)

    Raises
    ------
    Exception
        If deduplication processing fails.
    """
    try:
        # TODO(Devin): Make hash algo configurable
        task_config = {"hash_algorithm": "md5"}

        df_result = deduplicate_images_internal(
            df_ledger=df_ledger, task_config=task_config, mutate_config=validated_config, execution_trace_log=None
        )

        return df_result

    except Exception as e:
        err_msg = f"dedup_image_stage: Error during deduplication. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e


def generate_dedup_stage(
    c: Config,
    deduplicate_image_config: Dict[str, Any],
    task: str = "dedup",
    task_desc: str = "dedup_images",
    pe_count: int = 8,
) -> MultiProcessingBaseStage:
    """
    Generates a deduplication processing stage for images using multiprocessing.

    This function validates the deduplication configuration, wraps the `dedup_image_stage` function with the validated
    configuration, and then generates a `MultiProcessingBaseStage` for executing the deduplication task.

    Parameters
    ----------
    c : Config
        The configuration object used to set up the multiprocessing stage.
    deduplicate_image_config : dict of {str: Any}
        A dictionary containing the deduplication configuration parameters.
    task : str, optional
        The name of the task to be performed, by default "dedup".
    task_desc : str, optional
        A description of the task, by default "dedup_images".
    pe_count : int, optional
        The number of processing elements (workers) to use for the task, by default 8.

    Returns
    -------
    MultiProcessingBaseStage
        An instance of `MultiProcessingBaseStage` configured to perform the deduplication task.

    Notes
    -----
    - The `dedup_image_stage` function is partially applied with the validated configuration, allowing it to be used
      within the multiprocessing framework.
    - The task is configured specifically for processing images, as indicated by the `filter_properties`.

    Examples
    --------
    >>> c = Config()
    >>> dedup_config = {"filter": True}
    >>> stage = generate_dedup_stage(c, deduplicate_image_config)
    >>> stage.run()

    Raises
    ------
    Exception
        If an error occurs during stage generation.
    """
    try:
        validated_config = ImageDedupSchema(**deduplicate_image_config)
        _wrapped_dedup_image_stage = functools.partial(dedup_image_stage, validated_config=validated_config)
        logger.debug(f"generate_dedup_stage: Generating deduplication stage with config: {validated_config}")
        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=_wrapped_dedup_image_stage,
            filter_properties={"content_type": ContentTypeEnum.IMAGE.value},
        )
    except Exception as e:
        err_msg = f"generate_dedup_stage: Error generating deduplication stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
