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
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def add_info_message(x, info_msg):
    x["info_message_metadata"] = info_msg

    return x


def calculate_average_image_size(x):
    return (x["image_metadata"]["width"] + x["image_metadata"]["height"]) / 2


def calculate_aspect_ratio(x):
    return x["image_metadata"]["width"] / max(x["image_metadata"]["height"], 1e-9)


def _cpu_only_apply_filter(df: pd.DataFrame, task_params: dict) -> pd.DataFrame:
    """
    Applies a deduplication filter to images in the DataFrame.

    This function identifies duplicate images within a DataFrame based on content hashes and either filters out
    duplicates or marks them as informational messages, depending on the `filter_flag`.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be filtered. It must have a `document_type` column indicating content type
        and a `metadata` column containing content metadata.
    filter_flag : bool
        A flag indicating whether to filter out duplicates (`True`) or mark them with informational messages (`False`).

    Returns
    -------
    pd.DataFrame
        The DataFrame with duplicates either filtered out or marked as informational messages.

    Notes
    -----
    - The function operates only on rows where `document_type` is `ContentTypeEnum.IMAGE`.
    - When `filter_flag` is `False`, duplicate images are marked with an informational message and the `document_type`
      is updated to `ContentTypeEnum.INFO_MSG`.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "document_type": [ContentTypeEnum.IMAGE, ContentTypeEnum.IMAGE, ContentTypeEnum.TEXT],
    ...     "metadata": [{"content": "image1"}, {"content": "image1"}, {"content": "text"}]
    ... })
    >>> result_df = _cpu_only_apply_filter(df, filter_flag=True)
    >>> result_df
      document_type            metadata
    0       IMAGE  {'content': 'image1'}
    2        TEXT     {'content': 'text'}

    Raises
    ------
    ValueError
        If `df` does not contain the necessary columns `document_type` and `metadata`.
    """
    try:
        min_size = task_params.get("min_size")
        max_aspect_ratio = task_params.get("max_aspect_ratio")
        min_aspect_ratio = task_params.get("min_aspect_ratio")
        filter_images = task_params.get("filter", False)

        # Return if no images
        image_mask = df["document_type"] == ContentTypeEnum.IMAGE
        if not image_mask.any():
            return df[~image_mask]

        df_image = df.loc[image_mask].copy()

        avg_size = df_image["metadata"].apply(calculate_average_image_size)
        avg_size_mask = avg_size > min_size

        aspect_ratio = df_image["metadata"].apply(calculate_aspect_ratio)
        min_aspect_ratio_mask = aspect_ratio > min_aspect_ratio
        max_aspect_ratio_mask = aspect_ratio < max_aspect_ratio

        image_filter_mask = ~(avg_size_mask & min_aspect_ratio_mask & max_aspect_ratio_mask)
        filter_bool = image_filter_mask.any()

        if filter_bool:
            filtered_df = df_image.loc[image_filter_mask].copy()

            if filter_images:
                df.drop(labels=filtered_df.index, inplace=True)

                return df

            info_msg = {
                "task": TaskTypeEnum.FILTER.value,
                "status": StatusEnum.SUCCESS.value,
                "message": "Filtered due to image size.",
                "filter": True,
            }

            validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()

            filtered_df["info_message_metadata"] = [validated_info_msg] * filtered_df.shape[0]
            filtered_df["metadata"] = filtered_df["metadata"].apply(add_info_message, args=(info_msg,))

            df.loc[filtered_df.index, "metadata"] = filtered_df["metadata"]
            df.loc[filtered_df.index, "document_type"] = ContentTypeEnum.INFO_MSG

        return df

    except Exception as e:
        err_msg = f"_cpu_only_apply_filter: Error applying deduplication filter. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e


def image_filter_stage(df, task_props, validated_config) -> pd.DataFrame:
    try:
        if isinstance(task_props, BaseModel):
            task_props = task_props.model_dump()

        task_props.get("content_type")
        task_params = task_props.get("params", {})
        filter_flag = task_params.get("filter", True)

        logger.debug(f"Filtering images by scale with filter_flag={filter_flag}")

        df_result = _cpu_only_apply_filter(df, task_params)

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
