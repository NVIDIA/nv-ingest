# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Optional, List, Any

import pandas as pd

from nv_ingest_api.internal.enums.common import TaskTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import (
    ContentTypeEnum,
    InfoMessageMetadataSchema,
    StatusEnum,
)
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def _add_info_message(x, info_msg):
    x["info_message_metadata"] = info_msg

    return x


def _calculate_average_image_size(x):
    return (x["image_metadata"]["width"] + x["image_metadata"]["height"]) / 2


def _calculate_aspect_ratio(x):
    return x["image_metadata"]["width"] / max(x["image_metadata"]["height"], 1e-9)


def filter_images_internal(
    df_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    mutate_config: ImageFilterSchema = ImageFilterSchema(),
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Apply an image filtering operation to a DataFrame based on average image size and aspect ratio.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame to be filtered. Must contain 'document_type' and 'metadata' columns.
    task_config : dict
        Dictionary with the following keys:
            - "min_size": Minimum average image size threshold.
            - "max_aspect_ratio": Maximum allowed aspect ratio.
            - "min_aspect_ratio": Minimum allowed aspect ratio.
            - "filter": If True, rows failing the criteria are dropped; if False, they are flagged.
    mutate_config : ImageFilterSchema
    execution_trace_log : Optional[List[Any]], optional

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after applying the image filter.

    Raises
    ------
    ValueError
        If required columns are missing or if parameters are invalid.
    Exception
        For other errors encountered during filtering.
    """

    _ = mutate_config  # Unused variable
    _ = execution_trace_log  # TODO(Devin)

    try:
        required_columns = {"document_type", "metadata"}
        if not required_columns.issubset(df_ledger.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        min_size = task_config.get("min_size")
        max_aspect_ratio = task_config.get("max_aspect_ratio")
        min_aspect_ratio = task_config.get("min_aspect_ratio")
        filter_flag = task_config.get("filter", True)

        if not isinstance(min_size, (int, float)) or min_size < 0:
            raise ValueError("min_size must be a non-negative number")
        if not isinstance(max_aspect_ratio, (int, float)) or max_aspect_ratio <= 0:
            raise ValueError("max_aspect_ratio must be a positive number")
        if not isinstance(min_aspect_ratio, (int, float)) or min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio must be a positive number")
        if min_aspect_ratio > max_aspect_ratio:
            raise ValueError("min_aspect_ratio cannot be greater than max_aspect_ratio")

        image_mask = df_ledger["document_type"] == ContentTypeEnum.IMAGE
        if not image_mask.any():
            return df_ledger.copy()

        df_image = df_ledger.loc[image_mask].copy()
        avg_size = df_image["metadata"].apply(_calculate_average_image_size)
        avg_size_mask = avg_size > min_size

        aspect_ratio = df_image["metadata"].apply(_calculate_aspect_ratio)
        min_aspect_ratio_mask = aspect_ratio > min_aspect_ratio
        max_aspect_ratio_mask = aspect_ratio < max_aspect_ratio

        valid_mask = avg_size_mask & min_aspect_ratio_mask & max_aspect_ratio_mask
        image_filter_mask = ~valid_mask

        if image_filter_mask.any():
            filtered_df = df_image.loc[image_filter_mask].copy()
            if filter_flag:
                df_ledger.drop(labels=filtered_df.index, inplace=True)
                return df_ledger

            info_msg = {
                "task": TaskTypeEnum.FILTER.value,
                "status": StatusEnum.SUCCESS.value,
                "message": "Filtered due to image size or aspect ratio.",
                "filter": True,
            }
            validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()
            filtered_df["info_message_metadata"] = [validated_info_msg] * filtered_df.shape[0]
            filtered_df["metadata"] = filtered_df["metadata"].apply(_add_info_message, args=(info_msg,))
            df_ledger.loc[filtered_df.index, "metadata"] = filtered_df["metadata"]
            df_ledger.loc[filtered_df.index, "document_type"] = ContentTypeEnum.INFO_MSG

        result, execution_trace_log = df_ledger, {}

        return result

    except Exception as e:
        err_msg = f"filter_images_internal: Error applying image filter. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
