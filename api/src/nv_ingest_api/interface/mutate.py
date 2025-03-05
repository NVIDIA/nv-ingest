# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union, Dict

import pandas as pd

from nv_ingest_api.internal.mutate.deduplicate import deduplicate_images_internal
from nv_ingest_api.internal.mutate.filter import filter_images_internal
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


@unified_exception_handler
def filter_images(
    *,
    df_ledger: pd.DataFrame,
    min_size: int = 128,
    max_aspect_ratio: Union[float, int] = 5.0,
    min_aspect_ratio: Union[float, int] = 2.0,
) -> pd.DataFrame:
    """
    Apply an image filter to the ledger DataFrame based on size and aspect ratio criteria.

    This function builds a set of task parameters and then delegates the filtering work to
    `filter_images_internal`. If an exception occurs during filtering, the error is logged
    and re-raised with additional context.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing image metadata. It must include the columns 'document_type' and 'metadata'.
    min_size : int, optional
        Minimum average image size threshold. Images with an average size less than or equal to this
        value are considered for filtering. Default is 128.
    max_aspect_ratio : float or int, optional
        Maximum allowed image aspect ratio. Images with an aspect ratio greater than or equal to this value
        are considered for filtering. Default is 5.0.
    min_aspect_ratio : float or int, optional
        Minimum allowed image aspect ratio. Images with an aspect ratio less than or equal to this value
        are considered for filtering. Default is 2.0.
    execution_trace_log : Optional[List[Any]], optional

    Returns
    -------
    pd.DataFrame
        The DataFrame after applying the image filter.

    Raises
    ------
    Exception
        If an error occurs during the filtering process.
    """

    try:
        task_params: Dict[str, Union[int, float, bool]] = {
            "min_size": min_size,
            "max_aspect_ratio": max_aspect_ratio,
            "min_aspect_ratio": min_aspect_ratio,
            "filter": True,
        }
        mutate_config = ImageFilterSchema()

        result, _ = filter_images_internal(
            df_ledger, task_params, mutate_config=mutate_config, execution_trace_log=None
        )

        return result
    except Exception as e:
        err_msg = f"filter_images: Error applying deduplication filter. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


@unified_exception_handler
def deduplicate_images(
    *,
    df_ledger: pd.DataFrame,
    hash_algorithm: str = "md5",
) -> pd.DataFrame:
    """
    Deduplicate images in the DataFrame based on content hashes.

    This function builds a task configuration with the specified hashing algorithm and
    delegates processing to `deduplicate_images_internal`.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing image metadata. It must include at least the columns
        'document_type' and 'metadata'.
    hash_algorithm : str, optional
        Hashing algorithm to use for deduplication. Valid algorithms are those supported by
        Python's hashlib.new() function (e.g., "md5", "sha1", "sha256"). Default is "md5".
    execution_trace_log : Optional[List[Any]], optional
        Optional list for execution trace logging (currently unused).

    Returns
    -------
    pd.DataFrame
        The deduplicated DataFrame with duplicate images removed.

    Raises
    ------
    Exception
        Propagates any exceptions raised during the deduplication process.
    """

    task_config: Dict[str, Union[int, float, bool, str]] = {
        "hash_algorithm": hash_algorithm,
    }
    mutate_config = ImageDedupSchema()

    result, _ = deduplicate_images_internal(
        df_ledger=df_ledger,
        task_config=task_config,
        mutate_config=mutate_config,
        execution_trace_log=None,
    )

    return result
