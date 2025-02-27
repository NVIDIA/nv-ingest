# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union, Dict, Optional, Any

import pandas as pd

from nv_ingest_api.internal.mutate.filter import filter_images_internal

logger = logging.getLogger(__name__)


def filter_images(
    df_ledger: pd.DataFrame,
    min_size: int = 128,
    max_aspect_ratio: Union[float, int] = 5.0,
    min_aspect_ratio: Union[float, int] = 2.0,
    execution_trace_log: Optional[list[Any]] = None,
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
        return filter_images_internal(df_ledger, task_params, execution_trace_log)
    except Exception as e:
        err_msg = f"filter_images: Error applying deduplication filter. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def deduplicate_images():
    pass
