# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import hashlib
from typing import Any, Dict, Optional, List

import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema

logger = logging.getLogger(__name__)


def _hash_content(x: Any, algorithm: str = "md5") -> bytes:
    """
    Compute a hash of the content using the specified algorithm.

    Parameters
    ----------
    x : dict
        A dictionary containing the content under the key "content".
    algorithm : str, optional
        Hashing algorithm to use (default "md5").

    Returns
    -------
    bytes
        The computed hash.
    """
    try:
        return hashlib.new(algorithm, x["content"].encode()).digest()
    except Exception as e:
        msg = f"hash_content: Error computing hash: {e}"
        logger.error(msg, exc_info=True)
        raise type(e)(msg) from e


def deduplicate_images_internal(
    df_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    mutate_config: ImageDedupSchema = ImageDedupSchema(),
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Deduplicate images in a DataFrame based on content hashes.

    The function processes rows where the 'document_type' is IMAGE, computes a content hash for each,
    and then either removes duplicates or marks them based on the 'filter' flag in task_config.
    A 'hash_algorithm' flag in task_config determines the algorithm used for hashing.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing at least 'document_type' and 'metadata' columns.
    task_config : dict
        Configuration parameters, including:
            - "filter": bool, if True duplicate rows are removed; if False, duplicates are marked.
            - "hash_algorithm": str, the algorithm to use for hashing (default "md5").
    mutate_config : ImageDedupSchema, optional
    execution_trace_log : Optional[List[Any]], optional

    Returns
    -------
    pd.DataFrame
        The DataFrame with duplicate images either removed or marked.

    Raises
    ------
    ValueError
        If the required columns are missing.
    Exception
        For any other errors encountered during deduplication.
    """

    _ = mutate_config  # Unused variable
    _ = execution_trace_log  # TODO(Devin): Implement trace logging

    try:
        # Verify required columns exist.
        for col in ("document_type", "metadata"):
            if col not in df_ledger.columns:
                raise ValueError(f"Missing required column '{col}'.")

        # Select image rows.
        image_mask = df_ledger["document_type"] == ContentTypeEnum.IMAGE
        if not image_mask.any():
            return df_ledger[~image_mask]

        df_images = df_ledger.loc[image_mask].copy()
        hash_algorithm = task_config.get("hash_algorithm", "md5")

        # Compute content hash for each image.
        df_images["_image_content_hash"] = df_images["metadata"].apply(_hash_content, args=(hash_algorithm,))
        df_images_deduped = df_images.drop_duplicates(subset="_image_content_hash")
        deduped_indices = df_images_deduped.index

        non_image_rows = df_ledger.loc[~image_mask]
        deduped_images = df_images.loc[deduped_indices][df_ledger.columns.difference(["_image_content_hash"])]

        result, execution_trace_log = pd.concat([deduped_images, non_image_rows], axis=0), {}
        _ = execution_trace_log

        return result
    except Exception as e:
        msg = f"deduplicate_images_internal: Error applying deduplication filter: {e}"
        logger.error(msg, exc_info=True)
        raise type(e)(msg) from e
