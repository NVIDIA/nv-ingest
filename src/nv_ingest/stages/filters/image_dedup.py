# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import logging
from typing import Any
from typing import Dict

import pandas as pd
from morpheus.config import Config
from morpheus.utils.module_utils import ModuleLoaderFactory
from pydantic import BaseModel

import cudf

from nv_ingest.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage

logger = logging.getLogger(__name__)

MODULE_NAME = "dedup_images"
MODULE_NAMESPACE = "nv-ingest"
ImageDedupLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageDedupSchema)


def hash_content(x: Any, algorithm: str = "md5") -> bytes:
    """
    Computes a hash of the content using the specified algorithm.

    This function takes a dictionary containing content, encodes the content as bytes,
    and then computes the hash using the specified algorithm. The default algorithm is `md5`.

    Parameters
    ----------
    x : dict
        A dictionary containing the content to be hashed. The content is expected to be a string under the key
        `"content"`.
    algorithm : str, optional
        The hashing algorithm to use, by default "md5". Other valid algorithms can be provided,
        but the function currently always uses `md5`.

    Returns
    -------
    bytes
        The hash of the content as a byte sequence.

    Examples
    --------
    >>> x = {"content": "example content"}
    >>> hash_content(x)
    b'\x9a\x03\x8b\xad\x8cR\x0ea\xcd\r\x1e\xd6;+\x9c\xe0'

    Notes
    -----
    This function currently supports only the `md5` algorithm.
    """
    try:
        return hashlib.md5(x["content"].encode()).digest()
    except Exception as e:
        err_msg = f"hash_content: Error computing hash for content. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def _cpu_only_apply_dedup_filter(df: pd.DataFrame, filter_flag: bool) -> pd.DataFrame:
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
    >>> result_df = _cpu_only_apply_dedup_filter(df, filter_flag=True)
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
        for col in ["document_type", "metadata"]:
            if col not in df.columns:
                raise ValueError(f"_cpu_only_apply_dedup_filter: Missing required column '{col}'.")
        image_mask = df["document_type"] == ContentTypeEnum.IMAGE
        if not image_mask.any():
            return df[~image_mask]
        base_cols = df.columns
        df_images = df.loc[image_mask].copy()
        content_hash_sr = df_images["metadata"].apply(hash_content, args=("md5",))
        df_images.loc[content_hash_sr.index, "_image_content_hash"] = content_hash_sr
        df_images_deduped = df_images.drop_duplicates(subset="_image_content_hash")
        deduped_indices = df_images_deduped.index
        duplicate_indices = df_images.loc[~df_images.index.isin(deduped_indices)].index
        if filter_flag:
            df_result = pd.concat(
                [
                    df_images.loc[deduped_indices][df.columns.difference(["_image_content_hash"])],
                    df.loc[~image_mask],
                ],
                axis=0,
            )
            return df_result
        duplicate_images_df = df_images.loc[duplicate_indices]
        info_msg = {
            "task": TaskTypeEnum.FILTER.value,
            "status": StatusEnum.SUCCESS.value,
            "message": "Filtered duplicate image.",
            "filter": True,
        }
        validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()
        duplicate_images_df["info_message_metadata"] = [validated_info_msg] * duplicate_images_df.shape[0]
        df.loc[duplicate_images_df["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG
        df.drop(labels=df.columns.difference(base_cols), inplace=True, axis=1)
        return df
    except Exception as e:
        err_msg = f"_cpu_only_apply_dedup_filter: Error applying deduplication filter. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def _apply_dedup_filter(ctrl_msg: IngestControlMessage, filter_flag: bool) -> None:
    """
    Applies a deduplication filter to images within a DataFrame encapsulated in a IngestControlMessage.

    This function identifies duplicate images based on content hashes within a DataFrame,
    and either filters out the duplicates or marks them as informational messages depending on the `filter_flag`.

    Parameters
    ----------
    ctrl_msg : IngestControlMessage
        The control message containing the payload with the DataFrame to be filtered.
    filter_flag : bool
        A flag indicating whether to filter out duplicates (`True`) or mark them with informational messages (`False`).

    Returns
    -------
    None
        The function modifies the `ctrl_msg` in place, updating the payload with the filtered or marked DataFrame.

    Notes
    -----
    - The function operates only on rows where `document_type` is `ContentTypeEnum.IMAGE.value`.
    - When `filter_flag` is `True`, duplicates are removed from the DataFrame.
    - When `filter_flag` is `False`, duplicate images are marked with an informational message and the `document_type`
      is updated to `ContentTypeEnum.INFO_MSG.value`.
    - The `metadata` field in the DataFrame is exploded and restructured as needed.

    Examples
    --------
    >>> ctrl_msg = IngestControlMessage(payload=some_dataframe)
    >>> _apply_dedup_filter(ctrl_msg, filter_flag=True)
    >>> filtered_df = ctrl_msg.payload().dataframe()
    >>> print(filtered_df)

    Raises
    ------
    ValueError
        If the DataFrame does not contain the necessary columns `document_type` and `metadata`,
        or if other expected operations fail.
    """
    try:
        with ctrl_msg.payload().mutable_dataframe() as mdf:
            image_mask = mdf["document_type"] == ContentTypeEnum.IMAGE.value
            if not image_mask.any():
                return
            gdf = mdf.copy()
        base_cols = gdf.columns
        gdf_images = gdf.loc[image_mask]
        content_sr = gdf_images["metadata"].struct.field("content")
        content_hash_sr = content_sr.hash_values(method="md5", seed=None)
        gdf_images.loc[content_hash_sr.index, "_image_content_hash"] = content_hash_sr
        gdf_images_deduped = gdf_images.drop_duplicates(subset="_image_content_hash")
        deduped_indices = gdf_images_deduped.index
        duplicate_indices = gdf_images.loc[~gdf_images.index.isin(deduped_indices)].index
        if filter_flag:
            gdf_result = cudf.concat(
                [
                    gdf_images.loc[deduped_indices][gdf.columns.difference(["_image_content_hash"])],
                    gdf.loc[~image_mask],
                ],
                axis=0,
            )
            ctrl_msg.payload(gdf_result.to_pandas())
            return
        gdf_temp = gdf["metadata"].struct.explode()
        exploded_metadata_cols = list(gdf_temp.columns)
        gdf[exploded_metadata_cols] = gdf_temp
        duplicate_images_gdf = gdf_images.loc[duplicate_indices]
        info_msg = {
            "task": TaskTypeEnum.FILTER.value,
            "status": StatusEnum.SUCCESS.value,
            "message": "Filtered duplicate image.",
            "filter": True,
        }
        validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()
        duplicate_images_gdf["info_message_metadata"] = [validated_info_msg] * duplicate_images_gdf.shape[0]
        gdf.drop(labels=["info_message_metadata", "metadata"], inplace=True, axis=1)
        gdf["info_message_metadata"] = duplicate_images_gdf["info_message_metadata"]
        gdf.loc[duplicate_images_gdf["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG.value
        gdf["metadata"] = gdf[exploded_metadata_cols + ["info_message_metadata"]].to_struct()
        gdf.drop(labels=gdf.columns.difference(base_cols), inplace=True, axis=1)
        ctrl_msg.payload(gdf.to_pandas())

        return

    except Exception as e:
        err_msg = f"_apply_dedup_filter: Error applying deduplication filter to control message. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e


def dedup_image_stage(df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any) -> pd.DataFrame:
    """
    Deduplicates images in the provided DataFrame based on the task properties.

    This function processes a DataFrame containing images and applies a deduplication filter
    based on the `filter` parameter within the task properties. The deduplication is performed
    by identifying and removing duplicate images, or by marking them with informational messages,
    depending on the value of the `filter_flag`.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be deduplicated. It must have columns that include
        image metadata and document types.
    task_props : dict of {str: Any}
        A dictionary containing task properties, which may include the content type and parameters for filtering.
    validated_config : Any
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
    >>> result_df = dedup_image_stage(df, task_props, validated_config)
    >>> print(result_df)

    Raises
    ------
    Exception
        If deduplication processing fails.
    """
    try:
        if isinstance(task_props, BaseModel):
            task_props = task_props.model_dump()

        task_props.get("content_type")  # Preserve any side effects.
        task_params = task_props.get("params", {})
        filter_flag = task_params.get("filter", True)

        logger.debug(f"dedup_image_stage: De-duplicating images with filter_flag={filter_flag}")

        df_result = _cpu_only_apply_dedup_filter(df, filter_flag)

        return df_result

    except Exception as e:
        err_msg = f"dedup_image_stage: Error during deduplication. Original error: {e}"
        logger.error(err_msg, exc_info=True)

        raise type(e)(err_msg) from e


def generate_dedup_stage(
    c: Config,
    dedup_config: Dict[str, Any],
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
    dedup_config : dict of {str: Any}
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
    >>> stage = generate_dedup_stage(c, dedup_config)
    >>> stage.run()

    Raises
    ------
    Exception
        If an error occurs during stage generation.
    """
    try:
        validated_config = ImageDedupSchema(**dedup_config)
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
