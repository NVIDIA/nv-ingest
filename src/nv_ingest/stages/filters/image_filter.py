# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from functools import partial
from typing import Any
from typing import Dict

import mrc
import mrc.core.operators as ops
import pandas as pd
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "filter_images"
MODULE_NAMESPACE = "nv-ingest"
ImageFilterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageFilterSchema)


def add_info_message(x, info_msg):
    x["info_message_metadata"] = info_msg

    return x


def calculate_average_image_size(x):
    return (x["image_metadata"]["width"] + x["image_metadata"]["height"]) / 2


def calculate_aspect_ratio(x):
    return x["image_metadata"]["width"] / max(x["image_metadata"]["height"], 1e-9)


def _cpu_only_apply_filter(df: pd.DataFrame, task_params: dict):
    min_size = task_params.get("min_size")
    max_aspect_ratio = task_params.get("max_aspect_ratio")
    min_aspect_ratio = task_params.get("min_aspect_ratio")
    filter_images = task_params.get("filter", False)

    # return if no images
    image_mask = df["document_type"] == ContentTypeEnum.IMAGE
    if not image_mask.any():
        return df[~image_mask]

    df_image = df.loc[image_mask]
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
            "task": TaskTypeEnum.FILTER,
            "status": StatusEnum.SUCCESS,
            "message": "Filtered due to image size.",
            "filter": True,
        }

        validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()

        filtered_df["info_message_metadata"] = [validated_info_msg] * filtered_df.shape[0]
        filtered_df["metadata"] = filtered_df["metadata"].apply(add_info_message, args=(info_msg,))

        df.loc[filtered_df.index, "metadata"] = filtered_df["metadata"]
        df.loc[filtered_df.index, "document_type"] = ContentTypeEnum.INFO_MSG

    return df


def _apply_filter(ctrl_msg: ControlMessage, task_params: dict):
    min_size = task_params.get("min_size")
    max_aspect_ratio = task_params.get("max_aspect_ratio")
    min_aspect_ratio = task_params.get("min_aspect_ratio")
    filter_flag = task_params.get("filter", False)

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # return if no images
        image_mask = mdf["document_type"] == ContentTypeEnum.IMAGE.value
        if not image_mask.any():
            return

        # detect undesirable images
        base_cols = mdf.columns
        gdf_image = mdf.loc[image_mask]

        img_width = gdf_image["metadata"].struct.field("image_metadata").struct.field("width")

        img_height = gdf_image["metadata"].struct.field("image_metadata").struct.field("height")

        avg_size = (img_width + img_height) / 2
        aspect_ratio = (img_width / img_height).fillna(0)

        image_filter_mask = ~(
            (avg_size > min_size) & (aspect_ratio < max_aspect_ratio) & (aspect_ratio > min_aspect_ratio)
        )

        if image_filter_mask.any():
            # if we want do immediately remove undesireable images from payload
            if filter_flag:
                # Slow first time, jitify is performs a one-time only warm-up to populate the persistent cache.
                result_gdf = mdf[base_cols].drop(labels=gdf_image.loc[image_filter_mask].index, inplace=False)
                # Strange segfault if we don't do this...
                result_gdf = cudf.from_pandas(result_gdf.to_pandas())
                message_meta = MessageMeta(df=result_gdf)
                ctrl_msg.payload(message_meta)
                return

            # explode to extract individual metadata structs
            mdf_temp = mdf["metadata"].struct.explode()
            exploded_metadata_cols = list(mdf_temp.columns)
            mdf[exploded_metadata_cols] = mdf_temp
            filtered_images_gdf = gdf_image.loc[image_filter_mask]

            # define and validate `info_message_metadata`
            info_msg = {
                "task": TaskTypeEnum.FILTER.value,
                "status": StatusEnum.SUCCESS.value,
                "message": "Filtered due to image size.",
                "filter": True,
            }

            validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()

            # update payload with `info_message_metadata` and `document_type`
            filtered_images_gdf["info_message_metadata"] = [validated_info_msg] * filtered_images_gdf.shape[0]
            mdf.drop(labels=["info_message_metadata", "metadata"], inplace=True, axis=1)
            mdf["info_message_metadata"] = filtered_images_gdf["info_message_metadata"]
            mdf.loc[filtered_images_gdf["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG.value
            mdf["metadata"] = mdf[exploded_metadata_cols + ["info_message_metadata"]].to_struct()
            mdf.drop(labels=mdf.columns.difference(base_cols), inplace=True, axis=1)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _filter_images(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, ImageFilterSchema)

    @filter_by_task(["filter"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def filter_images_fn(ctrl_msg: ControlMessage):
        task_props = ctrl_msg.remove_task("filter")
        content_type = task_props.get("content_type")
        task_params = task_props.get("params", {})
        filter_flag = task_params.get("filter", True)

        logger.debug(f"Filtering images by scale with filter_flag={filter_flag}")

        if content_type != ContentTypeEnum.IMAGE:
            return ctrl_msg

        if validated_config.cpu_only:
            with ctrl_msg.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            df_result = _cpu_only_apply_filter(df, task_params)

            if not df_result.empty:
                gdf = cudf.from_pandas(df_result)
                msg_meta = MessageMeta(df=gdf)
                ctrl_msg.payload(msg_meta)

        else:
            _apply_filter(ctrl_msg, task_params)

        return ctrl_msg

    # Create a node for filtering incoming images
    input_node = builder.make_node(
        "image_filter",
        ops.map(filter_images_fn),
    )

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)


def image_filter_stage(df, task_props, validated_config) -> pd.DataFrame:
    task_props.get("content_type")
    task_params = task_props.get("params", {})
    filter_flag = task_params.get("filter", True)

    logger.debug(f"Filtering images by scale with filter_flag={filter_flag}")

    df_result = _cpu_only_apply_filter(df, task_params)

    return df_result


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
