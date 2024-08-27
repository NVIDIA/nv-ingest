# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging

import mrc
import mrc.core.operators as ops
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.modules.filters.image_filter import add_info_message
from nv_ingest.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "dedup_images"
MODULE_NAMESPACE = "nv-ingest"
ImageDedupLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageDedupSchema)


def hash_content(x, algorithm="md5"):
    return hashlib.md5(x["content"].encode()).digest()


def _cpu_only_apply_dedup_filter(df: pd.DataFrame, filter_flag: bool):
    # return if no images
    image_mask = df["document_type"] == ContentTypeEnum.IMAGE
    if not image_mask.any():
        return df[image_mask]

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

    # define and validate `info_message_metadata`
    info_msg = {
        "task": TaskTypeEnum.FILTER.value,
        "status": StatusEnum.SUCCESS.value,
        "message": "Filtered duplicate image.",
        "filter": True,
    }

    # update payload with `info_message_metadata` and `document_type`
    validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()

    duplicate_images_df["info_message_metadata"] = [validated_info_msg] * duplicate_images_df.shape[0]
    duplicate_images_df["metadata"] = duplicate_images_df["metadata"].apply(add_info_message, args=(info_msg,))

    df.loc[duplicate_images_df["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG
    df.drop(labels=df.columns.difference(base_cols), inplace=True, axis=1)

    return df


def _apply_dedup_filter(ctrl_msg: ControlMessage, filter_flag):
    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # return if no images
        image_mask = mdf["document_type"] == ContentTypeEnum.IMAGE.value
        if not image_mask.any():
            return

        gdf = mdf.copy()  # noqa

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

        message_meta = MessageMeta(df=gdf_result)
        ctrl_msg.payload(message_meta)

        return

    # explode to extract individual metadata structs
    gdf_temp = gdf["metadata"].struct.explode()
    exploded_metadata_cols = list(gdf_temp.columns)
    gdf[exploded_metadata_cols] = gdf_temp
    duplicate_images_gdf = gdf_images.loc[duplicate_indices]

    # define and validate `info_message_metadata`
    info_msg = {
        "task": TaskTypeEnum.FILTER.value,
        "status": StatusEnum.SUCCESS.value,
        "message": "Filtered duplicate image.",
        "filter": True,
    }

    # update payload with `info_message_metadata` and `document_type`
    validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()
    duplicate_images_gdf["info_message_metadata"] = [validated_info_msg] * duplicate_images_gdf.shape[0]
    gdf.drop(labels=["info_message_metadata", "metadata"], inplace=True, axis=1)
    gdf["info_message_metadata"] = duplicate_images_gdf["info_message_metadata"]
    gdf.loc[duplicate_images_gdf["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG.value
    gdf["metadata"] = gdf[exploded_metadata_cols + ["info_message_metadata"]].to_struct()
    gdf.drop(labels=gdf.columns.difference(base_cols), inplace=True, axis=1)

    message_meta = MessageMeta(df=gdf)
    ctrl_msg.payload(message_meta)

    return


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _dedup_images(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, ImageDedupSchema)

    @filter_by_task(["dedup"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def dedup_fn(ctrl_msg: ControlMessage):
        task_props = ctrl_msg.remove_task("dedup")
        content_type = task_props.get("content_type")
        task_params = task_props.get("params", {})
        filter_flag = task_params.get("filter", True)

        logger.info(f"Deduplicating images with filter_flag={filter_flag}")

        if content_type != ContentTypeEnum.IMAGE:
            return ctrl_msg

        if validated_config.cpu_only:
            with ctrl_msg.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()  # noqa

            df_result = _cpu_only_apply_dedup_filter(df, filter_flag)

            if not df_result.empty:
                gdf = cudf.from_pandas(df_result)
                msg_meta = MessageMeta(df=gdf)
                ctrl_msg.payload(msg_meta)

        else:
            _apply_dedup_filter(ctrl_msg, filter_flag)

        return ctrl_msg

    # Create a node for filtering incoming images
    input_node = builder.make_node(
        "image_dedup",
        ops.map(dedup_fn),  # noqa
    )

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
