# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import traceback

import mrc
import pandas as pd
import sklearn.neighbors
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

import cudf

from nv_ingest.schemas.associate_nearby_text_schema import AssociateNearbyTextSchema
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "associate_nearby_text"
MODULE_NAMESPACE = "nv_ingest"

AssociateNearbyTextLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, AssociateNearbyTextSchema)


def _get_center(bbox: tuple) -> float:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _is_nearby_text(row):
    if row.get("text_metadata") is not None:
        return row["text_metadata"].get("text_type") == TextTypeEnum.NEARBY_BLOCK

    return False


def _get_bbox(row):
    if row.get("text_metadata") is not None:
        return row["text_metadata"]["text_location"]
    elif row.get("image_metadata") is not None:
        return row["image_metadata"]["image_location"]
    else:
        return None


def _associate_nearby_text_blocks(df: pd.DataFrame, n_neighbors):
    # convert pandas dataframe to image list and text list

    metadata_sr = df["metadata"].apply(lambda x: json.loads(x))
    metadata_dict = metadata_sr.to_dict()

    # only consider pages w/ images
    metadata_df = pd.DataFrame()
    metadata_df["metadata"] = metadata_sr
    metadata_df["page"] = metadata_sr.apply(lambda x: x["content_metadata"]["hierarchy"]["page"])
    metadata_df["is_image"] = metadata_sr.apply(lambda x: x.get("image_metadata") is not None)
    metadata_df["is_nearby_text"] = metadata_sr.apply(_is_nearby_text)

    # filter to only possible data
    pages_with_images = metadata_df.loc[metadata_df["is_image"]]["page"].unique()
    filtered_df = metadata_df.loc[metadata_df["page"].isin(pages_with_images)]

    if filtered_df.empty:
        return df

    filtered_df["bbox"] = filtered_df["metadata"].apply(_get_bbox)
    filtered_df[["bbox_center_x", "bbox_center_y"]] = filtered_df["bbox"].apply(_get_center).tolist()

    for page in pages_with_images:
        page_df = filtered_df.loc[filtered_df["page"] == page]
        page_nearest_text_block_df = page_df.loc[page_df["is_nearby_text"] == True]  # noqa: E712
        page_nearest_text_block_centers_df = page_nearest_text_block_df[["bbox_center_x", "bbox_center_y"]]

        if page_nearest_text_block_centers_df.empty:
            continue

        page_image_df = page_df.loc[page_df["is_image"] == True]  # noqa: E712
        page_image_centers_df = page_image_df[["bbox_center_x", "bbox_center_y"]]

        knn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=min(page_nearest_text_block_centers_df.shape[0], 5))

        knn_model.fit(page_nearest_text_block_centers_df)

        _, indices_stack = knn_model.kneighbors(
            page_image_centers_df[["bbox_center_x", "bbox_center_y"]],
            n_neighbors=min(page_nearest_text_block_centers_df.shape[0], 5),
        )

        # image_idx (row) closest text blocks indices (cols)
        img_indices = page_image_centers_df.index
        text_block_indices = page_nearest_text_block_centers_df.index

        for row_idx in range(indices_stack.shape[0]):
            for col_idx in range(indices_stack.shape[1]):
                metadata_dict[img_indices[row_idx]]["content_metadata"]["hierarchy"]["nearby_objects"]["text"][
                    "content"
                ].append(metadata_dict[text_block_indices[indices_stack[row_idx, col_idx]]]["content"])

                metadata_dict[img_indices[row_idx]]["content_metadata"]["hierarchy"]["nearby_objects"]["text"][
                    "bbox"
                ].append(
                    metadata_dict[text_block_indices[indices_stack[row_idx, col_idx]]]["text_metadata"]["text_location"]
                )

            metadata_dict[img_indices[row_idx]] = validate_metadata(metadata_dict[img_indices[row_idx]]).dict()

        df["metadata"] = metadata_dict

    return df


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _associate_nearby_text(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """

    validated_config = fetch_and_validate_module_config(builder, AssociateNearbyTextSchema)

    @filter_by_task(["caption"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def associate_nearby_text_fn(message: ControlMessage):
        try:
            task_props = message.remove_task("caption")

            # Validate that all 'content' values are not None
            with message.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            n_neighbors = task_props.get("n_neighbors", validated_config.n_neighbors)

            logger.info(f"Associating text blocks with images with neighbors: {n_neighbors}")

            result_df = _associate_nearby_text_blocks(df, n_neighbors)

            # Work around until https://github.com/apache/arrow/pull/40412 is resolved
            result_gdf = cudf.from_pandas(result_df)

            message_meta = MessageMeta(df=result_gdf)
            message.payload(message_meta)

            # adding another caption task for inference
            task_props = message.add_task("caption", {"n_neighbors": n_neighbors})

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to associate text with images: {e}")

    association_node = builder.make_node("associate_nearby_text", ops.map(associate_nearby_text_fn))

    # Register the input and output of the module
    builder.register_module_input("input", association_node)
    builder.register_module_output("output", association_node)
