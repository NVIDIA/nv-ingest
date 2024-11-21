# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
import traceback
from io import BytesIO
from typing import Any
from typing import Dict
from urllib.parse import quote

import mrc
import mrc.core.operators as ops
import pandas as pd
from minio import Minio
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas.image_storage_schema import ImageStorageModuleSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "image_storage"
MODULE_NAMESPACE = "nv_ingest"

# TODO: Move these into pipeline.py to populate the stage and validate them using the pydantic schema on startup.
_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_READ_ADDRESS = os.environ.get("MINIO_PUBLIC_ADDRESS", "http://minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")

ImageStorageLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageStorageModuleSchema)


def upload_images(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identify contents (e.g., images) within a dataframe and uploads the data to MinIO.
    The image metadata in the metadata column is updated with the URL of the uploaded data.
    """
    content_types = params.get("content_types")
    endpoint = params.get("endpoint", _DEFAULT_ENDPOINT)
    bucket_name = params.get("bucket_name", _DEFAULT_BUCKET_NAME)

    client = Minio(
        endpoint,
        access_key=params.get("access_key", None),
        secret_key=params.get("secret_key", None),
        session_token=params.get("session_token", None),
        secure=params.get("secure", False),
        region=params.get("region", None),
    )

    bucket_found = client.bucket_exists(bucket_name)
    if not bucket_found:
        client.make_bucket(bucket_name)
        logger.debug("Created bucket %s", bucket_name)
    else:
        logger.debug("Bucket %s already exists", bucket_name)

    for idx, row in df.iterrows():
        if row["document_type"] not in content_types.keys():
            continue

        metadata = row["metadata"].copy()

        content = base64.b64decode(metadata["content"].encode())

        source_id = metadata["source_metadata"]["source_id"]

        image_type = "png"
        if row["document_type"] == ContentTypeEnum.IMAGE:
            image_type = metadata.get("image_metadata").get("image_type", "png")

        # URL-encode source_id and image_type to ensure they are safe for the URL path
        encoded_source_id = quote(source_id, safe="")
        encoded_image_type = quote(image_type, safe="")

        destination_file = f"{encoded_source_id}/{idx}.{encoded_image_type}"

        source_file = BytesIO(content)
        client.put_object(
            bucket_name,
            destination_file,
            source_file,
            length=len(content),
        )

        metadata["source_metadata"]["source_location"] = f"{_DEFAULT_READ_ADDRESS}/{bucket_name}/{destination_file}"
        if row["document_type"] == ContentTypeEnum.IMAGE:
            logger.debug("Storing image data to Minio")
            metadata["image_metadata"][
                "uploaded_image_url"
            ] = f"{_DEFAULT_READ_ADDRESS}/{bucket_name}/{destination_file}"
        elif row["document_type"] == ContentTypeEnum.STRUCTURED:
            logger.debug("Storing structured image data to Minio")
            metadata["table_metadata"][
                "uploaded_image_url"
            ] = f"{_DEFAULT_READ_ADDRESS}/{bucket_name}/{destination_file}"

        # TODO: validate metadata before putting it back in.
        df.at[idx, "metadata"] = metadata

    return df


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _storage_images(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, ImageStorageModuleSchema)

    @filter_by_task(["store"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: ControlMessage):
        try:
            task_props = ctrl_msg.remove_task("store")
            store_structured = task_props.get("structured", True)
            store_images = task_props.get("images", False)

            content_types = {}
            if store_structured:
                content_types[ContentTypeEnum.STRUCTURED] = store_structured
            if store_images:
                content_types[ContentTypeEnum.IMAGE] = store_images

            params = task_props.get("params", {})

            params["content_types"] = content_types

            # TODO(Matt) validate this resolves to the right filter criteria....
            logger.debug(f"Processing storage task with parameters: {params}")

            with ctrl_msg.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            storage_obj_mask = df["document_type"].isin(list(content_types.keys()))
            if (~storage_obj_mask).all():  # if there are no images, return immediately.
                logger.debug(f"No storage objects for '{content_types}' found in the dataframe.")
                return ctrl_msg

            df = upload_images(df, params)

            # Update control message with new payload
            gdf = cudf.from_pandas(df)
            msg_meta = MessageMeta(df=gdf)
            ctrl_msg.payload(msg_meta)
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to store extracted objects: {e}")

        return ctrl_msg

    input_node = builder.make_node("image_storage", ops.map(on_data))

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
