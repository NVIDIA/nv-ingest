# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
import uuid
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

from nv_ingest.schemas.embedding_storage_schema import EmbeddingStorageModuleSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "embedding_storage"
MODULE_NAMESPACE = "nv_ingest"

# TODO: Move these into pipeline.py to populate the stage and validate them using the pydantic schema on startup.
_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_READ_ADDRESS = os.environ.get("MINIO_PUBLIC_ADDRESS", "http://minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")

EmbeddingStorageLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, EmbeddingStorageModuleSchema)

def upload_embeddings(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identify contents (e.g., images) within a dataframe and uploads the data to MinIO.
    The image metadata in the metadata column is updated with the URL of the uploaded data.
    """
    
    access_key = params.get("access_key", None)
    secret_key = params.get("secret_key", None)

    content_types = params.get("content_types")
    endpoint = params.get("endpoint", _DEFAULT_ENDPOINT)
    bucket_name = params.get("bucket_name", _DEFAULT_BUCKET_NAME)
    bucket_path = params.get("bucket_path", "embeddings")

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
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

    storage_options = {
        "key":access_key,
        "secret": secret_key,
        "client_kwargs": {
        "endpoint_url": _DEFAULT_READ_ADDRESS
        }
    }

    meta = df["metadata"]
    emb_df_list = []  
    file_uuid = uuid.uuid4().hex
    destination_file = f"{bucket_path}/{file_uuid}.parquet"
    write_path = f"s3://{bucket_name}/{destination_file}"
    for idx, row in df.iterrows():
        uu_id = row["uuid"]
        metadata = row["metadata"].copy()

        metadata["source_metadata"]["source_location"] = write_path

        if row["document_type"] == ContentTypeEnum.EMBEDDING:
            logger.debug("Storing embedding data to Minio")
            metadata["embedding_metadata"][
                "uploaded_embedding_url"
            ] = write_path
        # TODO: validate metadata before putting it back in.
        # cm = str(metadata["content_metadata"])
        text = metadata["content"]
        # source_meta = str(metadata["source_metadata"])
        emb = metadata["embedding"]
        df.at[idx, "metadata"] = metadata

        emb_df_list.append([emb, text])

    emb_df = pd.DataFrame(emb_df_list, columns=["vector", "text"])
    emb_df.to_parquet(write_path, engine='pyarrow', storage_options=storage_options)

    return df


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _storage_embeddings(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, EmbeddingStorageModuleSchema)

    @filter_by_task(["store_embedding"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: ControlMessage):
        try:
            task_props = ctrl_msg.remove_task("store_embedding")
            store_embeddings = task_props.get("embedding", True)

            content_types = {}
            if store_embeddings:

                content_types[ContentTypeEnum.EMBEDDING] = store_embeddings

                params = task_props.get("extra_params", {})

                params["content_types"] = content_types

                # TODO(Matt) validate this resolves to the right filter criteria....
                logger.debug(f"Processing storage task with parameters: {params}")

                with ctrl_msg.payload().mutable_dataframe() as mdf:
                    df = mdf.to_pandas()

                df = upload_embeddings(df, params)
            
            # Update control message with new payload
            gdf = cudf.from_pandas(df)
            msg_meta = MessageMeta(df=gdf)
            ctrl_msg.payload(msg_meta)
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to store extracted objects: {e}")

        return ctrl_msg

    input_node = builder.make_node("embedding_storage", ops.map(on_data))

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
