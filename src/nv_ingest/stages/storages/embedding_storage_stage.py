# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import typing
from typing import Any
from typing import Dict
import functools
import uuid
from minio import Minio

import traceback
import mrc
from morpheus.config import Config

import pandas as pd
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.schemas.embedding_storage_schema import EmbeddingStorageModuleSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")


def upload_embeddings(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identify contents (e.g., images) within a dataframe and uploads the data to MinIO.
    The image metadata in the metadata column is updated with the URL of the uploaded data.
    """
    dimension = params.get("dim", 1024)
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
        if emb is not None:
            df.at[idx, "metadata"] = metadata
            emb_df_list.append([emb, text])
    
    emb_df = pd.DataFrame(emb_df_list, columns=["vector", "text"])
    logger.error(f"exporting: {emb_df}")
    emb_df.to_parquet(write_path, engine='pyarrow', storage_options=storage_options)

    return df


def _store_embeddings(df, task_props, validated_config, trace_info=None):
    logger.error(f"in store embedding")
    try:
        content_types = {}
        content_types[ContentTypeEnum.EMBEDDING] = True

        params = task_props.get("extra_params", {})
        params["content_types"] = content_types

        df = upload_embeddings(df, params)
        
        return df
    except Exception as e:
        traceback.print_exc()
        err_msg = f"Failed to store embeddings: {e}"
        logger.error(err_msg)
        raise

def generate_embedding_storage_stage(
    c: Config,
    task: str = "store_embedding",
    task_desc: str = "Store_embeddings_minio",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform pdf content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    embedding_storage_config : dict
        Configuration parameters for embedding storage.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pdf content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """
    validated_config = EmbeddingStorageModuleSchema()
    _wrapped_process_fn = functools.partial(_store_embeddings, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
