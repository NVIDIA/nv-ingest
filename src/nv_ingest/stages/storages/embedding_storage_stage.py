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
from pymilvus import Collection, connections
from pymilvus.bulk_writer.remote_bulk_writer import RemoteBulkWriter 
from pymilvus.bulk_writer.constants import BulkFileType
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
    collection_name = params.get("collection_name", "nv_ingest_collection")

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        session_token=params.get("session_token", None),
        secure=params.get("secure", False),
        region=params.get("region", None),
    )

    connections.connect(
            address= "milvus:19530",
            uri= "http://milvus:19530",
            host = "milvus",
            port= "19530"
        )
    schema = Collection(collection_name).schema

    bucket_found = client.bucket_exists(bucket_name)
    if not bucket_found:
        client.make_bucket(bucket_name)
        logger.debug("Created bucket %s", bucket_name)
    else:
        logger.debug("Bucket %s already exists", bucket_name)

    conn = RemoteBulkWriter.ConnectParam(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        secure=False
    )

    writer = RemoteBulkWriter(
        schema=schema,
        remote_path=bucket_path,
        connect_param=conn,
        file_type=BulkFileType.PARQUET
    )

    for idx, row in df.iterrows():
        uu_id = row["uuid"]
        metadata = row["metadata"].copy()
        metadata["embedding_metadata"] = {}
        metadata["embedding_metadata"]["uploaded_embedding_url"] = bucket_path
        doc_type = row["document_type"]
        content_replace = doc_type in [ContentTypeEnum.IMAGE, ContentTypeEnum.STRUCTURED]
        location = metadata["source_metadata"]["source_location"]
        content = metadata["content"]
        # TODO: validate metadata before putting it back in.
        if metadata["embedding"] is not None:
            logger.error(f"row type: {doc_type} -  {location} -  {len(content)}")
            df.at[idx, "metadata"] = metadata
            writer.append_row({
                "text":  location if content_replace  else content, 
                "source": metadata["source_metadata"], 
                "content_metadata": metadata["content_metadata"], 
                "vector": metadata["embedding"]}
            )
    
    writer.commit()

    return df


def _store_embeddings(df, task_props, validated_config, trace_info=None):
    try:
        content_types = {}
        content_types[ContentTypeEnum.EMBEDDING] = True

        params = task_props.get("params", {})
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
