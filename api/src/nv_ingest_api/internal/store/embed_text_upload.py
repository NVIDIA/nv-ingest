# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Union, Optional
from typing import Dict

import pandas as pd
from minio import Minio
from pymilvus import Collection
from pymilvus import connections
from pymilvus.bulk_writer.constants import BulkFileType
from pymilvus.bulk_writer.remote_bulk_writer import RemoteBulkWriter
from pydantic import BaseModel

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")


def _upload_text_embeddings(df_store_ledger: pd.DataFrame, task_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Uploads embeddings to MinIO for contents (e.g., images) contained in a DataFrame.
    The image metadata in the "metadata" column is updated with the URL (or path) of the uploaded data.

    This function performs the following steps:
      1. Initializes a MinIO client using the provided task configuration parameters.
      2. Connects to a Milvus instance and retrieves the collection schema.
      3. Ensures that the target bucket exists (creating it if necessary).
      4. Configures a RemoteBulkWriter to upload embedding data in PARQUET format.
      5. Iterates over each row in the DataFrame, updates the metadata with the bucket path, and appends
         rows to the writer if an embedding is present.
      6. Commits the writer, finalizing the upload process.

    Parameters
    ----------
    df_store_ledger : pd.DataFrame
        DataFrame containing the data to upload. Each row is expected to have:
          - A "metadata" column (a dictionary) that includes keys such as "content", "embedding",
            "source_metadata", and "content_metadata".
          - A "document_type" column indicating the type of document (e.g., IMAGE, STRUCTURED).
    task_config : Dict[str, Any]
        Dictionary of parameters for the upload. Expected keys include:
          - "minio_access_key": Optional[str]
                Access key for MinIO.
          - "minio_secret_key": Optional[str]
                Secret key for MinIO.
          - "minio_endpoint": str, default _DEFAULT_ENDPOINT
                MinIO endpoint URL.
          - "minio_bucket_name": str, default _DEFAULT_BUCKET_NAME
                Name of the bucket in MinIO.
          - "minio_bucket_path": str, default "embeddings"
                Path within the bucket where embeddings are stored.
          - "minio_session_token": Optional[str]
                (Optional) Session token for MinIO.
          - "minio_secure": bool, default False
                Whether to use a secure connection to MinIO.
          - "minio_region": Optional[str]
                (Optional) Region for the MinIO service.
          - "milvus_address": str, default "milvus"
                Address of the Milvus service.
          - "milvus_uri": str, default "http://milvus:19530"
                URI for Milvus.
          - "milvus_host": str, default "milvus"
                Host for Milvus.
          - "milvus_port": int, default 19530
                Port for Milvus.
          - "collection_name": str, default "nv_ingest_collection"
                Name of the Milvus collection from which to retrieve the schema.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with updated "metadata" columns containing the uploaded embedding URL
        (or bucket path).

    Raises
    ------
    Exception
        Propagates any exception encountered during the upload process, wrapping it with additional context.
    """
    try:
        # Retrieve connection parameters for MinIO
        minio_access_key: Optional[str] = task_config.get("minio_access_key")
        minio_secret_key: Optional[str] = task_config.get("minio_secret_key")
        minio_endpoint: str = task_config.get("minio_endpoint", _DEFAULT_ENDPOINT)
        minio_bucket_name: str = task_config.get("minio_bucket_name", _DEFAULT_BUCKET_NAME)
        minio_bucket_path: str = task_config.get("minio_bucket_path", "embeddings")

        # Retrieve connection parameters for Milvus
        milvus_address: str = task_config.get("milvus_address", "milvus")
        milvus_uri: str = task_config.get("milvus_uri", "http://milvus:19530")
        milvus_host: str = task_config.get("milvus_host", "milvus")
        milvus_port: int = task_config.get("milvus_port", 19530)
        milvus_collection_name: str = task_config.get("collection_name", "nv_ingest_collection")

        # Initialize MinIO client
        client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            session_token=task_config.get("minio_session_token"),
            secure=task_config.get("minio_secure", False),
            region=task_config.get("minio_region"),
        )

        # Connect to Milvus and retrieve collection schema
        connections.connect(
            address=milvus_address,
            uri=f"{milvus_uri}:{milvus_port}",
            host=milvus_host,
            port=milvus_port,
        )
        schema = Collection(milvus_collection_name).schema

        # Ensure bucket exists
        if not client.bucket_exists(minio_bucket_name):
            client.make_bucket(minio_bucket_name)
            logger.debug("Created bucket %s", minio_bucket_name)
        else:
            logger.debug("Bucket %s already exists", minio_bucket_name)

        # Setup connection parameters for RemoteBulkWriter
        conn = RemoteBulkWriter.ConnectParam(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            bucket_name=minio_bucket_name,
            secure=False,
        )
        writer = RemoteBulkWriter(
            schema=schema,
            remote_path=minio_bucket_path,
            connect_param=conn,
            file_type=BulkFileType.PARQUET,
        )

        # Process each row in the DataFrame
        for idx, row in df_store_ledger.iterrows():
            metadata: Dict[str, Any] = row["metadata"].copy()
            # Update embedding metadata with the bucket path
            metadata["embedding_metadata"] = {"uploaded_embedding_url": minio_bucket_path}

            doc_type = row["document_type"]
            content_replace: bool = doc_type in [ContentTypeEnum.IMAGE, ContentTypeEnum.STRUCTURED]
            location: str = metadata["source_metadata"]["source_location"]
            content = metadata["content"]

            # If an embedding exists, update metadata and append the row for upload
            if metadata.get("embedding") is not None:
                logger.error(f"row type: {doc_type} -  {location} -  {len(content)}")
                df_store_ledger.at[idx, "metadata"] = metadata

                writer.append_row(
                    {
                        "text": location if content_replace else content,
                        "source": metadata["source_metadata"],
                        "content_metadata": metadata["content_metadata"],
                        "vector": metadata["embedding"],
                    }
                )

        writer.commit()
        return df_store_ledger

    except Exception as e:
        err_msg = f"upload_embeddings: Error uploading embeddings. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e


def store_text_embeddings_internal(
    df_store_ledger: pd.DataFrame,
    task_config: Union[BaseModel, Dict[str, Any]],
    store_config: EmbeddingStorageSchema,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Stores embeddings by uploading content from a DataFrame to MinIO.

    This function prepares the necessary parameters for the upload based on the task configuration,
    invokes the upload routine, and returns the updated DataFrame.

    Parameters
    ----------
    df_store_ledger : pd.DataFrame
        DataFrame containing the data whose embeddings need to be stored.
    task_config : Union[BaseModel, Dict[str, Any]]
        Task configuration. If it is a Pydantic model, it will be converted to a dictionary.
    store_config : Dict[str, Any]
        Configuration parameters for storage (not directly used in the current implementation).
    execution_trace_log : Optional[Dict[str, Any]], default=None
        Optional dictionary for trace logging information.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after embeddings have been uploaded and metadata updated.

    Raises
    ------
    Exception
        If any error occurs during the storage process, it is logged and re-raised with additional context.
    """

    _ = store_config  # Unused
    _ = execution_trace_log  # Unused

    try:
        # Convert Pydantic model to dict if necessary
        if isinstance(task_config, BaseModel):
            task_config = task_config.model_dump()

        # Set content types for embeddings and update params
        content_types = {ContentTypeEnum.EMBEDDING: True}
        params: Dict[str, Any] = task_config.get("params", {})
        params["content_types"] = content_types

        # Perform the upload of embeddings
        df_store_ledger = _upload_text_embeddings(df_store_ledger, params)

        result, execution_trace_log = df_store_ledger, {}
        _ = execution_trace_log  # Unused

        return result

    except Exception as e:
        err_msg = f"_store_embeddings: Failed to store embeddings: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
