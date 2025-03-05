# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Optional

import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.store.embed_text_upload import store_text_embeddings_internal
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler


@unified_exception_handler
def store_embeddings(
    *,
    df_ledger: pd.DataFrame,
    milvus_address: Optional[str] = None,
    milvus_uri: Optional[str] = None,
    milvus_host: Optional[str] = None,
    milvus_port: Optional[int] = None,
    milvus_collection_name: Optional[str] = None,
    minio_access_key: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
    minio_session_token: Optional[str] = None,
    minio_endpoint: Optional[str] = None,
    minio_bucket_name: Optional[str] = None,
    minio_bucket_path: Optional[str] = None,
    minio_secure: Optional[bool] = None,
    minio_region: Optional[str] = None,
) -> pd.DataFrame:
    """
    Stores embeddings by configuring task parameters and invoking the internal storage routine.

    If any of the connection or configuration parameters are None, they will be omitted from the task
    configuration, allowing default values defined in the storage schema to be used.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing the data whose embeddings need to be stored.
    milvus_address : Optional[str], default=None
        The address of the Milvus service.
    milvus_uri : Optional[str], default=None
        The URI for the Milvus service.
    milvus_host : Optional[str], default=None
        The host for the Milvus service.
    milvus_port : Optional[int], default=None
        The port for the Milvus service.
    milvus_collection_name : Optional[str], default=None
        The name of the Milvus collection.
    minio_access_key : Optional[str], default=None
        The access key for MinIO.
    minio_secret_key : Optional[str], default=None
        The secret key for MinIO.
    minio_session_token : Optional[str], default=None
        The session token for MinIO.
    minio_endpoint : Optional[str], default=None
        The endpoint URL for MinIO.
    minio_bucket_name : Optional[str], default=None
        The name of the MinIO bucket.
    minio_bucket_path : Optional[str], default=None
        The bucket path where embeddings will be stored.
    minio_secure : Optional[bool], default=None
        Whether to use a secure connection to MinIO.
    minio_region : Optional[str], default=None
        The region of the MinIO service.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after embeddings have been stored.

    Raises
    ------
    Exception
        Propagates any exception raised during the storage process, wrapped with additional context.
    """
    params: Dict[str, Any] = {
        "milvus_address": milvus_address,
        "milvus_collection_name": milvus_collection_name,
        "milvus_host": milvus_host,
        "milvus_port": milvus_port,
        "milvus_uri": milvus_uri,
        "minio_access_key": minio_access_key,
        "minio_bucket_name": minio_bucket_name,
        "minio_bucket_path": minio_bucket_path,
        "minio_endpoint": minio_endpoint,
        "minio_region": minio_region,
        "minio_secret_key": minio_secret_key,
        "minio_secure": minio_secure,
        "minio_session_token": minio_session_token,
    }
    # Remove keys with None values so that default values in the storage schema are used.
    filtered_params = {key: value for key, value in params.items() if value is not None}
    task_config: Dict[str, Any] = {"params": filtered_params}

    store_config = EmbeddingStorageSchema()

    result, _ = store_text_embeddings_internal(
        df_ledger,
        task_config=task_config,
        store_config=store_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def store_images_to_minio(
    *,
    df_ledger: pd.DataFrame,
    store_structured: bool = True,
    store_unstructured: bool = False,
    minio_access_key: Optional[str] = None,
    minio_bucket_name: Optional[str] = None,
    minio_endpoint: Optional[str] = None,
    minio_region: Optional[str] = None,
    minio_secret_key: Optional[str] = None,
    minio_secure: bool = False,
    minio_session_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Store images to a Minio storage backend.

    This function prepares a flat configuration dictionary for storing images and structured
    data to a Minio storage system. It determines which content types to store based on the
    provided flags and delegates the storage operation to the internal function
    `store_images_to_minio_internal`.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing ledger information with document metadata.
    store_structured : bool, optional
        Flag indicating whether to store structured content. Defaults to True.
    store_unstructured : bool, optional
        Flag indicating whether to store unstructured image content. Defaults to False.
    minio_access_key : Optional[str], optional
        Access key for authenticating with Minio. Defaults to None.
    minio_bucket_name : Optional[str], optional
        Name of the Minio bucket where images will be stored. Defaults to None.
    minio_endpoint : Optional[str], optional
        Endpoint URL for the Minio service. Defaults to None.
    minio_region : Optional[str], optional
        Region identifier for the Minio service. Defaults to None.
    minio_secret_key : Optional[str], optional
        Secret key for authenticating with Minio. Defaults to None.
    minio_secure : bool, optional
        Whether to use a secure connection (HTTPS) with Minio. Defaults to False.
    minio_session_token : Optional[str], optional
        Session token for temporary credentials with Minio. Defaults to None.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after uploading images if matching objects were found;
        otherwise, the original DataFrame is returned.

    Raises
    ------
    Exception
        Any exceptions raised during the image storage process will be handled by the
        `unified_exception_handler` decorator.

    See Also
    --------
    store_images_to_minio_internal : Internal function that performs the actual image storage.
    _upload_images_to_minio : Function that uploads images to MinIO and updates the ledger metadata.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'document_type': ['IMAGE'],
    ...     'metadata': [{
    ...         'source_metadata': {'source_id': '123'},
    ...         'image_metadata': {'image_type': 'png'},
    ...         'content': 'base64_encoded_content'
    ...     }]
    ... })
    >>> result = store_images_to_minio(
    ...     df_ledger=df,
    ...     minio_access_key='ACCESS_KEY',
    ...     minio_secret_key='SECRET_KEY',
    ...     minio_bucket_name='mybucket'
    ... )
    """
    content_types = {
        ContentTypeEnum.STRUCTURED: store_structured,
        ContentTypeEnum.IMAGE: store_unstructured,
    }

    # Build the task configuration as a flat dictionary, matching the internal function's expectations.
    task_config = {
        "access_key": minio_access_key,
        "bucket_name": minio_bucket_name,
        "content_types": content_types,
        "endpoint": minio_endpoint,
        "region": minio_region,
        "secret_key": minio_secret_key,
        "secure": minio_secure,
        "session_token": minio_session_token,
    }

    storage_config = {}

    result, _ = store_images_to_minio_internal(
        df_storage_ledger=df_ledger,
        task_config=task_config,
        storage_config=storage_config,
        execution_trace_log=None,
    )

    return result
